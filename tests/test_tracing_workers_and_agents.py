import os
import unittest
from unittest.mock import patch

from src.lib.tracing import LocalTraceQueue, TraceEventType, TraceTracker
from src.lib.tracing import workers as tracing_workers


class WorkerTracingTests(unittest.TestCase):
    def setUp(self):
        self.original_tracer = tracing_workers.tracer
        self.tracker = TraceTracker(queue=LocalTraceQueue(), enabled=True)
        tracing_workers.tracer = self.tracker

    def tearDown(self):
        tracing_workers.tracer = self.original_tracer
        tracing_workers._task_context.runs = {}

    def test_worker_signals_record_start_sla_and_failure(self):
        tracing_workers._trace_task_prerun(
            sender=type("Task", (), {"name": "demo.task"})(),
            task_id="task-1",
            args=({"brain_id": "tenant-1"},),
            kwargs={},
        )
        stack = tracing_workers._task_context.runs["task-1"]
        started_at, token = stack[-1]
        stack[-1] = (started_at - 2, token)
        tracing_workers._trace_task_failure(
            sender=type("Task", (), {"name": "demo.task"})(),
            task_id="task-1",
            exception=RuntimeError("boom"),
            args=({"brain_id": "tenant-1"},),
            kwargs={},
        )
        tracing_workers._trace_task_postrun(
            sender=type("Task", (), {"name": "demo.task"})(),
            task_id="task-1",
            args=({"brain_id": "tenant-1"},),
            kwargs={},
            state="FAILURE",
        )

        events = self.tracker.queue.drain()
        self.assertEqual(events[0].event_type, TraceEventType.LATENCY)
        self.assertEqual(events[0].tenant_id, "tenant-1")
        self.assertEqual(events[1].event_type, TraceEventType.EXCEPTION)
        self.assertEqual(events[1].error_type, "RuntimeError")
        self.assertEqual(events[2].event_type, TraceEventType.SLA_BREACH)
        self.assertEqual(events[2].operation, "demo.task")


class AgentLoopTracingTests(unittest.TestCase):
    def test_agent_loop_records_model_exception(self):
        from src.core.agents.core import invoke_loop

        tracker = TraceTracker(queue=LocalTraceQueue(), enabled=True)

        class FailingModel:
            def invoke(self, *_args, **_kwargs):
                raise RuntimeError("model down")

        class Agent:
            class Tool:
                name = "demo_tool"
                description = "Demo tool"
                args_schema = {}

            tools = [Tool()]
            output_schema = None
            model = FailingModel()
            thinking = False
            _tools_bound = False
            system_prompt = "system"
            debug = False

            def _model_requires_thought_signatures(self):
                return False

        with patch.object(invoke_loop, "tracer", tracker):
            with self.assertRaises(RuntimeError):
                invoke_loop.run_invoke_loop(
                    Agent(),
                    [{"role": "user", "content": "hello"}],
                    {"metadata": {"brain_id": "tenant-2", "agent": "test-agent"}},
                )

        events = tracker.queue.drain()
        event_types = [event.event_type for event in events]
        self.assertIn(TraceEventType.EXCEPTION, event_types)
        self.assertTrue(
            any(event.name == "agent.model.invoke.failed" for event in events)
        )
        self.assertTrue(any(event.name == "agent.invoke_loop" for event in events))
        self.assertTrue(all(
            event.tenant_id == "tenant-2" for event in events if event.tenant_id
        ))

    def test_agent_loop_records_outer_loop_threshold(self):
        from src.core.agents.core import invoke_loop

        tracker = TraceTracker(queue=LocalTraceQueue(), enabled=True)

        class SimpleModel:
            def invoke(self, *_args, **_kwargs):
                return {"content": "done"}

        class Agent:
            tools = []
            output_schema = None
            model = SimpleModel()
            thinking = False
            _tools_bound = False
            system_prompt = "system"
            debug = False
            _get_effective_output_schema = None

            def _model_requires_thought_signatures(self):
                return False

        with patch.object(invoke_loop, "tracer", tracker):
            with patch.dict(os.environ, {"TRACE_AGENT_LOOP_ITERATIONS": "1"}):
                result = invoke_loop.run_invoke_loop(
                    Agent(),
                    [{"role": "user", "content": "hello"}],
                    {"metadata": {"brain_id": "tenant-3", "agent": "test-agent"}},
                )

        self.assertGreaterEqual(len(result["messages"]), 1)
        events = tracker.queue.drain()
        self.assertTrue(
            any(
                event.event_type == TraceEventType.EXPENSIVE_LOOP
                and event.name == "agent.invoke_loop.outer_loop"
                for event in events
            )
        )


if __name__ == "__main__":
    unittest.main()


def test_duplicate_delivery_keeps_pipeline_and_trace_context_on_each_thread():
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier
    from types import SimpleNamespace

    from celery import signals

    from src.core.saving.ingest_manifest import install_task_context
    from src.core.saving.pipeline_runtime import _context

    handlers = []

    def capture(**_):
        def register(handler):
            handlers.append(handler)
            return handler
        return register

    with patch.object(signals.task_prerun, "connect", side_effect=capture), patch.object(
        signals.task_postrun, "connect", side_effect=capture
    ):
        install_task_context()
    begin, end = handlers
    barrier = Barrier(2, timeout=5)

    def delivery(brain_id):
        headers = {"pipeline_operation": {"brain_id": brain_id}}
        task = SimpleNamespace(request=SimpleNamespace(headers=headers))
        tracing_workers._trace_task_prerun(task_id="duplicate", args=({"brain_id": brain_id},))
        begin(task_id="duplicate", task=task)
        barrier.wait()
        assert _context.get()["brain_id"] == brain_id
        end(task_id="duplicate")
        tracing_workers._trace_task_postrun(task_id="duplicate")
        assert _context.get() is None

    with ThreadPoolExecutor(max_workers=2) as pool:
        deliveries = [pool.submit(delivery, brain) for brain in ("brain-a", "brain-b")]
        for future in deliveries:
            future.result()
