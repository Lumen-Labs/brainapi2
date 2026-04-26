import asyncio
import unittest
from unittest.mock import patch

from src.lib.tracing import LocalTraceQueue, TraceEventType, TraceTracker
from src.lib.tracing import runtime as tracing_runtime
from src.lib.tracing.runtime import HealthProbe, RuntimeMonitor


class RuntimeMonitorTests(unittest.TestCase):
    def setUp(self):
        self.original_tracer = tracing_runtime.tracer
        self.tracker = TraceTracker(queue=LocalTraceQueue(), enabled=True)
        tracing_runtime.tracer = self.tracker

    def tearDown(self):
        tracing_runtime.tracer = self.original_tracer
        tracing_runtime._monitors.clear()

    def test_runtime_monitor_records_heartbeat_resources_and_shutdown(self):
        monitor = RuntimeMonitor(
            service_name="test-service",
            probes=[],
            resource_sampler=lambda: {"rss_kb": 12},
        )

        monitor.heartbeat()
        monitor.sample_resources()
        monitor.stop()

        events = self.tracker.queue.drain()
        self.assertEqual(events[0].event_type, TraceEventType.HEARTBEAT)
        self.assertEqual(events[1].event_type, TraceEventType.RESOURCE_SAMPLE)
        self.assertEqual(events[1].metadata["rss_kb"], 12)
        self.assertEqual(events[2].event_type, TraceEventType.PROCESS)
        self.assertEqual(events[2].name, "process.stopped")

    def test_health_check_records_success_and_downtime(self):
        monitor = RuntimeMonitor(
            service_name="test-service",
            probes=[
                HealthProbe("up", "127.0.0.1", 1),
                HealthProbe("down", "127.0.0.1", 2),
            ],
        )

        with patch.object(tracing_runtime, "probe_tcp", side_effect=[True, False]):
            monitor.check_health()

        events = self.tracker.queue.drain()
        self.assertEqual(events[0].event_type, TraceEventType.HEALTH_CHECK)
        self.assertEqual(events[0].name, "health.up")
        self.assertEqual(events[1].event_type, TraceEventType.DOWNTIME)
        self.assertEqual(events[1].name, "health.down.down")

    def test_unhandled_exception_hook_records_exception(self):
        monitor = RuntimeMonitor(service_name="test-service", probes=[])
        exc = RuntimeError("boom")

        monitor.record_unhandled_exception(exc, exc.__traceback__, "unit-test")

        events = self.tracker.queue.drain()
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].event_type, TraceEventType.EXCEPTION)
        self.assertEqual(events[0].severity.value, "critical")
        self.assertEqual(events[0].operation, "unit-test")

    def test_asyncio_exception_handler_records_context(self):
        async def run_test():
            loop = asyncio.get_running_loop()
            previous_handler = loop.get_exception_handler()
            loop.set_exception_handler(lambda _loop, _context: None)
            monitor = RuntimeMonitor(service_name="test-service", probes=[])
            monitor.install_asyncio_exception_handler(loop)
            try:
                loop.call_exception_handler({"message": "async problem"})
            finally:
                loop.set_exception_handler(previous_handler)

        asyncio.run(run_test())

        events = self.tracker.queue.drain()
        self.assertEqual(events[0].event_type, TraceEventType.ERROR)
        self.assertEqual(events[0].name, "runtime.asyncio.unhandled_error")


if __name__ == "__main__":
    unittest.main()
