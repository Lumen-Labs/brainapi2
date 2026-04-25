import asyncio
import unittest

from starlette.applications import Starlette
from starlette.responses import JSONResponse, PlainTextResponse
from starlette.middleware import Middleware
from starlette.routing import Route
from starlette.testclient import TestClient

from src.lib.tracing import (
    LocalTraceQueue,
    TraceEvent,
    TraceEventType,
    TraceSeverity,
    TraceSubscriber,
    TraceTracker,
)
from src.lib.tracing.middleware import TraceMiddleware
from src.lib.tracing import middleware as tracing_middleware


class MemorySubscriber(TraceSubscriber):
    def __init__(self):
        self.events = []

    def handle(self, event: TraceEvent) -> None:
        self.events.append(event)


class TraceTrackerTests(unittest.TestCase):
    def test_publish_records_event_with_tenant_and_trace_context(self):
        tracker = TraceTracker(queue=LocalTraceQueue(), enabled=True)
        tokens = tracker.set_context(trace_id="trace-1", tenant_id="tenant-1")
        try:
            event = tracker.error(
                "operation.failed",
                service="api",
                operation="GET /demo",
                message="failed",
            )
        finally:
            tracker.reset_context(tokens)

        self.assertIsNotNone(event)
        drained = tracker.queue.drain()
        self.assertEqual(len(drained), 1)
        self.assertEqual(drained[0].event_type, TraceEventType.ERROR)
        self.assertEqual(drained[0].severity, TraceSeverity.ERROR)
        self.assertEqual(drained[0].tenant_id, "tenant-1")
        self.assertEqual(drained[0].trace_id, "trace-1")
        self.assertEqual(drained[0].service, "api")
        self.assertEqual(drained[0].operation, "GET /demo")

    def test_span_records_exception_and_sla_breach(self):
        tracker = TraceTracker(
            queue=LocalTraceQueue(),
            enabled=True,
            slow_operation_ms=0,
        )

        with self.assertRaises(ValueError):
            with tracker.span("failing-span", tenant_id="brain1"):
                raise ValueError("boom")

        events = tracker.queue.drain()
        self.assertEqual([event.event_type for event in events], [
            TraceEventType.EXCEPTION,
            TraceEventType.SLA_BREACH,
        ])
        self.assertEqual(events[0].error_type, "ValueError")
        self.assertIn("boom", events[0].message)
        self.assertEqual(events[0].tenant_id, "brain1")
        self.assertIsNotNone(events[0].stack_trace)

    def test_track_loop_records_expensive_loop_when_threshold_is_met(self):
        tracker = TraceTracker(
            queue=LocalTraceQueue(),
            enabled=True,
            expensive_loop_iterations=3,
        )

        self.assertEqual(list(tracker.track_loop("loop", range(3))), [0, 1, 2])

        events = tracker.queue.drain()
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].event_type, TraceEventType.EXPENSIVE_LOOP)
        self.assertEqual(events[0].metadata["iterations"], 3)

    def test_dispatch_once_sends_event_to_subscriber(self):
        async def run_test():
            tracker = TraceTracker(queue=LocalTraceQueue(), enabled=True)
            subscriber = MemorySubscriber()
            tracker.subscribe(subscriber)
            event = tracker.error("error")

            dispatched = await tracker.dispatch_once()

            self.assertIs(dispatched, event)
            self.assertEqual(subscriber.events, [event])

        asyncio.run(run_test())


class TraceMiddlewareTests(unittest.TestCase):
    def setUp(self):
        self.original_tracer = tracing_middleware.tracer
        self.tracker = TraceTracker(queue=LocalTraceQueue(), enabled=True)
        tracing_middleware.tracer = self.tracker

    def tearDown(self):
        tracing_middleware.tracer = self.original_tracer

    def test_middleware_records_server_errors_and_tenant_id(self):
        async def error_route(_request):
            return JSONResponse({"detail": "bad"}, status_code=503)

        app = Starlette(
            routes=[Route("/bad", error_route)],
            middleware=[
                Middleware(
                    TraceMiddleware,
                    service_name="test-api",
                    slow_request_ms=100000,
                )
            ],
        )

        response = TestClient(app).get("/bad", headers={"X-Brain-ID": "tenant-a"})

        self.assertEqual(response.status_code, 503)
        events = self.tracker.queue.drain()
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].event_type, TraceEventType.ERROR)
        self.assertEqual(events[0].tenant_id, "tenant-a")
        self.assertEqual(events[0].service, "test-api")
        self.assertEqual(events[0].status_code, 503)

    def test_middleware_records_sla_breach(self):
        async def ok_route(_request):
            return PlainTextResponse("ok")

        app = Starlette(
            routes=[Route("/ok", ok_route)],
            middleware=[
                Middleware(
                    TraceMiddleware,
                    service_name="test-api",
                    slow_request_ms=0,
                )
            ],
        )

        response = TestClient(app).get("/ok?brain_id=tenant-b")

        self.assertEqual(response.status_code, 200)
        events = self.tracker.queue.drain()
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].event_type, TraceEventType.SLA_BREACH)
        self.assertEqual(events[0].tenant_id, "tenant-b")
        self.assertEqual(events[0].operation, "GET /ok")


if __name__ == "__main__":
    unittest.main()
