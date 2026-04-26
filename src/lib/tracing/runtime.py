import atexit
import asyncio
import os
import resource
import socket
import sys
import threading
import time
import traceback
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Callable

from src.lib.tracing.events import TraceEventType, TraceSeverity
from src.lib.tracing.tracker import tracer


@dataclass(frozen=True)
class HealthProbe:
    name: str
    host: str
    port: int
    timeout_seconds: float = 1.0


class RuntimeMonitor:
    def __init__(
        self,
        *,
        service_name: str,
        heartbeat_interval_seconds: float | None = None,
        health_interval_seconds: float | None = None,
        resource_interval_seconds: float | None = None,
        probes: list[HealthProbe] | None = None,
        resource_sampler: Callable[[], dict] | None = None,
    ):
        self.service_name = service_name
        self.heartbeat_interval_seconds = _interval(
            heartbeat_interval_seconds, "TRACE_HEARTBEAT_INTERVAL_SECONDS", 30
        )
        self.health_interval_seconds = _interval(
            health_interval_seconds, "TRACE_HEALTH_INTERVAL_SECONDS", 30
        )
        self.resource_interval_seconds = _interval(
            resource_interval_seconds, "TRACE_RESOURCE_INTERVAL_SECONDS", 30
        )
        self.probes = probes or default_health_probes()
        self.resource_sampler = resource_sampler or sample_process_resources
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._hooks_installed = False

    def start(self) -> None:
        self.install_exception_hooks()
        tracer.publish(
            TraceEventType.PROCESS,
            "process.started",
            service=self.service_name,
            operation="runtime.start",
            metadata=_process_metadata(),
        )
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(
            target=self._run,
            name=f"{self.service_name}-runtime-monitor",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2)
        tracer.publish(
            TraceEventType.PROCESS,
            "process.stopped",
            service=self.service_name,
            operation="runtime.stop",
            metadata=_process_metadata(),
        )

    def install_exception_hooks(self) -> None:
        if self._hooks_installed:
            return
        previous_sys_hook = sys.excepthook
        previous_thread_hook = getattr(threading, "excepthook", None)

        def sys_hook(exc_type, exc_value, exc_traceback):
            self.record_unhandled_exception(exc_value, exc_traceback, "sys.excepthook")
            previous_sys_hook(exc_type, exc_value, exc_traceback)

        def thread_hook(args):
            self.record_unhandled_exception(
                args.exc_value,
                args.exc_traceback,
                f"threading.excepthook:{getattr(args.thread, 'name', 'unknown')}",
            )
            if previous_thread_hook:
                previous_thread_hook(args)

        sys.excepthook = sys_hook
        if previous_thread_hook:
            threading.excepthook = thread_hook
        self._hooks_installed = True

    def install_asyncio_exception_handler(self, loop: asyncio.AbstractEventLoop) -> None:
        previous_handler = loop.get_exception_handler()

        def handler(event_loop, context):
            exc = context.get("exception")
            if exc:
                self.record_unhandled_exception(exc, exc.__traceback__, "asyncio")
            else:
                tracer.error(
                    "runtime.asyncio.unhandled_error",
                    service=self.service_name,
                    operation="asyncio",
                    severity=TraceSeverity.ERROR,
                    message=context.get("message"),
                    metadata={k: str(v) for k, v in context.items() if k != "exception"},
                )
            if previous_handler:
                previous_handler(event_loop, context)
            else:
                event_loop.default_exception_handler(context)

        loop.set_exception_handler(handler)

    def record_unhandled_exception(
        self,
        exc: BaseException,
        exc_traceback,
        operation: str,
    ) -> None:
        tracer.publish(
            TraceEventType.EXCEPTION,
            "runtime.unhandled_exception",
            service=self.service_name,
            operation=operation,
            severity=TraceSeverity.CRITICAL,
            message=str(exc),
            exception=exc,
            metadata={
                **_process_metadata(),
                "traceback": "".join(
                    traceback.format_exception(type(exc), exc, exc_traceback)
                ),
            },
        )

    def heartbeat(self) -> None:
        tracer.publish(
            TraceEventType.HEARTBEAT,
            "runtime.heartbeat",
            service=self.service_name,
            operation="heartbeat",
            metadata=_process_metadata(),
        )

    def sample_resources(self) -> None:
        tracer.publish(
            TraceEventType.RESOURCE_SAMPLE,
            "runtime.resources",
            service=self.service_name,
            operation="resource_sampler",
            metadata=self.resource_sampler(),
        )

    def check_health(self) -> None:
        for probe in self.probes:
            started_at = time.perf_counter()
            ok = probe_tcp(probe)
            duration_ms = (time.perf_counter() - started_at) * 1000
            metadata = {
                "probe": probe.name,
                "host": probe.host,
                "port": probe.port,
                "ok": ok,
            }
            if ok:
                tracer.publish(
                    TraceEventType.HEALTH_CHECK,
                    f"health.{probe.name}",
                    service=self.service_name,
                    operation="health_check",
                    duration_ms=duration_ms,
                    metadata=metadata,
                )
            else:
                tracer.downtime(
                    f"health.{probe.name}.down",
                    duration_ms,
                    service=self.service_name,
                    operation="health_check",
                    metadata=metadata,
                )

    def _run(self) -> None:
        last_heartbeat = last_health = last_resource = 0.0
        while not self._stop_event.is_set():
            now = time.monotonic()
            if _due(now, last_heartbeat, self.heartbeat_interval_seconds):
                self.heartbeat()
                last_heartbeat = now
            if _due(now, last_health, self.health_interval_seconds):
                self.check_health()
                last_health = now
            if _due(now, last_resource, self.resource_interval_seconds):
                self.sample_resources()
                last_resource = now
            self._stop_event.wait(1)


_monitors: dict[str, RuntimeMonitor] = {}


def start_runtime_monitoring(service_name: str, **kwargs) -> RuntimeMonitor:
    monitor = _monitors.get(service_name)
    if monitor is None:
        monitor = RuntimeMonitor(service_name=service_name, **kwargs)
        _monitors[service_name] = monitor
        atexit.register(monitor.stop)
    monitor.start()
    return monitor


def stop_runtime_monitoring(service_name: str) -> None:
    monitor = _monitors.get(service_name)
    if monitor:
        monitor.stop()


def runtime_tracing_lifespan(service_name: str, nested_lifespan=None):
    @asynccontextmanager
    async def lifespan(app):
        start_runtime_monitoring(service_name)
        if nested_lifespan is None:
            try:
                yield
            finally:
                stop_runtime_monitoring(service_name)
            return
        async with nested_lifespan(app):
            try:
                yield
            finally:
                stop_runtime_monitoring(service_name)

    return lifespan


def default_health_probes() -> list[HealthProbe]:
    return [
        _probe_from_env("redis", "REDIS_HOST", "REDIS_PORT"),
        _probe_from_env("mongo", "MONGO_HOST", "MONGO_PORT"),
        _probe_from_env("neo4j", "NEO4J_HOST", "NEO4J_PORT"),
        _probe_from_env("milvus", "MILVUS_HOST", "MILVUS_PORT"),
        _probe_from_env("rabbitmq", "RABBITMQ_HOST", "RABBITMQ_PORT"),
    ]


def _probe_from_env(name: str, host_var: str, port_var: str) -> HealthProbe:
    defaults = {
        "redis": ("localhost", 6379),
        "mongo": ("localhost", 27017),
        "neo4j": ("localhost", 7687),
        "milvus": ("localhost", 19530),
        "rabbitmq": ("localhost", 5672),
    }
    default_host, default_port = defaults[name]
    return HealthProbe(
        name=name,
        host=os.getenv(host_var, default_host),
        port=int(os.getenv(port_var, str(default_port))),
        timeout_seconds=float(os.getenv("TRACE_HEALTH_TIMEOUT_SECONDS", "1")),
    )


def probe_tcp(probe: HealthProbe) -> bool:
    try:
        with socket.create_connection(
            (probe.host, probe.port), timeout=probe.timeout_seconds
        ):
            return True
    except OSError:
        return False


def sample_process_resources() -> dict:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return {
        **_process_metadata(),
        "rss_kb": usage.ru_maxrss,
        "user_cpu_seconds": usage.ru_utime,
        "system_cpu_seconds": usage.ru_stime,
        "threads": threading.active_count(),
        "load_avg": os.getloadavg() if hasattr(os, "getloadavg") else None,
        **_proc_status(),
    }


def _proc_status() -> dict:
    status_path = "/proc/self/status"
    if not os.path.exists(status_path):
        return {}
    keys = {
        "VmRSS": "vm_rss",
        "VmSize": "vm_size",
        "Threads": "proc_threads",
    }
    values = {}
    try:
        with open(status_path, encoding="utf-8") as status_file:
            for line in status_file:
                key, _, value = line.partition(":")
                if key in keys:
                    values[keys[key]] = value.strip()
    except OSError:
        return {}
    return values


def _process_metadata() -> dict:
    return {
        "pid": os.getpid(),
        "process": os.path.basename(sys.argv[0] or "python"),
    }


def _interval(value: float | None, env_name: str, default: float) -> float:
    return value if value is not None else float(os.getenv(env_name, str(default)))


def _due(now: float, last: float, interval: float) -> bool:
    return interval >= 0 and now - last >= interval
