from dataclasses import dataclass, field
from enum import Enum
import time
from typing import Any


class TraceEventType(str, Enum):
    ERROR = "error"
    EXCEPTION = "exception"
    DOWNTIME = "downtime"
    EXPENSIVE_LOOP = "expensive_loop"
    SLA_BREACH = "sla_breach"
    LATENCY = "latency"


class TraceSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass(frozen=True)
class TraceEvent:
    event_type: TraceEventType
    name: str
    severity: TraceSeverity = TraceSeverity.INFO
    service: str | None = None
    operation: str | None = None
    tenant_id: str | None = None
    trace_id: str | None = None
    duration_ms: float | None = None
    threshold_ms: float | None = None
    status_code: int | None = None
    error_type: str | None = None
    message: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    stack_trace: str | None = None
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_type": self.event_type.value,
            "name": self.name,
            "severity": self.severity.value,
            "service": self.service,
            "operation": self.operation,
            "tenant_id": self.tenant_id,
            "trace_id": self.trace_id,
            "duration_ms": self.duration_ms,
            "threshold_ms": self.threshold_ms,
            "status_code": self.status_code,
            "error_type": self.error_type,
            "message": self.message,
            "metadata": dict(self.metadata),
            "stack_trace": self.stack_trace,
            "created_at": self.created_at,
        }
