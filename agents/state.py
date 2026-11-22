"""
agents/state.py
================
Structured schemas defining IncidentState, the shared object that flows
through every node of the LangGraph workflow. Using a TypedDict (rather than
a pydantic model) as the graph state is a LangGraph convention that keeps
partial-update dict merging cheap; nested fields use Pydantic models for
validation.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional, TypedDict

from pydantic import BaseModel, Field

from config import RiskLevel


class IncidentStatus(str, Enum):
    TRIAGING = "triaging"
    INVESTIGATING = "investigating"
    HYPOTHESIZING = "hypothesizing"
    AWAITING_APPROVAL = "awaiting_approval"
    REMEDIATING = "remediating"
    VERIFYING = "verifying"
    RESOLVED = "resolved"
    ESCALATED = "escalated"
    FAILED = "failed"


class AlertPayload(BaseModel):
    """Normalized representation of a single Alertmanager alert."""
    alertname: str
    service: str
    namespace: str = "default"
    severity: str = "warning"
    summary: Optional[str] = None
    description: Optional[str] = None
    starts_at: Optional[str] = None
    labels: Dict[str, str] = Field(default_factory=dict)
    annotations: Dict[str, str] = Field(default_factory=dict)


class TelemetrySnapshot(BaseModel):
    logs: List[str] = Field(default_factory=list)
    metrics: Dict[str, Any] = Field(default_factory=dict)
    collection_errors: List[str] = Field(default_factory=list)
    window_minutes: int = 15


class RunbookMatch(BaseModel):
    title: str
    content: str
    similarity_score: float


class RemediationAction(BaseModel):
    action_type: str  # "restart_pod" | "rollback_deployment" | "scale_replicas" | "noop"
    target: str
    namespace: str
    params: Dict[str, Any] = Field(default_factory=dict)
    risk_level: RiskLevel
    justification: str


class RemediationResult(BaseModel):
    executed: bool
    success: bool
    message: str
    executed_at: Optional[str] = None
    dry_run: bool = True


class IncidentState(TypedDict, total=False):
    incident_id: str
    alert: AlertPayload
    service: str
    namespace: str
    time_window_minutes: int

    telemetry: TelemetrySnapshot
    runbook_matches: List[RunbookMatch]

    root_cause_hypothesis: str
    confidence: float

    remediation_plan: Optional[RemediationAction]
    requires_human_approval: bool
    human_approved: Optional[bool]

    remediation_result: Optional[RemediationResult]
    verification_status: str  # "pending" | "verified_recovered" | "verification_failed" | "not_applicable"
    verification_attempts: int

    history: List[str]
    status: str
    error: Optional[str]


def new_incident_state(incident_id: str, alert: AlertPayload) -> IncidentState:
    """Factory for a fresh IncidentState from an inbound alert."""
    return IncidentState(
        incident_id=incident_id,
        alert=alert,
        service=alert.service,
        namespace=alert.namespace,
        time_window_minutes=15,
        telemetry=TelemetrySnapshot(),
        runbook_matches=[],
        root_cause_hypothesis="",
        confidence=0.0,
        remediation_plan=None,
        requires_human_approval=False,
        human_approved=None,
        remediation_result=None,
        verification_status="pending",
        verification_attempts=0,
        history=[f"Incident created from alert '{alert.alertname}' on service '{alert.service}'."],
        status=IncidentStatus.TRIAGING.value,
        error=None,
    )
