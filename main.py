"""
main.py
=======
FastAPI server exposing PROJECT Sentinel's entrypoints:

  POST /webhook/alert          Ingest a Prometheus Alertmanager webhook payload
  POST /incident/{id}/approve  Human confirmation for a HIGH-risk remediation
  GET  /incident/{id}          Fetch current incident state / history
  GET  /incidents              List known incidents
  GET  /healthz                Liveness probe
"""

from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional

from fastapi import BackgroundTasks, FastAPI, HTTPException
from pydantic import BaseModel, Field

from agents.graph import sentinel_graph
from agents.state import AlertPayload, IncidentState, new_incident_state
from config import settings

app = FastAPI(
    title="PROJECT Sentinel",
    description="Autonomous Incident-Response Copilot for Distributed Systems",
    version="0.1.0",
)

# In-memory incident store for the MVP. Swap for Redis/Postgres in production
# so incident state survives process restarts (see README "Productionizing").
INCIDENTS: Dict[str, IncidentState] = {}


# ---------------------------------------------------------------------------
# Alertmanager webhook schema
# https://prometheus.io/docs/alerting/latest/configuration/#webhook_config
# ---------------------------------------------------------------------------
class AlertmanagerAlert(BaseModel):
    status: str
    labels: Dict[str, str] = Field(default_factory=dict)
    annotations: Dict[str, str] = Field(default_factory=dict)
    startsAt: Optional[str] = None
    endsAt: Optional[str] = None
    generatorURL: Optional[str] = None
    fingerprint: Optional[str] = None


class AlertmanagerWebhook(BaseModel):
    version: Optional[str] = None
    groupKey: Optional[str] = None
    status: str
    receiver: Optional[str] = None
    alerts: List[AlertmanagerAlert]


class ApprovalRequest(BaseModel):
    approved: bool
    approver: Optional[str] = None
    comment: Optional[str] = None


def _alert_to_payload(alert: AlertmanagerAlert) -> AlertPayload:
    labels = alert.labels
    annotations = alert.annotations
    return AlertPayload(
        alertname=labels.get("alertname", "UnknownAlert"),
        service=labels.get("service") or labels.get("app") or labels.get("job", "unknown-service"),
        namespace=labels.get("namespace", "default"),
        severity=labels.get("severity", "warning"),
        summary=annotations.get("summary"),
        description=annotations.get("description"),
        starts_at=alert.startsAt,
        labels=labels,
        annotations=annotations,
    )


async def _run_graph(incident_id: str, state: IncidentState) -> None:
    """Background task: invoke the LangGraph workflow and persist the result."""
    config = {"configurable": {"thread_id": incident_id}}
    try:
        final_state = await sentinel_graph.ainvoke(state, config=config)
        INCIDENTS[incident_id] = final_state
    except Exception as exc:  # noqa: BLE001 - top-level safety net for background task
        state["status"] = "failed"
        state["error"] = str(exc)
        state.setdefault("history", []).append(f"FATAL ERROR: {exc}")
        INCIDENTS[incident_id] = state


@app.post("/webhook/alert")
async def webhook_alert(payload: AlertmanagerWebhook, background_tasks: BackgroundTasks) -> Dict[str, Any]:
    """
    Entry point for Prometheus Alertmanager. Only 'firing' alerts spawn a new
    incident investigation; 'resolved' notifications are logged and ignored.
    """
    if payload.status != "firing":
        return {"status": "ignored", "reason": f"Alertmanager status was '{payload.status}', not 'firing'."}

    if not payload.alerts:
        raise HTTPException(status_code=400, detail="Webhook payload contained no alerts.")

    created_incidents = []
    for alert in payload.alerts:
        alert_payload = _alert_to_payload(alert)
        incident_id = str(uuid.uuid4())
        state = new_incident_state(incident_id, alert_payload)
        INCIDENTS[incident_id] = state
        background_tasks.add_task(_run_graph, incident_id, state)
        created_incidents.append(incident_id)

    return {"status": "accepted", "incident_ids": created_incidents}


@app.post("/incident/{incident_id}/approve")
async def approve_incident(
    incident_id: str, approval: ApprovalRequest, background_tasks: BackgroundTasks
) -> Dict[str, Any]:
    """
    Human-in-the-loop confirmation endpoint. Resumes a paused HIGH-risk
    incident from the LangGraph checkpoint using the same thread_id.
    """
    if incident_id not in INCIDENTS:
        raise HTTPException(status_code=404, detail="Unknown incident_id.")

    state = INCIDENTS[incident_id]
    if not state.get("requires_human_approval"):
        raise HTTPException(status_code=409, detail="This incident is not awaiting approval.")

    state["human_approved"] = approval.approved
    state.setdefault("history", []).append(
        f"Human decision recorded: approved={approval.approved} "
        f"by='{approval.approver or 'unknown'}' comment='{approval.comment or ''}'."
    )

    if not approval.approved:
        state["status"] = "escalated"
        INCIDENTS[incident_id] = state
        return {"status": "rejected", "incident_id": incident_id, "message": "Remediation declined by human operator."}

    background_tasks.add_task(_run_graph, incident_id, state)
    return {"status": "resumed", "incident_id": incident_id}


@app.get("/incident/{incident_id}")
async def get_incident(incident_id: str) -> Dict[str, Any]:
    if incident_id not in INCIDENTS:
        raise HTTPException(status_code=404, detail="Unknown incident_id.")
    state = INCIDENTS[incident_id]
    # Serialize pydantic sub-models for JSON response.
    return _serialize_state(state)


@app.get("/incidents")
async def list_incidents() -> Dict[str, Any]:
    return {
        "count": len(INCIDENTS),
        "incidents": [
            {
                "incident_id": iid,
                "service": s.get("service"),
                "status": s.get("status"),
                "requires_human_approval": s.get("requires_human_approval"),
            }
            for iid, s in INCIDENTS.items()
        ],
    }


@app.get("/healthz")
async def healthz() -> Dict[str, str]:
    return {"status": "ok", "dry_run": str(settings.dry_run)}


def _serialize_state(state: IncidentState) -> Dict[str, Any]:
    out: Dict[str, Any] = dict(state)
    for key, value in list(out.items()):
        if hasattr(value, "model_dump"):
            out[key] = value.model_dump()
        elif isinstance(value, list):
            out[key] = [v.model_dump() if hasattr(v, "model_dump") else v for v in value]
    return out


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host=settings.host, port=settings.port, reload=True)
