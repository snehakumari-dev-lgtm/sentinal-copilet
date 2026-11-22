"""
agents/graph.py
================
The complete LangGraph state machine for PROJECT Sentinel.

Flow:
    triage_agent
        -> telemetry_agent
        -> hypothesis_agent
        -> safety_agent
            -> [HIGH risk, not yet approved]  -> END (paused; resumes via /approve)
            -> [LOW/MEDIUM risk, or approved] -> remediation_executor
        -> remediation_executor
        -> verification_agent
            -> [recovered]                     -> END (resolved)
            -> [not recovered, retries left]   -> telemetry_agent (loop)
            -> [not recovered, retries exhausted] -> END (escalated)

Human-in-the-loop is implemented via LangGraph's checkpointer: the graph is
compiled with a MemorySaver keyed by incident_id (thread_id), so it can be
invoked, pause at the safety gate, and be resumed later from main.py's
/approve endpoint without losing state.
"""

from __future__ import annotations

import json
from typing import Literal

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from config import RiskLevel, settings, resolve_risk
from agents.state import (
    IncidentState,
    IncidentStatus,
    RemediationAction,
    RemediationResult,
)
from agents.runbooks import search_runbooks
from tools.telemetry import gather_service_telemetry, query_prometheus_instant, TelemetryError
from tools.k8s_actions import (
    restart_pod,
    rollback_deployment,
    scale_replicas,
    RemediationBlockedError,
)

_llm = None


def _get_llm() -> ChatAnthropic:
    global _llm
    if _llm is None:
        _llm = ChatAnthropic(
            model=settings.llm_model,
            api_key=settings.anthropic_api_key,
            temperature=0,
            max_tokens=1024,
        )
    return _llm


def _log(state: IncidentState, message: str) -> None:
    state.setdefault("history", []).append(message)


# ---------------------------------------------------------------------------
# Node: Alert Triage Agent
# ---------------------------------------------------------------------------
async def triage_agent(state: IncidentState) -> IncidentState:
    """Extract and normalize the initial failure scope from the raw alert."""
    alert = state["alert"]
    state["status"] = IncidentStatus.TRIAGING.value
    state["service"] = alert.service
    state["namespace"] = alert.namespace

    # Widen the investigation window for critical severity alerts.
    state["time_window_minutes"] = 30 if alert.severity.lower() == "critical" else 15

    _log(
        state,
        f"Triage complete: service='{alert.service}', namespace='{alert.namespace}', "
        f"severity='{alert.severity}', window={state['time_window_minutes']}m.",
    )
    return state


# ---------------------------------------------------------------------------
# Node: Telemetry Agent
# ---------------------------------------------------------------------------
async def telemetry_agent(state: IncidentState) -> IncidentState:
    """Fetch logs and metrics for the affected service/time window."""
    state["status"] = IncidentStatus.INVESTIGATING.value
    bundle = await gather_service_telemetry(
        service=state["service"],
        namespace=state["namespace"],
        window_minutes=state["time_window_minutes"],
    )
    from agents.state import TelemetrySnapshot

    state["telemetry"] = TelemetrySnapshot(
        logs=bundle["logs"],
        metrics=bundle["metrics"],
        collection_errors=bundle["errors"],
        window_minutes=state["time_window_minutes"],
    )
    _log(
        state,
        f"Telemetry gathered: {len(bundle['logs'])} log lines, "
        f"{len(bundle['metrics'])} metric series, {len(bundle['errors'])} collection errors.",
    )
    return state


# ---------------------------------------------------------------------------
# Node: Hypothesis Agent
# ---------------------------------------------------------------------------
_HYPOTHESIS_SYSTEM_PROMPT = """You are a Principal SRE diagnosing a production incident.
You will be given: the alert details, recent logs, recent metrics, and matching runbook
excerpts. Correlate these signals and respond with STRICT JSON only, no prose, matching
this schema:

{
  "root_cause_hypothesis": "<one to three sentence explanation>",
  "confidence": <float 0.0-1.0>,
  "recommended_action": "restart_pod" | "rollback_deployment" | "scale_replicas" | "noop",
  "action_target": "<pod name, deployment name, or service name>",
  "action_params": {},
  "justification": "<why this action addresses the root cause>"
}

Rules:
- Only recommend "rollback_deployment" if logs/metrics clearly correlate with a recent
  deployment (e.g. error spike beginning right after a release marker).
- Only recommend "scale_replicas" if metrics show load/latency pressure without errors
  indicating a bad release.
- Use "restart_pod" for crash-looping / OOM / stuck-process signatures.
- Use "noop" if evidence is insufficient to safely recommend any action.
- action_params for scale_replicas must include {"replicas": <int>}.
"""


async def hypothesis_agent(state: IncidentState) -> IncidentState:
    """Correlate telemetry + runbooks and propose a root cause and action."""
    state["status"] = IncidentStatus.HYPOTHESIZING.value
    telemetry = state["telemetry"]

    query_text = (
        f"{state['alert'].alertname} {state['alert'].description or ''} "
        f"{' '.join(telemetry.logs[-10:])}"
    )
    try:
        runbook_matches = search_runbooks(query_text, top_k=3)
    except Exception as exc:  # vector store may be uninitialized on first run
        runbook_matches = []
        _log(state, f"Runbook search failed (non-fatal): {exc}")
    state["runbook_matches"] = runbook_matches

    user_prompt = json.dumps(
        {
            "alert": state["alert"].model_dump(),
            "recent_logs": telemetry.logs[-30:],
            "metrics": telemetry.metrics,
            "runbooks": [m.model_dump() for m in runbook_matches],
        },
        default=str,
    )

    llm = _get_llm()
    response = await llm.ainvoke(
        [
            SystemMessage(content=_HYPOTHESIS_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ]
    )

    try:
        raw = response.content if isinstance(response.content, str) else str(response.content)
        # Strip accidental markdown fences if the model adds them.
        raw = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        parsed = json.loads(raw)
    except (json.JSONDecodeError, AttributeError) as exc:
        _log(state, f"Hypothesis Agent: failed to parse LLM response ({exc}); defaulting to noop.")
        parsed = {
            "root_cause_hypothesis": "Unable to determine root cause from available signals.",
            "confidence": 0.0,
            "recommended_action": "noop",
            "action_target": state["service"],
            "action_params": {},
            "justification": "LLM output could not be parsed; defaulting to no-op for safety.",
        }

    state["root_cause_hypothesis"] = parsed.get("root_cause_hypothesis", "")
    state["confidence"] = float(parsed.get("confidence", 0.0))

    action_type = parsed.get("recommended_action", "noop")
    risk = resolve_risk(action_type, state["namespace"])

    state["remediation_plan"] = RemediationAction(
        action_type=action_type,
        target=parsed.get("action_target", state["service"]),
        namespace=state["namespace"],
        params=parsed.get("action_params", {}),
        risk_level=risk,
        justification=parsed.get("justification", ""),
    )

    _log(
        state,
        f"Hypothesis: {state['root_cause_hypothesis']} (confidence={state['confidence']:.2f}). "
        f"Proposed action: {action_type} on '{state['remediation_plan'].target}' "
        f"[risk={risk.value}].",
    )
    return state


# ---------------------------------------------------------------------------
# Node: Safety / Remediation Gate Agent
# ---------------------------------------------------------------------------
async def safety_agent(state: IncidentState) -> IncidentState:
    """
    Evaluate the proposed remediation's risk. LOW/MEDIUM proceed automatically;
    HIGH always requires explicit human approval before execution.
    """
    plan = state.get("remediation_plan")
    if plan is None or plan.action_type == "noop":
        state["requires_human_approval"] = False
        _log(state, "Safety Agent: no actionable remediation proposed (noop). Skipping execution.")
        return state

    if plan.risk_level == RiskLevel.HIGH:
        state["requires_human_approval"] = True
        if state.get("human_approved") is not True:
            state["status"] = IncidentStatus.AWAITING_APPROVAL.value
            _log(
                state,
                f"Safety Agent: action '{plan.action_type}' is HIGH risk. "
                f"Halting for human confirmation before proceeding.",
            )
        else:
            _log(state, "Safety Agent: HIGH risk action was previously human-approved. Proceeding.")
    else:
        state["requires_human_approval"] = False
        _log(state, f"Safety Agent: action '{plan.action_type}' is {plan.risk_level.value} risk. Auto-approved.")

    return state


def route_after_safety(state: IncidentState) -> Literal["remediate", "await_approval", "skip"]:
    plan = state.get("remediation_plan")
    if plan is None or plan.action_type == "noop":
        return "skip"
    if state.get("requires_human_approval") and state.get("human_approved") is not True:
        return "await_approval"
    return "remediate"


# ---------------------------------------------------------------------------
# Node: Remediation Executor
# ---------------------------------------------------------------------------
async def remediation_executor(state: IncidentState) -> IncidentState:
    """Execute the approved remediation action via the Kubernetes API tools."""
    state["status"] = IncidentStatus.REMEDIATING.value
    plan = state["remediation_plan"]

    try:
        if plan.action_type == "restart_pod":
            result = restart_pod(
                namespace=plan.namespace,
                pod_name=plan.params.get("pod_name"),
                label_selector=plan.params.get("label_selector") or f"app={plan.target}",
            )
        elif plan.action_type == "rollback_deployment":
            result = rollback_deployment(
                namespace=plan.namespace,
                deployment_name=plan.target,
                to_revision=plan.params.get("to_revision"),
            )
        elif plan.action_type == "scale_replicas":
            result = scale_replicas(
                namespace=plan.namespace,
                deployment_name=plan.target,
                replicas=int(plan.params.get("replicas", 1)),
            )
        else:
            result = {"executed": False, "success": False, "message": "Unknown action type.", "dry_run": settings.dry_run}
    except RemediationBlockedError as exc:
        result = {"executed": False, "success": False, "message": f"Blocked by policy: {exc}", "dry_run": settings.dry_run}

    state["remediation_result"] = RemediationResult(**result)
    _log(state, f"Remediation executed: {result['message']} (success={result['success']}).")
    return state


async def skip_remediation(state: IncidentState) -> IncidentState:
    """No-op path: nothing safe/confident enough to remediate automatically."""
    state["status"] = IncidentStatus.ESCALATED.value
    state["verification_status"] = "not_applicable"
    _log(state, "No remediation executed. Escalating to on-call for manual investigation.")
    return state


# ---------------------------------------------------------------------------
# Node: Post-Fix Verification Agent
# ---------------------------------------------------------------------------
async def verification_agent(state: IncidentState) -> IncidentState:
    """Poll key metrics after remediation to confirm the incident actually recovered."""
    state["status"] = IncidentStatus.VERIFYING.value
    state["verification_attempts"] = state.get("verification_attempts", 0) + 1

    service, namespace = state["service"], state["namespace"]
    error_rate_query = (
        f'sum(rate(http_requests_total{{app="{service}",namespace="{namespace}",status=~"5.."}}[5m]))'
    )
    try:
        data = await query_prometheus_instant(error_rate_query)
        result_vector = data.get("result", [])
        current_error_rate = float(result_vector[0]["value"][1]) if result_vector else 0.0
    except (TelemetryError, IndexError, ValueError, KeyError) as exc:
        _log(state, f"Verification Agent: metric check failed ({exc}); treating as unresolved.")
        current_error_rate = float("inf")

    recovered = current_error_rate < 0.01  # near-zero 5xx rate threshold
    if recovered:
        state["verification_status"] = "verified_recovered"
        state["status"] = IncidentStatus.RESOLVED.value
        _log(state, f"Verification: error rate {current_error_rate:.4f} — incident RESOLVED.")
    else:
        state["verification_status"] = "verification_failed"
        _log(state, f"Verification: error rate {current_error_rate:.4f} — incident NOT yet resolved.")

    return state


def route_after_verification(state: IncidentState) -> Literal["resolved", "retry", "escalate"]:
    if state.get("verification_status") == "verified_recovered":
        return "resolved"
    if state.get("verification_attempts", 0) >= settings.max_verification_retries:
        return "escalate"
    return "retry"


async def escalate_agent(state: IncidentState) -> IncidentState:
    state["status"] = IncidentStatus.ESCALATED.value
    _log(
        state,
        f"Verification retries exhausted ({state.get('verification_attempts')}). "
        f"Escalating to human on-call engineer.",
    )
    return state


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------
def build_graph():
    graph = StateGraph(IncidentState)

    graph.add_node("triage_agent", triage_agent)
    graph.add_node("telemetry_agent", telemetry_agent)
    graph.add_node("hypothesis_agent", hypothesis_agent)
    graph.add_node("safety_agent", safety_agent)
    graph.add_node("remediation_executor", remediation_executor)
    graph.add_node("skip_remediation", skip_remediation)
    graph.add_node("verification_agent", verification_agent)
    graph.add_node("escalate_agent", escalate_agent)

    graph.set_entry_point("triage_agent")
    graph.add_edge("triage_agent", "telemetry_agent")
    graph.add_edge("telemetry_agent", "hypothesis_agent")
    graph.add_edge("hypothesis_agent", "safety_agent")

    graph.add_conditional_edges(
        "safety_agent",
        route_after_safety,
        {
            "remediate": "remediation_executor",
            "await_approval": END,  # graph pauses here; resumed by main.py /approve
            "skip": "skip_remediation",
        },
    )

    graph.add_edge("remediation_executor", "verification_agent")
    graph.add_edge("skip_remediation", END)

    graph.add_conditional_edges(
        "verification_agent",
        route_after_verification,
        {
            "resolved": END,
            "retry": "telemetry_agent",  # loop back to re-gather signals
            "escalate": "escalate_agent",
        },
    )
    graph.add_edge("escalate_agent", END)

    checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer)


# Compiled singleton used by main.py
sentinel_graph = build_graph()
