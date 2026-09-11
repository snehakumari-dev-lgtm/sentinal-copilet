# PROJECT Sentinel

Autonomous Incident-Response Copilot for Distributed Systems. Ingests
Prometheus Alertmanager webhooks, investigates via Loki/Prometheus, proposes
a root cause using an LLM + runbook vector search, and — under a strict,
deterministic risk policy — either auto-remediates via the Kubernetes API or
pauses for human confirmation.

## Architecture

```
Alertmanager --webhook--> FastAPI (/webhook/alert)
                              |
                              v
                     LangGraph workflow
   triage -> telemetry -> hypothesis -> safety_gate
                                            |-- LOW/MED risk --> remediate -> verify -> resolved/retry/escalate
                                            |-- HIGH risk ------> PAUSE (await human /approve) -> remediate -> verify
```

Risk is resolved deterministically in `config.py::resolve_risk` — the LLM
proposes *what* to do, never *how risky* it is. Production-looking
namespaces are force-escalated to HIGH regardless of action type.

## Local Setup

1. **Python environment**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Configure environment**
   ```bash
   cp .env.example .env
   # edit .env: set ANTHROPIC_API_KEY, PROMETHEUS_URL, LOKI_URL, KUBE_CONFIG_PATH
   ```

   `SENTINEL_DRY_RUN=true` by default — Kubernetes actions are logged but not
   executed. Flip to `false` only once you've validated behavior against a
   staging cluster.

3. **Kubernetes access**
   Point `KUBE_CONFIG_PATH` at a kubeconfig with a service account scoped to
   only the namespaces in `SENTINEL_ALLOWED_NAMESPACES`. Do not grant
   cluster-admin. Minimum RBAC: `get/list/delete` on pods, `get/patch` on
   deployments and replicasets.

4. **Run the server**
   ```bash
   uvicorn main:app --reload --port 8000
   ```

5. **Point Alertmanager at Sentinel** — in `alertmanager.yml`:
   ```yaml
   receivers:
     - name: sentinel-copilot
       webhook_configs:
         - url: http://<sentinel-host>:8000/webhook/alert
           send_resolved: false
   ```

## Testing locally without a real cluster

```bash
curl -X POST http://localhost:8000/webhook/alert \
  -H "Content-Type: application/json" \
  -d '{
    "status": "firing",
    "alerts": [{
      "status": "firing",
      "labels": {
        "alertname": "HighErrorRate",
        "service": "checkout-api",
        "namespace": "staging",
        "severity": "critical"
      },
      "annotations": {
        "summary": "Checkout API 5xx rate above threshold",
        "description": "Error rate spiked after latest deploy."
      },
      "startsAt": "2026-08-01T10:00:00Z"
    }]
  }'
```

Check progress:
```bash
curl http://localhost:8000/incidents
curl http://localhost:8000/incident/<incident_id>
```

If the proposed action is HIGH risk, approve it:
```bash
curl -X POST http://localhost:8000/incident/<incident_id>/approve \
  -H "Content-Type: application/json" \
  -d '{"approved": true, "approver": "jane@company.com"}'
```

## Productionizing (post-MVP)

- **Checkpointer**: swap `MemorySaver` in `agents/graph.py` for a
  Postgres/Redis-backed LangGraph checkpointer so paused (awaiting-approval)
  incidents survive process restarts.
- **Incident store**: replace the in-memory `INCIDENTS` dict in `main.py`
  with a real datastore.
- **Auth**: add authentication to `/incident/{id}/approve` — right now
  anyone who can reach the API can approve a HIGH-risk action.
- **Notifications**: wire the "awaiting approval" transition to Slack/
  PagerDuty so a human actually sees the request.
- **Vector store**: `agents/runbooks.py` isolates Chroma behind a small
  interface — swap for Pinecone/Weaviate without touching graph logic.
- **Observability**: instrument the graph nodes themselves (structured logs
  per node, LangSmith tracing) so Sentinel's own decisions are debuggable.

## Safety Notes

- All Kubernetes actions default to `SENTINEL_DRY_RUN=true`.
- Namespace allow-list (`SENTINEL_ALLOWED_NAMESPACES`) is enforced
  independently in both the graph's safety gate and inside
  `tools/k8s_actions.py` (defense in depth).
- `rollback_deployment` is HIGH risk by default and cannot be auto-executed
  under any configuration without a `human_approved=True` flag on the
  incident state.
- Any namespace containing "prod", "production", or "live" is force-escalated
  to HIGH risk regardless of action type.
