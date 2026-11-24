"""
tools/telemetry.py
===================
Async retrieval functions for logs (Loki) and metrics (Prometheus).
These are the "eyes" of the Telemetry Agent — pure I/O, no business logic.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from config import settings


class TelemetryError(Exception):
    """Raised when a telemetry backend cannot be reached or errors out."""


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=0.5, max=4))
async def query_loki_logs(
    service: str,
    namespace: str,
    window_minutes: int = 15,
    limit: int = 200,
    extra_label_filters: Optional[Dict[str, str]] = None,
) -> List[str]:
    """
    Query Loki for recent log lines for a service, using LogQL.
    Returns a flat list of raw log lines (most recent last).
    """
    end_ns = int(time.time() * 1e9)
    start_ns = end_ns - int(window_minutes * 60 * 1e9)

    label_filters = {"namespace": namespace, "app": service}
    if extra_label_filters:
        label_filters.update(extra_label_filters)
    selector = ",".join(f'{k}="{v}"' for k, v in label_filters.items())
    logql = f'{{{selector}}}'

    params = {
        "query": logql,
        "start": str(start_ns),
        "end": str(end_ns),
        "limit": str(limit),
        "direction": "backward",
    }

    url = f"{settings.loki_url.rstrip('/')}/loki/api/v1/query_range"
    try:
        async with httpx.AsyncClient(timeout=settings.telemetry_timeout_seconds) as client:
            resp = await client.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()
    except httpx.HTTPError as exc:
        raise TelemetryError(f"Loki query failed: {exc}") from exc

    lines: List[str] = []
    for stream in data.get("data", {}).get("result", []):
        for _, log_line in stream.get("values", []):
            lines.append(log_line)
    lines.reverse()  # chronological order
    return lines[-limit:]


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=0.5, max=4))
async def query_prometheus_instant(promql: str) -> Dict[str, Any]:
    """Run an instant PromQL query and return the raw result vector."""
    url = f"{settings.prometheus_url.rstrip('/')}/api/v1/query"
    try:
        async with httpx.AsyncClient(timeout=settings.telemetry_timeout_seconds) as client:
            resp = await client.get(url, params={"query": promql})
            resp.raise_for_status()
            payload = resp.json()
    except httpx.HTTPError as exc:
        raise TelemetryError(f"Prometheus instant query failed: {exc}") from exc

    if payload.get("status") != "success":
        raise TelemetryError(f"Prometheus returned error: {payload}")
    return payload["data"]


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=0.5, max=4))
async def query_prometheus_range(
    promql: str, window_minutes: int = 15, step_seconds: int = 30
) -> Dict[str, Any]:
    """Run a range PromQL query over the given window and return raw matrix data."""
    end = time.time()
    start = end - (window_minutes * 60)
    url = f"{settings.prometheus_url.rstrip('/')}/api/v1/query_range"
    params = {"query": promql, "start": start, "end": end, "step": step_seconds}
    try:
        async with httpx.AsyncClient(timeout=settings.telemetry_timeout_seconds) as client:
            resp = await client.get(url, params=params)
            resp.raise_for_status()
            payload = resp.json()
    except httpx.HTTPError as exc:
        raise TelemetryError(f"Prometheus range query failed: {exc}") from exc

    if payload.get("status") != "success":
        raise TelemetryError(f"Prometheus returned error: {payload}")
    return payload["data"]


async def gather_service_telemetry(
    service: str, namespace: str, window_minutes: int = 15
) -> Dict[str, Any]:
    """
    Convenience aggregator: pulls a standard SRE golden-signal bundle
    (error rate, latency p99, restart count) plus recent logs for a service.
    Failures in one signal do not block the others — partial telemetry beats
    none during an active incident.
    """
    result: Dict[str, Any] = {"logs": [], "metrics": {}, "errors": []}

    try:
        result["logs"] = await query_loki_logs(service, namespace, window_minutes)
    except TelemetryError as exc:
        result["errors"].append(str(exc))

    metric_queries = {
        "error_rate": f'sum(rate(http_requests_total{{app="{service}",namespace="{namespace}",status=~"5.."}}[5m]))',
        "latency_p99": f'histogram_quantile(0.99, sum(rate(http_request_duration_seconds_bucket{{app="{service}",namespace="{namespace}"}}[5m])) by (le))',
        "restart_count": f'sum(kube_pod_container_status_restarts_total{{namespace="{namespace}",pod=~"{service}.*"}})',
        "cpu_usage": f'sum(rate(container_cpu_usage_seconds_total{{namespace="{namespace}",pod=~"{service}.*"}}[5m]))',
        "memory_usage": f'sum(container_memory_working_set_bytes{{namespace="{namespace}",pod=~"{service}.*"}})',
    }
    for name, q in metric_queries.items():
        try:
            result["metrics"][name] = await query_prometheus_instant(q)
        except TelemetryError as exc:
            result["errors"].append(f"{name}: {exc}")

    return result
