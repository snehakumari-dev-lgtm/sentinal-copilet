"""
tools/k8s_actions.py
=====================
Policy-wrapped Kubernetes remediation actions.

Every function here is a guarded write operation. NONE of these functions
should ever be called directly by an LLM tool-call without first passing
through the Safety Agent's risk gate in agents/graph.py. This module
enforces its OWN independent checks too (defense in depth):
  1. Namespace allow-list check.
  2. Dry-run mode short-circuit (default ON).
  3. Bounds checking (e.g. max replica count).
  4. Structured, auditable return values — never silent failure.
"""

from __future__ import annotations

import datetime as dt
from typing import Any, Dict, Optional

from kubernetes import client, config as k8s_config
from kubernetes.client import ApiException

from config import settings, is_namespace_allowed


class RemediationBlockedError(Exception):
    """Raised when a policy check refuses to execute an action."""


def _load_kube_client() -> None:
    """Load kube config once. Supports in-cluster, kubeconfig file, or named context."""
    try:
        if settings.kube_config_path:
            k8s_config.load_kube_config(
                config_file=settings.kube_config_path,
                context=settings.kube_context or None,
            )
        else:
            try:
                k8s_config.load_incluster_config()
            except k8s_config.ConfigException:
                k8s_config.load_kube_config(context=settings.kube_context or None)
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RemediationBlockedError(f"Unable to load Kubernetes config: {exc}") from exc


def _now_iso() -> str:
    return dt.datetime.utcnow().isoformat() + "Z"


def _guard(namespace: str, action_name: str) -> None:
    if not is_namespace_allowed(namespace):
        raise RemediationBlockedError(
            f"Namespace '{namespace}' is not in the allowed remediation list "
            f"({settings.allowed_namespaces}). Refusing to execute '{action_name}'."
        )


def _result(executed: bool, success: bool, message: str) -> Dict[str, Any]:
    return {
        "executed": executed,
        "success": success,
        "message": message,
        "executed_at": _now_iso(),
        "dry_run": settings.dry_run,
    }


def restart_pod(namespace: str, pod_name: Optional[str] = None, label_selector: Optional[str] = None) -> Dict[str, Any]:
    """
    Restart a pod by deleting it (the owning controller recreates it).
    Either pod_name or label_selector must be provided.
    LOW risk action per default policy — still namespace-gated.
    """
    _guard(namespace, "restart_pod")
    if not pod_name and not label_selector:
        raise RemediationBlockedError("restart_pod requires pod_name or label_selector.")

    if settings.dry_run:
        target = pod_name or f"label_selector={label_selector}"
        return _result(False, True, f"[DRY RUN] Would restart pod '{target}' in '{namespace}'.")

    _load_kube_client()
    v1 = client.CoreV1Api()
    try:
        if pod_name:
            v1.delete_namespaced_pod(name=pod_name, namespace=namespace)
            return _result(True, True, f"Deleted pod '{pod_name}' in '{namespace}' for restart.")
        else:
            pods = v1.list_namespaced_pod(namespace=namespace, label_selector=label_selector)
            if not pods.items:
                return _result(True, False, f"No pods matched selector '{label_selector}'.")
            deleted = []
            for pod in pods.items:
                v1.delete_namespaced_pod(name=pod.metadata.name, namespace=namespace)
                deleted.append(pod.metadata.name)
            return _result(True, True, f"Deleted pods for restart: {deleted}")
    except ApiException as exc:
        return _result(True, False, f"Kubernetes API error restarting pod: {exc.reason}")


def rollback_deployment(namespace: str, deployment_name: str, to_revision: Optional[int] = None) -> Dict[str, Any]:
    """
    Roll back a Deployment to the previous (or specified) ReplicaSet revision.
    HIGH risk action by default policy — this function assumes the caller
    (graph.py Safety Agent) has already confirmed human approval.
    """
    _guard(namespace, "rollback_deployment")

    if settings.dry_run:
        rev = to_revision if to_revision is not None else "previous"
        return _result(False, True, f"[DRY RUN] Would roll back deployment '{deployment_name}' to revision {rev} in '{namespace}'.")

    _load_kube_client()
    apps_v1 = client.AppsV1Api()
    try:
        if to_revision is not None:
            # Rollback to a specific revision requires patching via the
            # deployment's revisionHistoryLimit + ReplicaSet template copy.
            replica_sets = apps_v1.list_namespaced_replica_set(
                namespace=namespace,
                label_selector=f"app={deployment_name}",
            )
            target_rs = None
            for rs in replica_sets.items:
                rev_annotation = rs.metadata.annotations.get("deployment.kubernetes.io/revision")
                if rev_annotation == str(to_revision):
                    target_rs = rs
                    break
            if target_rs is None:
                return _result(True, False, f"Revision {to_revision} not found for '{deployment_name}'.")
            patch_body = {"spec": {"template": target_rs.spec.template.to_dict()}}
            apps_v1.patch_namespaced_deployment(name=deployment_name, namespace=namespace, body=patch_body)
            return _result(True, True, f"Rolled back '{deployment_name}' to revision {to_revision}.")
        else:
            # Roll back to previous revision: use the standard rollback
            # trick of patching the deployment to its prior ReplicaSet.
            deployment = apps_v1.read_namespaced_deployment(name=deployment_name, namespace=namespace)
            current_revision = deployment.metadata.annotations.get("deployment.kubernetes.io/revision")
            replica_sets = apps_v1.list_namespaced_replica_set(
                namespace=namespace,
                label_selector=f"app={deployment_name}",
            )
            candidates = [
                rs for rs in replica_sets.items
                if rs.metadata.annotations.get("deployment.kubernetes.io/revision") != current_revision
            ]
            if not candidates:
                return _result(True, False, f"No previous revision found for '{deployment_name}'.")
            candidates.sort(
                key=lambda rs: int(rs.metadata.annotations.get("deployment.kubernetes.io/revision", "0")),
                reverse=True,
            )
            previous_rs = candidates[0]
            patch_body = {"spec": {"template": previous_rs.spec.template.to_dict()}}
            apps_v1.patch_namespaced_deployment(name=deployment_name, namespace=namespace, body=patch_body)
            prev_rev = previous_rs.metadata.annotations.get("deployment.kubernetes.io/revision")
            return _result(True, True, f"Rolled back '{deployment_name}' to previous revision {prev_rev}.")
    except ApiException as exc:
        return _result(True, False, f"Kubernetes API error rolling back deployment: {exc.reason}")


def scale_replicas(namespace: str, deployment_name: str, replicas: int) -> Dict[str, Any]:
    """
    Scale a Deployment's replica count. MEDIUM risk by default policy.
    Bounded by settings.max_scale_replicas to prevent runaway scale-ups.
    """
    _guard(namespace, "scale_replicas")
    if replicas < 0:
        raise RemediationBlockedError("Replica count cannot be negative.")
    if replicas > settings.max_scale_replicas:
        raise RemediationBlockedError(
            f"Requested replica count {replicas} exceeds policy max "
            f"({settings.max_scale_replicas}). Refusing to execute."
        )

    if settings.dry_run:
        return _result(False, True, f"[DRY RUN] Would scale '{deployment_name}' in '{namespace}' to {replicas} replicas.")

    _load_kube_client()
    apps_v1 = client.AppsV1Api()
    try:
        apps_v1.patch_namespaced_deployment_scale(
            name=deployment_name,
            namespace=namespace,
            body={"spec": {"replicas": replicas}},
        )
        return _result(True, True, f"Scaled '{deployment_name}' in '{namespace}' to {replicas} replicas.")
    except ApiException as exc:
        return _result(True, False, f"Kubernetes API error scaling deployment: {exc.reason}")
