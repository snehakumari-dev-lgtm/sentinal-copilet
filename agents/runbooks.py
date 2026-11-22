"""
agents/runbooks.py
===================
Thin wrapper around a Chroma vector store holding SRE runbooks. Kept as an
isolated module so swapping Chroma for Pinecone/Weaviate later only touches
this file.
"""

from __future__ import annotations

from typing import List

import chromadb
from chromadb.utils import embedding_functions

from config import settings
from agents.state import RunbookMatch

_client = None
_collection = None

_DEFAULT_RUNBOOKS = [
    {
        "title": "High error rate after deployment",
        "content": (
            "If a service shows a spike in 5xx error rate immediately after a "
            "new deployment, the most common cause is a bad release. Check the "
            "deployment's revision history. If errors correlate tightly with the "
            "deploy timestamp, roll back to the previous revision."
        ),
    },
    {
        "title": "Pod crash-looping due to OOM",
        "content": (
            "Pods repeatedly restarting with OOMKilled status indicate memory "
            "limits are too low or there is a memory leak. Restarting the pod is "
            "a short-term mitigation; if restarts recur within minutes, escalate "
            "for a resource limit increase or code-level investigation."
        ),
    },
    {
        "title": "Latency spike under load",
        "content": (
            "P99 latency spikes correlated with rising request volume usually "
            "indicate the service is under-provisioned. Scaling replicas "
            "horizontally is the standard first mitigation before deeper "
            "profiling."
        ),
    },
    {
        "title": "Database connection exhaustion",
        "content": (
            "Logs showing 'connection pool exhausted' or timeouts to the "
            "database, without corresponding CPU/memory pressure, suggest a "
            "connection leak or an undersized pool. Restarting pods can provide "
            "temporary relief, but the underlying pool configuration should be "
            "reviewed and is not something Sentinel should auto-remediate."
        ),
    },
]


def _get_collection():
    global _client, _collection
    if _collection is not None:
        return _collection

    _client = chromadb.PersistentClient(path=settings.vector_db_path)
    embed_fn = embedding_functions.DefaultEmbeddingFunction()
    _collection = _client.get_or_create_collection(
        name=settings.runbook_collection_name,
        embedding_function=embed_fn,
    )

    if _collection.count() == 0:
        _collection.add(
            ids=[f"runbook-{i}" for i in range(len(_DEFAULT_RUNBOOKS))],
            documents=[rb["content"] for rb in _DEFAULT_RUNBOOKS],
            metadatas=[{"title": rb["title"]} for rb in _DEFAULT_RUNBOOKS],
        )
    return _collection


def search_runbooks(query_text: str, top_k: int = 3) -> List[RunbookMatch]:
    """Semantic search over the runbook collection."""
    collection = _get_collection()
    results = collection.query(query_texts=[query_text], n_results=top_k)

    matches: List[RunbookMatch] = []
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    dists = results.get("distances", [[]])[0]
    for doc, meta, dist in zip(docs, metas, dists):
        similarity = max(0.0, 1.0 - dist)  # cosine distance -> similarity
        matches.append(
            RunbookMatch(
                title=meta.get("title", "Untitled runbook"),
                content=doc,
                similarity_score=round(similarity, 4),
            )
        )
    return matches


def add_runbook(title: str, content: str) -> None:
    """Allow operators to append new runbooks post-incident (learning loop)."""
    collection = _get_collection()
    existing_count = collection.count()
    collection.add(
        ids=[f"runbook-custom-{existing_count}"],
        documents=[content],
        metadatas=[{"title": title}],
    )
