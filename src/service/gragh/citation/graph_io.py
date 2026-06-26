"""
Read ``citation_network.json`` into sparse structures for F2 and F3.

The whole graph supplements the (few) local papers when computing co-citation
(F2) and PageRank (F3). Edges mean ``source_doi -> target_doi`` ("source cites
target"). For F2/F3 we use only the *in-graph* sub-network (both endpoints are
collected nodes), because out-of-graph citing papers were never collected.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
from scipy.sparse import csr_matrix

from src.core.utils import load_json
from src.service.gragh.citation.index import normalize_doi


@dataclass
class GraphData:
    """In-graph citation network as sparse adjacency + bookkeeping."""

    # node_doi -> graph integer index (0..n_nodes-1), normalized DOIs.
    node_doi_to_idx: Dict[str, int] = field(default_factory=dict)
    node_dois: List[str] = field(default_factory=list)
    node_titles: List[str] = field(default_factory=list)

    # A[s, t] = 1 when node s cites node t (both in graph). CSR.
    A_inner: Optional[csr_matrix] = None

    in_degree: Optional[np.ndarray] = None
    out_degree: Optional[np.ndarray] = None

    n_nodes: int = 0
    n_inner_edges: int = 0
    n_total_edges: int = 0
    n_out_of_graph_targets: int = 0

    network_metadata: dict = field(default_factory=dict)
    network_sha256: str = ""

    def index_of(self, doi: Optional[str]) -> Optional[int]:
        """Return the graph index of a (normalized) DOI, or None if absent."""
        if not doi:
            return None
        return self.node_doi_to_idx.get(doi)


def _hash_file(path: str) -> str:
    """SHA256 of the upstream network file for provenance (§5.4)."""
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return ""


def load_graph(citation_network_path: str) -> GraphData:
    """
    Load the citation network and build the in-graph sparse adjacency.

    Args:
        citation_network_path: Path to ``citation_network.json``.

    Returns:
        Populated :class:`GraphData`. ``A_inner`` only contains edges whose
        source *and* target are collected nodes; out-of-graph targets are
        counted but excluded (they cannot contribute in-graph co-citation).
    """
    raw = load_json(citation_network_path)
    nodes = raw.get("nodes", []) if raw else []
    edges = raw.get("edges", []) if raw else []

    graph = GraphData()
    graph.network_metadata = raw.get("metadata", {}) if raw else {}
    graph.network_sha256 = _hash_file(citation_network_path)

    # 1. Register nodes (normalized DOI -> idx), de-duplicating (AC-5).
    for node in nodes:
        doi = normalize_doi(node.get("doi"))
        if not doi or doi in graph.node_doi_to_idx:
            continue
        idx = len(graph.node_dois)
        graph.node_doi_to_idx[doi] = idx
        graph.node_dois.append(doi)
        graph.node_titles.append(node.get("title") or "")

    n = len(graph.node_dois)
    graph.n_nodes = n

    # 2. Build in-graph edge list (both endpoints are nodes).
    rows: List[int] = []
    cols: List[int] = []
    seen = set()
    out_of_graph = 0
    total = 0
    for edge in edges:
        total += 1
        s = graph.node_doi_to_idx.get(normalize_doi(edge.get("source_doi")))
        t_doi = normalize_doi(edge.get("target_doi"))
        t = graph.node_doi_to_idx.get(t_doi)
        if t is None:
            out_of_graph += 1
        if s is None or t is None or s == t:
            continue
        key = (s, t)
        if key in seen:
            continue
        seen.add(key)
        rows.append(s)
        cols.append(t)

    data = np.ones(len(rows), dtype=np.float64)
    graph.A_inner = csr_matrix(
        (data, (np.asarray(rows, dtype=np.int64), np.asarray(cols, dtype=np.int64))),
        shape=(n, n),
    )
    graph.n_inner_edges = len(rows)
    graph.n_total_edges = total
    graph.n_out_of_graph_targets = out_of_graph

    # 3. Degrees (in-graph).
    if n > 0:
        graph.out_degree = np.asarray(graph.A_inner.sum(axis=1)).ravel().astype(np.int64)
        graph.in_degree = np.asarray(graph.A_inner.sum(axis=0)).ravel().astype(np.int64)
    else:
        graph.out_degree = np.zeros(0, dtype=np.int64)
        graph.in_degree = np.zeros(0, dtype=np.int64)

    return graph
