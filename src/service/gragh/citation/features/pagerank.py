"""
PageRank: node prestige on the whole in-graph citation network, projected to
edge-level features for the local candidate edges (FR-F3).

PageRank is computed over the full directed graph (all collected nodes) so the
score reflects the broader citation structure. Dangling nodes (out-degree 0)
are handled by uniform redistribution, matching ``networkx`` (FR-F3.2). The
per-node score is exposed for the local papers, and three edge transforms are
emitted: ``pagerank_target`` (cited-side prestige = PR(j)), ``pagerank_min``,
``pagerank_product`` (FR-F3.4). Rows are directed edges ``i -> j`` (i cites j),
so the cited endpoint for ``pagerank_target`` is always ``j``.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from src.service.gragh.citation.config import CitationFeatureConfig
from src.service.gragh.citation.features import FeatureResult
from src.service.gragh.citation.graph_io import GraphData
from src.service.gragh.citation.index import PaperIndex


PAGERANK_COLUMNS = ("pagerank_target", "pagerank_min", "pagerank_product")


def pagerank_networkx(graph: GraphData, damping: float, max_iter: int, tol: float) -> np.ndarray:
    """Reference PageRank via ``networkx.pagerank`` over the full graph."""
    import networkx as nx

    n = graph.n_nodes
    G = nx.DiGraph()
    G.add_nodes_from(range(n))
    coo = graph.A_inner.tocoo()
    G.add_edges_from(zip(coo.row.tolist(), coo.col.tolist()))
    pr = nx.pagerank(G, alpha=damping, max_iter=max_iter, tol=tol)
    out = np.zeros(n, dtype=np.float64)
    for node, score in pr.items():
        out[node] = score
    return out


def pagerank_scipy(graph: GraphData, damping: float, max_iter: int, tol: float) -> np.ndarray:
    """
    PageRank via explicit power iteration (scipy sparse), for large graphs.

    Implements ``PR = (1-d)/N + d (M PR + dangling redistribution)`` with the
    column-stochastic transition ``M[i, j] = A[j, i] / out_degree(j)`` and
    uniform dangling-mass redistribution (FR-F3.1/F3.2).
    """
    n = graph.n_nodes
    if n == 0:
        return np.zeros(0, dtype=np.float64)

    A = graph.A_inner.tocsr().astype(np.float64)
    out_deg = np.asarray(A.sum(axis=1)).ravel()
    dangling = out_deg == 0
    inv = np.zeros(n, dtype=np.float64)
    nonzero = ~dangling
    inv[nonzero] = 1.0 / out_deg[nonzero]

    # Row-normalize A so row i distributes paper i's rank to its out-neighbours.
    from scipy.sparse import diags
    P = diags(inv) @ A  # P[i, k] = share of i's rank going to k

    pr = np.full(n, 1.0 / n, dtype=np.float64)
    teleport = (1.0 - damping) / n
    for _ in range(max_iter):
        dangling_mass = damping * pr[dangling].sum() / n
        new = teleport + dangling_mass + damping * (P.T @ pr)
        if np.abs(new - pr).sum() < tol:
            pr = new
            break
        pr = new
    s = pr.sum()
    return pr / s if s > 0 else pr


def compute_pagerank(
    index: PaperIndex,
    graph: GraphData,
    candidate_pairs,
    config: CitationFeatureConfig,
    logger=None,
    method: str = "networkx",
) -> FeatureResult:
    """
    Compute node PageRank (full graph) and the directed-edge transforms.

    Args:
        method: "networkx" (reference) or "scipy" (power iteration).
    """
    if method == "scipy":
        pr = pagerank_scipy(graph, config.pagerank_damping,
                            config.pagerank_max_iter, config.pagerank_tol)
    else:
        pr = pagerank_networkx(graph, config.pagerank_damping,
                               config.pagerank_max_iter, config.pagerank_tol)

    result = FeatureResult(columns=PAGERANK_COLUMNS)
    g_idx: Dict[int, Optional[int]] = {
        pid: graph.index_of(index.doi(pid)) for pid in range(index.n)
    }

    # Node-level scores for local papers.
    for pid in range(index.n):
        gi = g_idx[pid]
        result.node_values.setdefault(pid, {})
        result.node_values[pid]["pagerank"] = float(pr[gi]) if gi is not None else None
        result.node_values[pid]["in_degree"] = int(graph.in_degree[gi]) if gi is not None else None
        result.node_values[pid]["out_degree"] = int(graph.out_degree[gi]) if gi is not None else None

    for (i, j) in candidate_pairs:
        gi, gj = g_idx[i], g_idx[j]
        if gi is None or gj is None:
            result.edge_values[(i, j)] = {c: None for c in PAGERANK_COLUMNS}
            continue
        pri, prj = float(pr[gi]), float(pr[gj])

        # Rows are directed edges i -> j (i cites j), so the cited endpoint is j.
        values: Dict[str, Optional[float]] = {}
        if config.pagerank_emit_target:
            values["pagerank_target"] = prj
        if config.pagerank_emit_min:
            values["pagerank_min"] = min(pri, prj)
        if config.pagerank_emit_product:
            values["pagerank_product"] = pri * prj
        result.edge_values[(i, j)] = values

    result.metadata = {
        "source": "citation_network.json full directed graph",
        "method": method,
        "damping": config.pagerank_damping,
        "max_iter": config.pagerank_max_iter,
        "tol": config.pagerank_tol,
        "dangling_handling": "uniform redistribution (networkx-consistent)",
        "pagerank_sum": float(pr.sum()),
        "pagerank_target_direction": "PR of cited endpoint j on directed edge i -> j",
    }
    if logger is not None:
        logger.info("pagerank: method={m}, sum={s:.6f}", m=method, s=float(pr.sum()))
    return result
