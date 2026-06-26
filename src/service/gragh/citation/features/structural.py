"""
F1 (bibliographic coupling) and F2 (co-citation) -- the structural features.

F1 uses each local paper's own ``references.json`` reference list (out-references,
including out-of-graph cited works, FR-F1.4); it does NOT depend on the citation
network. F2 uses the in-graph citation network (FR-F2) to count shared citers,
restricted to the local candidate pairs. Both use sparse matrix products
(GC-2.1): no O(N^2) double loops over the full graph.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from scipy.sparse import csr_matrix

from src.core.utils import load_json, exists
from src.service.gragh.citation.features import FeatureResult
from src.service.gragh.citation.graph_io import GraphData
from src.service.gragh.citation.index import PaperIndex, local_paths, normalize_doi


BIBCOUPLING_COLUMNS = ("bibcoupling_raw", "bibcoupling_jaccard")
COCITATION_COLUMNS = ("cocitation_raw", "cocitation_salton")

_WS_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[^a-z0-9 ]")


def _normalize_title_key(title: Optional[str]) -> Optional[str]:
    """Normalized title used as a reference identity when a DOI is absent."""
    if not title:
        return None
    text = _PUNCT_RE.sub(" ", title.lower())
    text = _WS_RE.sub(" ", text).strip()
    return f"title::{text}" if text else None


def load_reference_sets(index: PaperIndex, logger=None) -> Dict[int, Optional[Set[str]]]:
    """
    Build R(i): the set of cited-work identifiers for each local paper.

    Identity is the normalized reference DOI when present, else a normalized
    title (FR-F1.1/F1.4). Out-of-graph cited works are intentionally kept.

    Returns:
        paper_id -> set of identifiers, or ``None`` when reference data is
        unavailable (missing/empty references.json) so F1 is ``null`` (GC-4).
    """
    ref_sets: Dict[int, Optional[Set[str]]] = {}
    for paper in index.papers:
        path = local_paths(index, paper.id)["references_json"]
        if not exists(path):
            ref_sets[paper.id] = None
            if logger is not None:
                logger.warning("F1: references.json missing for {doi}", doi=paper.doi)
            continue
        refs = load_json(path) or {}
        if not refs:
            ref_sets[paper.id] = None
            continue
        ids: Set[str] = set()
        for raw_key, info in refs.items():
            info = info or {}
            ident = normalize_doi(info.get("doi"))
            if not ident:
                ident = _normalize_title_key(info.get("title") or raw_key)
            if ident:
                ids.add(ident)
        ref_sets[paper.id] = ids
    return ref_sets


def compute_bibcoupling(index: PaperIndex, candidate_pairs, logger=None) -> FeatureResult:
    """
    Compute bibliographic coupling: shared references + Jaccard (FR-F1).

    Raw value is ``|R(i) ∩ R(j)|`` via a sparse ``R R^T`` product; normalized
    value is the Jaccard ``|∩| / (|R(i)| + |R(j)| - |∩|)`` (FR-F1.3).
    """
    ref_sets = load_reference_sets(index, logger=logger)

    # Build the sparse reference matrix R (n_local x M) over the union of refs.
    vocab: Dict[str, int] = {}
    rows: List[int] = []
    cols: List[int] = []
    for pid in range(index.n):
        s = ref_sets.get(pid)
        if not s:
            continue
        for ident in s:
            col = vocab.setdefault(ident, len(vocab))
            rows.append(pid)
            cols.append(col)

    m = len(vocab)
    R = csr_matrix(
        (np.ones(len(rows)), (rows, cols)),
        shape=(index.n, max(m, 1)),
    )
    inter = (R @ R.T).toarray()  # |R(i) ∩ R(j)|; n_local is tiny
    sizes = {pid: (len(s) if s else 0) for pid, s in ref_sets.items()}

    result = FeatureResult(columns=BIBCOUPLING_COLUMNS)
    for (i, j) in candidate_pairs:
        if ref_sets.get(i) is None or ref_sets.get(j) is None:
            result.edge_values[(i, j)] = {c: None for c in BIBCOUPLING_COLUMNS}
            continue
        raw = int(inter[i, j])
        union = sizes[i] + sizes[j] - raw
        jaccard = (raw / union) if union > 0 else 0.0
        result.edge_values[(i, j)] = {
            "bibcoupling_raw": raw,
            "bibcoupling_jaccard": float(jaccard),
        }

    for pid in range(index.n):
        result.node_values.setdefault(pid, {})["reference_count"] = sizes.get(pid, 0)

    covered = sum(1 for pid in range(index.n) if ref_sets.get(pid) is not None)
    result.metadata = {
        "source": "local references.json (out-references incl. out-of-graph)",
        "normalization": "jaccard",
        "papers_with_reference_data": covered,
        "reference_vocab_size": m,
    }
    if logger is not None:
        logger.info("bibcoupling: reference vocab={m}, papers_with_refs={c}", m=m, c=covered)
    return result


def compute_cocitation(
    index: PaperIndex,
    graph: GraphData,
    candidate_pairs,
    logger=None,
) -> FeatureResult:
    """
    Compute co-citation: shared citers + Salton cosine (FR-F2).

    Raw value is ``|C(i) ∩ C(j)|`` (shared in-graph citers); normalized value
    is the Salton cosine ``|∩| / sqrt(|C(i)| |C(j)|)`` (FR-F2.3). This is a
    lower bound on the true co-citation (out-of-graph citers excluded, FR-F2.4).
    """
    result = FeatureResult(columns=COCITATION_COLUMNS)

    # Local graph indices (None when a local paper is not a graph node).
    g_idx = {pid: graph.index_of(index.doi(pid)) for pid in range(index.n)}
    local_nodes = [g for g in g_idx.values() if g is not None]

    cocite = None
    if local_nodes and graph.A_inner is not None:
        # Columns of A_inner are citer-incidence per cited node; selecting the
        # local columns and multiplying gives shared-citer counts (sparse).
        sub = graph.A_inner[:, local_nodes]            # (n_nodes x k)
        gram = (sub.T @ sub).toarray()                 # (k x k) shared citers
        node_to_local = {g: pos for pos, g in enumerate(local_nodes)}
        cocite = (gram, node_to_local)

    for (i, j) in candidate_pairs:
        gi, gj = g_idx[i], g_idx[j]
        if gi is None or gj is None or cocite is None:
            result.edge_values[(i, j)] = {c: None for c in COCITATION_COLUMNS}
            continue
        gram, node_to_local = cocite
        raw = int(gram[node_to_local[gi], node_to_local[gj]])
        ci = int(graph.in_degree[gi])
        cj = int(graph.in_degree[gj])
        denom = (ci * cj) ** 0.5
        salton = (raw / denom) if denom > 0 else 0.0
        result.edge_values[(i, j)] = {
            "cocitation_raw": raw,
            "cocitation_salton": float(salton),
        }

    result.metadata = {
        "source": "citation_network.json in-graph in-edges",
        "normalization": "salton_cosine",
        "warning": (
            "F2 is the in-graph co-citation, a lower-bound estimate of true "
            "co-citation: out-of-graph (uncollected) citing papers cannot "
            "contribute. Reliability depends on the network's coverage of the "
            "field (see coverage audit)."
        ),
    }
    return result


# ── Brute-force reference implementations (AC-1 validation) ─────────────────

def brute_force_f1(ref_sets: Dict[int, Set[str]], i: int, j: int) -> Tuple[int, float]:
    """Direct set-based bibliographic coupling for testing (AC-1)."""
    a, b = ref_sets[i], ref_sets[j]
    inter = len(a & b)
    union = len(a | b)
    return inter, (inter / union if union else 0.0)


def brute_force_f2(adj: np.ndarray, i: int, j: int) -> Tuple[int, float]:
    """
    Direct co-citation from a dense adjacency ``adj[s, t]=1`` (s cites t).

    Used in unit tests to verify the sparse implementation (AC-1).
    """
    citers_i = adj[:, i]
    citers_j = adj[:, j]
    raw = int(np.dot(citers_i, citers_j))
    ci, cj = int(citers_i.sum()), int(citers_j.sum())
    denom = (ci * cj) ** 0.5
    return raw, (raw / denom if denom > 0 else 0.0)
