"""
Citation frequency (CF) -- directed in-text citation intensity.

For a directed candidate edge ``i -> j`` ("paper ``i`` cites paper ``j``"):

    N_ij = number of times paper i mentions paper j in its body text
    T_i  = sum over all references j' in R(i) of N_ij'  (total in-text mentions)
    citation_freq(i, j) = N_ij / T_i

i.e. the share of paper ``i``'s overall in-text citation discussion devoted to
``j``. This is a *directed* feature: only the ``i -> j`` value is emitted (there
is no ``j -> i`` column on the same row; a mutually-citing pair appears as a
separate ``j -> i`` row).

Data source: each local paper's ``citation_by_article.json`` (produced by the
in-text citation extractor), where every cited article carries a normalized DOI
and a ``citation_count`` (= N for that article).
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

from src.core.utils import load_json, exists
from src.service.gragh.citation.features import FeatureResult
from src.service.gragh.citation.index import PaperIndex, local_paths, normalize_doi


CITATION_FREQ_COLUMNS = ("citation_count", "citation_total", "citation_freq")


def load_citation_counts(
    index: PaperIndex,
    logger=None,
) -> Dict[int, Optional[Tuple[Dict[str, int], int]]]:
    """
    Build per-paper in-text citation counts from ``citation_by_article.json``.

    Returns:
        paper_id -> (counts, total) where ``counts`` maps a normalized cited DOI
        to N (its in-text mention count) and ``total`` is T_i (sum over all
        cited articles). Returns ``None`` for a paper whose file is missing/empty
        so CF is ``null`` for edges out of it (GC-4).
    """
    out: Dict[int, Optional[Tuple[Dict[str, int], int]]] = {}
    for paper in index.papers:
        path = local_paths(index, paper.id)["citation_by_article_json"]
        if not exists(path):
            out[paper.id] = None
            if logger is not None:
                logger.warning(
                    "citation_freq: citation_by_article.json missing for {doi}",
                    doi=paper.doi,
                )
            continue
        data = load_json(path) or {}
        articles = data.get("articles") or []
        counts: Dict[str, int] = {}
        total = 0
        for article in articles:
            article = article or {}
            n = int(article.get("citation_count") or 0)
            total += n
            norm = normalize_doi(article.get("doi"))
            if norm:
                counts[norm] = counts.get(norm, 0) + n
        out[paper.id] = (counts, total)
    return out


def compute_citation_freq(
    index: PaperIndex,
    candidate_pairs,
    logger=None,
) -> FeatureResult:
    """Compute the directed citation-frequency feature for candidate edges."""
    per_paper = load_citation_counts(index, logger=logger)

    result = FeatureResult(columns=CITATION_FREQ_COLUMNS)

    # Node-level T_i for the node table.
    for pid in range(index.n):
        entry = per_paper.get(pid)
        total = entry[1] if entry is not None else None
        result.node_values.setdefault(pid, {})["citation_total"] = total

    for (i, j) in candidate_pairs:
        entry = per_paper.get(i)
        if entry is None:
            result.edge_values[(i, j)] = {c: None for c in CITATION_FREQ_COLUMNS}
            continue
        counts, total = entry
        n_ij = int(counts.get(index.doi(j), 0))
        freq = (n_ij / total) if total > 0 else None
        result.edge_values[(i, j)] = {
            "citation_count": n_ij,
            "citation_total": total,
            "citation_freq": freq,
        }

    covered = sum(1 for v in per_paper.values() if v is not None)
    result.metadata = {
        "source": "local citation_by_article.json (in-text citation_count per cited article)",
        "definition": "citation_freq(i,j) = N_ij / T_i; N_ij = in-text mentions of j by i; "
                      "T_i = sum of N over all cited articles in i",
        "direction": "directed edge i -> j (i cites j)",
        "papers_with_citation_data": covered,
        "null_semantics": (
            "N_ij=0 is a true zero (reference present but no located in-text "
            "marker); null means missing citation_by_article.json or T_i=0"
        ),
    }
    if logger is not None:
        logger.info(
            "citation_freq: papers_with_citation_data={c}/{n}",
            c=covered, n=index.n,
        )
    return result
