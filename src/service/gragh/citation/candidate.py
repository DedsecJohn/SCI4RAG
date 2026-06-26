"""
Candidate-row generation (GC-5).

The candidate universe is, by default, the set of *directed citation edges*
between local papers: one row ``(i, j)`` for every "paper ``i`` cites paper
``j``" relation discovered from local reference data. Each row therefore has a
clear direction (``i`` = citing/source, ``j`` = cited/target), which lets the
in-text Citation-frequency feature carry a single ``i -> j`` value and lets
PageRank pick the cited endpoint unambiguously. Mutually-citing papers yield two
rows (``(i, j)`` and ``(j, i)``).

A legacy ``"all_local"`` scope is kept for comparison: it emits every unordered
local pair ``(i, j), i < j`` (the old ``C(N, 2)`` behaviour).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Dict, List, Optional, Set, Tuple

from src.service.gragh.citation.index import PaperIndex, load_local_out_dois


@dataclass
class CandidateSet:
    """Candidate rows (local integer ids).

    For ``candidate_scope="cited_local"`` the rows are *directed* edges
    ``(i, j)`` meaning ``i`` cites ``j``. For ``"all_local"`` they are unordered
    pairs ``(i, j), i < j``.
    """

    pairs: List[Tuple[int, int]] = field(default_factory=list)
    directed_edges: Set[Tuple[int, int]] = field(default_factory=set)
    n_papers: int = 0
    scope: str = "cited_local"
    over_threshold: bool = False

    def __len__(self) -> int:
        return len(self.pairs)


def build_candidates(
    index: PaperIndex,
    scope: str = "cited_local",
    threshold: int = 5_000_000,
    logger=None,
) -> CandidateSet:
    """
    Enumerate candidate rows (GC-5).

    Args:
        index: Local paper index (candidate universe).
        scope: ``"cited_local"`` (directed local citation edges, default) or
            ``"all_local"`` (legacy unordered local pairs).
        threshold: Soft cap; warns when exceeded (GC-2.3).
        logger: Optional loguru logger for progress/warnings (GC-6).

    Returns:
        Populated :class:`CandidateSet`.
    """
    n = index.n

    if scope == "all_local":
        pairs = list(combinations(range(n), 2))  # i < j, de-duplicated
        directed: Set[Tuple[int, int]] = set()
    else:
        out_dois = load_local_out_dois(index)
        directed = set()
        for i in range(n):
            for cited_doi in out_dois.get(i, set()):
                j = index.doi_to_id.get(cited_doi)
                if j is not None and j != i:
                    directed.add((i, j))
        pairs = sorted(directed)

    over = len(pairs) > threshold
    if logger is not None:
        if over:
            logger.warning(
                "candidate: row count {c} exceeds threshold {t}; consider "
                "chunked write-out (GC-2.3)",
                c=len(pairs), t=threshold,
            )
        if scope == "all_local":
            logger.info(
                "candidate: scope=all_local {n} local papers -> {c} unordered pairs",
                n=n, c=len(pairs),
            )
        else:
            logger.info(
                "candidate: scope=cited_local {n} local papers -> {c} directed "
                "citation edges", n=n, c=len(pairs),
            )

    return CandidateSet(
        pairs=pairs,
        directed_edges=directed,
        n_papers=n,
        scope=scope,
        over_threshold=over,
    )
