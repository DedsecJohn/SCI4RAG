"""
Build the local directed citation graph (paper i cites paper j).

This reuses the citation feature-engineering primitives:
- :func:`discover_local_papers` for the local paper universe (DOI + doi.json).
- :func:`build_candidates` with ``scope="cited_local"`` to enumerate directed
  local citation edges ``(i, j)`` (i cites j), derived from the union of each
  paper's ``references.json`` and ``citation_by_article.json``.

Only the in-graph edges (both endpoints are local papers) are produced, which is
exactly what the relation classifier needs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from src.service.gragh.citation.candidate import build_candidates
from src.service.gragh.citation.index import PaperIndex, discover_local_papers


@dataclass
class CitationGraph:
    """Local directed citation graph.

    ``edges`` are directed integer pairs ``(i, j)`` meaning local paper ``i``
    cites local paper ``j``. ``index`` resolves integer ids to DOI/title/file_id.
    """

    index: PaperIndex
    edges: List[Tuple[int, int]] = field(default_factory=list)
    scope: str = "cited_local"

    @property
    def n_nodes(self) -> int:
        return self.index.n

    @property
    def n_edges(self) -> int:
        return len(self.edges)


def build_citation_graph(
    username: str,
    dataset_name: str,
    scope: str = "cited_local",
    threshold: int = 5_000_000,
    logger=None,
) -> CitationGraph:
    """
    Discover local papers and enumerate directed local citation edges.

    Args:
        username: Target user.
        dataset_name: Target dataset.
        scope: ``"cited_local"`` (directed local citation edges, default) or
            ``"all_local"`` (legacy unordered local pairs).
        threshold: Soft cap forwarded to candidate enumeration.
        logger: Optional loguru logger for progress/warnings.

    Returns:
        Populated :class:`CitationGraph`.
    """
    index = discover_local_papers(username, dataset_name)
    if logger is not None:
        logger.info(
            "relation-eval: discovered {n} local papers for {u}/{d}",
            n=index.n, u=username, d=dataset_name,
        )

    candidate = build_candidates(
        index, scope=scope, threshold=threshold, logger=logger,
    )

    return CitationGraph(index=index, edges=candidate.pairs, scope=candidate.scope)
