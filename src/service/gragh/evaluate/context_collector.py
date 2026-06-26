"""
Collect in-text citation contexts for a directed edge ``i -> j``.

For edge ``i -> j`` (paper ``i`` cites paper ``j``), the evidence lives in the
*citing* paper's ``citation_by_article.json``: that file groups every in-text
citation by the cited article, exposing each occurrence's ``context`` (the
citation sentence plus its neighbours). We locate the article whose normalized
DOI equals paper ``j``'s DOI and gather its citation contexts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from src.core.paths import clean_citation_by_article_json
from src.core.utils import load_json
from src.service.gragh.citation.index import PaperIndex, normalize_doi


@dataclass
class CitationContext:
    """One in-text occurrence where the citing paper references the cited one."""

    citation_id: Optional[int] = None
    markers: List[str] = field(default_factory=list)
    citation_sentence: str = ""
    context: str = ""


@dataclass
class EdgeEvidence:
    """All citation contexts supporting a directed edge ``i -> j``."""

    source_id: int                 # citing paper (i)
    target_id: int                 # cited paper (j)
    contexts: List[CitationContext] = field(default_factory=list)

    @property
    def has_context(self) -> bool:
        return len(self.contexts) > 0


# Per-paper cache so we read each citation_by_article.json at most once.
_CitationByArticleCache = Dict[str, dict]


def _load_citation_by_article(
    index: PaperIndex, paper_id: int, cache: _CitationByArticleCache
) -> dict:
    """Load (and cache) the citation_by_article.json of a local paper."""
    file_id = index.file_id(paper_id)
    if file_id not in cache:
        path = clean_citation_by_article_json(
            index.username, index.dataset_name, file_id
        )
        cache[file_id] = load_json(path) or {}
    return cache[file_id]


def collect_edge_evidence(
    index: PaperIndex,
    source_id: int,
    target_id: int,
    cache: Optional[_CitationByArticleCache] = None,
    max_contexts: int = 12,
) -> EdgeEvidence:
    """
    Gather the citation contexts for edge ``source_id -> target_id``.

    Args:
        index: Local paper index.
        source_id: Citing paper integer id (``i``).
        target_id: Cited paper integer id (``j``).
        cache: Optional shared cache of loaded citation_by_article.json files.
        max_contexts: Cap on contexts returned (guards prompt length).

    Returns:
        Populated :class:`EdgeEvidence`. ``contexts`` is empty when the citing
        paper has no in-text occurrence resolvable to the cited paper's DOI
        (e.g. the edge was inferred from references.json only).
    """
    cache = cache if cache is not None else {}
    evidence = EdgeEvidence(source_id=source_id, target_id=target_id)

    data = _load_citation_by_article(index, source_id, cache)
    target_doi = index.doi(target_id)

    for article in (data.get("articles") or []):
        if normalize_doi((article or {}).get("doi")) != target_doi:
            continue
        for cit in (article.get("citations") or []):
            evidence.contexts.append(
                CitationContext(
                    citation_id=cit.get("citation_id"),
                    markers=list(cit.get("markers") or []),
                    citation_sentence=cit.get("citation_sentence", "") or "",
                    context=cit.get("context", "") or "",
                )
            )
            if len(evidence.contexts) >= max_contexts:
                break
        # A cited DOI should appear in a single aggregated article entry.
        break

    return evidence
