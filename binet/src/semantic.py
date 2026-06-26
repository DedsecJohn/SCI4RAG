"""
Semantic Scholar data source (forward / cited-by fallback, §2).

Used as a fallback for OpenAlex on the forward direction. The Graph API exposes
both ``references`` and ``citations`` for a paper identified by DOI.

API notes:
    - Paper lookup:  GET /graph/v1/paper/DOI:{doi}
    - citations:     GET /graph/v1/paper/DOI:{doi}/citations (paged)
    - references:    GET /graph/v1/paper/DOI:{doi}/references (paged)
"""

from __future__ import annotations

import logging
from typing import List, Optional

from binet.doi_utils import normalize_doi
from binet.errors import DeterministicFailure
from binet.models import PaperMeta
from binet.src.base import BaseSource

logger = logging.getLogger("binet.semantic")

_API = "https://api.semanticscholar.org/graph/v1"
_LIMIT = 100


class SemanticScholarSource(BaseSource):
    """Semantic Scholar implementation (metadata + references + citations)."""

    name = "semantic_scholar"

    def fetch_metadata(self, doi: str) -> Optional[PaperMeta]:
        url = f"{_API}/paper/DOI:{doi}"
        params = {"fields": "title,year,authors,externalIds"}
        try:
            data = self.http.get_json(url, params=params)
        except DeterministicFailure:
            return None
        if not data:
            return None
        authors = [a.get("name") for a in data.get("authors", []) if a.get("name")]
        return PaperMeta(
            doi=doi,
            title=data.get("title"),
            year=data.get("year"),
            authors=authors,
        )

    def fetch_references(self, doi: str) -> List[str]:
        return self._paged_dois(doi, "references", "citedPaper")

    def fetch_citations(self, doi: str) -> List[str]:
        return self._paged_dois(doi, "citations", "citingPaper")

    # ------------------------------------------------------------------ #

    def _paged_dois(self, doi: str, endpoint: str, item_key: str) -> List[str]:
        """
        Page through references/citations and collect target DOIs.

        Args:
            doi: Source DOI.
            endpoint: ``references`` or ``citations``.
            item_key: ``citedPaper`` or ``citingPaper`` (the nested paper key).

        Returns:
            List of normalized DOIs (entries without DOI are dropped & counted).
        """
        url = f"{_API}/paper/DOI:{doi}/{endpoint}"
        dois: List[str] = []
        offset = 0
        while True:
            params = {
                "fields": f"{item_key}.externalIds,{item_key}.title",
                "limit": _LIMIT,
                "offset": offset,
            }
            try:
                data = self.http.get_json(url, params=params)
            except DeterministicFailure:
                break
            if not data:
                break
            items = data.get("data", [])
            if not items:
                break
            for entry in items:
                paper = entry.get(item_key) or {}
                ext = paper.get("externalIds") or {}
                target = normalize_doi(ext.get("DOI"))
                if target:
                    dois.append(target)
                else:
                    self.dropped_no_doi += 1
            # Semantic Scholar returns ``next`` offset when more pages exist.
            nxt = data.get("next")
            if nxt is None:
                break
            offset = nxt
        return dois
