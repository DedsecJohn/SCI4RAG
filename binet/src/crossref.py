"""
CrossRef data source (references / backward only, §2).

CrossRef's ``/works/{doi}`` payload contains a ``reference`` list with the out-
edge DOIs. CrossRef does NOT provide cited-by, so ``fetch_citations`` MUST raise
``NotSupportedError`` (§5.2) rather than silently returning wrong data.
"""

from __future__ import annotations

import logging
from typing import List, Optional

from binet.doi_utils import normalize_doi
from binet.errors import DeterministicFailure, NotSupportedError
from binet.models import PaperMeta
from binet.src.base import BaseSource

logger = logging.getLogger("binet.crossref")

_API = "https://api.crossref.org/works"


class CrossRefSource(BaseSource):
    """CrossRef implementation (references only)."""

    name = "crossref"

    def _get_message(self, doi: str) -> Optional[dict]:
        url = f"{_API}/{doi}"
        params = {"mailto": self.http.email}
        data = self.http.get_json(url, params=params)
        if not data:
            return None
        return data.get("message")

    def fetch_metadata(self, doi: str) -> Optional[PaperMeta]:
        try:
            message = self._get_message(doi)
        except DeterministicFailure:
            return None
        if not message:
            return None
        title_list = message.get("title") or []
        authors = []
        for a in message.get("author", []):
            given = (a.get("given") or "").strip()
            family = (a.get("family") or "").strip()
            full = f"{given} {family}".strip()
            if full:
                authors.append(full)
        year = None
        for key in ("published-print", "published", "issued", "published-online"):
            parts = message.get(key, {}).get("date-parts")
            if parts and parts[0] and parts[0][0]:
                year = parts[0][0]
                break
        return PaperMeta(
            doi=normalize_doi(message.get("DOI")) or doi,
            title=title_list[0] if title_list else None,
            year=year,
            authors=authors,
        )

    def fetch_references(self, doi: str) -> List[str]:
        try:
            message = self._get_message(doi)
        except DeterministicFailure:
            return []
        if not message:
            return []
        dois: List[str] = []
        for ref in message.get("reference", []):
            target = normalize_doi(ref.get("DOI"))
            if target:
                dois.append(target)
            else:
                # Reference without a usable DOI is dropped and counted (FR-4.3).
                self.dropped_no_doi += 1
        return dois

    def fetch_citations(self, doi: str) -> List[str]:
        raise NotSupportedError(
            "CrossRef does not provide cited-by; use OpenAlex or Semantic Scholar."
        )
