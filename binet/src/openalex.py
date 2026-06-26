"""
OpenAlex data source (primary, §2).

OpenAlex is the recommended primary source because a single API provides both
references (``referenced_works``) and cited-by (``filter=cites:{id}``),
simplifying the bidirectional logic.

API notes:
    - Look up a work by DOI:  GET /works/https://doi.org/{doi}
    - referenced_works are OpenAlex IDs (e.g. ``https://openalex.org/W123``);
      we resolve them back to DOIs in batches via ``filter=openalex_id:...``.
    - cited-by: GET /works?filter=cites:{openalex_id} with cursor paging.
"""

from __future__ import annotations

import logging
from typing import List, Optional

from binet.doi_utils import normalize_doi
from binet.errors import DeterministicFailure
from binet.models import PaperMeta
from binet.src.base import BaseSource

logger = logging.getLogger("binet.openalex")

_API = "https://api.openalex.org"
# OpenAlex allows up to 50 ids in an OR filter; keep a safe batch size.
_BATCH = 50
_PER_PAGE = 200


def _short_id(openalex_id: Optional[str]) -> Optional[str]:
    """Reduce a full OpenAlex URL id to its bare ``W...`` form."""
    if not openalex_id:
        return None
    return openalex_id.rstrip("/").split("/")[-1]


def _meta_from_work(work: dict) -> PaperMeta:
    """Build a PaperMeta from an OpenAlex work object."""
    doi = normalize_doi(work.get("doi"))
    authors = [
        a.get("author", {}).get("display_name")
        for a in work.get("authorships", [])
        if a.get("author", {}).get("display_name")
    ]
    return PaperMeta(
        doi=doi or "",
        title=work.get("title") or work.get("display_name"),
        openalex_id=_short_id(work.get("id")),
        year=work.get("publication_year"),
        authors=authors,
    )


class OpenAlexSource(BaseSource):
    """OpenAlex implementation of the CitationSource contract."""

    name = "openalex"

    def __init__(self, http):
        super().__init__(http)
        # Cache work objects keyed by DOI to avoid duplicate lookups across
        # metadata / references / citations within one node's processing.
        self._work_cache: dict = {}

    # ------------------------------------------------------------------ #
    # Internal: fetch a full work object by DOI
    # ------------------------------------------------------------------ #

    def _get_work(self, doi: str) -> Optional[dict]:
        if doi in self._work_cache:
            return self._work_cache[doi]
        url = f"{_API}/works/https://doi.org/{doi}"
        params = {"mailto": self.http.email}
        try:
            work = self.http.get_json(url, params=params)
        except DeterministicFailure:
            self._work_cache[doi] = None
            raise
        self._work_cache[doi] = work
        return work

    # ------------------------------------------------------------------ #
    # Contract methods
    # ------------------------------------------------------------------ #

    def fetch_metadata(self, doi: str) -> Optional[PaperMeta]:
        work = self._get_work(doi)
        if not work:
            return None
        return _meta_from_work(work)

    def fetch_references(self, doi: str) -> List[str]:
        work = self._get_work(doi)
        if not work:
            return []
        referenced = work.get("referenced_works") or []
        ids = [_short_id(r) for r in referenced if r]
        ids = [i for i in ids if i]
        if not ids:
            return []
        return self._resolve_ids_to_dois(ids)

    def fetch_citations(self, doi: str) -> List[str]:
        work = self._get_work(doi)
        if not work:
            return []
        oa_id = _short_id(work.get("id"))
        if not oa_id:
            return []
        return self._cited_by(oa_id)

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _resolve_ids_to_dois(self, ids: List[str]) -> List[str]:
        """
        Resolve OpenAlex work IDs to DOIs in batches.

        IDs whose work has no DOI are dropped and counted (FR-4.3).
        """
        dois: List[str] = []
        for start in range(0, len(ids), _BATCH):
            chunk = ids[start:start + _BATCH]
            url = f"{_API}/works"
            params = {
                "filter": f"openalex_id:{'|'.join(chunk)}",
                "select": "id,doi",
                "per-page": _BATCH,
                "mailto": self.http.email,
            }
            try:
                data = self.http.get_json(url, params=params)
            except DeterministicFailure:
                continue
            if not data:
                continue
            returned = {}
            for w in data.get("results", []):
                returned[_short_id(w.get("id"))] = normalize_doi(w.get("doi"))
            for cid in chunk:
                doi = returned.get(cid)
                if doi:
                    dois.append(doi)
                else:
                    self.dropped_no_doi += 1
        return dois

    def _cited_by(self, openalex_id: str) -> List[str]:
        """Page through all works that cite ``openalex_id`` (cursor paging)."""
        dois: List[str] = []
        cursor = "*"
        url = f"{_API}/works"
        while cursor:
            params = {
                "filter": f"cites:{openalex_id}",
                "select": "id,doi",
                "per-page": _PER_PAGE,
                "cursor": cursor,
                "mailto": self.http.email,
            }
            try:
                data = self.http.get_json(url, params=params)
            except DeterministicFailure:
                break
            if not data:
                break
            for w in data.get("results", []):
                doi = normalize_doi(w.get("doi"))
                if doi:
                    dois.append(doi)
                else:
                    self.dropped_no_doi += 1
            cursor = data.get("meta", {}).get("next_cursor")
            if not data.get("results"):
                break
        return dois
