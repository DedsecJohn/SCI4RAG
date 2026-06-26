"""
DOI normalization (GC-1.2) and the local paper index (GC-1.3).

The candidate-pair universe is the set of *local* papers: those registered in
``documents.json`` that have a DOI and a cleaned ``doi.json`` on disk. Every
matrix/array operation downstream uses the integer ids assigned here; all
outward-facing products write back normalized DOIs.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

from src.core.paths import (
    clean_citation_by_article_json,
    clean_doi_json,
    clean_document_md,
    clean_references_json,
)
from src.core.utils import load_json, exists
from src.service.document.load_document import load_document_metadata


# Prefixes stripped during DOI normalization (case-insensitive), GC-1.2.
_DOI_PREFIX_RE = re.compile(r'^\s*(?:https?://(?:dx\.)?doi\.org/|doi:)\s*', re.IGNORECASE)


def normalize_doi(raw: Optional[str]) -> Optional[str]:
    """
    Normalize a DOI to the project-wide canonical form (GC-1.2).

    Rules (authoritative, self-contained):
        1. Strip the prefixes ``https://doi.org/``, ``http://doi.org/``,
           ``doi:`` (case-insensitive).
        2. Lowercase.
        3. Strip surrounding whitespace.

    Args:
        raw: Raw DOI string (possibly None / URL-prefixed).

    Returns:
        Normalized DOI, or None if input is empty/None.

    Example:
        ``https://doi.org/10.48550/arXiv.2410.05779`` -> ``10.48550/arxiv.2410.05779``
    """
    if not raw:
        return None
    text = _DOI_PREFIX_RE.sub("", str(raw))
    text = text.strip().lower()
    return text or None


@dataclass
class PaperRecord:
    """One local paper in the candidate-pair universe."""
    id: int                       # internal integer id (matrix row/col)
    doi: str                      # normalized DOI (unique primary key, GC-1.1)
    title: str
    file_id: str                  # data_clean directory key
    raw_doi: Optional[str] = None  # DOI exactly as stored in documents.json


@dataclass
class PaperIndex:
    """Bidirectional map: int id <-> normalized DOI <-> title (GC-1.3)."""

    username: str
    dataset_name: str
    papers: List[PaperRecord] = field(default_factory=list)
    doi_to_id: Dict[str, int] = field(default_factory=dict)

    @property
    def n(self) -> int:
        return len(self.papers)

    @property
    def dois(self) -> List[str]:
        return [p.doi for p in self.papers]

    def title(self, paper_id: int) -> str:
        return self.papers[paper_id].title

    def doi(self, paper_id: int) -> str:
        return self.papers[paper_id].doi

    def file_id(self, paper_id: int) -> str:
        return self.papers[paper_id].file_id


def discover_local_papers(
    username: str,
    dataset_name: str,
    require_doi_json: bool = True,
) -> PaperIndex:
    """
    Build the :class:`PaperIndex` for the target dataset.

    A paper is included when it has a DOI in ``documents.json`` and (optionally)
    a cleaned ``doi.json`` on disk. Duplicate normalized DOIs collapse to a
    single node (GC-1.1 / AC-5) keeping the first occurrence.

    Args:
        username: Target user.
        dataset_name: Target dataset.
        require_doi_json: When True, only papers with an existing doi.json are
            included (doi.json carries title/authors/abstract used by F4/F6).

    Returns:
        Populated PaperIndex (integer ids assigned in registration order).
    """
    metadata = load_document_metadata(username, dataset_name) or {}

    index = PaperIndex(username=username, dataset_name=dataset_name)
    next_id = 0

    for file_id, meta in metadata.items():
        raw_doi = meta.get("doi")
        norm = normalize_doi(raw_doi)
        if not norm:
            continue
        if norm in index.doi_to_id:
            continue  # de-duplicate (AC-5)
        if require_doi_json and not exists(clean_doi_json(username, dataset_name, file_id)):
            continue

        # Prefer the cleaned doi.json title; fall back to documents.json file_name.
        doi_meta = load_json(clean_doi_json(username, dataset_name, file_id)) or {}
        title = doi_meta.get("title") or meta.get("file_name") or ""

        record = PaperRecord(
            id=next_id,
            doi=norm,
            title=title,
            file_id=file_id,
            raw_doi=raw_doi,
        )
        index.papers.append(record)
        index.doi_to_id[norm] = next_id
        next_id += 1

    return index


def build_doi_to_file_id(username: str, dataset_name: str) -> Dict[str, str]:
    """
    Build a reverse lookup: normalized DOI -> file_id, from documents.json.

    Useful for resolving graph-node DOIs back to local cleaned-data directories.
    """
    metadata = load_document_metadata(username, dataset_name) or {}
    mapping: Dict[str, str] = {}
    for file_id, meta in metadata.items():
        norm = normalize_doi(meta.get("doi"))
        if norm and norm not in mapping:
            mapping[norm] = file_id
    return mapping


def local_paths(index: PaperIndex, paper_id: int) -> Dict[str, object]:
    """Return the on-disk cleaned-data paths for a local paper."""
    fid = index.file_id(paper_id)
    u, d = index.username, index.dataset_name
    return {
        "doi_json": clean_doi_json(u, d, fid),
        "references_json": clean_references_json(u, d, fid),
        "document_md": clean_document_md(u, d, fid),
        "citation_by_article_json": clean_citation_by_article_json(u, d, fid),
    }


def load_local_out_dois(index: PaperIndex) -> Dict[int, Set[str]]:
    """
    Build each local paper's set of *local* cited DOIs (out-references).

    A local paper ``i`` "cites" local paper ``j`` when ``j``'s normalized DOI
    appears among ``i``'s references. The reference DOIs are gathered from the
    union of ``references.json`` (reference entries' ``doi``) and
    ``citation_by_article.json`` (resolved cited-article ``doi``), then
    intersected with the local DOI universe so only local->local edges remain.
    Self-references are excluded.

    Args:
        index: Local paper index (candidate universe).

    Returns:
        local paper id -> set of normalized DOIs of *other local* papers it cites.
    """
    local_dois = set(index.dois)
    out: Dict[int, Set[str]] = {}
    for paper in index.papers:
        paths = local_paths(index, paper.id)
        cited: Set[str] = set()

        refs = load_json(paths["references_json"]) or {}
        if isinstance(refs, dict):
            for info in refs.values():
                norm = normalize_doi((info or {}).get("doi"))
                if norm:
                    cited.add(norm)

        cba = load_json(paths["citation_by_article_json"]) or {}
        for article in (cba.get("articles") or []):
            norm = normalize_doi((article or {}).get("doi"))
            if norm:
                cited.add(norm)

        # Keep only edges that point to *other* local papers.
        out[paper.id] = {d for d in (cited & local_dois) if d != paper.doi}
    return out
