"""
F4 author overlap (Author Jaccard, FR-F4).

Author sets come from each local paper's own ``doi.json`` author field. When an
ORCID / OpenAlex author id is present it is used directly (FR-F4.2); otherwise a
light name normalization is applied as a documented degradation (FR-F4.3):
``surname + first-initial`` (lowercased, punctuation removed). This does NOT do
heavy author disambiguation, so the metadata flags the homonym/synonym risk.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Set

from src.core.utils import load_json, exists
from src.service.gragh.citation.features import FeatureResult
from src.service.gragh.citation.index import PaperIndex, local_paths


AUTHOR_COLUMNS = ("author_jaccard",)

_PUNCT_RE = re.compile(r"[^\w\s\-]", re.UNICODE)
_WS_RE = re.compile(r"\s+")


def _normalize_name(name: str) -> Optional[str]:
    """
    Light name normalization to ``surname firstinitial`` (FR-F4.3).

    Examples:
        "Zhaohui Jin"  -> "jin z"
        "K. Lu"        -> "lu k"
        "X. Y. Li"     -> "li x"
        "H.L. Fu"      -> "fu h"
    """
    if not name:
        return None
    cleaned = _PUNCT_RE.sub(" ", str(name))
    cleaned = _WS_RE.sub(" ", cleaned).strip().lower()
    if not cleaned:
        return None
    tokens = cleaned.split(" ")
    if len(tokens) == 1:
        return tokens[0]
    surname = tokens[-1]
    first_initial = next((ch for ch in tokens[0] if ch.isalpha()), "")
    return f"{surname} {first_initial}".strip()


def _author_set(author_field, logger=None) -> Optional[Set[str]]:
    """
    Build the normalized author identity set for one paper.

    Accepts a list of name strings, or dicts carrying ``orcid`` / ``id`` /
    ``name``. Returns None when no author data is available (-> null, FR-F4.5).
    """
    if not author_field:
        return None
    ids: Set[str] = set()
    for entry in author_field:
        if isinstance(entry, dict):
            ident = entry.get("orcid") or entry.get("openalex_id") or entry.get("id")
            if ident:
                ids.add(f"id::{str(ident).strip().lower()}")
                continue
            name = entry.get("name") or entry.get("display_name")
        else:
            name = entry
        norm = _normalize_name(name)
        if norm:
            ids.add(norm)
    return ids or None


def load_author_sets(index: PaperIndex, logger=None) -> Dict[int, Optional[Set[str]]]:
    """Build the author set for every local paper (from doi.json)."""
    out: Dict[int, Optional[Set[str]]] = {}
    used_ids = False
    for paper in index.papers:
        path = local_paths(index, paper.id)["doi_json"]
        if not exists(path):
            out[paper.id] = None
            continue
        meta = load_json(path) or {}
        s = _author_set(meta.get("author"), logger=logger)
        if s and any(x.startswith("id::") for x in s):
            used_ids = True
        out[paper.id] = s
    out["_used_ids"] = used_ids  # type: ignore[index]
    return out


def compute_author(index: PaperIndex, candidate_pairs, logger=None) -> FeatureResult:
    """Compute author Jaccard for the local candidate edges (FR-F4.4)."""
    author_sets = load_author_sets(index, logger=logger)
    used_ids = bool(author_sets.pop("_used_ids", False))  # type: ignore[arg-type]

    result = FeatureResult(columns=AUTHOR_COLUMNS)
    for (i, j) in candidate_pairs:
        a, b = author_sets.get(i), author_sets.get(j)
        if not a or not b:
            result.edge_values[(i, j)] = {"author_jaccard": None}
            continue
        inter = len(a & b)
        union = len(a | b)
        result.edge_values[(i, j)] = {
            "author_jaccard": (inter / union) if union > 0 else 0.0
        }

    for pid in range(index.n):
        s = author_sets.get(pid)
        result.node_values.setdefault(pid, {})["authors"] = sorted(s) if s else []

    result.metadata = {
        "source": "local doi.json author field",
        "normalization": "jaccard",
        "id_based": used_ids,
        "warning": (
            "Authors are NOT disambiguated; name-based normalization "
            "(surname + first initial) risks homonyms (same name, different "
            "people) and synonyms (same person, different spellings)."
        ) if not used_ids else "Author ids used where available.",
    }
    return result
