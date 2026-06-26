"""
Raw per-edge feature vectors for Stage-2 adjudication.

The :class:`FeatureVector` keeps features grouped by their symmetry, because the
symmetry structure itself encodes edge directionality (the core design
constraint). It is the single input contract shared by every
:class:`EdgeClassifier` implementation (rule-based now, learned later), so the
features are never collapsed into one scalar.

``load_feature_table`` reads ``edge_features.parquet`` and indexes rows by the
directed DOI pair ``(paper_i_doi, paper_j_doi)`` so a Stage-1 edge
``source_doi -> target_doi`` joins directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import pandas as pd

from src.core.paths import citation_edge_features
from src.service.gragh.citation.index import normalize_doi


# Column groups (kept explicit so a learned model can consume the same split).
SYMMETRIC_COLUMNS = (
    "bibcoupling_jaccard",
    "cocitation_salton",
    "author_jaccard",
    "title_sim",
    "abstract_sim",
)
ASYMMETRIC_COLUMNS = (
    "citation_freq",
    "pagerank_target",
)


def _to_float(value) -> Optional[float]:
    """Coerce a parquet cell to float, mapping NaN/None to None."""
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


@dataclass
class FeatureVector:
    """Raw feature values for one directed edge ``i -> j``, grouped by symmetry."""

    # Symmetric family (encodes "parallel / related work").
    bibcoupling_jaccard: Optional[float] = None
    cocitation_salton: Optional[float] = None
    author_jaccard: Optional[float] = None
    title_sim: Optional[float] = None
    abstract_sim: Optional[float] = None

    # Asymmetric / directed family (encodes "inheritance / importance").
    citation_freq: Optional[float] = None
    pagerank_target: Optional[float] = None

    # True when no feature row was found for this edge.
    missing: bool = False

    @property
    def symmetric(self) -> Dict[str, Optional[float]]:
        return {c: getattr(self, c) for c in SYMMETRIC_COLUMNS}

    @property
    def asymmetric(self) -> Dict[str, Optional[float]]:
        return {c: getattr(self, c) for c in ASYMMETRIC_COLUMNS}

    def snapshot(self) -> Dict[str, Optional[float]]:
        """Flat dict of all feature values for the audit feature_snapshot."""
        data = {c: getattr(self, c) for c in SYMMETRIC_COLUMNS + ASYMMETRIC_COLUMNS}
        data["missing"] = self.missing
        return data

    @classmethod
    def from_row(cls, row: pd.Series) -> "FeatureVector":
        return cls(
            bibcoupling_jaccard=_to_float(row.get("bibcoupling_jaccard")),
            cocitation_salton=_to_float(row.get("cocitation_salton")),
            author_jaccard=_to_float(row.get("author_jaccard")),
            title_sim=_to_float(row.get("title_sim")),
            abstract_sim=_to_float(row.get("abstract_sim")),
            citation_freq=_to_float(row.get("citation_freq")),
            pagerank_target=_to_float(row.get("pagerank_target")),
            missing=False,
        )


# Directed-pair key -> feature vector.
FeatureTable = Dict[Tuple[str, str], FeatureVector]


def load_feature_table(
    username: str,
    dataset_name: str,
    features_path: Optional[str] = None,
    logger=None,
) -> FeatureTable:
    """
    Load ``edge_features.parquet`` into a directed-pair feature lookup.

    Args:
        username: Target user.
        dataset_name: Target dataset.
        features_path: Optional explicit parquet path (defaults to the dataset's
            ``citation-features/edge_features.parquet``).
        logger: Optional loguru logger.

    Returns:
        Mapping ``(norm_source_doi, norm_target_doi) -> FeatureVector``.
    """
    path = features_path or str(citation_edge_features(username, dataset_name))
    df = pd.read_parquet(path)

    table: FeatureTable = {}
    for _, row in df.iterrows():
        i_doi = normalize_doi(row.get("paper_i_doi"))
        j_doi = normalize_doi(row.get("paper_j_doi"))
        if not i_doi or not j_doi:
            continue
        table[(i_doi, j_doi)] = FeatureVector.from_row(row)

    if logger is not None:
        logger.info(
            "stage2: loaded {n} edge feature rows from {p}",
            n=len(table), p=path,
        )
    return table


def lookup(
    table: FeatureTable, source_doi: str, target_doi: str
) -> FeatureVector:
    """Fetch the feature vector for a directed edge, or a missing placeholder."""
    key = (normalize_doi(source_doi), normalize_doi(target_doi))
    fv = table.get(key)
    if fv is None:
        return FeatureVector(missing=True)
    return fv
