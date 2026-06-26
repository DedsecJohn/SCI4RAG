"""Feature computation modules for citation feature engineering.

Features: bibliographic coupling, co-citation, pagerank, author jaccard,
semantic (title/abstract) similarity, and in-text citation frequency (CF).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple


@dataclass
class FeatureResult:
    """
    Uniform return type for every feature computation.

    Attributes:
        edge_values: pair (i, j) [i < j local ids] -> {column_name: value|None}.
            ``None`` means ``null`` (cannot compute, GC-4); ``0`` means a true
            zero. Pairs absent from the dict are "not computed".
        node_values: local paper id -> {column_name: value} (e.g. pagerank).
        metadata: feature-specific provenance for GC-3 / §5.4.
        columns: ordered edge column names this feature contributes.
    """

    edge_values: Dict[Tuple[int, int], Dict[str, Optional[float]]] = field(default_factory=dict)
    node_values: Dict[int, Dict[str, object]] = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)
    columns: tuple = ()
    # Optional dense per-paper embedding matrices (semantic feature); row index
    # recorded in node_values[pid]["title_embedding_row"]/["abstract_embedding_row"].
    title_embeddings: object = None
    abstract_embeddings: object = None
