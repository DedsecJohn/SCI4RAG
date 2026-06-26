"""
Core data structures for binet.

These dataclasses define the in-memory representation of the citation graph
and the BFS queue elements (§5.3).
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import List, Optional


class Direction(str, Enum):
    """
    Expansion direction semantics.

    - BACKWARD: follow the paper's references (out-edges, "who it cites").
    - FORWARD:  follow the paper's citations / cited-by (in-edges, "who cites it").
    - BOTH:     expand in both directions.
    """

    BACKWARD = "backward"
    FORWARD = "forward"
    BOTH = "both"


@dataclass
class PaperMeta:
    """
    Minimal paper metadata returned by a data source.

    A node MUST have at least ``doi`` and ``title`` (§3 / §4.1). When the title
    is missing, callers fill it with a placeholder; that logic lives in the
    crawler, not here, so a source MAY legitimately return ``title=None``.
    """

    doi: str
    title: Optional[str] = None
    openalex_id: Optional[str] = None
    year: Optional[int] = None
    authors: List[str] = field(default_factory=list)


@dataclass
class QueueItem:
    """
    A BFS queue element (§5.3 MUST).

    Carries at least ``(doi, depth, direction_to_expand)`` so that each
    direction can honour its own depth budget (FR-2.3).
    """

    doi: str
    depth: int
    direction: Direction

    def to_dict(self) -> dict:
        return {"doi": self.doi, "depth": self.depth, "direction": self.direction.value}

    @classmethod
    def from_dict(cls, d: dict) -> "QueueItem":
        return cls(doi=d["doi"], depth=int(d["depth"]), direction=Direction(d["direction"]))


@dataclass
class NodeRecord:
    """
    A graph node as stored by the crawler and serialized to JSON (§4.1).
    """

    doi: str
    title: str = "Unknown Title"
    depth: int = 0
    is_seed: bool = False

    def to_dict(self) -> dict:
        return {
            "doi": self.doi,
            "title": self.title,
            "depth": self.depth,
            "is_seed": self.is_seed,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "NodeRecord":
        return cls(
            doi=d["doi"],
            title=d.get("title", "Unknown Title"),
            depth=int(d.get("depth", 0)),
            is_seed=bool(d.get("is_seed", False)),
        )


@dataclass(frozen=True)
class Edge:
    """
    A directed edge: ``source_doi`` cites ``target_doi`` (FR-4.1).

    Frozen so it can live in a set for O(1) de-duplication (FR-4.2).
    """

    source_doi: str
    target_doi: str

    def to_dict(self) -> dict:
        return {"source_doi": self.source_doi, "target_doi": self.target_doi}

    @classmethod
    def from_dict(cls, d: dict) -> "Edge":
        return cls(source_doi=d["source_doi"], target_doi=d["target_doi"])
