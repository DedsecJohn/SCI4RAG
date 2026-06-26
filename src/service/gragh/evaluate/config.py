"""
Configuration and defaults for the citation-edge relation evaluation pipeline.

All tunable parameters live here so the entry script only assembles a config and
calls :func:`src.service.gragh.evaluate.pipeline.run`. Edges are *directed*
citation edges ``i -> j`` (paper ``i`` cites paper ``j``); the candidate
universe is restricted to local papers that actually cite one another
(``candidate_scope="cited_local"``).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Tuple


# Version stamped into every product's metadata for traceability.
RELATION_EVAL_VERSION = "1.0.0"

# Canonical label set for this run (requirement: inheritance vs unknown).
INHERITANCE_LABEL = "inheritance"
UNKNOWN_LABEL = "unknown"


@dataclass
class RelationEvalConfig:
    """Full configuration for one relation-evaluation run."""

    # ── Target dataset (defines the candidate universe) ────────────────
    username: str = "administrator"
    dataset_name: str = "leiting"

    # ── Output ─────────────────────────────────────────────────────────
    # When empty, products are written to users/{username}/{dataset}/graph/.
    output_dir: str = ""

    # ── Candidate edges ────────────────────────────────────────────────
    # "cited_local": only directed edges between local papers that cite each
    # other (default). "all_local": every unordered local pair (legacy).
    candidate_scope: str = "cited_local"
    candidate_pair_threshold: int = 5_000_000

    # ── LLM ────────────────────────────────────────────────────────────
    # Low temperature keeps the binary judgement stable.
    llm_temperature: float = 0.1
    # Cap on how many citation contexts are sent for a single edge (guards
    # against an over-long prompt when a pair is cited very many times).
    max_contexts_per_edge: int = 12

    # ── Labels ─────────────────────────────────────────────────────────
    label_set: Tuple[str, ...] = (INHERITANCE_LABEL, UNKNOWN_LABEL)

    # ── Logging ────────────────────────────────────────────────────────
    log_level: str = "INFO"

    def to_metadata(self) -> dict:
        """Return a JSON-serialisable snapshot for product metadata."""
        data = asdict(self)
        data["label_set"] = list(self.label_set)
        data["relation_eval_version"] = RELATION_EVAL_VERSION
        return data
