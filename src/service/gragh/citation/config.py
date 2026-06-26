"""
Configuration and defaults for the citation feature engineering pipeline.

All tunable parameters live here so the entry script only assembles a config and
calls :func:`src.service.gragh.citation.pipeline.run`. Defaults satisfy the
requirements (damping 0.85, candidate-row threshold 5e6, semantic fixed seed,
etc.).

Edge rows are *directed* citation edges ``i -> j`` (paper ``i`` cites paper
``j``); the candidate universe is restricted to local papers that actually cite
one another (``candidate_scope="cited_local"``).
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import List


# Project version stamped into every product's metadata (GC-3 / §5.4).
FEATUREFORGE_VERSION = "3.0.0"


@dataclass
class CitationFeatureConfig:
    """Full configuration for one feature-engineering run."""

    # ── Target dataset (defines the candidate universe) ────────────────
    username: str = "administrator"
    dataset_name: str = "leiting"

    # ── Inputs ─────────────────────────────────────────────────────────
    # Upstream directed citation network (binet output).
    citation_network_path: str = "binet/output/citation_network.json"

    # ── Output ─────────────────────────────────────────────────────────
    # When empty, products are written to users/{username}/{dataset}/citation-features/.
    output_dir: str = ""

    # ── Feature toggles ────────────────────────────────────────────────
    enable_bibcoupling: bool = True     # bibliographic coupling (shared references)
    enable_cocitation: bool = True      # co-citation (shared citers)
    enable_pagerank: bool = True        # pagerank prestige
    enable_author: bool = True          # author jaccard
    enable_semantic: bool = True        # title / abstract embedding cosine
    enable_citation_freq: bool = True   # in-text citation frequency (CF)

    # ── Candidate rows (GC-2 / GC-5) ───────────────────────────────────
    # "cited_local": only directed edges between local papers that cite each
    # other (default). "all_local": every unordered local pair (legacy C(N,2)).
    candidate_scope: str = "cited_local"
    # Hard cap above which the pipeline warns (GC-2.3). Local-only rows are
    # tiny, but the guard is kept for forward compatibility.
    candidate_pair_threshold: int = 5_000_000

    # ── PageRank ───────────────────────────────────────────────────────
    pagerank_method: str = "networkx"   # "networkx" (reference) or "scipy"
    pagerank_damping: float = 0.85
    pagerank_max_iter: int = 200
    pagerank_tol: float = 1e-9
    # Edge-level transforms of node PageRank to emit (FR-F3.4).
    pagerank_emit_target: bool = True   # pagerank_target  = PR(cited j)
    pagerank_emit_min: bool = True      # pagerank_min     = min(PR(i), PR(j))
    pagerank_emit_product: bool = True  # pagerank_product = PR(i) * PR(j)

    # ── Semantic similarity (title / abstract embeddings) ──────────────
    semantic_model_name: str = "allenai/specter2_base"
    # Fallback chain tried in order if the primary model fails to load.
    semantic_model_fallbacks: List[str] = field(
        default_factory=lambda: ["sentence-transformers/allenai-specter"]
    )
    semantic_max_seq_length: int = 512
    semantic_random_seed: int = 42

    # ── Logging ────────────────────────────────────────────────────────
    log_level: str = "INFO"

    def to_metadata(self) -> dict:
        """Return a JSON-serialisable snapshot for product metadata (GC-3)."""
        data = asdict(self)
        data["featureforge_version"] = FEATUREFORGE_VERSION
        return data
