"""
Configuration and tunable thresholds for Stage-2 feature adjudication.

Every threshold lives here (the single tuning surface) with a comment stating
its domain rationale. Initial values are set from domain priors; they are meant
to be re-tuned once a labeled edge set of a few hundred exists. The entry script
only assembles a :class:`Stage2Config` and calls
:func:`src.service.gragh.evaluate.stage2.pipeline.run_stage2`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


# Version stamped into every Stage-2 product's metadata for traceability.
STAGE2_VERSION = "1.0.0"

# Final label set after Stage-2 adjudication.
INHERITANCE_LABEL = "inheritance"
PARALLEL_LABEL = "parallel"
PERIPHERAL_LABEL = "peripheral"

# Confidence tiers attached to each decision for human audit.
TIER_HIGH = "high"
TIER_MEDIUM = "medium"
TIER_LOW = "low"


@dataclass
class Stage2Config:
    """Full configuration for one Stage-2 adjudication run."""

    # ── Target dataset ─────────────────────────────────────────────────
    username: str = "administrator"
    dataset_name: str = "leiting"

    # When empty, products are written to users/{username}/{dataset}/graph/.
    output_dir: str = ""

    # ── Subtask A: false-negative inheritance recovery (asymmetric) ─────
    # citation_freq = N_ij / T_i = share of paper i's in-text citation
    # discussion devoted to j. A high share means i engages deeply with j,
    # the hallmark of building upon (inheriting) j's work.
    # Domain prior: >=0.10 (one tenth of all in-text discussion on a single
    # reference) is a strong "deep engagement" signal worth recovering.
    cf_recover_threshold: float = 0.10
    # >=0.15 is an even more decisive share -> high-confidence recovery.
    cf_high_threshold: float = 0.15
    # pagerank_target is the prestige/centrality of the cited paper j. A more
    # central target is more likely to be a foundational work being inherited;
    # used only as a confidence booster (never as the sole trigger), since its
    # absolute scale is dataset-dependent.
    pagerank_target_boost: float = 0.05

    # ── Subtask B: parallel vs peripheral (symmetric coupling) ─────────
    # Primary symmetric signals of "related/parallel work":
    # cocitation_salton: how often i and j are cited together by others
    # (normalized). Strong co-citation => the community treats them as related.
    # Domain prior: Salton index >=0.30 is a conventional "related" threshold.
    cocitation_salton_threshold: float = 0.30
    # bibcoupling_jaccard: shared references between i and j (Jaccard). Sharing
    # >=5% of the combined bibliography indicates overlapping foundations.
    bibcoupling_jaccard_threshold: float = 0.05
    # author_jaccard: author overlap. >=0.25 implies a shared research group,
    # a strong indicator of parallel/related (not peripheral) work.
    author_jaccard_threshold: float = 0.25

    # Content-similarity boosters (NOT primary): in tight topical corpora these
    # are uniformly high, so they only raise confidence, never decide alone.
    title_sim_threshold: float = 0.85
    abstract_sim_threshold: float = 0.90

    # ── Logging ────────────────────────────────────────────────────────
    log_level: str = "INFO"

    def to_metadata(self) -> dict:
        """Return a JSON-serialisable snapshot for product metadata."""
        data = asdict(self)
        data["stage2_version"] = STAGE2_VERSION
        return data
