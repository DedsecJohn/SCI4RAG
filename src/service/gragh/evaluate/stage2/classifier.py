"""
Stage-2 edge classifiers.

``EdgeClassifier`` is the stable contract: it consumes a raw
:class:`FeatureVector` (plus the Stage-1 label) and returns a
:class:`Stage2Decision`. The current :class:`RuleBasedClassifier` is a
transparent, per-edge auditable decision table; a future ``LearnedClassifier``
can implement the same Protocol, consuming the *same* feature vector and
learning the weights, without touching the pipeline or the product schema.

Decision order (each step records an auditable trace entry):
1. Stage-1 ``inheritance`` is final (kept).
2. Subtask A - false-negative recovery (asymmetric family dominant).
3. Subtask B - parallel vs peripheral (symmetric family dominant).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Protocol

from src.service.gragh.evaluate.stage2.config import (
    INHERITANCE_LABEL,
    PARALLEL_LABEL,
    PERIPHERAL_LABEL,
    TIER_HIGH,
    TIER_LOW,
    TIER_MEDIUM,
    Stage2Config,
)
from src.service.gragh.evaluate.stage2.features import FeatureVector


@dataclass
class Stage2Decision:
    """Auditable outcome of classifying one directed edge."""

    label: str
    confidence_tier: str
    decision_path: List[dict] = field(default_factory=list)
    feature_snapshot: dict = field(default_factory=dict)


class EdgeClassifier(Protocol):
    """Stable Stage-2 classifier contract (rule-based now, learned later)."""

    def classify(
        self, features: FeatureVector, stage1_label: str
    ) -> Stage2Decision:
        ...


def _ge(value: Optional[float], threshold: float) -> bool:
    """True iff value is present and >= threshold."""
    return value is not None and value >= threshold


def _step(rule: str, fired: bool, detail: str, **features) -> dict:
    """Build one decision-path trace entry."""
    return {"rule": rule, "fired": fired, "detail": detail, "features": features}


class RuleBasedClassifier:
    """Transparent threshold + decision-table adjudicator (Stage-2)."""

    def __init__(self, config: Stage2Config):
        self.config = config

    def classify(
        self, features: FeatureVector, stage1_label: str
    ) -> Stage2Decision:
        cfg = self.config
        path: List[dict] = []

        # ── Step 0: Stage-1 inheritance is final. ──────────────────────
        if stage1_label == INHERITANCE_LABEL:
            path.append(_step(
                "stage1_inheritance", True,
                "Stage-1 LLM already classified this edge as inheritance; kept.",
            ))
            return Stage2Decision(
                label=INHERITANCE_LABEL,
                confidence_tier=TIER_HIGH,
                decision_path=path,
                feature_snapshot=features.snapshot(),
            )

        path.append(_step(
            "stage1_unknown", True,
            f"Stage-1 label='{stage1_label}'; entering Stage-2 adjudication.",
        ))

        # ── Subtask A: false-negative inheritance recovery (asymmetric). ─
        cf = features.citation_freq
        pr = features.pagerank_target
        recovered = _ge(cf, cfg.cf_recover_threshold)
        path.append(_step(
            "A_citation_freq_recovery", recovered,
            (
                f"citation_freq={cf} vs recover>={cfg.cf_recover_threshold}: "
                "high in-text discussion share signals deep engagement / "
                "inheritance." if recovered else
                f"citation_freq={cf} < recover {cfg.cf_recover_threshold}: "
                "no asymmetric recovery."
            ),
            citation_freq=cf, pagerank_target=pr,
        ))
        if recovered:
            strong_cf = _ge(cf, cfg.cf_high_threshold)
            pr_boost = _ge(pr, cfg.pagerank_target_boost)
            tier = TIER_HIGH if (strong_cf or pr_boost) else TIER_MEDIUM
            path.append(_step(
                "A_confidence", True,
                (
                    f"cf_high(>={cfg.cf_high_threshold})={strong_cf}, "
                    f"pagerank_target_boost(>={cfg.pagerank_target_boost})="
                    f"{pr_boost} -> tier={tier}."
                ),
                citation_freq=cf, pagerank_target=pr,
            ))
            return Stage2Decision(
                label=INHERITANCE_LABEL,
                confidence_tier=tier,
                decision_path=path,
                feature_snapshot=features.snapshot(),
            )

        # ── Subtask B: parallel vs peripheral (symmetric family). ───────
        cocit = features.cocitation_salton
        bibc = features.bibcoupling_jaccard
        auth = features.author_jaccard
        primary = {
            "cocitation_salton": _ge(cocit, cfg.cocitation_salton_threshold),
            "bibcoupling_jaccard": _ge(bibc, cfg.bibcoupling_jaccard_threshold),
            "author_jaccard": _ge(auth, cfg.author_jaccard_threshold),
        }
        n_fired = sum(primary.values())

        # Content similarity is a confidence booster only (uniformly high in
        # tight topical corpora), never a primary trigger.
        content_high = (
            _ge(features.title_sim, cfg.title_sim_threshold)
            or _ge(features.abstract_sim, cfg.abstract_sim_threshold)
        )

        if n_fired >= 1:
            fired_names = [k for k, v in primary.items() if v]
            tier = TIER_HIGH if (n_fired >= 2 or content_high) else TIER_MEDIUM
            path.append(_step(
                "B_parallel", True,
                (
                    f"symmetric coupling fired={fired_names} "
                    f"(cocitation={cocit}, bibcoupling={bibc}, author={auth}); "
                    f"content_high={content_high} -> parallel, tier={tier}."
                ),
                cocitation_salton=cocit, bibcoupling_jaccard=bibc,
                author_jaccard=auth, title_sim=features.title_sim,
                abstract_sim=features.abstract_sim,
            ))
            return Stage2Decision(
                label=PARALLEL_LABEL,
                confidence_tier=tier,
                decision_path=path,
                feature_snapshot=features.snapshot(),
            )

        # No symmetric coupling and no asymmetric recovery -> peripheral.
        # Confidence is high when the symmetric primaries are all observed and
        # clearly low and content is not high; lower when some are missing.
        any_missing = any(
            getattr(features, c) is None
            for c in ("cocitation_salton", "bibcoupling_jaccard", "author_jaccard")
        )
        if features.missing:
            tier = TIER_LOW
        elif any_missing or content_high:
            tier = TIER_MEDIUM
        else:
            tier = TIER_HIGH
        path.append(_step(
            "B_peripheral", True,
            (
                f"all symmetric primaries below thresholds "
                f"(cocitation={cocit}, bibcoupling={bibc}, author={auth}) and "
                f"citation_freq={cf} low; content_high={content_high}, "
                f"any_missing={any_missing} -> peripheral, tier={tier}."
            ),
            cocitation_salton=cocit, bibcoupling_jaccard=bibc,
            author_jaccard=auth, citation_freq=cf,
        ))
        return Stage2Decision(
            label=PERIPHERAL_LABEL,
            confidence_tier=tier,
            decision_path=path,
            feature_snapshot=features.snapshot(),
        )
