"""
Reporting: coverage rates, coverage audit, and feature correlation matrix (§5.3).

- Effective coverage per feature = non-null pairs / total candidate pairs (GC-4.2).
- Coverage audit = mean in-graph in-degree of the local papers, with reliability
  tiers for F2/F3 (AC-9).
- Correlation = 6x6 Pearson + Spearman over one representative column per
  feature, for downstream collinearity diagnostics (AC-8).
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd


# One representative (normalized) column per feature for the correlation matrix.
FEATURE_REPRESENTATIVE = {
    "bibcoupling": ["bibcoupling_jaccard"],
    "cocitation": ["cocitation_salton"],
    "pagerank": ["pagerank_min", "pagerank_target", "pagerank_product"],
    "author": ["author_jaccard"],
    "title_sim": ["title_sim"],
    "abstract_sim": ["abstract_sim"],
    "citation_freq": ["citation_freq"],
}


def compute_coverage(edge_df: pd.DataFrame, feature_columns: List[str]) -> Dict[str, dict]:
    """Effective coverage per feature column (GC-4.2)."""
    total = len(edge_df)
    coverage: Dict[str, dict] = {}
    for col in feature_columns:
        if col not in edge_df.columns:
            continue
        non_null = int(edge_df[col].notna().sum())
        coverage[col] = {
            "non_null_pairs": non_null,
            "total_pairs": total,
            "coverage_rate": (non_null / total) if total else 0.0,
        }
    return coverage


def coverage_audit(
    index,
    graph,
    high_threshold: int = 50,
    medium_threshold: int = 10,
) -> dict:
    """
    Coverage-completeness audit for F2/F3 (AC-9).

    Computes the mean in-graph in-degree of the local papers and assigns a
    reliability tier (the higher the in-degree, the more in-graph co-citers and
    the more meaningful the PageRank prestige).
    """
    in_degrees = []
    per_paper = {}
    for paper in index.papers:
        gi = graph.index_of(paper.doi)
        kin = int(graph.in_degree[gi]) if gi is not None else None
        per_paper[paper.doi] = kin
        if kin is not None:
            in_degrees.append(kin)

    mean_kin = float(np.mean(in_degrees)) if in_degrees else 0.0
    if mean_kin >= high_threshold:
        tier = "high"
    elif mean_kin >= medium_threshold:
        tier = "medium"
    else:
        tier = "low"

    return {
        "local_papers": index.n,
        "papers_in_graph": len(in_degrees),
        "mean_in_graph_in_degree": mean_kin,
        "per_paper_in_degree": per_paper,
        "f2_f3_reliability_tier": tier,
        "thresholds": {"high": high_threshold, "medium": medium_threshold},
        "note": (
            "Higher mean in-degree => more in-graph co-citers (F2) and more "
            "meaningful PageRank prestige (F3). Low tier means F2/F3 should be "
            "treated cautiously."
        ),
    }


def _representative_columns(edge_df: pd.DataFrame) -> Dict[str, str]:
    """Pick the first available representative column for each feature."""
    chosen: Dict[str, str] = {}
    for feat, candidates in FEATURE_REPRESENTATIVE.items():
        for col in candidates:
            if col in edge_df.columns:
                chosen[feat] = col
                break
    return chosen


def correlation_matrix(edge_df: pd.DataFrame) -> dict:
    """Pearson + Spearman correlation over representative feature columns (AC-8)."""
    chosen = _representative_columns(edge_df)
    feats = list(chosen.keys())
    cols = [chosen[f] for f in feats]
    sub = edge_df[cols].apply(pd.to_numeric, errors="coerce")

    def _matrix(method: str) -> List[List[Optional[float]]]:
        corr = sub.corr(method=method, min_periods=2)
        out = []
        for a in cols:
            row = []
            for b in cols:
                val = corr.loc[a, b] if (a in corr.index and b in corr.columns) else np.nan
                row.append(None if pd.isna(val) else float(val))
            out.append(row)
        return out

    return {
        "features": feats,
        "columns": cols,
        "pearson": _matrix("pearson"),
        "spearman": _matrix("spearman"),
    }


def build_report(
    edge_df: pd.DataFrame,
    feature_columns: List[str],
    index,
    graph,
    feature_metadata: Dict[str, dict],
    run_metadata: dict,
) -> dict:
    """Assemble the full feature_report.json content (§5.3)."""
    return {
        "feature_coverage": compute_coverage(edge_df, feature_columns),
        "coverage_audit": coverage_audit(index, graph),
        "correlation": correlation_matrix(edge_df),
        "feature_metadata": feature_metadata,
        "model_parameter_locks": {
            "pagerank": feature_metadata.get("pagerank", {}),
            "semantic": feature_metadata.get("semantic", {}),
        },
        "run_metadata": run_metadata,
    }
