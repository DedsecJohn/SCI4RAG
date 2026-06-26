"""
End-to-end orchestration for Stage-2 feature adjudication.

``run_stage2(config)`` wires the stages: read Stage-1 edges/labels (from
``citation_edgelist.csv``) -> join the raw feature vectors from
``edge_features.parquet`` -> adjudicate each edge with an
:class:`EdgeClassifier` (the rule-based one by default) -> serialize the final
products into the graph directory.

The entry script only assembles a :class:`Stage2Config` and calls this function;
no core logic lives in the entry file. Edge direction is ``i -> j`` (paper i
cites paper j); final labels are ``inheritance`` / ``parallel`` / ``peripheral``.
"""

from __future__ import annotations

import csv
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from tqdm import tqdm

from src.core.logger import get_user_logger
from src.core.paths import (
    graph_citation_edgelist_csv,
    graph_citation_edgelist_final_csv,
    graph_citation_graph_final_gexf,
    graph_citation_relations_final_json,
    graph_dir,
)
from src.service.gragh.evaluate import serialize
from src.service.gragh.evaluate.stage2.classifier import (
    EdgeClassifier,
    RuleBasedClassifier,
)
from src.service.gragh.evaluate.stage2.config import STAGE2_VERSION, Stage2Config
from src.service.gragh.evaluate.stage2.features import load_feature_table, lookup


def _read_stage1_edges(path) -> List[dict]:
    """Read Stage-1 citation_edgelist.csv into edge dicts."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Stage-1 edge list not found: {path}. Run Stage-1 first."
        )
    edges: List[dict] = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            edges.append({
                "source_doi": (row.get("source_doi") or "").strip(),
                "target_doi": (row.get("target_doi") or "").strip(),
                "source_title": row.get("source_title") or "",
                "target_title": row.get("target_title") or "",
                "stage1_label": (row.get("label") or "").strip(),
            })
    return edges


def _build_nodes(edges: List[dict]) -> List[dict]:
    """Derive unique nodes (doi/title) from the union of edge endpoints."""
    titles: Dict[str, str] = {}
    for e in edges:
        for doi_key, title_key in (
            ("source_doi", "source_title"),
            ("target_doi", "target_title"),
        ):
            doi = e[doi_key]
            if not doi:
                continue
            if doi not in titles or (not titles[doi] and e[title_key]):
                titles[doi] = e[title_key]
    return [{"doi": doi, "title": title} for doi, title in titles.items()]


def run_stage2(
    config: Optional[Stage2Config] = None,
    classifier: Optional[EdgeClassifier] = None,
) -> dict:
    """
    Run the full Stage-2 adjudication and write all final products.

    Args:
        config: Full configuration; defaults are used when omitted.
        classifier: Edge classifier implementing the Stage-2 contract. Defaults
            to :class:`RuleBasedClassifier`. Pass a learned classifier (same
            Protocol, same FeatureVector input) to upgrade without other changes.

    Returns:
        A summary dict with output paths, counts and label distribution.
    """
    config = config or Stage2Config()
    classifier = classifier or RuleBasedClassifier(config)
    logger = get_user_logger(config.username, config.dataset_name)
    started = datetime.now(timezone.utc)
    logger.info(
        "stage2: start user={u} dataset={d} version={v} classifier={c}",
        u=config.username, d=config.dataset_name, v=STAGE2_VERSION,
        c=type(classifier).__name__,
    )

    # 1. Stage-1 edges + labels.
    stage1_path = graph_citation_edgelist_csv(config.username, config.dataset_name)
    edges = _read_stage1_edges(stage1_path)
    stage1_counter = Counter(e["stage1_label"] for e in edges)
    logger.info(
        "stage2: read {n} Stage-1 edges from {p} (stage1 labels={c})",
        n=len(edges), p=str(stage1_path), c=dict(stage1_counter),
    )

    # 2. Raw feature vectors (directed-pair lookup).
    table = load_feature_table(config.username, config.dataset_name, logger=logger)

    # 3. Adjudicate each edge.
    final_edges: List[dict] = []
    label_counter: Counter = Counter()
    tier_counter: Counter = Counter()

    for edge in tqdm(edges, desc="stage2 adjudicate", unit="edge", ncols=100):
        features = lookup(table, edge["source_doi"], edge["target_doi"])
        decision = classifier.classify(features, edge["stage1_label"])
        label_counter[decision.label] += 1
        tier_counter[decision.confidence_tier] += 1

        logger.info(
            "stage2: edge {s} -> {t} | stage1={s1} -> final={l} ({tier})",
            s=edge["source_doi"], t=edge["target_doi"],
            s1=edge["stage1_label"], l=decision.label,
            tier=decision.confidence_tier,
        )

        final_edges.append({
            "paper_i_doi": edge["source_doi"],
            "paper_j_doi": edge["target_doi"],
            # Aliases kept for the flat edge list / gexf writers.
            "source_doi": edge["source_doi"],
            "target_doi": edge["target_doi"],
            "source_title": edge["source_title"],
            "target_title": edge["target_title"],
            "stage1_label": edge["stage1_label"],
            "label": decision.label,
            "confidence_tier": decision.confidence_tier,
            "decision_path": decision.decision_path,
            "feature_snapshot": decision.feature_snapshot,
        })

    nodes = _build_nodes(edges)

    # 4. Run metadata for provenance.
    run_metadata = {
        "stage2_version": STAGE2_VERSION,
        "generated_at": started.isoformat(),
        "username": config.username,
        "dataset_name": config.dataset_name,
        "classifier": type(classifier).__name__,
        "n_edges": len(edges),
        "stage1_label_distribution": dict(stage1_counter),
        "final_label_distribution": dict(label_counter),
        "confidence_tier_distribution": dict(tier_counter),
        "row_semantics": "directed edge i -> j (paper_i cites paper_j); final "
                         "label in {inheritance, parallel, peripheral}",
        "config": config.to_metadata(),
    }

    relations = {
        "metadata": run_metadata,
        "nodes": nodes,
        "edges": [
            {
                "paper_i_doi": e["paper_i_doi"],
                "paper_j_doi": e["paper_j_doi"],
                "source_title": e["source_title"],
                "target_title": e["target_title"],
                "stage1_label": e["stage1_label"],
                "label": e["label"],
                "confidence_tier": e["confidence_tier"],
                "decision_path": e["decision_path"],
                "feature_snapshot": e["feature_snapshot"],
            }
            for e in final_edges
        ],
    }

    # 5. Serialize final products into the graph directory.
    if config.output_dir:
        base = Path(config.output_dir)
        relations_path = base / "citation_relations_final.json"
        edgelist_path = base / "citation_edgelist_final.csv"
        gexf_path = base / "citation_graph_final.gexf"
    else:
        base = graph_dir(config.username, config.dataset_name)
        relations_path = graph_citation_relations_final_json(config.username, config.dataset_name)
        edgelist_path = graph_citation_edgelist_final_csv(config.username, config.dataset_name)
        gexf_path = graph_citation_graph_final_gexf(config.username, config.dataset_name)

    serialize.write_relations_final_json(relations, relations_path)
    serialize.write_final_edgelist_csv(final_edges, edgelist_path)
    serialize.write_final_gexf(nodes, final_edges, gexf_path)

    logger.success(
        "stage2: done edges={e} final_labels={labels} tiers={tiers} -> {d}",
        e=len(edges), labels=dict(label_counter), tiers=dict(tier_counter),
        d=str(base),
    )

    return {
        "output_dir": str(base),
        "citation_relations_final": str(relations_path),
        "citation_edgelist_final": str(edgelist_path),
        "citation_graph_final_gexf": str(gexf_path),
        "n_edges": len(edges),
        "stage1_label_distribution": dict(stage1_counter),
        "final_label_distribution": dict(label_counter),
        "confidence_tier_distribution": dict(tier_counter),
    }
