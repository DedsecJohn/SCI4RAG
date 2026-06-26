"""
Coverage audit / crawl report (AC-6).

Computes the seed papers' average in-corpus in-degree (how many papers within
the crawled corpus cite each seed) to judge whether Co-Citation / PageRank
features will be meaningful.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Set

from binet.models import Edge, NodeRecord

logger = logging.getLogger("binet.report")


def build_report(
    nodes: Dict[str, NodeRecord],
    edges: Set[Edge],
    seed_dois: List[str],
    failed: Dict[str, str],
    dropped_edges_no_doi: int,
    source_hits: Dict[str, int],
    status: str,
) -> dict:
    """
    Build the coverage-audit report dict (AC-6).

    Returns:
        A report dict including per-seed in-degree, average in-corpus in-degree,
        node/edge totals, failure stats and per-source hit counts.
    """
    node_set = set(nodes.keys())

    # In-corpus in-degree: count edges whose target is the seed AND whose source
    # is also a node in the corpus.
    in_degree: Dict[str, int] = {d: 0 for d in node_set}
    out_degree: Dict[str, int] = {d: 0 for d in node_set}
    for e in edges:
        if e.target_doi in node_set and e.source_doi in node_set:
            in_degree[e.target_doi] = in_degree.get(e.target_doi, 0) + 1
            out_degree[e.source_doi] = out_degree.get(e.source_doi, 0) + 1

    seed_in_degrees = {d: in_degree.get(d, 0) for d in seed_dois}
    avg_seed_in_degree = (
        sum(seed_in_degrees.values()) / len(seed_in_degrees)
        if seed_in_degrees else 0.0
    )

    # Depth distribution.
    depth_dist: Dict[int, int] = {}
    for n in nodes.values():
        depth_dist[n.depth] = depth_dist.get(n.depth, 0) + 1

    report = {
        "status": status,
        "total_nodes": len(nodes),
        "total_edges": len(edges),
        "seed_count": len(seed_dois),
        "failed_count": len(failed),
        "dropped_edges_no_doi": dropped_edges_no_doi,
        "source_hits": source_hits,
        "depth_distribution": {str(k): v for k, v in sorted(depth_dist.items())},
        "avg_seed_in_corpus_in_degree": round(avg_seed_in_degree, 3),
        "seed_in_corpus_in_degree": seed_in_degrees,
    }
    return report


def save_report(path: Path, report: dict) -> None:
    """Persist the report to ``crawl_report.json``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    logger.info("crawl_report.json saved to %s", path)


def print_summary(report: dict) -> None:
    """Print a concise terminal summary of the crawl (§4.2 SHOULD)."""
    print("\n" + "=" * 56)
    print(" binet crawl summary")
    print("=" * 56)
    print(f"  status                  : {report['status']}")
    print(f"  total nodes             : {report['total_nodes']}")
    print(f"  total edges             : {report['total_edges']}")
    print(f"  seeds                   : {report['seed_count']}")
    print(f"  failed                  : {report['failed_count']}")
    print(f"  dropped edges (no DOI)  : {report['dropped_edges_no_doi']}")
    print(f"  avg seed in-degree      : {report['avg_seed_in_corpus_in_degree']}")
    print(f"  source hits             : {report['source_hits']}")
    print(f"  depth distribution      : {report['depth_distribution']}")
    print("=" * 56 + "\n")
