"""
Serialization of the relation-evaluation products (written to the graph dir).

Three artifacts are produced:
- ``citation_relations.json``: the main product (run metadata + nodes + directed
  edges with label, reason and evidence contexts).
- ``citation_graph.gexf``: the directed graph for visualization (Gephi etc.).
- ``citation_edgelist.csv``: a flat directed edge list with labels.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import List

import networkx as nx

from src.core.paths import ensure_dir


def write_relations_json(relations: dict, path) -> Path:
    """Write the main citation_relations.json product."""
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(relations, f, ensure_ascii=False, indent=2)
    return path


def write_gexf(nodes: List[dict], edges: List[dict], path) -> Path:
    """
    Write a directed GEXF graph for visualization.

    Node attributes: title, file_id. Edge attributes: label, reason,
    n_contexts (scalar only, to stay GEXF-compatible).
    """
    path = Path(path)
    ensure_dir(path.parent)

    g = nx.DiGraph()
    for node in nodes:
        g.add_node(
            node["doi"],
            title=node.get("title", "") or "",
            file_id=node.get("file_id", "") or "",
        )
    for edge in edges:
        g.add_edge(
            edge["source_doi"],
            edge["target_doi"],
            label=edge.get("label", "") or "",
            reason=edge.get("reason", "") or "",
            n_contexts=int(edge.get("n_contexts", 0) or 0),
        )
    nx.write_gexf(g, str(path))
    return path


def write_edgelist_csv(edges: List[dict], path) -> Path:
    """Write a flat directed edge list CSV with labels."""
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["source_doi", "target_doi", "source_title", "target_title",
             "label", "n_contexts", "reason"]
        )
        for edge in edges:
            writer.writerow([
                edge.get("source_doi", ""),
                edge.get("target_doi", ""),
                edge.get("source_title", ""),
                edge.get("target_title", ""),
                edge.get("label", ""),
                edge.get("n_contexts", 0),
                edge.get("reason", ""),
            ])
    return path


# ── Stage-2 final products ─────────────────────────────────────────────

def write_relations_final_json(relations: dict, path) -> Path:
    """Write citation_relations_final.json (Stage-2 adjudicated edges)."""
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(relations, f, ensure_ascii=False, indent=2)
    return path


def write_final_edgelist_csv(edges: List[dict], path) -> Path:
    """Write the Stage-2 final edge list CSV (label only, no audit columns)."""
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["source_doi", "target_doi", "source_title", "target_title", "label"]
        )
        for edge in edges:
            writer.writerow([
                edge.get("source_doi", ""),
                edge.get("target_doi", ""),
                edge.get("source_title", ""),
                edge.get("target_title", ""),
                edge.get("label", ""),
            ])
    return path


def write_final_gexf(nodes: List[dict], edges: List[dict], path) -> Path:
    """
    Write the Stage-2 directed GEXF graph for visualization.

    Node attributes: title. Edge attributes: label, confidence_tier (scalar
    only, to stay GEXF-compatible).
    """
    path = Path(path)
    ensure_dir(path.parent)

    g = nx.DiGraph()
    for node in nodes:
        g.add_node(
            node["doi"],
            title=node.get("title", "") or "",
        )
    for edge in edges:
        g.add_edge(
            edge["source_doi"],
            edge["target_doi"],
            label=edge.get("label", "") or "",
            confidence_tier=edge.get("confidence_tier", "") or "",
        )
    nx.write_gexf(g, str(path))
    return path
