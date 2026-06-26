"""
Graph serialization (§4.1, §4.3).

Decouples graph serialization from the crawl logic so additional downstream
formats (edgelist.csv, GraphML) can be appended easily for later networkx /
PageRank work.
"""

from __future__ import annotations

import csv
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Set

from binet.config import BinetConfig
from binet.models import Edge, NodeRecord

logger = logging.getLogger("binet.serialize")


def build_network_dict(
    config: BinetConfig,
    nodes: Dict[str, NodeRecord],
    edges: Set[Edge],
    seed_dois: List[str],
    failed: Dict[str, str],
    dropped_edges_no_doi: int,
    status: str,
) -> dict:
    """
    Assemble the ``citation_network.json`` structure (§4.1).

    Returns:
        A dict with ``metadata`` / ``nodes`` / ``edges`` sections.
    """
    node_list = [n.to_dict() for n in nodes.values()]
    edge_list = [e.to_dict() for e in edges]

    metadata = {
        "seed_dois": seed_dois,
        "max_depth": config.max_depth,
        "backward_depth": config.backward_depth,
        "forward_depth": config.forward_depth,
        "max_papers": config.max_papers,
        "total_nodes": len(node_list),
        "total_edges": len(edge_list),
        "failed_count": len(failed),
        "dropped_edges_no_doi": dropped_edges_no_doi,
        "data_sources": {
            "references": config.reference_sources,
            "citations": config.citation_sources,
        },
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
    }

    return {"metadata": metadata, "nodes": node_list, "edges": edge_list}


def save_network_json(path: Path, network: dict) -> None:
    """Persist the network dict to ``citation_network.json``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(network, f, ensure_ascii=False, indent=2)
    logger.info("citation_network.json saved to %s", path)


def save_failed_dois(path: Path, failed: Dict[str, str]) -> None:
    """Write failed DOIs and their reasons (§4.2, MUST)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for doi, reason in failed.items():
            f.write(f"{doi}\t{reason}\n")
    logger.info("failed_dois.txt saved to %s (%d entries)", path, len(failed))


# ---------------------------------------------------------------------- #
# Optional downstream graph formats (§4.3)
# ---------------------------------------------------------------------- #

def export_edgelist_csv(path: Path, edges: Set[Edge]) -> None:
    """Export a simple ``source_doi,target_doi`` edge list CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["source_doi", "target_doi"])
        for e in edges:
            writer.writerow([e.source_doi, e.target_doi])
    logger.info("edgelist.csv saved to %s", path)


def export_graphml(path: Path, nodes: Dict[str, NodeRecord], edges: Set[Edge]) -> None:
    """
    Export a minimal GraphML file (directed) for networkx / Gephi.

    Implemented without a hard networkx dependency so the core tool stays light.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def esc(text: str) -> str:
        out = str(text)
        out = out.replace("&", "&" + "amp;")
        out = out.replace("<", "&" + "lt;")
        out = out.replace(">", "&" + "gt;")
        out = out.replace('"', "&" + "quot;")
        return out

    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<graphml xmlns="http://graphml.graphdrawing.org/xmlns">',
        '  <key id="title" for="node" attr.name="title" attr.type="string"/>',
        '  <key id="depth" for="node" attr.name="depth" attr.type="int"/>',
        '  <key id="is_seed" for="node" attr.name="is_seed" attr.type="boolean"/>',
        '  <graph edgedefault="directed">',
    ]
    for n in nodes.values():
        lines.append(f'    <node id="{esc(n.doi)}">')
        lines.append(f'      <data key="title">{esc(n.title)}</data>')
        lines.append(f'      <data key="depth">{n.depth}</data>')
        lines.append(f'      <data key="is_seed">{str(n.is_seed).lower()}</data>')
        lines.append("    </node>")
    for i, e in enumerate(edges):
        lines.append(
            f'    <edge id="e{i}" source="{esc(e.source_doi)}" '
            f'target="{esc(e.target_doi)}"/>'
        )
    lines.append("  </graph>")
    lines.append("</graphml>")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    logger.info("GraphML saved to %s", path)
