"""
End-to-end orchestration for citation-edge relation evaluation.

``run(config)`` wires the stages: build the local directed citation graph ->
collect each edge's in-text citation contexts -> classify each edge with one LLM
call (inheritance / unknown) -> serialize products into the graph directory.

The entry script only assembles a config and calls this function; no core logic
lives in the entry file. Edge direction is ``i -> j`` (paper i cites paper j);
``inheritance`` means i inherits knowledge from j.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from tqdm import tqdm

from src.core.logger import get_user_logger
from src.core.paths import (
    graph_citation_edgelist_csv,
    graph_citation_gexf,
    graph_citation_relations_json,
    graph_dir,
)
from src.service.gragh.evaluate import serialize
from src.service.gragh.evaluate.classifier import classify_edge
from src.service.gragh.evaluate.config import (
    RELATION_EVAL_VERSION,
    RelationEvalConfig,
)
from src.service.gragh.evaluate.context_collector import collect_edge_evidence
from src.service.gragh.evaluate.graph_builder import build_citation_graph


def _context_to_dict(ctx) -> dict:
    return {
        "citation_id": ctx.citation_id,
        "markers": ctx.markers,
        "citation_sentence": ctx.citation_sentence,
        "context": ctx.context,
    }


def run(config: Optional[RelationEvalConfig] = None) -> dict:
    """
    Run the full relation-evaluation pipeline and write all products.

    Args:
        config: Full configuration; defaults are used when omitted.

    Returns:
        A summary dict with output paths, counts and label distribution.
    """
    config = config or RelationEvalConfig()
    logger = get_user_logger(config.username, config.dataset_name)
    started = datetime.now(timezone.utc)
    logger.info(
        "relation-eval: start user={u} dataset={d} version={v}",
        u=config.username, d=config.dataset_name, v=RELATION_EVAL_VERSION,
    )

    # 1. Build the local directed citation graph.
    graph = build_citation_graph(
        config.username, config.dataset_name,
        scope=config.candidate_scope,
        threshold=config.candidate_pair_threshold,
        logger=logger,
    )
    index = graph.index
    logger.info(
        "relation-eval: graph nodes={n} directed_edges={e}",
        n=graph.n_nodes, e=graph.n_edges,
    )
    if graph.n_nodes < 2:
        logger.warning(
            "relation-eval: fewer than 2 local papers ({n}); nothing to judge",
            n=graph.n_nodes,
        )
    if graph.n_edges == 0:
        logger.warning("relation-eval: no local citation edges to classify")

    # 2. Per-edge: collect contexts and classify in a single LLM call.
    cache: dict = {}
    edge_records = []
    label_counter: Counter = Counter()

    for (i, j) in tqdm(
        graph.edges,
        desc="relation-eval edges",
        unit="edge",
        ncols=100,
    ):
        evidence = collect_edge_evidence(
            index, i, j, cache=cache, max_contexts=config.max_contexts_per_edge,
        )
        source_title = index.title(i)
        target_title = index.title(j)

        result = classify_edge(
            source_title=source_title,
            target_title=target_title,
            contexts=evidence.contexts,
            temperature=config.llm_temperature,
            label_set=config.label_set,
            logger=logger,
        )
        label_counter[result.label] += 1

        logger.info(
            "relation-eval: edge {s} -> {t} | contexts={c} | label={l}",
            s=index.doi(i), t=index.doi(j),
            c=len(evidence.contexts), l=result.label,
        )

        edge_records.append({
            "source_doi": index.doi(i),
            "target_doi": index.doi(j),
            "source_title": source_title,
            "target_title": target_title,
            "label": result.label,
            "reason": result.reason,
            "n_contexts": len(evidence.contexts),
            "evidence_contexts": [_context_to_dict(c) for c in evidence.contexts],
        })

    # 3. Node records.
    node_records = [
        {
            "doi": index.doi(pid),
            "title": index.title(pid),
            "file_id": index.file_id(pid),
        }
        for pid in range(index.n)
    ]

    # 4. Run metadata for provenance.
    run_metadata = {
        "relation_eval_version": RELATION_EVAL_VERSION,
        "generated_at": started.isoformat(),
        "username": config.username,
        "dataset_name": config.dataset_name,
        "n_local_papers": graph.n_nodes,
        "n_directed_edges": graph.n_edges,
        "candidate_scope": graph.scope,
        "row_semantics": "directed edge i -> j (paper_i cites paper_j); "
                         "inheritance means i inherits knowledge from j",
        "label_distribution": dict(label_counter),
        "config": config.to_metadata(),
    }

    relations = {
        "metadata": run_metadata,
        "nodes": node_records,
        "edges": edge_records,
    }

    # 5. Serialize products into the graph directory.
    if config.output_dir:
        base = Path(config.output_dir)
        relations_path = base / "citation_relations.json"
        gexf_path = base / "citation_graph.gexf"
        edgelist_path = base / "citation_edgelist.csv"
    else:
        base = graph_dir(config.username, config.dataset_name)
        relations_path = graph_citation_relations_json(config.username, config.dataset_name)
        gexf_path = graph_citation_gexf(config.username, config.dataset_name)
        edgelist_path = graph_citation_edgelist_csv(config.username, config.dataset_name)

    serialize.write_relations_json(relations, relations_path)
    serialize.write_gexf(node_records, edge_records, gexf_path)
    serialize.write_edgelist_csv(edge_records, edgelist_path)

    logger.success(
        "relation-eval: done papers={n} edges={e} labels={labels} -> {d}",
        n=graph.n_nodes, e=graph.n_edges, labels=dict(label_counter), d=str(base),
    )

    return {
        "output_dir": str(base),
        "citation_relations": str(relations_path),
        "citation_graph_gexf": str(gexf_path),
        "citation_edgelist": str(edgelist_path),
        "n_local_papers": graph.n_nodes,
        "n_directed_edges": graph.n_edges,
        "label_distribution": dict(label_counter),
    }
