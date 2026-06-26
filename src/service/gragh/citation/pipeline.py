"""
End-to-end orchestration for citation feature engineering (§7.2).

``run(config)`` wires the stages together: discover local papers -> load graph
(for co-citation/pagerank) -> build candidate rows (directed local citation
edges) -> compute features -> assemble edge/node tables -> report -> serialize.
The entry script only assembles a config and calls this function; no core logic
lives in the entry file.

Edge rows are directed citation edges ``i -> j`` (``paper_i_doi`` cites
``paper_j_doi``). Symmetric features take the same value for either direction;
the citation-frequency feature is directional (``i -> j`` only).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import pandas as pd

from src.core.logger import get_user_logger
from src.core.paths import (
    citation_abstract_embeddings_npy,
    citation_edge_features,
    citation_feature_report,
    citation_features_dir,
    citation_node_features,
    citation_title_embeddings_npy,
)
from src.service.gragh.citation.candidate import build_candidates
from src.service.gragh.citation.config import CitationFeatureConfig, FEATUREFORGE_VERSION
from src.service.gragh.citation.graph_io import GraphData, load_graph
from src.service.gragh.citation.index import discover_local_papers
from src.service.gragh.citation.features import (
    structural, pagerank, author, semantic, citation_freq,
)
from src.service.gragh.citation import report as report_mod
from src.service.gragh.citation import serialize


# Edge feature column order (§5.1). Directed edge i -> j (i cites j).
EDGE_FEATURE_COLUMNS = [
    "bibcoupling_raw", "bibcoupling_jaccard",
    "cocitation_raw", "cocitation_salton",
    "pagerank_target", "pagerank_min", "pagerank_product",
    "author_jaccard",
    "title_sim", "abstract_sim",
    "citation_count", "citation_total", "citation_freq",
]


def _merge_node_values(target: Dict[int, dict], result) -> None:
    for pid, vals in result.node_values.items():
        target.setdefault(pid, {}).update(vals)


def run(config: Optional[CitationFeatureConfig] = None) -> dict:
    """
    Run the full pipeline and write all §5 products.

    Args:
        config: Full configuration; defaults are used when omitted.

    Returns:
        A summary dict with output paths and key counts.
    """
    config = config or CitationFeatureConfig()
    logger = get_user_logger(config.username, config.dataset_name)
    started = datetime.now(timezone.utc)
    logger.info(
        "citation-features: start user={u} dataset={d} version={v}",
        u=config.username, d=config.dataset_name, v=FEATUREFORGE_VERSION,
    )

    # 1. Local paper universe.
    index = discover_local_papers(config.username, config.dataset_name)
    logger.info("citation-features: discovered {n} local papers", n=index.n)
    if index.n < 2:
        logger.warning(
            "citation-features: fewer than 2 local papers ({n}); no edges to "
            "compute", n=index.n,
        )

    # 2. Graph (only needed for co-citation / pagerank).
    graph = GraphData()
    if config.enable_cocitation or config.enable_pagerank:
        logger.info("citation-features: loading graph {p}", p=config.citation_network_path)
        graph = load_graph(config.citation_network_path)
        logger.info(
            "citation-features: graph nodes={n} inner_edges={e}",
            n=graph.n_nodes, e=graph.n_inner_edges,
        )

    # 3. Candidate rows (directed local citation edges by default).
    candidate = build_candidates(
        index, scope=config.candidate_scope,
        threshold=config.candidate_pair_threshold, logger=logger,
    )
    pairs: List[Tuple[int, int]] = candidate.pairs

    # 4. Features.
    node_values: Dict[int, dict] = {}
    feature_metadata: Dict[str, dict] = {}
    edge_results = []
    title_embeddings = None
    abstract_embeddings = None

    if config.enable_bibcoupling:
        r = structural.compute_bibcoupling(index, pairs, logger=logger)
        edge_results.append(r); _merge_node_values(node_values, r)
        feature_metadata["bibcoupling"] = r.metadata
    if config.enable_cocitation:
        r = structural.compute_cocitation(index, graph, pairs, logger=logger)
        edge_results.append(r); _merge_node_values(node_values, r)
        feature_metadata["cocitation"] = r.metadata
    if config.enable_pagerank:
        r = pagerank.compute_pagerank(index, graph, pairs, config, logger=logger,
                                      method=config.pagerank_method)
        edge_results.append(r); _merge_node_values(node_values, r)
        feature_metadata["pagerank"] = r.metadata
    if config.enable_author:
        r = author.compute_author(index, pairs, logger=logger)
        edge_results.append(r); _merge_node_values(node_values, r)
        feature_metadata["author"] = r.metadata
    if config.enable_semantic:
        r = semantic.compute_semantic(index, pairs, config, logger=logger)
        edge_results.append(r); _merge_node_values(node_values, r)
        feature_metadata["semantic"] = r.metadata
        title_embeddings = r.title_embeddings
        abstract_embeddings = r.abstract_embeddings
    if config.enable_citation_freq:
        r = citation_freq.compute_citation_freq(index, pairs, logger=logger)
        edge_results.append(r); _merge_node_values(node_values, r)
        feature_metadata["citation_freq"] = r.metadata

    # 5. Assemble edge table (§5.1).
    present_cols = [c for c in EDGE_FEATURE_COLUMNS
                    if any(c in r.columns for r in edge_results)]
    edge_rows = []
    for (i, j) in pairs:
        row = {"paper_i_doi": index.doi(i), "paper_j_doi": index.doi(j)}
        for r in edge_results:
            row.update(r.edge_values.get((i, j), {}))
        for c in present_cols:
            row.setdefault(c, None)
        edge_rows.append(row)
    edge_df = pd.DataFrame(edge_rows, columns=["paper_i_doi", "paper_j_doi"] + present_cols)

    # 6. Assemble node table (§5.2).
    node_rows = []
    for pid in range(index.n):
        gi = graph.index_of(index.doi(pid)) if graph.n_nodes else None
        vals = node_values.get(pid, {})
        node_rows.append({
            "doi": index.doi(pid),
            "title": index.title(pid),
            "pagerank": vals.get("pagerank"),
            "in_degree": vals.get("in_degree", int(graph.in_degree[gi]) if gi is not None else None),
            "out_degree": vals.get("out_degree", int(graph.out_degree[gi]) if gi is not None else None),
            "authors": vals.get("authors", []),
            "reference_count": vals.get("reference_count"),
            "citation_total": vals.get("citation_total"),
            "title_embedding_row": vals.get("title_embedding_row", -1),
            "abstract_embedding_row": vals.get("abstract_embedding_row", -1),
        })
    node_df = pd.DataFrame(node_rows)

    # 7. Run metadata for provenance (§5.4).
    run_metadata = {
        "featureforge_version": FEATUREFORGE_VERSION,
        "generated_at": started.isoformat(),
        "username": config.username,
        "dataset_name": config.dataset_name,
        "citation_network_path": config.citation_network_path,
        "citation_network_sha256": graph.network_sha256,
        "citation_network_metadata": graph.network_metadata,
        "n_local_papers": index.n,
        "n_candidate_rows": len(pairs),
        "n_directed_local_edges": len(candidate.directed_edges),
        "candidate_scope": candidate.scope,
        "row_semantics": "directed edge i -> j (paper_i cites paper_j)",
        "config": config.to_metadata(),
    }

    # 8. Report (§5.3).
    report = report_mod.build_report(
        edge_df=edge_df,
        feature_columns=present_cols,
        index=index,
        graph=graph,
        feature_metadata=feature_metadata,
        run_metadata=run_metadata,
    )

    # 9. Serialize all products.
    out_dir = config.output_dir or str(citation_features_dir(config.username, config.dataset_name))
    edge_path = citation_edge_features(config.username, config.dataset_name)
    node_path = citation_node_features(config.username, config.dataset_name)
    title_emb_path = citation_title_embeddings_npy(config.username, config.dataset_name)
    abstract_emb_path = citation_abstract_embeddings_npy(config.username, config.dataset_name)
    report_path = citation_feature_report(config.username, config.dataset_name)
    if config.output_dir:
        from pathlib import Path
        base = Path(config.output_dir)
        edge_path = base / "edge_features.parquet"
        node_path = base / "node_features.parquet"
        title_emb_path = base / "title_embeddings.npy"
        abstract_emb_path = base / "abstract_embeddings.npy"
        report_path = base / "feature_report.json"

    serialize.write_edge_features(edge_df, edge_path, run_metadata)
    serialize.write_node_features(node_df, node_path, run_metadata)
    title_written = serialize.write_embeddings(title_embeddings, title_emb_path)
    abstract_written = serialize.write_embeddings(abstract_embeddings, abstract_emb_path)
    serialize.write_report(report, report_path)

    logger.success(
        "citation-features: done papers={n} edges={p} -> {d}",
        n=index.n, p=len(pairs), d=out_dir,
    )

    return {
        "output_dir": out_dir,
        "edge_features": str(edge_path),
        "node_features": str(node_path),
        "title_embeddings": str(title_written) if title_written else None,
        "abstract_embeddings": str(abstract_written) if abstract_written else None,
        "feature_report": str(report_path),
        "n_local_papers": index.n,
        "n_candidate_rows": len(pairs),
        "edge_columns": list(edge_df.columns),
    }
