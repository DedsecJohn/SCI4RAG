"""
Entry point for citation-edge relation evaluation (two-stage cascade).

This script only assembles configurations and calls the pipelines; all core
logic lives in ``src/service/gragh/evaluate``.

- Stage-1 (default): build the local directed citation graph (paper i cites
  paper j) and ask an LLM to judge each edge's relation type from its in-text
  citation contexts (inheritance / unknown).
- Stage-2: adjudicate the Stage-1 ``unknown`` edges from their raw feature
  vectors (edge_features.parquet) into the final label set
  (inheritance / parallel / peripheral).

Run:
    # Stage-1 (default; both forms equivalent)
    python -m example.5evaluate.citation_relation
    python -m example.5evaluate.citation_relation stage1 --temperature 0.0

    # Stage-2 (requires Stage-1 products + edge_features.parquet)
    python -m example.5evaluate.citation_relation stage2
    python -m example.5evaluate.citation_relation stage2 --username administrator --dataset leiting
"""

import argparse
import os
import sys

# Ensure the project root is importable when run as a plain script.
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.service.gragh.evaluate.config import RelationEvalConfig
from src.service.gragh.evaluate.pipeline import run
from src.service.gragh.evaluate.stage2.config import Stage2Config
from src.service.gragh.evaluate.stage2.pipeline import run_stage2


def _add_stage1_args(parser: argparse.ArgumentParser) -> None:
    defaults = RelationEvalConfig()
    parser.add_argument("--username", default=defaults.username)
    parser.add_argument("--dataset", dest="dataset_name", default=defaults.dataset_name)
    parser.add_argument("--output-dir", dest="output_dir", default=defaults.output_dir,
                        help="Override output directory (default: dataset graph/).")
    parser.add_argument("--candidate-scope", dest="candidate_scope",
                        choices=["cited_local", "all_local"],
                        default=defaults.candidate_scope,
                        help="cited_local: directed local citation edges; "
                             "all_local: every local pair.")
    parser.add_argument("--temperature", dest="llm_temperature", type=float,
                        default=defaults.llm_temperature)
    parser.add_argument("--max-contexts", dest="max_contexts_per_edge", type=int,
                        default=defaults.max_contexts_per_edge)


def _add_stage2_args(parser: argparse.ArgumentParser) -> None:
    defaults = Stage2Config()
    parser.add_argument("--username", default=defaults.username)
    parser.add_argument("--dataset", dest="dataset_name", default=defaults.dataset_name)
    parser.add_argument("--output-dir", dest="output_dir", default=defaults.output_dir,
                        help="Override output directory (default: dataset graph/).")
    # Threshold overrides (Subtask A - asymmetric recovery).
    parser.add_argument("--cf-recover", dest="cf_recover_threshold", type=float,
                        default=defaults.cf_recover_threshold)
    parser.add_argument("--cf-high", dest="cf_high_threshold", type=float,
                        default=defaults.cf_high_threshold)
    parser.add_argument("--pagerank-boost", dest="pagerank_target_boost", type=float,
                        default=defaults.pagerank_target_boost)
    # Threshold overrides (Subtask B - symmetric parallel/peripheral split).
    parser.add_argument("--cocitation-th", dest="cocitation_salton_threshold", type=float,
                        default=defaults.cocitation_salton_threshold)
    parser.add_argument("--bibcoupling-th", dest="bibcoupling_jaccard_threshold", type=float,
                        default=defaults.bibcoupling_jaccard_threshold)
    parser.add_argument("--author-th", dest="author_jaccard_threshold", type=float,
                        default=defaults.author_jaccard_threshold)
    parser.add_argument("--title-sim-th", dest="title_sim_threshold", type=float,
                        default=defaults.title_sim_threshold)
    parser.add_argument("--abstract-sim-th", dest="abstract_sim_threshold", type=float,
                        default=defaults.abstract_sim_threshold)


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Two-stage citation-edge relation evaluation.",
    )
    subparsers = parser.add_subparsers(dest="stage")
    _add_stage1_args(subparsers.add_parser(
        "stage1", help="LLM citation-context intent classification (default)."))
    _add_stage2_args(subparsers.add_parser(
        "stage2", help="Feature-based adjudication of Stage-1 unknown edges."))
    # Default to stage1 when no subcommand is given (backward compatible).
    _add_stage1_args(parser)
    return parser.parse_args(argv)


def run_stage1_main(args) -> dict:
    config = RelationEvalConfig(
        username=args.username,
        dataset_name=args.dataset_name,
        output_dir=args.output_dir,
        candidate_scope=args.candidate_scope,
        llm_temperature=args.llm_temperature,
        max_contexts_per_edge=args.max_contexts_per_edge,
    )
    summary = run(config)
    print("\n=== Stage-1 citation edge relation evaluation complete ===")
    print(f"Local papers        : {summary['n_local_papers']}")
    print(f"Directed edges      : {summary['n_directed_edges']}")
    print(f"Label distribution  : {summary['label_distribution']}")
    print(f"Output directory    : {summary['output_dir']}")
    print(f"  citation_relations: {summary['citation_relations']}")
    print(f"  citation_graph    : {summary['citation_graph_gexf']}")
    print(f"  citation_edgelist : {summary['citation_edgelist']}")
    return summary


def run_stage2_main(args) -> dict:
    config = Stage2Config(
        username=args.username,
        dataset_name=args.dataset_name,
        output_dir=args.output_dir,
        cf_recover_threshold=args.cf_recover_threshold,
        cf_high_threshold=args.cf_high_threshold,
        pagerank_target_boost=args.pagerank_target_boost,
        cocitation_salton_threshold=args.cocitation_salton_threshold,
        bibcoupling_jaccard_threshold=args.bibcoupling_jaccard_threshold,
        author_jaccard_threshold=args.author_jaccard_threshold,
        title_sim_threshold=args.title_sim_threshold,
        abstract_sim_threshold=args.abstract_sim_threshold,
    )
    summary = run_stage2(config)
    print("\n=== Stage-2 feature adjudication complete ===")
    print(f"Edges               : {summary['n_edges']}")
    print(f"Stage-1 labels      : {summary['stage1_label_distribution']}")
    print(f"Final labels        : {summary['final_label_distribution']}")
    print(f"Confidence tiers    : {summary['confidence_tier_distribution']}")
    print(f"Output directory    : {summary['output_dir']}")
    print(f"  relations_final   : {summary['citation_relations_final']}")
    print(f"  edgelist_final    : {summary['citation_edgelist_final']}")
    print(f"  graph_final       : {summary['citation_graph_final_gexf']}")
    return summary


def main(argv=None) -> dict:
    args = _parse_args(argv)
    if args.stage == "stage2":
        return run_stage2_main(args)
    # stage1 explicitly, or no subcommand (default).
    return run_stage1_main(args)


if __name__ == "__main__":
    main()
