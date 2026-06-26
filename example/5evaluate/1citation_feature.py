"""
Entry point for citation feature engineering (FeatureForge, §7.2).

This script only assembles a configuration and calls the pipeline; all core
logic lives in ``src/service/gragh/citation``. It supports both a no-arg trial
run (code-internal defaults) and command-line overrides for scripted batches.

Edge rows are directed citation edges ``i -> j`` (paper i cites paper j), and
the candidate universe is restricted to local papers that cite one another.

Run:
    python -m example.5evaluate.citation_feature
    python example/5evaluate/citation_feature.py --username administrator --dataset leiting
    python example/5evaluate/citation_feature.py --disable-semantic   # skip embeddings
    python example/5evaluate/citation_feature.py --candidate-scope all_local
"""

import argparse
import os
import sys

# Ensure the project root is importable when run as a plain script.
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.service.gragh.citation.config import CitationFeatureConfig
from src.service.gragh.citation.pipeline import run


def build_config_from_args(argv=None) -> CitationFeatureConfig:
    """Assemble a CitationFeatureConfig from CLI args over code defaults."""
    defaults = CitationFeatureConfig()
    parser = argparse.ArgumentParser(
        description="Build citation relatedness features for a dataset's papers.",
    )
    parser.add_argument("--username", default=defaults.username)
    parser.add_argument("--dataset", dest="dataset_name", default=defaults.dataset_name)
    parser.add_argument("--citation-network", dest="citation_network_path",
                        default=defaults.citation_network_path)
    parser.add_argument("--output-dir", dest="output_dir", default=defaults.output_dir,
                        help="Override output directory (default: dataset citation-features/).")
    parser.add_argument("--candidate-scope", dest="candidate_scope",
                        choices=["cited_local", "all_local"], default=defaults.candidate_scope,
                        help="cited_local: directed local citation edges; all_local: every local pair.")
    parser.add_argument("--pagerank-method", choices=["networkx", "scipy"], default="networkx")
    parser.add_argument("--semantic-model", dest="semantic_model_name",
                        default=defaults.semantic_model_name)
    parser.add_argument("--semantic-seed", dest="semantic_random_seed", type=int,
                        default=defaults.semantic_random_seed)
    for f in ("bibcoupling", "cocitation", "pagerank", "author", "semantic", "citation-freq"):
        parser.add_argument(f"--disable-{f}", action="store_true",
                            help=f"Disable the {f} feature.")
    args = parser.parse_args(argv)

    config = CitationFeatureConfig(
        username=args.username,
        dataset_name=args.dataset_name,
        citation_network_path=args.citation_network_path,
        output_dir=args.output_dir,
        candidate_scope=args.candidate_scope,
        semantic_model_name=args.semantic_model_name,
        semantic_random_seed=args.semantic_random_seed,
        enable_bibcoupling=not args.disable_bibcoupling,
        enable_cocitation=not args.disable_cocitation,
        enable_pagerank=not args.disable_pagerank,
        enable_author=not args.disable_author,
        enable_semantic=not args.disable_semantic,
        enable_citation_freq=not getattr(args, "disable_citation_freq"),
        pagerank_method=args.pagerank_method,
    )
    return config


def main(argv=None) -> dict:
    config = build_config_from_args(argv)

    # Step 1: assemble config (done above). Step 2: run the pipeline.
    summary = run(config)

    print("\n=== Citation feature engineering complete ===")
    print(f"Local papers       : {summary['n_local_papers']}")
    print(f"Candidate edges    : {summary['n_candidate_rows']}")
    print(f"Output directory   : {summary['output_dir']}")
    print(f"  edge_features    : {summary['edge_features']}")
    print(f"  node_features    : {summary['node_features']}")
    print(f"  title_embeddings : {summary['title_embeddings']}")
    print(f"  abstract_embeds  : {summary['abstract_embeddings']}")
    print(f"  feature_report   : {summary['feature_report']}")
    return summary


if __name__ == "__main__":
    main()
