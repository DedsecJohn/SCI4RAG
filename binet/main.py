"""
Command-line entry point for binet (FR-6).

Supports both CLI arguments and in-code defaults. Seeds can be supplied either
directly (``--seeds``) or from a file (``--seed-file``: one DOI per line or a
JSON array).

Run:
    python -m binet.main --seeds 10.18653/v1/2025.acl-long.907
    python -m binet.main --seed-file seeds.txt --max-depth 2 --resume
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List

from binet.config import BinetConfig, CONTACT_EMAIL
from binet.crawler import Crawler
from binet.checkpoint import clear_checkpoint
from binet.serialize import (
    build_network_dict,
    save_network_json,
    save_failed_dois,
    export_edgelist_csv,
    export_graphml,
)
from binet.report import build_report, save_report, print_summary


# ---------------------------------------------------------------------- #
# Logging
# ---------------------------------------------------------------------- #

def _setup_logging(log_path: Path, verbose: bool) -> None:
    """Configure console + file logging (FR-5.5)."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    level = logging.DEBUG if verbose else logging.INFO

    root = logging.getLogger("binet")
    root.setLevel(logging.DEBUG)
    root.handlers.clear()

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console = logging.StreamHandler(sys.stderr)
    console.setLevel(level)
    console.setFormatter(fmt)
    root.addHandler(console)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(fmt)
    root.addHandler(file_handler)


# ---------------------------------------------------------------------- #
# Seed loading (FR-6.2)
# ---------------------------------------------------------------------- #

def load_seeds(seed_dois: List[str], seed_file: str | None) -> List[str]:
    """
    Resolve seeds from a direct list and/or a file.

    File format: either one DOI per line, or a JSON array of DOIs.

    Args:
        seed_dois: DOIs passed directly.
        seed_file: Optional file path.

    Returns:
        A merged list of seed DOI strings (raw; normalization happens later).
    """
    seeds: List[str] = list(seed_dois or [])

    if seed_file:
        path = Path(seed_file)
        if not path.exists():
            raise FileNotFoundError(f"Seed file not found: {seed_file}")
        text = path.read_text(encoding="utf-8").strip()
        if text.startswith("["):
            seeds.extend(json.loads(text))
        else:
            seeds.extend(
                line.strip() for line in text.splitlines() if line.strip()
            )

    return seeds


# ---------------------------------------------------------------------- #
# Argument parsing
# ---------------------------------------------------------------------- #

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="binet",
        description="Bidirectional citation network crawler.",
    )
    p.add_argument("--seeds", nargs="*", default=[], help="Seed DOIs.")
    p.add_argument("--seed-file", default=None,
                   help="File with seeds (one DOI per line or a JSON array).")
    p.add_argument("--max-depth", type=int, default=2,
                   help="Max hops from seeds (default 2).")
    p.add_argument("--backward-depth", type=int, default=None,
                   help="Per-direction depth for references (default = max-depth).")
    p.add_argument("--forward-depth", type=int, default=None,
                   help="Per-direction depth for citations (default = max-depth).")
    p.add_argument("--direction", choices=["backward", "forward", "both"],
                   default="both", help="Which directions to expand (default both).")
    p.add_argument("--max-papers", type=int, default=10000,
                   help="Hard cap on total nodes (default 10000).")
    p.add_argument("--reference-sources", nargs="*",
                   default=["openalex", "crossref"],
                   help="Ordered fallback chain for references.")
    p.add_argument("--citation-sources", nargs="*",
                   default=["openalex", "semantic_scholar"],
                   help="Ordered fallback chain for citations.")
    p.add_argument("--email", default=CONTACT_EMAIL,
                   help="Polite-pool contact email.")
    p.add_argument("--delay-min", type=float, default=0.5,
                   help="Min per-request delay seconds (default 0.5).")
    p.add_argument("--delay-max", type=float, default=1.0,
                   help="Max per-request delay seconds (default 1.0).")
    p.add_argument("--max-retries", type=int, default=3,
                   help="Max retry attempts (default 3).")
    p.add_argument("--checkpoint-every", type=int, default=50,
                   help="Checkpoint every N processed nodes (default 50).")
    p.add_argument("--output-dir", default=None,
                   help="Output directory (default binet/output).")
    p.add_argument("--resume", action="store_true",
                   help="Resume from an existing checkpoint.")
    p.add_argument("--export-edgelist", action="store_true",
                   help="Also export edgelist.csv.")
    p.add_argument("--export-graphml", action="store_true",
                   help="Also export GraphML.")
    p.add_argument("--verbose", action="store_true", help="Verbose logging.")
    return p


def config_from_args(args: argparse.Namespace) -> BinetConfig:
    kwargs = dict(
        seed_dois=args.seeds,
        seed_file=args.seed_file,
        max_depth=args.max_depth,
        backward_depth=args.backward_depth,
        forward_depth=args.forward_depth,
        direction=args.direction,
        max_papers=args.max_papers,
        reference_sources=args.reference_sources,
        citation_sources=args.citation_sources,
        email=args.email,
        delay_range=(args.delay_min, args.delay_max),
        max_retries=args.max_retries,
        checkpoint_every=args.checkpoint_every,
        resume=args.resume,
    )
    if args.output_dir:
        kwargs["output_dir"] = Path(args.output_dir)
    return BinetConfig(**kwargs)


# ---------------------------------------------------------------------- #
# Orchestration
# ---------------------------------------------------------------------- #

def run(config: BinetConfig, seeds: List[str],
        export_edgelist: bool = False, export_graphml_flag: bool = False) -> dict:
    """
    Execute a full crawl and write all artifacts.

    Args:
        config: Runtime configuration.
        seeds: Raw seed DOIs.
        export_edgelist: Also write edgelist.csv.
        export_graphml_flag: Also write GraphML.

    Returns:
        The report dict.
    """
    config.output_dir.mkdir(parents=True, exist_ok=True)

    crawler = Crawler(config)

    interrupted = False
    try:
        crawler.crawl(seeds)
    except KeyboardInterrupt:
        interrupted = True

    # Build & persist outputs regardless of interruption (partial graph is useful).
    network = build_network_dict(
        config=config,
        nodes=crawler.nodes,
        edges=crawler.edges,
        seed_dois=crawler.seed_dois,
        failed=crawler.failed,
        dropped_edges_no_doi=crawler.dropped_edges_no_doi,
        status=crawler.status,
    )
    save_network_json(config.network_json, network)
    save_failed_dois(config.failed_txt, crawler.failed)

    report = build_report(
        nodes=crawler.nodes,
        edges=crawler.edges,
        seed_dois=crawler.seed_dois,
        failed=crawler.failed,
        dropped_edges_no_doi=crawler.dropped_edges_no_doi,
        source_hits=crawler._source_hits,
        status=crawler.status,
    )
    save_report(config.report_json, report)

    if export_edgelist:
        export_edgelist_csv(config.edgelist_csv, crawler.edges)
    if export_graphml_flag:
        export_graphml(config.graphml, crawler.nodes, crawler.edges)

    # Clean up checkpoint only on a clean, non-interrupted finish (§4.2).
    if not interrupted and crawler.status != "interrupted":
        clear_checkpoint(config.checkpoint_json)

    print_summary(report)
    return report


def main(argv: List[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    config = config_from_args(args)

    _setup_logging(config.crawl_log, args.verbose)

    seeds = load_seeds(args.seeds, args.seed_file)
    if not seeds and not args.resume:
        print("Error: no seeds provided. Use --seeds or --seed-file.",
              file=sys.stderr)
        return 2

    run(
        config,
        seeds,
        export_edgelist=args.export_edgelist,
        export_graphml_flag=args.export_graphml,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
