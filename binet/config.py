"""
Configuration and default values for binet (FR-6.1).

All defaults live here so the tool can run with zero arguments (AC-2). The CLI
(``main.py``) overrides these via argparse.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

# Polite-pool contact email (§2). Sent to OpenAlex / CrossRef so we land in the
# faster, more reliable rate-limit pool.
CONTACT_EMAIL = "dedsecjohn@163.com"

# Default output directory (relative to the binet package).
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "output"


@dataclass
class BinetConfig:
    """
    Runtime configuration for a crawl.

    Attributes:
        seed_dois: Seed DOIs (depth=0). May be empty if ``seed_file`` is set.
        seed_file: Optional path to a file with seeds (one DOI per line or a
            JSON array).
        max_depth: Max hops from the seeds (FR-2.1). Default 1 (FR-2.2).
        backward_depth: Per-direction depth for references; defaults to
            ``max_depth`` when None (FR-2.3).
        forward_depth: Per-direction depth for citations; defaults to
            ``max_depth`` when None (FR-2.3).
        direction: Which directions to expand: backward / forward / both.
        max_papers: Hard circuit-breaker on node count (FR-2.4). Default 2000.
        reference_sources: Ordered fallback chain for references (backward).
        citation_sources: Ordered fallback chain for citations (forward).
        email: Polite-pool contact email.
        delay_range: Random per-request delay range in seconds (FR-5.3).
        max_retries: Max retry attempts with exponential backoff (FR-5.1).
        checkpoint_every: Save a checkpoint every N processed nodes (FR-5.4).
        output_dir: Directory for all output artifacts.
        resume: Whether to resume from an existing checkpoint (AC-4).
    """

    seed_dois: List[str] = field(default_factory=list)
    seed_file: Optional[str] = None

    max_depth: int = 2
    backward_depth: Optional[int] = None
    forward_depth: Optional[int] = None
    direction: str = "both"

    max_papers: int = 10000

    reference_sources: List[str] = field(default_factory=lambda: ["openalex", "crossref"])
    citation_sources: List[str] = field(default_factory=lambda: ["openalex", "semantic_scholar"])

    email: str = CONTACT_EMAIL
    delay_range: Tuple[float, float] = (0.5, 1.0)
    max_retries: int = 3
    checkpoint_every: int = 50

    output_dir: Path = field(default_factory=lambda: DEFAULT_OUTPUT_DIR)
    resume: bool = False

    def __post_init__(self) -> None:
        # Resolve per-direction depths from max_depth when unset (FR-2.3).
        if self.backward_depth is None:
            self.backward_depth = self.max_depth
        if self.forward_depth is None:
            self.forward_depth = self.max_depth
        self.output_dir = Path(self.output_dir)

    # ---- Output artifact paths (§4.1 / §4.2) ----

    @property
    def network_json(self) -> Path:
        return self.output_dir / "citation_network.json"

    @property
    def failed_txt(self) -> Path:
        return self.output_dir / "failed_dois.txt"

    @property
    def checkpoint_json(self) -> Path:
        return self.output_dir / "checkpoint.json"

    @property
    def crawl_log(self) -> Path:
        return self.output_dir / "crawl.log"

    @property
    def report_json(self) -> Path:
        return self.output_dir / "crawl_report.json"

    @property
    def edgelist_csv(self) -> Path:
        return self.output_dir / "edgelist.csv"

    @property
    def graphml(self) -> Path:
        return self.output_dir / "citation_network.graphml"
