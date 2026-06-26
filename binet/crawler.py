"""
Bidirectional BFS citation crawler (FR-1 ~ FR-4).

Starting from a seed set S (depth=0), the crawler expands the citation network
in both directions:
    - backward: a paper's references (out-edges; source=current, target=ref)
    - forward:  a paper's cited-by  (in-edges; source=citing, target=current)

Key behaviours:
    - BFS, level by level; a node's ``depth`` is its first-discovery min depth
      (FR-1.3, FR-3.3).
    - Per-direction depth budgets backward_depth / forward_depth (FR-2.3).
    - Hard circuit-breaker on node count via ``max_papers`` (FR-2.4, AC-5).
    - Source fallback chains for both directions (§2 fallback).
    - Periodic checkpointing for resume (FR-5.4, AC-4).
    - Edge direction semantics are preserved (FR-1.4); edges are de-duplicated
      (FR-4.2).
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Dict, List, Optional, Set, Tuple

from tqdm import tqdm

from binet.config import BinetConfig
from binet.checkpoint import save_checkpoint, load_checkpoint
from binet.doi_utils import normalize_doi
from binet.errors import DeterministicFailure, NotSupportedError
from binet.http_client import HttpClient
from binet.models import Direction, Edge, NodeRecord, QueueItem
from binet.src.base import CitationSource
from binet.src.openalex import OpenAlexSource
from binet.src.crossref import CrossRefSource
from binet.src.semantic import SemanticScholarSource

logger = logging.getLogger("binet.crawler")


# Status values for the output metadata (§4.1).
STATUS_COMPLETED = "completed"
STATUS_MAX_PAPERS = "max_papers_reached"
STATUS_INTERRUPTED = "interrupted"

# Registry mapping source names to their classes.
_SOURCE_REGISTRY = {
    "openalex": OpenAlexSource,
    "crossref": CrossRefSource,
    "semantic_scholar": SemanticScholarSource,
}


class Crawler:
    """The bidirectional BFS engine."""

    def __init__(self, config: BinetConfig):
        self.config = config
        self.http = HttpClient(
            email=config.email,
            delay_range=config.delay_range,
            max_retries=config.max_retries,
        )

        # Instantiate one source object per unique name and reuse across chains
        # so per-node work caches (OpenAlex) stay effective.
        self._sources: Dict[str, CitationSource] = {}
        for name in set(config.reference_sources) | set(config.citation_sources):
            cls = _SOURCE_REGISTRY.get(name)
            if cls is None:
                logger.warning("Unknown source '%s' ignored", name)
                continue
            self._sources[name] = cls(self.http)

        self.ref_chain = [n for n in config.reference_sources if n in self._sources]
        self.cit_chain = [n for n in config.citation_sources if n in self._sources]

        # Graph state.
        self.nodes: Dict[str, NodeRecord] = {}
        self.edges: Set[Edge] = set()
        self.queue: deque[QueueItem] = deque()
        self.processed: Set[str] = set()  # DOIs whose expansion is done
        self.failed: Dict[str, str] = {}  # doi -> reason
        self.seed_dois: List[str] = []

        # Counters.
        self.dropped_edges_no_doi = 0
        self._source_hits: Dict[str, int] = {}
        self._processed_since_ckpt = 0

        self.status = STATUS_COMPLETED

    # ------------------------------------------------------------------ #
    # Public entry
    # ------------------------------------------------------------------ #

    def crawl(self, seeds: List[str]) -> None:
        """
        Run the bidirectional crawl.

        Args:
            seeds: Seed DOIs (any form; will be normalized).
        """
        if self.config.resume and load_checkpoint(self.config.checkpoint_json):
            self._restore()
        else:
            self._seed(seeds)

        direction_filter = self.config.direction

        try:
            self._run_bfs(direction_filter)
        except KeyboardInterrupt:
            logger.warning("Interrupted by user; saving checkpoint...")
            self.status = STATUS_INTERRUPTED
            self._checkpoint()
            raise
        finally:
            # Aggregate dropped-no-doi counters from sources (FR-4.3).
            for src in self._sources.values():
                self.dropped_edges_no_doi += getattr(src, "dropped_no_doi", 0)

    # ------------------------------------------------------------------ #
    # Seeding
    # ------------------------------------------------------------------ #

    def _seed(self, seeds: List[str]) -> None:
        """Initialize the queue with the seed set at depth 0 (FR-1.1)."""
        try:
            initial = Direction(self.config.direction)
        except ValueError:
            initial = Direction.BOTH

        for raw in seeds:
            doi = normalize_doi(raw)
            if not doi:
                logger.warning("Skipping invalid seed DOI: %s", raw)
                continue
            if doi in self.nodes:
                continue  # de-dup seeds by normalized DOI (AC-3)
            self.seed_dois.append(doi)
            self.nodes[doi] = NodeRecord(doi=doi, depth=0, is_seed=True)
            self.queue.append(QueueItem(doi=doi, depth=0, direction=initial))

    # ------------------------------------------------------------------ #
    # BFS main loop
    # ------------------------------------------------------------------ #

    def _run_bfs(self, direction_filter: str) -> None:
        pbar = tqdm(desc="binet BFS", unit="node")
        while self.queue:
            if len(self.nodes) >= self.config.max_papers:
                self.status = STATUS_MAX_PAPERS
                logger.warning(
                    "max_papers=%d reached, stopping (nodes=%d)",
                    self.config.max_papers, len(self.nodes),
                )
                break

            item = self.queue.popleft()
            if item.doi in self.processed:
                continue

            self._process_node(item)
            self.processed.add(item.doi)
            pbar.update(1)
            pbar.set_postfix(nodes=len(self.nodes), edges=len(self.edges))

            self._processed_since_ckpt += 1
            if self._processed_since_ckpt >= self.config.checkpoint_every:
                self._checkpoint()
                self._processed_since_ckpt = 0

        pbar.close()

    def _process_node(self, item: QueueItem) -> None:
        """Fetch metadata and expand the node in its assigned direction(s)."""
        doi = item.doi

        # 1. Metadata (title); fill placeholder if missing (§4.1).
        meta = self._fetch_metadata(doi)
        node = self.nodes.get(doi) or NodeRecord(doi=doi, depth=item.depth)
        if meta and meta.title:
            node.title = meta.title
        elif not node.title:
            node.title = "Unknown Title"
        node.depth = min(node.depth, item.depth)
        self.nodes[doi] = node

        # 2. Expand directions, each honouring its own depth budget.
        expand = item.direction
        do_backward = expand in (Direction.BACKWARD, Direction.BOTH)
        do_forward = expand in (Direction.FORWARD, Direction.BOTH)

        if do_backward and item.depth < self.config.backward_depth:
            targets = self._fetch_with_fallback(doi, self.ref_chain, "fetch_references")
            for tgt in targets:
                self._add_edge(doi, tgt)  # current cites tgt
                self._discover(tgt, item.depth + 1)

        if do_forward and item.depth < self.config.forward_depth:
            sources = self._fetch_with_fallback(doi, self.cit_chain, "fetch_citations")
            for src in sources:
                self._add_edge(src, doi)  # src cites current
                self._discover(src, item.depth + 1)

    # ------------------------------------------------------------------ #
    # Graph mutation helpers
    # ------------------------------------------------------------------ #

    def _add_edge(self, source_doi: str, target_doi: str) -> None:
        """Add a directed edge (source cites target), de-duplicated (FR-4.2)."""
        if not source_doi or not target_doi or source_doi == target_doi:
            return
        self.edges.add(Edge(source_doi=source_doi, target_doi=target_doi))

    def _discover(self, doi: str, depth: int) -> None:
        """
        Register a newly discovered node and enqueue it for expansion.

        ``depth`` is kept at its minimum across paths (FR-3.3). Discovery
        respects the max_papers cap so we never create more nodes than allowed.
        """
        if not doi:
            return
        if doi in self.nodes:
            # Update min depth even if already known.
            self.nodes[doi].depth = min(self.nodes[doi].depth, depth)
            return
        if len(self.nodes) >= self.config.max_papers:
            # Cap reached; do not create new nodes. Mark status (AC-5) unless we
            # were interrupted.
            if self.status != STATUS_INTERRUPTED:
                self.status = STATUS_MAX_PAPERS
            return

        self.nodes[doi] = NodeRecord(doi=doi, depth=depth, is_seed=False)

        # Only enqueue if there is remaining depth budget in some direction.
        try:
            base_dir = Direction(self.config.direction)
        except ValueError:
            base_dir = Direction.BOTH
        if depth < max(self.config.backward_depth, self.config.forward_depth):
            self.queue.append(QueueItem(doi=doi, depth=depth, direction=base_dir))

    # ------------------------------------------------------------------ #
    # Source dispatch with fallback chains
    # ------------------------------------------------------------------ #

    def _fetch_metadata(self, doi: str):
        for name in self.ref_chain + self.cit_chain:
            src = self._sources.get(name)
            if not src:
                continue
            try:
                meta = src.fetch_metadata(doi)
            except DeterministicFailure as exc:
                self.failed[doi] = exc.reason
                return None
            except NotSupportedError:
                continue
            if meta:
                return meta
        return None

    def _fetch_with_fallback(
        self, doi: str, chain: List[str], method: str
    ) -> List[str]:
        """
        Try each source in ``chain`` until one yields results.

        Args:
            doi: Source DOI.
            chain: Ordered source names.
            method: 'references' or 'citations'.

        Returns:
            A de-duplicated list of DOIs (order preserved).
        """
        for name in chain:
            src = self._sources.get(name)
            if not src:
                continue
            try:
                result = getattr(src, method)(doi)
            except NotSupportedError:
                continue
            except DeterministicFailure as exc:
                self.failed[doi] = exc.reason
                return []
            if result:
                self._source_hits[name] = self._source_hits.get(name, 0) + 1
                # De-duplicate while preserving order.
                seen: Set[str] = set()
                deduped = []
                for d in result:
                    if d and d not in seen:
                        seen.add(d)
                        deduped.append(d)
                return deduped
        return []

    # ------------------------------------------------------------------ #
    # Checkpoint (de)serialization
    # ------------------------------------------------------------------ #

    def _checkpoint(self) -> None:
        state = {
            "seed_dois": self.seed_dois,
            "queue": [q.to_dict() for q in self.queue],
            "nodes": {d: n.to_dict() for d, n in self.nodes.items()},
            "edges": [e.to_dict() for e in self.edges],
            "processed": list(self.processed),
            "failed": self.failed,
            "dropped_edges_no_doi": self.dropped_edges_no_doi,
            "source_hits": self._source_hits,
            "status": self.status,
        }
        save_checkpoint(self.config.checkpoint_json, state)

    def _restore(self) -> None:
        state = load_checkpoint(self.config.checkpoint_json)
        if not state:
            return
        self.seed_dois = state.get("seed_dois", [])
        self.queue = deque(QueueItem.from_dict(q) for q in state.get("queue", []))
        self.nodes = {
            d: NodeRecord.from_dict(n) for d, n in state.get("nodes", {}).items()
        }
        self.edges = {Edge.from_dict(e) for e in state.get("edges", [])}
        self.processed = set(state.get("processed", []))
        self.failed = state.get("failed", {})
        self.dropped_edges_no_doi = state.get("dropped_edges_no_doi", 0)
        self._source_hits = state.get("source_hits", {})
        logger.info(
            "Resumed from checkpoint: nodes=%d edges=%d queue=%d processed=%d",
            len(self.nodes), len(self.edges), len(self.queue), len(self.processed),
        )
