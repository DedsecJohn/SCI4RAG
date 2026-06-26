"""
Semantic similarity (embedding cosine, FR-F6) -- split into Title and Abstract.

Each local paper's Title and Abstract are encoded *separately* with a
scientific-domain embedding model (default SPECTER2; configurable), producing
two similarities per candidate edge: ``title_sim`` and ``abstract_sim``.
Embeddings are computed once, cached, and cosine is evaluated only on candidate
edges (FR-F6.4). The model name/version/dim and random seed are locked into
metadata (GC-3.2/3.3).

Self-citation-loop warning (FR-F6.5): SPECTER-family models are trained so that
cited papers are similar; using these similarities to predict/weight citations
and then rebuild the graph risks label leakage. Downstream evaluation must use
citation-independent labels. This warning is emitted in metadata.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from src.core.utils import load_json, read_text, exists
from src.service.gragh.citation.config import CitationFeatureConfig
from src.service.gragh.citation.features import FeatureResult
from src.service.gragh.citation.index import PaperIndex, local_paths


SEMANTIC_COLUMNS = ("title_sim", "abstract_sim")

_SELF_LOOP_WARNING = (
    "Semantic similarity uses embeddings trained on citation signals "
    "(SPECTER-family). Downstream evaluation must use labels independent of "
    "citations to avoid a self-confirming loop / label leakage."
)


def _abstract_from_markdown(text: str, max_chars: int = 2000) -> str:
    """Fallback abstract: leading prose paragraph(s) of document.md."""
    paras = []
    for raw in text.split("\n\n"):
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith("$$"):
            continue
        paras.append(line)
        if sum(len(p) for p in paras) >= max_chars:
            break
    return " ".join(paras)[:max_chars]


def _paper_title_abstract(index: PaperIndex, pid: int) -> Tuple[str, str]:
    """Return ``(title, abstract)`` for one paper (abstract may fall back to md)."""
    paths = local_paths(index, pid)
    meta = load_json(paths["doi_json"]) or {}
    title = (meta.get("title") or index.title(pid) or "").strip()
    abstract = (meta.get("abstract") or "").strip()
    if not abstract and exists(paths["document_md"]):
        abstract = _abstract_from_markdown(read_text(paths["document_md"]))
    return title, abstract


def _load_model(config: CitationFeatureConfig, logger=None):
    """Load the embedding model, trying the primary then the fallbacks."""
    from sentence_transformers import SentenceTransformer

    candidates = [config.semantic_model_name] + list(config.semantic_model_fallbacks)
    last_err = None
    for name in candidates:
        try:
            model = SentenceTransformer(name)
            if config.semantic_max_seq_length:
                model.max_seq_length = config.semantic_max_seq_length
            if logger is not None and name != config.semantic_model_name:
                logger.warning("semantic: primary model unavailable, using fallback {n}", n=name)
            return model, name
        except Exception as exc:  # noqa: BLE001 - report and try next
            last_err = exc
            if logger is not None:
                logger.warning("semantic: failed to load {n}: {e}", n=name, e=str(exc))
    raise RuntimeError(f"semantic: could not load any embedding model: {last_err}")


def _encode_field(
    model,
    index: PaperIndex,
    texts_of: Dict[int, str],
    dim: int,
) -> Tuple[np.ndarray, Dict[int, int]]:
    """Encode the non-empty texts of one field; return (matrix, pid->row)."""
    texts: List[str] = []
    row_of: Dict[int, int] = {}
    for pid in range(index.n):
        text = (texts_of.get(pid) or "").strip()
        if text:
            row_of[pid] = len(texts)
            texts.append(text)
    if texts:
        emb = model.encode(
            texts,
            batch_size=16,
            convert_to_numpy=True,
            normalize_embeddings=True,  # cosine == dot product on unit vectors
            show_progress_bar=False,
        ).astype(np.float32)
    else:
        emb = np.zeros((0, dim), dtype=np.float32)
    return emb, row_of


def compute_semantic(
    index: PaperIndex,
    candidate_pairs,
    config: CitationFeatureConfig,
    logger=None,
) -> FeatureResult:
    """Encode Title and Abstract separately and compute per-edge cosines."""
    import torch
    np.random.seed(config.semantic_random_seed)
    torch.manual_seed(config.semantic_random_seed)

    result = FeatureResult(columns=SEMANTIC_COLUMNS)
    model, model_name = _load_model(config, logger=logger)

    title_of: Dict[int, str] = {}
    abstract_of: Dict[int, str] = {}
    for pid in range(index.n):
        title, abstract = _paper_title_abstract(index, pid)
        title_of[pid] = title
        abstract_of[pid] = abstract

    _dim_fn = getattr(model, "get_embedding_dimension", None) or model.get_sentence_embedding_dimension
    dim = int(_dim_fn())

    title_emb, title_row = _encode_field(model, index, title_of, dim)
    abstract_emb, abstract_row = _encode_field(model, index, abstract_of, dim)
    result.title_embeddings = title_emb
    result.abstract_embeddings = abstract_emb

    for pid in range(index.n):
        result.node_values.setdefault(pid, {})
        result.node_values[pid]["title_embedding_row"] = title_row.get(pid, -1)
        result.node_values[pid]["abstract_embedding_row"] = abstract_row.get(pid, -1)

    def _cosine(emb, row_of, i, j) -> Optional[float]:
        ri, rj = row_of.get(i), row_of.get(j)
        if ri is None or rj is None:
            return None
        return float(np.dot(emb[ri], emb[rj]))  # unit-normalized -> cosine

    for (i, j) in candidate_pairs:
        result.edge_values[(i, j)] = {
            "title_sim": _cosine(title_emb, title_row, i, j),
            "abstract_sim": _cosine(abstract_emb, abstract_row, i, j),
        }

    result.metadata = {
        "source": "local doi.json Title and Abstract (document.md fallback for abstract)",
        "model_name": model_name,
        "model_requested": config.semantic_model_name,
        "embedding_dim": dim,
        "similarity": "cosine",
        "random_seed": config.semantic_random_seed,
        "max_seq_length": config.semantic_max_seq_length,
        "papers_with_title": len(title_row),
        "papers_with_abstract": len(abstract_row),
        "warning": _SELF_LOOP_WARNING,
    }
    if logger is not None:
        logger.info(
            "semantic: model={m} dim={d} titles={t} abstracts={a}",
            m=model_name, d=dim, t=len(title_row), a=len(abstract_row),
        )
    return result
