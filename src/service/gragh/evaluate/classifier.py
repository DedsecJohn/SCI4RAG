"""
LLM-based relation classifier for a single directed citation edge.

Each edge is judged in one LLM call: all its citation contexts are packed into a
single query (:func:`build_edge_query`) and the model returns a minified JSON
``{"label": ..., "reason": ...}``. Parsing is defensive: malformed output or an
out-of-vocabulary label falls back to ``unknown``.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import List, Optional

from src.llm.chat.response import llm_response
from src.service.gragh.evaluate.config import (
    INHERITANCE_LABEL,
    UNKNOWN_LABEL,
)
from src.service.gragh.evaluate.context_collector import CitationContext
from src.service.gragh.evaluate.prompt import (
    RELATION_SYSTEM_PROMPT,
    build_edge_query,
)


_JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)


@dataclass
class RelationResult:
    """Outcome of classifying one directed edge."""

    label: str
    reason: str


def _parse_label(raw: str, label_set) -> RelationResult:
    """Parse the model output into a (label, reason), defaulting to unknown."""
    text = (raw or "").strip()

    label: Optional[str] = None
    reason = ""

    match = _JSON_OBJ_RE.search(text)
    if match:
        try:
            obj = json.loads(match.group(0))
            label = str(obj.get("label", "")).strip().lower() or None
            reason = str(obj.get("reason", "")).strip()
        except (json.JSONDecodeError, AttributeError):
            label = None

    # Fallback: scan the raw text for a known label token.
    if label not in label_set:
        lowered = text.lower()
        if INHERITANCE_LABEL in lowered:
            label = INHERITANCE_LABEL
        else:
            label = UNKNOWN_LABEL
        if not reason:
            reason = "fallback parse from raw model output"

    return RelationResult(label=label, reason=reason)


def classify_edge(
    source_title: str,
    target_title: str,
    contexts: List[CitationContext],
    temperature: float = 0.1,
    label_set=(INHERITANCE_LABEL, UNKNOWN_LABEL),
    logger=None,
) -> RelationResult:
    """
    Classify one directed edge from its citation contexts via a single LLM call.

    Args:
        source_title: Title of the citing paper.
        target_title: Title of the cited paper.
        contexts: Citation contexts where the citing paper references the cited.
        temperature: Sampling temperature (low keeps the judgement stable).
        label_set: Allowed labels.
        logger: Optional loguru logger.

    Returns:
        A :class:`RelationResult`. Edges without any context, or whose LLM call
        fails, are returned as ``unknown``.
    """
    if not contexts:
        return RelationResult(
            label=UNKNOWN_LABEL,
            reason="no in-text citation context found for this edge",
        )

    query = build_edge_query(source_title, target_title, contexts)
    try:
        response = llm_response(
            query=query,
            system_prompt=RELATION_SYSTEM_PROMPT,
            temperature=temperature,
        )
        raw = response.get("content", "") if response else ""
    except Exception as exc:  # noqa: BLE001 - never abort the whole batch
        if logger is not None:
            logger.exception(
                "relation-eval: LLM call failed for edge '{s}' -> '{t}': {e}",
                s=source_title[:40], t=target_title[:40], e=exc,
            )
        return RelationResult(label=UNKNOWN_LABEL, reason=f"llm error: {exc}")

    return _parse_label(raw, label_set)
