"""
Prompt engineering for citation-edge relation classification.

The model receives, for a single directed edge ``i -> j`` (paper ``i`` cites
paper ``j``), all the in-text citation contexts where ``i`` references ``j``,
and must decide whether the edge represents *knowledge inheritance*.

It returns a single-line JSON object ``{"label": ..., "reason": ...}`` with
``label`` constrained to ``inheritance`` or ``unknown``.
"""

from __future__ import annotations

from typing import List

from src.service.gragh.evaluate.config import INHERITANCE_LABEL, UNKNOWN_LABEL
from src.service.gragh.evaluate.context_collector import CitationContext


RELATION_SYSTEM_PROMPT = (
    "You are an expert in scientific citation analysis.\n"
    "You are given the in-text citation contexts in which a CITING paper refers "
    "to a CITED paper. Knowledge flows from the cited paper to the citing paper.\n"
    "Decide the relation type of this directed citation edge "
    "(citing -> cited).\n\n"
    "Classify into EXACTLY ONE of the following labels:\n\n"
    f"{INHERITANCE_LABEL}: Knowledge inheritance. The citing paper directly "
    "builds upon, adopts, extends, reuses, or continues a method, model, theory, "
    "concept, framework, dataset, or technique that originates from the cited "
    "paper. The cited work is a foundation the citing work inherits from. "
    "Example: the Transformer paper inherits the attention method from "
    "'Attention is all you need' (method inheritance, a typical case of "
    "knowledge inheritance).\n"
    f"{UNKNOWN_LABEL}: It cannot be determined from the given contexts that the "
    "relation is knowledge inheritance (e.g. the citation is only a passing "
    "mention, background, comparison, contrast, or the evidence is insufficient/"
    "ambiguous).\n\n"
    "Rules:\n"
    "1. Base your judgement ONLY on the provided citation contexts.\n"
    f"2. If you are not clearly confident it is inheritance, return "
    f"'{UNKNOWN_LABEL}'.\n"
    "3. Respond with ONE single-line minified JSON object and nothing else, "
    'in the form {"label": "<label>", "reason": "<short reason>"}.\n'
    f"4. 'label' MUST be exactly '{INHERITANCE_LABEL}' or '{UNKNOWN_LABEL}'.\n"
    "5. Keep 'reason' concise (one sentence)."
)


def build_edge_query(
    source_title: str,
    target_title: str,
    contexts: List[CitationContext],
) -> str:
    """
    Build the per-edge user query from all citation contexts of the edge.

    Args:
        source_title: Title of the citing paper (``i``).
        target_title: Title of the cited paper (``j``).
        contexts: Citation contexts where ``i`` references ``j``.

    Returns:
        A single query string sent to the LLM in one call.
    """
    lines: List[str] = [
        f"CITING paper (source): {source_title or '(unknown title)'}",
        f"CITED paper (target):  {target_title or '(unknown title)'}",
        "",
        "In-text citation contexts where the citing paper references the cited "
        "paper:",
    ]
    for idx, ctx in enumerate(contexts, start=1):
        marker = " ".join(ctx.markers) if ctx.markers else ""
        text = ctx.context.strip() or ctx.citation_sentence.strip()
        header = f"[Context {idx}]"
        if marker:
            header += f" markers={marker}"
        lines.append(header)
        lines.append(text)
        lines.append("")

    lines.append(
        "Question: Does this directed edge (citing -> cited) represent "
        "knowledge inheritance?"
    )
    return "\n".join(lines)
