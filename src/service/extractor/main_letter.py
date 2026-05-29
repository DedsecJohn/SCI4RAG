import re
from src.llm.chat.response import llm_response


_MAIN_LETTER_SYSTEM_PROMPT = (
    "You are an expert in academic paper structure analysis.\n"
    "Classify the given text into ONE of the following categories ONLY:\n\n"
    "main_letter or other\n\n"
    "Category definitions:\n"
    "main_letter: Main body text from Introduction, Methods, Results, Discussion, "
    "or Conclusion sections. Contains scientific content with technical details, "
    "research methodology, findings, or analysis. Written in complete sentences with "
    "domain-specific terminology. Citations are optional — a paragraph can be "
    "main_letter even without references.\n"
    "other: Anything else — author names, affiliations, acknowledgments, funding statements, "
    "figure/table captions, equations, metadata (DOI, dates), or any non-body content.\n\n"
    "Rules:\n"
    "1. Return ONLY the category name in lowercase.\n"
    "2. Do NOT add explanations or extra text.\n"
    "3. If uncertain, return 'other'.\n"
)


def check_section_heading(chunk: str) -> bool | None:
    """
    Check if a chunk is a section heading that determines main_letter status.

    Args:
        chunk (str): Markdown chunk text.

    Returns:
        bool | None: True if it is a wanted heading (main_letter),
            False if it is an unwanted heading (other),
            None if not a heading.
    """
    heading_match = re.match(r'^#\s+(.+)$', chunk.strip())
    if not heading_match:
        return None

    text = heading_match.group(1).strip()

    # Wanted: numbered sections like "# 1. Introduction", "# 3.2. Results", "# 3.3.1. Thermal stabilities"
    if re.match(r'^\d+(\.\d+)*\..+', text):
        return True

    # Wanted: common section names
    if re.match(r'(?i)^(introduction|experimental|methods?|results?|discussion|conclusions?|summary)$', text):
        return True

    # Unwanted: uppercase metadata sections
    if re.match(r'^(ASSOCIATED CONTENT|AUTHOR INFORMATION|ACKNOWLEDGMENTS)$', text):
        return False

    # Unwanted: specific author-related headings
    if re.match(r'(?i)^(corresponding authors?|author contributions)', text):
        return False

    # Unwanted: supporting information (may contain markdown formatting)
    if re.search(r'(?i)supporting information', text):
        return False

    return None


def should_skip_main_letter_check(chunk: str) -> tuple[bool, str]:
    """
    Pre-filter to avoid unnecessary LLM calls for obvious non-main-body content.

    Args:
        chunk (str): Markdown chunk text.

    Returns:
        tuple[bool, str]: (should_skip, reason)
            - should_skip: True if this chunk is definitely not main_letter
            - reason: explanation for the decision
    """
    # Too short (< 50 chars)
    if len(chunk.strip()) < 50:
        return True, "too_short"

    # Contains obvious metadata markers
    if re.search(r'(?i)(\bDOI\b|doi\.org|email:|@)', chunk):
        return True, "metadata"

    # Pure equation block
    if chunk.count('$$') >= 2:
        return True, "equation_only"

    return False, "needs_check"


def has_fig_or_citations(chunk: str) -> bool:
    """
    Check if a long chunk contains figure references or citations.

    Args:
        chunk (str): Markdown chunk text.

    Returns:
        bool: True if chunk contains Fig/Figure references or citation brackets.
    """
    if len(chunk) < 200:
        return False

    has_fig_ref = bool(re.search(r'(?i)\b(fig\.|figure)\s*\d+', chunk))

    has_citation = bool(re.search(r'\[\d+[\d,\s\-–]+\]|\[\d+\]', chunk))

    return has_fig_ref or has_citation


def is_likely_main_body(chunk: str) -> bool:
    """
    Heuristic check if content is very likely main body text (avoid LLM call).

    Args:
        chunk (str): Markdown chunk text.

    Returns:
        bool: True if chunk is very likely main_letter.
    """
    # Long paragraph (> 200 chars)
    if len(chunk) < 200:
        return False

    # Contains citations
    has_citations = bool(re.search(r'\[\d+\]|\(\d{4}\)|\bet al\.', chunk))

    # Multiple complete sentences
    sentence_count = len(re.findall(r'[.!?]\s+[A-Z]', chunk))

    # Does not start with figure/table markers
    starts_with_fig_table = bool(re.match(r'(?i)^\s*(fig\.|figure|table)\s+\d+', chunk))

    return has_citations and sentence_count >= 2 and not starts_with_fig_table


def identify_main_letter(chunk: str) -> bool:
    """
    Check if a markdown chunk is main body text using rule-based pre-filtering + LLM.

    Args:
        chunk (str): Markdown chunk text.

    Returns:
        bool: True if the chunk is identified as main_letter.
    """
    # Section heading regex check (runs first to avoid being filtered by too_short)
    heading_result = check_section_heading(chunk)
    if heading_result is not None:
        return heading_result

    # Rule-based pre-filtering: explicit exclusion
    should_skip, reason = should_skip_main_letter_check(chunk)
    if should_skip:
        return False

    # Rule-based pre-filtering: explicit inclusion
    if is_likely_main_body(chunk):
        return True

    # Long chunk with Fig/citations → very likely main body
    if has_fig_or_citations(chunk):
        return True

    # Ambiguous case: call LLM
    result = llm_response(query=chunk, system_prompt=_MAIN_LETTER_SYSTEM_PROMPT, temperature=0.1)["content"].strip().lower()
    return result == "main_letter"
