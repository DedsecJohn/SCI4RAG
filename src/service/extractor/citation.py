"""
In-text citation extraction.

Pipeline:
    1. Read the cleaned body text (``document.md``) and split it into sentences.
    2. Locate sentences that contain citation markers together with their
       previous and next sentence (context). Supported marker styles:
         - numeric bracket: ``[1-3]``, ``[4,5]``
         - numeric parenthesis: ``(9-12)``, ``(11, 13-15)`` (italic markers
           whose formatting is lost after markdown parsing)
         - numeric superscript: ``crystals.8,20`` (superscript whose formatting
           is partially lost after markdown parsing)
         - author-year: ``(Smith et al., 2020)``, ``Smith et al. (2020)``
    3. Resolve each marker to the cited article title:
         - numeric: marker number -> raw reference string in the ``# References``
           section of ``full.md`` -> title in ``references.json``.
         - author-year: (first-author surname, year) -> matching entry in
           ``references.json``.
    4. Save a structured ``citation.json``.
"""

import re
from typing import List, Dict, Tuple, Optional

from src.core.paths import (
    parse_path_info,
    parse_full_md,
    clean_document_md,
    clean_references_json,
    clean_citation_json,
    clean_citation_by_article_json,
)
from src.core.utils import load_json, save_json, read_text, exists
from src.core.logger import get_user_logger
from src.service.document.load_document import update_document_metadata


# ============================================================================
# References section parsing (from full.md)
# ============================================================================

REF_HEADER = re.compile(
    r'^\s*#+\s*(?:references|bibliography)(?:\s+and\s+notes?)?\s*:?\s*$',
    re.IGNORECASE | re.MULTILINE,
)

# Reference entry numbering at line start: [N] , (N) or N.
_REF_ENTRY_LINE = re.compile(r'^\s*(?:\[(\d+)\]|\((\d+)\)|(\d{1,3})\.)\s+(.+)$')


def extract_reference_section(full_md_text: str) -> str:
    """
    Return the reference section text of full.md.

    Prefers the text following a ``# References`` / ``# Bibliography`` header.
    When no such header exists (some parsed papers drop it), falls back to the
    trailing contiguous block of numbered entries (``[N]`` / ``(N)`` / ``N.``).

    Args:
        full_md_text (str): Full markdown text of the paper.

    Returns:
        str: Reference section text, or "" if it cannot be located.
    """
    if not full_md_text:
        return ""

    match = REF_HEADER.search(full_md_text)
    if match:
        return full_md_text[match.end():]

    # Fallback: no header -> locate the trailing block of numbered entries.
    lines = full_md_text.splitlines()
    entry_idx = {i for i, line in enumerate(lines) if _REF_ENTRY_LINE.match(line)}
    if len(entry_idx) < 3:
        return ""

    last = max(entry_idx)
    start = last
    nonentry_streak = 0
    i = last - 1
    while i >= 0:
        if i in entry_idx:
            start = i
            nonentry_streak = 0
        elif lines[i].strip() == "":
            pass  # tolerate blank lines between entries
        else:
            nonentry_streak += 1
            if nonentry_streak > 1:  # 2+ prose lines => block has ended
                break
        i -= 1

    return "\n".join(lines[start:])


def parse_reference_entries(ref_section: str) -> Dict[int, str]:
    """
    Build a mapping from reference number to its raw reference string.

    Each entry in the reference section is numbered as ``[N]``, ``(N)`` or
    ``N.``. Continuation lines (a wrapped reference) are appended to the
    current entry.

    Args:
        ref_section (str): Reference section text.

    Returns:
        Dict[int, str]: number -> raw reference string.
    """
    mapping: Dict[int, str] = {}
    current: Optional[int] = None

    for line in ref_section.splitlines():
        m = _REF_ENTRY_LINE.match(line)
        if m:
            number = m.group(1) or m.group(2) or m.group(3)
            current = int(number)
            mapping[current] = m.group(4).strip()
        elif current is not None:
            extra = line.strip()
            if extra:
                mapping[current] = (mapping[current] + " " + extra).strip()

    return mapping


# ============================================================================
# Title matching (against references.json)
# ============================================================================

def _normalize(text: str) -> str:
    """Lowercase, drop punctuation and collapse whitespace for fuzzy matching."""
    text = text.lower()
    text = re.sub(r'[^a-z0-9 ]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def match_reference(raw_ref: Optional[str], references_json: dict) -> Optional[dict]:
    """
    Resolve a raw reference string to its full info dict in references.json.

    Tries exact key match first, then a normalized exact match, then a
    normalized containment match as a fallback.

    Args:
        raw_ref (Optional[str]): Raw reference string from full.md.
        references_json (dict): references.json content (raw string -> info).

    Returns:
        Optional[dict]: Matched reference info dict, or None.
    """
    if not raw_ref or not references_json:
        return None

    if raw_ref in references_json:
        return references_json[raw_ref]

    nraw = _normalize(raw_ref)

    for key, info in references_json.items():
        if _normalize(key) == nraw:
            return info

    for key, info in references_json.items():
        nkey = _normalize(key)
        if nraw and (nraw in nkey or nkey in nraw):
            return info

    return None


def match_title(raw_ref: Optional[str], references_json: dict) -> Optional[str]:
    """Resolve a raw reference string to its title using references.json."""
    info = match_reference(raw_ref, references_json)
    return info.get("title") if info else None


# ============================================================================
# Marker extraction
# ============================================================================

# Body of a pure-numeric marker: digits joined by commas / dashes
_NUM_BODY = r'\d+(?:\s*[–—\-,]\s*\d+)*'

# Numeric bracket marker: [1-3], [4,5]
_BRACKET_RE = re.compile(rf'\[({_NUM_BODY})\]')

# Numeric parenthesis marker: (9-12), (11, 13-15) (italic markers, format lost)
_PAREN_NUM_RE = re.compile(rf'\(({_NUM_BODY})\)')

# Superscript citation sentinel, injected by _normalize_superscripts()
_SUP_SENTINEL_RE = re.compile(r'⟦SUP:([\d,\-–—\s]+)⟧')

# Reference-like keywords that precede a non-citation parenthesis, e.g. "Eq. (9)"
_REF_KEYWORD_RE = re.compile(
    r'(?i)\b(?:eqs?|eqn|figs?|figures?|tables?|refs?|sections?|sec|steps?'
    r'|nos?|chapters?|parts?|panels?|columns?|rows?|lines?|items?)\.?\s*$'
)

# Crystallographic context that follows Miller-index parentheses like
# "(100), (010), and (001) lattice directions".
_CRYSTALLO_RE = re.compile(
    r'(?i)\b(?:lattice|planes?|directions?|facets?|orientations?|surfaces?'
    r'|reflections?|miller|axes|axis|crystallographic)\b'
)


def _is_citation_token(token: str) -> bool:
    """A valid reference number: digits without zero-padding (no leading 0)."""
    return token.isdigit() and not (len(token) > 1 and token[0] == "0")


def _expand_numeric(inner: str) -> List[int]:
    """Expand a numeric marker body like ``1-3``, ``4,5``, ``7,9,10`` to ints."""
    nums: List[int] = []
    inner = inner.replace('–', '-').replace('—', '-')
    for part in inner.split(','):
        part = part.strip()
        if not part:
            continue
        if '-' in part:
            a, _, b = part.partition('-')
            a, b = a.strip(), b.strip()
            if _is_citation_token(a) and _is_citation_token(b):
                lo, hi = int(a), int(b)
                if lo <= hi:
                    nums.extend(range(lo, hi + 1))
        elif _is_citation_token(part):
            nums.append(int(part))
    return nums


def _filter_valid(nums: List[int], max_ref: Optional[int]) -> List[int]:
    """Keep only numbers within the valid reference range [1, max_ref]."""
    if max_ref is None:
        return nums
    return [n for n in nums if 1 <= n <= max_ref]


def extract_numeric_markers(
    sentence: str, max_ref: Optional[int] = None
) -> List[Tuple[str, List[int], str]]:
    """
    Extract numeric citation markers (bracket, parenthesis, superscript).

    Bracket markers are always accepted. Parenthesis and superscript markers
    are ambiguous (enumerations, equation/figure references, exponents), so
    they are validated against the valid reference range when ``max_ref`` is
    provided, and parenthesis markers are filtered with heuristics to drop
    enumeration items ``(1) ...`` and equation references ``Eq. (9)``.

    Args:
        sentence (str): A single sentence.
        max_ref (Optional[int]): Largest valid reference number.

    Returns:
        List[Tuple[str, List[int], str]]: (marker text, numbers, subtype).
    """
    results: List[Tuple[str, List[int], str]] = []

    # 1. Bracket markers (unambiguous)
    for m in _BRACKET_RE.finditer(sentence):
        nums = _expand_numeric(m.group(1))
        if nums:
            results.append((m.group(0), nums, "bracket"))

    # 2. Superscript markers (pre-normalized sentinels)
    for m in _SUP_SENTINEL_RE.finditer(sentence):
        nums = _filter_valid(_expand_numeric(m.group(1)), max_ref)
        if nums:
            marker = "^{" + re.sub(r'\s+', '', m.group(1)) + "}"
            results.append((marker, nums, "superscript"))

    # 3. Parenthesis markers (ambiguous, heuristics applied)
    for m in _PAREN_NUM_RE.finditer(sentence):
        if _REF_KEYWORD_RE.search(sentence[:m.start()]):
            continue  # e.g. "Eq. (9)", "Fig. (3)"
        if _CRYSTALLO_RE.search(sentence[m.end():m.end() + 40]):
            continue  # e.g. "(100), (010) lattice directions"
        inner = m.group(1)
        nums = _filter_valid(_expand_numeric(inner), max_ref)
        if not nums:
            continue
        is_multi = bool(re.search(r'[,–—\-]', inner))
        # A single-number parenthesis directly followed by a word is most
        # likely an enumeration item like "(1) open pores".
        if not is_multi and re.match(r'\s+[A-Za-z]', sentence[m.end():]):
            continue
        results.append((m.group(0), nums, "paren"))

    return results


_AUTHOR = r"[A-Z][A-Za-z'’\-]+"
_NAMES = (
    rf"{_AUTHOR}(?:\s+(?:et\s+al\.?|and|&)\s+{_AUTHOR})*(?:\s+et\s+al\.?)?"
)

# Narrative form: "Smith et al. (2020)" / "Smith and Jones (2019)"
_NARRATIVE_RE = re.compile(rf"({_NAMES})\s+\((\d{{4}})[a-z]?\)")

# Parenthetical block containing at least one 4-digit year
_PAREN_RE = re.compile(r"\(([^()]*\b\d{4}[a-z]?\b[^()]*)\)")

# A single author-year entry inside a parenthetical block
_PAREN_ENTRY_RE = re.compile(rf"({_NAMES})\s*,?\s+(\d{{4}})[a-z]?")


def _surname(names: str) -> str:
    """Return the first-author surname from a names string."""
    m = re.match(_AUTHOR, names)
    return m.group(0) if m else names.strip()


def extract_author_year_markers(sentence: str) -> List[Dict]:
    """
    Extract author-year citation markers from a sentence.

    Handles narrative form (``Smith et al. (2020)``) and parenthetical form
    (``(Smith, 2020; Jones, 2021)``).

    Args:
        sentence (str): A single sentence.

    Returns:
        List[Dict]: dicts with keys ``marker``, ``surname``, ``year``.
    """
    results: List[Dict] = []

    for m in _NARRATIVE_RE.finditer(sentence):
        results.append({
            "marker": m.group(0),
            "surname": _surname(m.group(1)),
            "year": m.group(2),
        })

    for m in _PAREN_RE.finditer(sentence):
        block = m.group(1)
        for em in _PAREN_ENTRY_RE.finditer(block):
            results.append({
                "marker": m.group(0),
                "surname": _surname(em.group(1)),
                "year": em.group(2),
            })

    return results


def find_author_year_title(
    surname: str, year: str, references_json: dict
) -> Optional[Tuple[str, Optional[str]]]:
    """
    Match an (surname, year) author-year citation to a references.json entry.

    Args:
        surname (str): First-author surname.
        year (str): Publication year (string).
        references_json (dict): references.json content.

    Returns:
        Optional[Tuple[str, Optional[str]]]: (raw reference, title) or None.
    """
    if not references_json:
        return None

    surname_l = surname.lower()
    for raw_key, info in references_json.items():
        if str(info.get("year")) != str(year):
            continue
        authors = info.get("author") or []
        in_authors = any(surname_l in str(a).lower() for a in authors)
        in_key = surname_l in raw_key.lower()
        if in_authors or in_key:
            return raw_key, info.get("title")

    return None


# ============================================================================
# Sentence splitting (from document.md)
# ============================================================================

_ABBREVIATIONS = [
    "e.g.", "i.e.", "et al.", "etc.", "vs.", "cf.", "viz.", "approx.",
    "Fig.", "Figs.", "Eq.", "Eqs.", "Ref.", "Refs.", "No.", "Nos.",
    "Sec.", "Eqn.", "Tab.", "Dr.", "Prof.", "Mr.", "Ms.",
]

# Words that, when followed by a period and digits, are NOT superscript
# citations (e.g. "Fig. 3", "Eq. 5", "et al. 2020").
_NON_SUPERSCRIPT_WORDS = {
    "fig", "figs", "figure", "figures", "eq", "eqs", "eqn", "ref", "refs",
    "no", "nos", "sec", "section", "tab", "table", "tables", "al", "etc",
    "vs", "cf", "viz", "approx", "dr", "prof", "mr", "ms", "vol", "p", "pp",
    "ch", "chap", "chapter", "step", "eqn",
}

# A content word + period + trailing digits (lost superscript), where the
# digits sit between two sentences, e.g. "crystals.8,20 Therefore".
_SUPERSCRIPT_RE = re.compile(
    r'([A-Za-z]{2,})\.\s*(\d+(?:\s*[,\-–—]\s*\d+)*)(?=\s+[A-Z(]|\s*$)'
)


def _normalize_superscripts(text: str) -> str:
    """
    Rewrite lost-superscript citations into sentinels before sentence splitting.

    ``... single crystals.8,20 Therefore ...`` becomes
    ``... single crystals ⟦SUP:8,20⟧. Therefore ...`` so that the period keeps
    acting as a sentence boundary and the digits stay with their own sentence.
    """
    def _repl(m: "re.Match") -> str:
        word, digits = m.group(1), m.group(2)
        if word.lower() in _NON_SUPERSCRIPT_WORDS:
            return m.group(0)
        digits_clean = re.sub(r'\s+', '', digits)
        return f"{word} ⟦SUP:{digits_clean}⟧."

    return _SUPERSCRIPT_RE.sub(_repl, text)


def render_markers(text: str) -> str:
    """Render internal superscript sentinels back to a readable ``^{...}`` form."""
    return _SUP_SENTINEL_RE.sub(
        lambda m: "^{" + re.sub(r'\s+', '', m.group(1)) + "}", text
    )


def split_sentences(text: str) -> List[str]:
    """
    Split cleaned markdown body text into a flat list of sentences.

    Block math, inline math, image links and headings are stripped first; the
    remaining non-empty lines are treated as paragraphs and split into
    sentences in reading order (so adjacent sentences across paragraphs are
    still neighbours).

    Args:
        text (str): Body markdown text (document.md).

    Returns:
        List[str]: Sentences in reading order.
    """
    if not text:
        return []

    # Remove block math, image links and inline math
    text = re.sub(r'\$\$.*?\$\$', ' ', text, flags=re.DOTALL)
    text = re.sub(r'!\[[^\]]*\]\([^)]*\)', ' ', text)
    text = re.sub(r'\$[^$]*\$', ' ', text)

    # Recover lost superscript citations (e.g. "crystals.8,20 Therefore")
    text = _normalize_superscripts(text)

    sentences: List[str] = []

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith('#'):          # markdown heading
            continue
        if not re.search(r'[A-Za-z]', line):  # leftover symbols only
            continue
        sentences.extend(_split_paragraph(line))

    return sentences


def _split_paragraph(paragraph: str) -> List[str]:
    """Split a single paragraph into sentences, protecting abbreviations."""
    protected = paragraph
    for abbr in _ABBREVIATIONS:
        protected = protected.replace(abbr, abbr.replace('.', '<DOT>'))
    # Protect decimal points (e.g. 0.026)
    protected = re.sub(r'(\d)\.(\d)', r'\1<DOT>\2', protected)

    parts = re.split(r'(?<=[.!?])\s+', protected)

    sentences = []
    for part in parts:
        part = part.replace('<DOT>', '.').strip()
        if part:
            sentences.append(part)
    return sentences


# ============================================================================
# Core extraction
# ============================================================================

def extract_citations(file_data: dict) -> dict:
    """
    Extract in-text citations with context and resolved titles for one file.

    Args:
        file_data (dict): File metadata (must contain file_path and file_id).

    Returns:
        dict: Structured citation result.
    """
    username, dataset_name = parse_path_info(file_data["file_path"])
    file_id = file_data["file_id"]
    logger = get_user_logger(username, dataset_name)

    document_md_path = clean_document_md(username, dataset_name, file_id)
    full_md_path = parse_full_md(username, dataset_name, file_id)
    references_path = clean_references_json(username, dataset_name, file_id)

    if not exists(document_md_path):
        logger.warning(
            "citation: document.md not found for {fid}, no citation sentences "
            "can be located", fid=file_id
        )
        body_text = ""
    else:
        body_text = read_text(document_md_path)

    if not exists(full_md_path):
        logger.warning(
            "citation: full.md not found for {fid}, reference numbers cannot "
            "be resolved", fid=file_id
        )
    full_text = read_text(full_md_path) if exists(full_md_path) else ""
    references_json = load_json(references_path)
    if not references_json:
        logger.warning(
            "citation: references.json empty/missing for {fid}, titles cannot "
            "be matched", fid=file_id
        )

    num_to_raw = parse_reference_entries(extract_reference_section(full_text))
    max_ref = max(num_to_raw) if num_to_raw else None

    sentences = split_sentences(body_text)
    n = len(sentences)

    citations: List[Dict] = []
    citation_id = 0
    style_counts = {"numeric": 0, "author-year": 0}
    subtype_counts: Dict[str, int] = {}
    missing_titles = 0

    for i, sentence in enumerate(sentences):
        numeric = extract_numeric_markers(sentence, max_ref)
        author_year = extract_author_year_markers(sentence)

        if not numeric and not author_year:
            continue

        prev_sentence = render_markers(sentences[i - 1]) if i > 0 else ""
        next_sentence = render_markers(sentences[i + 1]) if i < n - 1 else ""
        cur_sentence = render_markers(sentence)
        context = " ".join(
            s for s in (prev_sentence, cur_sentence, next_sentence) if s
        )

        if numeric:
            citation_id += 1
            entry = _build_numeric_entry(
                citation_id, numeric, cur_sentence,
                prev_sentence, next_sentence, context,
                num_to_raw, references_json,
            )
            citations.append(entry)
            style_counts["numeric"] += 1
            for _, _, subtype in numeric:
                subtype_counts[subtype] = subtype_counts.get(subtype, 0) + 1
            missing_titles += sum(
                1 for a in entry["cited_articles"] if not a["title"]
            )

        if author_year:
            citation_id += 1
            entry = _build_author_year_entry(
                citation_id, author_year, cur_sentence,
                prev_sentence, next_sentence, context,
                references_json,
            )
            citations.append(entry)
            style_counts["author-year"] += 1
            missing_titles += sum(
                1 for a in entry["cited_articles"] if not a["title"]
            )

    logger.info(
        "citation: {fid} sentences={sents} citations={cites} "
        "(numeric={num} author-year={ay}) subtypes={subs} missing_titles={miss}",
        fid=file_id, sents=n, cites=len(citations),
        num=style_counts["numeric"], ay=style_counts["author-year"],
        subs=subtype_counts, miss=missing_titles,
    )

    return {
        "file_id": file_id,
        "file_name": file_data.get("file_name", ""),
        "total_citations": len(citations),
        "citations": citations,
    }


def _build_numeric_entry(
    citation_id, numeric, sentence, prev_sentence, next_sentence, context,
    num_to_raw, references_json,
) -> Dict:
    markers = []
    forms: List[str] = []
    ref_numbers: List[int] = []
    for marker, nums, subtype in numeric:
        markers.append(marker)
        if subtype not in forms:
            forms.append(subtype)
        for num in nums:
            if num not in ref_numbers:
                ref_numbers.append(num)
    ref_numbers.sort()

    cited_articles = []
    for num in ref_numbers:
        raw_ref = num_to_raw.get(num)
        cited_articles.append({
            "ref_key": num,
            "raw_reference": raw_ref,
            "title": match_title(raw_ref, references_json),
        })

    return {
        "citation_id": citation_id,
        "style": "numeric",
        "forms": forms,
        "markers": markers,
        "ref_keys": ref_numbers,
        "prev_sentence": prev_sentence,
        "citation_sentence": sentence,
        "next_sentence": next_sentence,
        "context": context,
        "cited_articles": cited_articles,
    }


def _build_author_year_entry(
    citation_id, author_year, sentence, prev_sentence, next_sentence, context,
    references_json,
) -> Dict:
    markers: List[str] = []
    ref_keys: List[List[str]] = []
    cited_articles = []

    for item in author_year:
        if item["marker"] not in markers:
            markers.append(item["marker"])
        key = [item["surname"], item["year"]]
        if key in ref_keys:
            continue
        ref_keys.append(key)
        match = find_author_year_title(
            item["surname"], item["year"], references_json
        )
        raw_ref, title = match if match else (None, None)
        cited_articles.append({
            "ref_key": key,
            "raw_reference": raw_ref,
            "title": title,
        })

    return {
        "citation_id": citation_id,
        "style": "author-year",
        "markers": markers,
        "ref_keys": ref_keys,
        "prev_sentence": prev_sentence,
        "citation_sentence": sentence,
        "next_sentence": next_sentence,
        "context": context,
        "cited_articles": cited_articles,
    }


def process_citations(file_data: dict, reidentify: bool = False) -> dict:
    """
    Extract citations for a file, persist citation.json and update metadata.

    When ``reidentify`` is False, files already marked as done / Not_Found are
    skipped to avoid redundant re-processing.

    Args:
        file_data (dict): File metadata.
        reidentify (bool): Force re-extraction even if already processed.

    Returns:
        dict: Updated file_data.
    """
    username, dataset_name = parse_path_info(file_data["file_path"])
    logger = get_user_logger(username, dataset_name)

    if not reidentify:
        state = file_data.get("citation_state")
        if state in ("done", "Not_Found"):
            logger.info(
                "citation: skip {fid} (state={state})",
                fid=file_data["file_id"], state=state,
            )
            return file_data

    result = extract_citations(file_data)

    citation_path = clean_citation_json(username, dataset_name, file_data["file_id"])
    save_json(result, citation_path, info=False)

    file_data["citation_state"] = "done" if result["citations"] else "Not_Found"
    update_document_metadata(username, dataset_name, file_data, info=False)
    logger.info(
        "citation: {fid} -> {state} ({n} citations) saved to {path}",
        fid=file_data["file_id"], state=file_data["citation_state"],
        n=result["total_citations"], path=str(citation_path),
    )

    return file_data


# ============================================================================
# Aggregation by cited article (doi / title)
# ============================================================================

def _article_group_key(doi: Optional[str], title: Optional[str],
                       raw_ref: Optional[str], ref_key) -> str:
    """Build a stable grouping key, preferring DOI, then title, then raw ref."""
    if doi:
        return f"doi::{doi.strip().lower()}"
    if title:
        return f"title::{_normalize(title)}"
    if raw_ref:
        return f"raw::{_normalize(raw_ref)}"
    return f"key::{ref_key}"


def _article_sort_key(article: dict) -> Tuple[int, str]:
    """Sort articles by smallest numeric ref number, then title."""
    numbers = [k for k in article["ref_keys"] if isinstance(k, int)]
    first = min(numbers) if numbers else 10 ** 9
    return first, (article.get("title") or "")


def aggregate_citations(file_data: dict) -> dict:
    """
    Aggregate citation sentences/context by cited article (DOI preferred).

    Reads the existing ``citation.json``, supplements each cited article with
    its DOI from ``references.json``, and groups every citation occurrence
    (sentence + context) under the article it refers to.

    Args:
        file_data (dict): File metadata (must contain file_path and file_id).

    Returns:
        dict: Article-grouped citation result.
    """
    username, dataset_name = parse_path_info(file_data["file_path"])
    file_id = file_data["file_id"]
    logger = get_user_logger(username, dataset_name)

    citation_path = clean_citation_json(username, dataset_name, file_id)
    references_path = clean_references_json(username, dataset_name, file_id)

    if not exists(citation_path):
        logger.warning(
            "citation-agg: citation.json not found for {fid}, run extraction "
            "first", fid=file_id
        )
        return {
            "file_id": file_id,
            "file_name": file_data.get("file_name", ""),
            "total_articles": 0,
            "articles": [],
        }

    citation_data = load_json(citation_path)
    references_json = load_json(references_path)

    groups: Dict[str, Dict] = {}
    order: List[str] = []

    for entry in citation_data.get("citations", []):
        occurrence = {
            "citation_id": entry.get("citation_id"),
            "style": entry.get("style"),
            "markers": entry.get("markers", []),
            "prev_sentence": entry.get("prev_sentence", ""),
            "citation_sentence": entry.get("citation_sentence", ""),
            "next_sentence": entry.get("next_sentence", ""),
            "context": entry.get("context", ""),
        }

        for article in entry.get("cited_articles", []):
            ref_key = article.get("ref_key")
            raw_ref = article.get("raw_reference")
            title = article.get("title")

            info = match_reference(raw_ref, references_json) if raw_ref else None
            doi = info.get("doi") if info else None
            if not title and info:
                title = info.get("title")

            key = _article_group_key(doi, title, raw_ref, ref_key)
            if key not in groups:
                groups[key] = {
                    "ref_keys": [],
                    "doi": doi,
                    "title": title,
                    "raw_reference": raw_ref,
                    "citation_count": 0,
                    "citations": [],
                    "_seen": set(),
                }
                order.append(key)

            group = groups[key]
            if ref_key not in group["ref_keys"]:
                group["ref_keys"].append(ref_key)
            cid = occurrence["citation_id"]
            if cid not in group["_seen"]:
                group["_seen"].add(cid)
                group["citations"].append(occurrence)

    articles: List[Dict] = []
    for key in order:
        group = groups[key]
        group.pop("_seen", None)
        group["citation_count"] = len(group["citations"])
        articles.append(group)

    articles.sort(key=_article_sort_key)

    with_doi = sum(1 for a in articles if a["doi"])
    logger.info(
        "citation-agg: {fid} articles={n} (with_doi={wd}) from {c} citations",
        fid=file_id, n=len(articles), wd=with_doi,
        c=len(citation_data.get("citations", [])),
    )

    return {
        "file_id": file_id,
        "file_name": file_data.get("file_name", ""),
        "total_articles": len(articles),
        "articles": articles,
    }


def process_citation_aggregation(file_data: dict) -> dict:
    """
    Build and persist the article-grouped citation file (citation_by_article.json).

    Args:
        file_data (dict): File metadata.

    Returns:
        dict: The article-grouped result.
    """
    username, dataset_name = parse_path_info(file_data["file_path"])
    logger = get_user_logger(username, dataset_name)

    result = aggregate_citations(file_data)

    output_path = clean_citation_by_article_json(
        username, dataset_name, file_data["file_id"]
    )
    save_json(result, output_path, info=False)
    logger.info(
        "citation-agg: {fid} saved {n} articles to {path}",
        fid=file_data["file_id"], n=result["total_articles"],
        path=str(output_path),
    )

    return result
