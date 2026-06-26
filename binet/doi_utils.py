"""
DOI normalization and validation utilities (FR-3.2).

Normalization rules (MUST be unified across the whole project):
    - strip the ``https://doi.org/`` / ``http://dx.doi.org/`` / ``doi:`` prefix
    - lowercase
    - strip leading/trailing whitespace
"""

import re
from typing import Optional

# A DOI always starts with ``10.`` followed by a registrant code and a suffix.
_DOI_CORE_RE = re.compile(r"10\.\d{4,9}/\S+", re.IGNORECASE)

# Prefixes that may wrap a bare DOI.
_PREFIX_RE = re.compile(
    r"^\s*(?:https?://(?:dx\.)?doi\.org/|doi:\s*)",
    re.IGNORECASE,
)


def normalize_doi(raw: Optional[str]) -> Optional[str]:
    """
    Normalize a DOI string to its canonical bare lowercase form.

    Args:
        raw: A DOI in any common form (bare, URL, or ``doi:`` prefixed).

    Returns:
        The normalized DOI, or None if the input is empty / not a DOI.

    Examples:
        >>> normalize_doi("https://doi.org/10.1234/ABC")
        '10.1234/abc'
        >>> normalize_doi("  DOI: 10.1/X  ")
        '10.1/x'
        >>> normalize_doi(None) is None
        True
    """
    if not raw or not isinstance(raw, str):
        return None

    text = raw.strip()
    # Remove URL / doi: prefix (possibly repeated).
    text = _PREFIX_RE.sub("", text).strip()
    text = text.lower()

    # Drop a trailing URL fragment / query that some sources append.
    # (Keep it conservative: only strip whitespace already handled above.)
    if not text.startswith("10."):
        # Maybe the prefix removal failed because of an unusual wrapper;
        # try to locate the DOI core anywhere in the string.
        match = _DOI_CORE_RE.search(text)
        if not match:
            return None
        text = match.group(0)

    return text or None


def is_valid_doi(doi: Optional[str]) -> bool:
    """
    Check whether a (preferably normalized) string looks like a valid DOI.

    Args:
        doi: Candidate DOI string.

    Returns:
        True if it matches the DOI pattern, False otherwise.
    """
    if not doi or not isinstance(doi, str):
        return False
    return bool(re.fullmatch(r"10\.\d{4,9}/\S+", doi.strip().lower()))


def doi_to_url(doi: str) -> str:
    """
    Render a normalized DOI as a ``https://doi.org/`` URL.

    Args:
        doi: A normalized (bare) DOI.

    Returns:
        The DOI URL form.
    """
    return f"https://doi.org/{doi}"
