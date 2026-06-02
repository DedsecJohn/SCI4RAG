"""
Fetch article metadata from CrossRef and Semantic Scholar APIs.
Includes: title, authors, journal, year, volume, pages, abstract, references.

Example DOI: 10.1103/PhysRevLett.127.136101
"""

import re
import time
import random
import requests
from pathlib import Path
from typing import Optional
from src.core.logger import get_logger, get_user_logger
from src.core.paths import clean_doi_json
from src.core.utils import load_json, save_json
from src.core.states import DoiStatus
from src.service.document.load_document import parse_path_info, update_document_metadata
from src.service.document.delete_document import delete_document


# ──────────────────────────────────────────────
# 0. HTTP Retry Helper
# ──────────────────────────────────────────────

def _request_with_retry(url, headers=None, params=None, timeout=15, max_retries=3):
    """
    GET request with exponential backoff for transient errors.
    Retries on 429 (rate limit), 503 (unavailable), connection errors, timeouts.
    Does NOT retry on 404 (invalid resource).

    Args:
        url: Request URL.
        headers: Request headers.
        params: Query parameters.
        timeout: Request timeout in seconds.
        max_retries: Maximum number of retry attempts.

    Returns:
        Response or None: Response object on success, None after all retries exhausted.
    """
    for attempt in range(max_retries):
        try:
            r = requests.get(url, headers=headers, params=params, timeout=timeout)
            if r.status_code == 404:
                return r
            r.raise_for_status()
            return r
        except requests.exceptions.HTTPError as e:
            status = e.response.status_code
            if status in {429, 503} and attempt < max_retries - 1:
                wait = 2 ** attempt + random.uniform(0, 1)
                get_logger().warning("HTTP {status} for {url}, retry {n}/{max} in {wait:.1f}s",
                                     status=status, url=url, n=attempt+1, max=max_retries, wait=wait)
                time.sleep(wait)
                continue
            raise
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            if attempt < max_retries - 1:
                wait = 2 ** attempt + random.uniform(0, 1)
                get_logger().warning("{error} for {url}, retry {n}/{max} in {wait:.1f}s",
                                     error=type(e).__name__, url=url, n=attempt+1, max=max_retries, wait=wait)
                time.sleep(wait)
                continue
            raise
    return None


# ──────────────────────────────────────────────
# 1. CrossRef API
# ──────────────────────────────────────────────

def fetch_article_info_by_doi(doi: str) -> Optional[dict]:
    """
    Fetch article metadata from Crossref using DOI with retry support.

    Args:
        doi: The DOI of the article.

    Returns:
        dict or None: Article metadata from CrossRef, or None if request fails.
    """
    url = f"https://api.crossref.org/works/{doi}"
    headers = {
        "User-Agent": "SCI4RAG/1.0 (mailto:sci_email@example.com)"
    }
    try:
        r = _request_with_retry(url, headers=headers, timeout=15)
        if r is None:
            return None
        if r.status_code == 404:
            get_logger().warning("DOI not found: {doi}", doi=doi)
            return None
        return r.json()["message"]
    except Exception as e:
        get_logger().warning("Error fetching DOI {doi}: {error}", doi=doi, error=str(e))
        return None


def query_crossref(reference: str) -> Optional[dict]:
    """
    Use Crossref to search metadata using a reference string with retry support.

    Args:
        reference: Reference string to search for.

    Returns:
        dict or None: Metadata dictionary if found, None otherwise.
    """
    url = "https://api.crossref.org/works"
    params = {
        "query.bibliographic": reference,
        "rows": 1
    }
    try:
        r = _request_with_retry(url, params=params, timeout=10)
        if r is None or r.status_code != 200:
            return None
        items = r.json().get("message", {}).get("items", [])
        if items:
            return items[0]
    except Exception as e:
        get_logger().warning("Error querying Crossref: {error}", error=str(e))
    return None


# ──────────────────────────────────────────────
# 2. Semantic Scholar API
# ──────────────────────────────────────────────

def fetch_abstract_from_semantic_scholar(doi: str) -> Optional[str]:
    """
    Fetch abstract from Semantic Scholar API.
    CrossRef abstract coverage is limited; Semantic Scholar is better.

    Args:
        doi: The DOI of the article.

    Returns:
        str or None: Abstract text if found, None otherwise.
    """
    url = f"https://api.semanticscholar.org/graph/v1/paper/DOI:{doi}"
    params = {"fields": "abstract"}
    try:
        r = requests.get(url, params=params, timeout=15)
        if r.status_code == 200:
            return r.json().get("abstract")
    except Exception as e:
        get_logger().warning("Error querying Semantic Scholar: {error}", error=str(e))
    return None


def fetch_references_from_semantic_scholar(doi: str) -> Optional[list]:
    """
    Fetch references with title/authors/DOI from Semantic Scholar.
    Useful when CrossRef reference entries lack titles.

    Args:
        doi: The DOI of the article.

    Returns:
        list or None: List of reference dictionaries if found, None otherwise.
    """
    url = f"https://api.semanticscholar.org/graph/v1/paper/DOI:{doi}"
    params = {
        "fields": "references,references.title,references.authors,references.externalIds,references.year,references.venue"
    }
    try:
        r = requests.get(url, params=params, timeout=15)
        if r.status_code == 200:
            return r.json().get("references")
    except Exception as e:
        get_logger().warning("Error querying Semantic Scholar references: {error}", error=str(e))
    return None


# ──────────────────────────────────────────────
# 3. Field Extraction Helpers
# ──────────────────────────────────────────────

def extract_author(data: dict) -> list:
    """
    Extract author list from CrossRef metadata.

    Args:
        data: CrossRef metadata dictionary.

    Returns:
        list: List of author names in "Given Family" format.
    """
    authors = []
    for a in data.get("author", []):
        family = a.get("family", "").strip()
        given = a.get("given", "").strip()
        if family and given:
            authors.append(f"{given} {family}")
        elif family:
            authors.append(family)
        elif given:
            authors.append(given)
    return authors


def extract_journal(data: dict) -> Optional[str]:
    """
    Extract journal name from CrossRef metadata.

    Args:
        data: CrossRef metadata dictionary.

    Returns:
        str or None: Journal name if found.
    """
    return (
        (data.get("container-title") or [None])[0]
        or (data.get("short-container-title") or [None])[0]
        or data.get("publisher")
    )


def extract_year(data: dict) -> int | str:
    """
    Extract publication year from CrossRef metadata.

    Args:
        data: CrossRef metadata dictionary.

    Returns:
        int or str: Publication year, or "XXXX" if not found.
    """
    for key in [
        "published-print",
        "published",
        "issued",
        "published-online",
    ]:
        parts = data.get(key, {}).get("date-parts")
        if parts and parts[0] and parts[0][0]:
            return parts[0][0]
    return "XXXX"


def extract_volume(data: dict) -> Optional[str]:
    """
    Extract volume number from CrossRef metadata.

    Args:
        data: CrossRef metadata dictionary.

    Returns:
        str or None: Volume number if found.
    """
    return data.get("volume")


def extract_issue(data: dict) -> Optional[str]:
    """
    Extract issue number from CrossRef metadata.

    Args:
        data: CrossRef metadata dictionary.

    Returns:
        str or None: Issue number if found.
    """
    return (
        data.get("issue")
        or data.get("journal-issue", {}).get("issue")
    )


def extract_pages(data: dict) -> Optional[str]:
    """
    Extract page numbers from CrossRef metadata.

    Args:
        data: CrossRef metadata dictionary.

    Returns:
        str or None: Page numbers if found.
    """
    return (
        data.get("page")
        or data.get("article-number")
        or data.get("eLocator")
    )


def extract_abstract(data: dict, doi: str) -> Optional[str]:
    """
    Try to get abstract from CrossRef first; fall back to Semantic Scholar.
    CrossRef abstracts may contain XML tags, so we strip them.

    Args:
        data: CrossRef metadata dictionary.
        doi: The DOI of the article.

    Returns:
        str or None: Abstract text if found.
    """
    abstract = data.get("abstract")
    if abstract:
        # CrossRef abstracts often contain JATS XML tags, strip them
        abstract = re.sub(r"<[^>]+>", "", abstract).strip()
        return abstract

    # Fallback: Semantic Scholar
    # get_logger().info("Abstract not found in CrossRef, trying Semantic Scholar for DOI: {doi}", doi=doi)
    return fetch_abstract_from_semantic_scholar(doi)


def extract_references_from_crossref(data: dict) -> list:
    """
    Extract reference list from CrossRef metadata.
    Each reference may contain: DOI, unstructured text, author, year, volume-title, journal-title, etc.

    Note: This function is preserved for future use but not currently used in the main pipeline.
    References are extracted from the document itself via extractor/reference.py for accuracy.

    Args:
        data: CrossRef metadata dictionary.

    Returns:
        list: List of reference dictionaries.
    """
    refs = []
    for ref in data.get("reference", []):
        entry = {
            "key": ref.get("key"),
            "doi": ref.get("DOI"),
            "unstructured": ref.get("unstructured"),
            "author": ref.get("author"),
            "year": ref.get("year"),
            "volume_title": ref.get("volume-title"),
            "journal_title": ref.get("journal-title"),
            "first_page": ref.get("first-page"),
            "volume": ref.get("volume"),
        }
        # Remove None values for cleaner output
        entry = {k: v for k, v in entry.items() if v is not None}
        refs.append(entry)
    return refs


# ──────────────────────────────────────────────
# 4. Citation Key Generation
# ──────────────────────────────────────────────

def make_citation_key(data: dict) -> str:
    """
    Generate a citation key for a given article metadata.
    Format: FirstAuthorLastNameYearFirstTitleWord (e.g., Smith2021Quantum)

    Args:
        data: CrossRef metadata dictionary.

    Returns:
        str: Citation key.
    """
    authors = data.get("author", [])
    if not authors:
        first_author = "Unknown"
    else:
        first_author_data = authors[0]
        if "family" in first_author_data:
            first_author = first_author_data["family"]
        elif "name" in first_author_data:
            first_author = first_author_data["name"]
        else:
            first_author = "Unknown"
    first_author = re.sub(r'[^a-zA-Z0-9]', '', first_author)

    year = extract_year(data)

    title_list = data.get("title", [""])
    title = title_list[0] if title_list else ""
    words = title.split()
    keyword = re.sub(r"\W+", "", words[0]) if words else "NoTitle"
    return f"{first_author}{year}{keyword}"


# ──────────────────────────────────────────────
# 5. Unified Output
# ──────────────────────────────────────────────

def to_custom_bibjson(data: dict, doi: str) -> dict:
    """
    Convert article metadata to a custom BibJSON format.
    Includes abstract but not references (references are extracted from document).

    Args:
        data: CrossRef metadata dictionary.
        doi: The DOI of the article.

    Returns:
        dict: Custom BibJSON format with title, author, journal, year, volume, pages, abstract, etc.
    """
    return {
        "title": (data.get("title") or [None])[0],
        "author": extract_author(data),
        "journal": extract_journal(data),
        "doi": data.get("DOI"),
        "url": data.get("URL"),
        "year": extract_year(data),
        "volume": extract_volume(data),
        "number": extract_issue(data),
        "pages": extract_pages(data),
        "bibkey": make_citation_key(data),
        "abstract": extract_abstract(data, doi),
    }


# ──────────────────────────────────────────────
# 6. Public Interface
# ──────────────────────────────────────────────

def get_doi_info(doi: str) -> Optional[dict]:
    """
    Get article metadata from DOI.
    Includes abstract fetched from CrossRef or Semantic Scholar.

    Args:
        doi: The DOI of the article.

    Returns:
        dict or None: Article metadata in custom BibJSON format, or None if fetch fails.
    """
    time.sleep(0.5)  # Rate limiting to avoid hitting API limits
    raw = fetch_article_info_by_doi(doi)
    if raw is None:
        return None
    return to_custom_bibjson(raw, doi)


def get_title_info(title: str) -> Optional[dict]:
    """
    Get article metadata from title string.

    Args:
        title: Title string to search for.

    Returns:
        dict or None: Article metadata in custom BibJSON format, or None if not found.
    """
    time.sleep(0.5)  # Rate limiting to avoid hitting API limits
    raw = query_crossref(title)
    if raw is None:
        get_logger().warning("Title not found in CrossRef: {title}", title=title[:100])
        return None
    
    doi = raw.get("DOI")
    if not doi:
        get_logger().warning("No DOI found in CrossRef result for title: {title}", title=title[:100])
        return None
    
    bibjson = to_custom_bibjson(raw, doi)
    return bibjson


def get_reference_info(title: str) -> Optional[dict]:
    """
    Get article metadata from title string (without abstract).
    Uses CrossRef only; does NOT call Semantic Scholar for abstract.

    Args:
        title (str): Title string to search for.

    Returns:
        Optional[dict]: Article metadata (title, author, journal, doi, url,
            year, volume, number, pages, bibkey) or None if not found.
    """
    time.sleep(0.5)
    raw = query_crossref(title)
    if raw is None:
        get_logger().warning("Title not found in CrossRef: {title}", title=title[:100])
        return None

    doi = raw.get("DOI")
    if not doi:
        get_logger().warning("No DOI found in CrossRef result for title: {title}", title=title[:100])
        return None

    return {
        "title": (raw.get("title") or [None])[0],
        "author": extract_author(raw),
        "journal": extract_journal(raw),
        "doi": raw.get("DOI"),
        "url": raw.get("URL"),
        "year": extract_year(raw),
        "volume": extract_volume(raw),
        "number": extract_issue(raw),
        "pages": extract_pages(raw),
        "bibkey": make_citation_key(raw),
    }


def fetch_DOI_metadata(file_data: dict) -> dict:
    """
    Ensure DOI is resolved and metadata is saved.
    If DOI not in file_data, extract from layout first.
    If DOI is invalid, fallback to title identification.
    Then fetch metadata from Crossref and save as JSON.

    Args:
        file_data: File metadata. See `FileData` (src/core/states.py)

    Returns:
        dict: Updated file_data with DOI_state and doi fields.
    """
    username, dataset_name = parse_path_info(file_data["file_path"])
    logger = get_user_logger(username, dataset_name)

    # ---- Guard: already processed ----
    if file_data["DOI_state"] in {
        DoiStatus.METADATA_FETCHED,
        DoiStatus.UPDATED
    }:
        logger.info("Already {state}, skip metadata fetch", state=file_data['DOI_state'])
        return file_data

    # ---- Ensure DOI exists ----
    if not file_data.get("doi"):
        from src.service.extractor.doi import identify_DOI
        file_data = identify_DOI(file_data)

    if not file_data.get("doi") or is_empty_doi(file_data["doi"]):
        return file_data

    # ---- Fetch and save metadata ----
    doi_path = clean_doi_json(username, dataset_name, file_data['file_id'])

    bibjson = get_doi_info(file_data["doi"])

    # ---- Fallback: invalid DOI → try title identification ----
    if bibjson is None:
        logger.info("DOI {doi} invalid for {file_name}, trying title identification...",
                    doi=file_data['doi'], file_name=file_data['file_name'])
        from src.service.extractor.title import identify_title
        file_data = identify_title(file_data)

        if not file_data.get("doi") or is_empty_doi(file_data["doi"]):
            file_data["DOI_state"] = DoiStatus.NOT_DOI
            update_document_metadata(username, dataset_name, file_data, info=False)
            logger.error("DOI not found via title identification in {file_name}",
                            file_name=file_data['file_name'])
            return file_data

        bibjson = get_doi_info(file_data["doi"])
        if bibjson is None:
            file_data["DOI_state"] = DoiStatus.NOT_DOI
            update_document_metadata(username, dataset_name, file_data, info=False)
            logger.error("Title-identified DOI also invalid: {doi}",
                            doi=file_data['doi'])
            return file_data

    save_json(bibjson, doi_path)
    file_data["DOI_state"] = DoiStatus.METADATA_FETCHED
    update_document_metadata(username, dataset_name, file_data, info=False)

    return file_data


# ──────────────────────────────────────────────
# 7. Utilities
# ──────────────────────────────────────────────

def is_empty_doi(doi) -> bool:
    """
    Check if DOI is empty or invalid.

    Args:
        doi: DOI value to check.

    Returns:
        bool: True if DOI is empty/invalid, False otherwise.
    """
    return doi in (None, "", "null", "Null", "NULL")

# ──────────────────────────────────────────────
# 8. update_doc_info
# ──────────────────────────────────────────────

def update_doc_info(file_data: dict) -> None:
    """
    Rename the PDF file based on the title from doi.json and update metadata.

    Operates only on the given file_data; does not touch other documents.

    Args:
        file_data: File metadata. See `FileData` (src/core/states.py)

    Returns:
        None
    """
    username, dataset_name = parse_path_info(file_data["file_path"])
    logger = get_user_logger(username, dataset_name)

    # ---- guard: only METADATA_FETCHED files ----
    if file_data.get("DOI_state") != DoiStatus.METADATA_FETCHED:
        return

    # ---- load title from doi.json ----
    doi_path = clean_doi_json(username, dataset_name, file_data["file_id"])
    doi_data = load_json(doi_path)
    raw_title = doi_data.get("title")
    if not raw_title:
        logger.warning("No title in doi.json for {name}", name=file_data["file_name"])
        return

    # ---- sanitise title for use as filename ----
    title = re.sub(r'[<>:"/\\|?*]', '_', raw_title)
    title = re.sub(r'\s+', ' ', title).strip()[:200]

    # ---- rename PDF ----
    old_pdf = Path(file_data["file_path"])
    if not old_pdf.exists():
        logger.warning("PDF not found: {path}", path=old_pdf)
        return

    new_pdf = old_pdf.with_stem(title)
    if old_pdf == new_pdf:
        file_data["DOI_state"] = DoiStatus.UPDATED
        update_document_metadata(username, dataset_name, file_data, info=False)
        return

    if new_pdf.exists():
        logger.warning("Duplicate title, removing file: {path}", path=old_pdf)
        delete_document(file_data)
        return

    try:
        old_pdf.rename(new_pdf)
    except OSError as e:
        logger.error("Rename failed: {error}", error=str(e))
        return

    # ---- update metadata ----
    file_data["file_name"] = title
    file_data["file_path"] = str(new_pdf)
    file_data["DOI_state"] = DoiStatus.UPDATED
    update_document_metadata(username, dataset_name, file_data, info=False)

    logger.info("Renamed PDF: {old} → {new}", old=old_pdf.name, new=new_pdf.name)
