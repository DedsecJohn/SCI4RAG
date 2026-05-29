import re
from src.core.logger import get_user_logger
from src.core.paths import parse_layout_json
from src.core.utils import load_json
from src.core.states import DoiStatus
from src.service.document.load_document import parse_path_info, update_document_metadata
from src.service.extractor.title import identify_title

DOI_PATTERN = re.compile(
    r'(?:doi\s*[:：]?\s*|https?://doi\.org/)'  # DOI prefix
    r'(10\.\d{4,9}/[^\s\]\)]+)',              # DOI number
    re.IGNORECASE
)


def find_first_doi(obj):
    """
    Recursively search for the first DOI in a nested data structure.

    Args:
        obj: A dict, list, or str to search for DOI patterns.

    Returns:
        str or None: The first DOI found, or None if no DOI is found.
    """
    if isinstance(obj, dict):
        for v in obj.values():
            doi = find_first_doi(v)
            if doi:
                return doi
    elif isinstance(obj, list):
        for item in obj:
            doi = find_first_doi(item)
            if doi:
                return doi
    elif isinstance(obj, str):
        match = DOI_PATTERN.search(obj)
        if match:
            return match.group(1)
    return None


def identify_DOI(file_data: dict) -> dict:
    """
    Extract DOI from document layout and update file metadata.

    Args:
        file_data: File metadata. See `FileData` (src/core/states.py)

    Returns:
        dict: Updated file_data with DOI_state and doi fields.
    """
    # ---- paths ----
    username, dataset_name = parse_path_info(file_data["file_path"])
    logger = get_user_logger(username, dataset_name)

    # ---- DOI_state guard ----
    if file_data["DOI_state"] not in {
        DoiStatus.NOT_DOI
    }:
        logger.info("Already {state}, skip DOI identification", state=file_data['DOI_state'])
        return file_data

    # ---- load layout.json ----
    parse_layout_path = parse_layout_json(username, dataset_name, file_data['file_id'])
    parse_layout = load_json(parse_layout_path)

    doi = find_first_doi(parse_layout)

    # ---- save ----
    file_data["doi"] = doi
    if doi:
        file_data["DOI_state"] = DoiStatus.DOI_EXTRACTED
        update_document_metadata(username, dataset_name, file_data, info=False)
    else:
        logger.info("DOI not found in layout, trying title identification...")
        file_data = identify_title(file_data)
        if not file_data.get("doi"):
            file_data["DOI_state"] = DoiStatus.NOT_DOI
            update_document_metadata(username, dataset_name, file_data)
            logger.error("DOI not found via title identification in {file_name}", file_name=file_data['file_name'])
    
    return file_data
