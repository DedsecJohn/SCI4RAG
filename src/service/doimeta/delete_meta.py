"""
Meta management: delete DOI metadata files and reset state.
"""
import os
from src.core.paths import clean_doi_json, parse_path_info
from src.core.states import DoiStatus
from src.core.logger import get_user_logger
from src.service.document.load_document import updata_document_metadata, load_document_metadata


def delete_meta(file_data):
    """
    Delete doi.json file and reset DOI state to NOT_DOI.

    Args:
        file_data: File metadata dict conforming to FileData schema.

    Returns:
        int: 1 on success, 0 if file did not exist, -1 on error.
    """
    username, dataset_name = parse_path_info(file_data["file_path"])
    logger = get_user_logger(username, dataset_name)
    file_id = file_data["file_id"]

    doi_file = clean_doi_json(username, dataset_name, file_id)
    if doi_file.exists():
        os.remove(doi_file)
        logger.info("Deleted doi.json: {path}", path=str(doi_file))
    else:
        logger.info("No doi.json found for {name}", name=file_data['file_name'])

    metadata = load_document_metadata(username, dataset_name).get(file_id, file_data)
    metadata["DOI_state"] = DoiStatus.NOT_DOI
    metadata["doi"] = None
    metadata["bibjson"] = {}
    updata_document_metadata(username, dataset_name, metadata, info=False)

    logger.info("Reset DOI status for {name}", name=file_data['file_name'])
    return 1
