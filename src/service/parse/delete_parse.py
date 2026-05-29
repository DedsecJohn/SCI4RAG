"""
Parse management: delete parse results and reset state.
"""
import shutil
from src.core.paths import parse_dir, parse_path_info
from src.core.states import ParseStatus
from src.core.logger import get_user_logger
from src.service.document.load_document import update_document_metadata, load_document_metadata


def delete_parse(file_data):
    """
    Delete parse folder and reset parsing status to NOT_PARSED.

    Args:
        file_data: File metadata dict conforming to FileData schema.

    Returns:
        int: 1 on success, 0 if folder did not exist, -1 on error.
    """
    username, dataset_name = parse_path_info(file_data["file_path"])
    logger = get_user_logger(username, dataset_name)
    file_id = file_data["file_id"]

    # Delete the parse folder
    parse_folder = parse_dir(username, dataset_name, file_id)
    if parse_folder.exists():
        shutil.rmtree(parse_folder, ignore_errors=True)
        logger.info("Deleted parse folder: {folder}", folder=str(parse_folder))
    else:
        logger.info("No parse folder found for {name}", name=file_data['file_name'])

    # Reset parsing status
    metadata = load_document_metadata(username, dataset_name).get(file_id, file_data)
    metadata["parsing_status"] = ParseStatus.NOT_PARSED
    metadata["batch_id"] = None
    update_document_metadata(username, dataset_name, metadata, info=False)

    logger.info("Reset parsing status for {name}", name=file_data['file_name'])
    return 1
