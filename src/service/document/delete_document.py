import os
import json
import secrets
import shutil
from pathlib import Path
from glob import glob
from datetime import datetime
from src.core.logger import get_logger, get_user_logger
from src.core.paths import (
    documents_json, documents_dir, parse_dir, clean_dir,
    parse_path_info, ensure_parent_dir
)
from src.core.utils import load_json, save_json
from src.service.document.load_document import load_document_metadata

def delete_document(file_data: dict) -> None:
    """
    Fully delete a document: PDF, parse dir, clean dir, and metadata entry.

    Args:
        file_data: File metadata. See `FileData` (src/core/states.py)

    Returns:
        None
    """
    username, dataset_name = parse_path_info(file_data["file_path"])
    file_id = file_data.get("file_id")
    logger = get_user_logger(username, dataset_name)

    # 1. delete physical files: PDF, parse dir, clean dir
    file_path = Path(file_data["file_path"])
    if file_path.exists():
        file_path.unlink()

    parse_folder = parse_dir(username, dataset_name, file_id)
    if parse_folder.exists():
        shutil.rmtree(parse_folder, ignore_errors=True)

    clean_folder = clean_dir(username, dataset_name, file_id)
    if clean_folder.exists():
        shutil.rmtree(clean_folder, ignore_errors=True)

    # 2. remove metadata entry from documents.json
    json_path = documents_json(username, dataset_name)
    all_meta = load_json(json_path)
    all_meta.pop(file_id, None)
    save_json(all_meta, json_path, info=False)

    logger.info("Deleted document: {name} ({id})", name=file_data["file_name"], id=file_id)

def delete_none_dir(username: str, dataset_name: str) -> None:
    """
    Delete directories that do not have corresponding metadata.

    Args:
        username (str): The username of the user.
        dataset_name (str): The name of the dataset.
    """
    data_info = load_document_metadata(username, dataset_name)
    logger = get_user_logger(username, dataset_name)

    # ---- parse ----
    parse_base = parse_dir(username, dataset_name, "")
    parse_base = parse_base.parent  # Get the parse directory itself
    if parse_base.exists():
        for folder in parse_base.iterdir():
            if folder.is_dir() and folder.name not in data_info:
                logger.warning("Orphan parse folder: {folder}", folder=folder)
                shutil.rmtree(folder, ignore_errors=True)

    # ---- data_clean ----
    clean_base = clean_dir(username, dataset_name, "")
    clean_base = clean_base.parent  # Get the data_clean directory itself
    if clean_base.exists():
        for folder in clean_base.iterdir():
            if folder.is_dir() and folder.name not in data_info:
                logger.warning("Orphan clean folder: {folder}", folder=folder)
                shutil.rmtree(folder, ignore_errors=True)