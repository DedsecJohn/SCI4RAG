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

def delete_file(file_data: dict) -> None:
    """
    Delete a file.    

    Args:
        file_data: File metadata. See `FileData` (src/core/states.py)
    """
    file_path = Path(file_data["file_path"])
    username, dataset_name = parse_path_info(file_data["file_path"])
    file_id = file_data.get("file_id")

    # 1. delete file_data["file_path"]
    if file_path.exists():
        file_path.unlink()
    # 2. delete username/dataset_name/parse/file_data["file_id"]
    parse_folder = parse_dir(username, dataset_name, file_id)
    if parse_folder.exists():
        shutil.rmtree(parse_folder, ignore_errors=True)    
    # 3. delete username/dataset_name/data_clean/file_data["file_id"]
    clean_folder = clean_dir(username, dataset_name, file_id)
    if clean_folder.exists():
        shutil.rmtree(clean_folder, ignore_errors=True)

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