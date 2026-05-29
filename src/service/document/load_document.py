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
    parse_path_info
)
from src.core.utils import load_json, save_json
from src.core.states import FileData
from dataclasses import asdict


def load_document_metadata(username: str, dataset_name: str) -> dict:
    """
    Load the documents.json metadata file for a given user and dataset.
    """
    return load_json(documents_json(username, dataset_name))

def update_document_metadata(username: str, dataset_name: str, metadata: dict, info = True) -> None:
    """
    update the documents.json metadata file for a given user and dataset.

    Args:
        username (str): The username of the user.
        dataset_name (str): The name of the dataset.
        metadata (dict): The metadata to update.
        info (bool, optional): Whether to print info message. Defaults to True.

    Returns:
        None
    """
    json_path = documents_json(username, dataset_name)
    doc_info = load_json(json_path)
    doc_info[metadata["file_id"]] = metadata
    doc_info[metadata["file_id"]]["update_time"] = datetime.now().strftime("%a, %d %b %Y %H:%M")
    save_json(doc_info, json_path, info=info)

def load_PDF_file(username: str, dataset_name: str, file_path: str) -> dict:
    """
    Load a specific PDF file and update its metadata.

    Args:
        username (str): The username of the user.
        dataset_name (str): The name of the dataset.
        file_path (str): The path to the PDF file.

    Returns:
        dict: The metadata of the PDF file.
    """
    logger = get_user_logger(username, dataset_name)

    if not os.path.exists(file_path):
        logger.error("PDF file not found: {path}", path=file_path)
        raise FileNotFoundError(f"PDF file not found: {file_path}")

    if not file_path.lower().endswith(".pdf"):
        logger.error("File is not a PDF: {path}", path=file_path)
        raise ValueError(f"File is not a PDF: {file_path}")

    update_time = datetime.now().strftime("%a, %d %b %Y %H:%M")
    file_id = f"{datetime.now().strftime('%Y%m%dT%H%M%S')}_{secrets.token_hex(5)}"
    file_name = os.path.splitext(os.path.basename(file_path))[0]
    file_size = os.path.getsize(file_path)

    file_meta = asdict(FileData(
        file_id=file_id,
        file_name=file_name,
        file_type="pdf",
        file_path=file_path,
        file_size=file_size,
        update_time=update_time,
    ))

    json_path = documents_json(username, dataset_name)
    doc_info = load_json(json_path)

    # Check duplicate (same name + same size)
    for fid, meta in doc_info.items():
        if meta["file_name"] == file_name and meta["file_size"] == file_size:
            logger.info("File already exists, skip: {name}", name=file_name)
            return meta

    # New file or updated file
    doc_info[file_id] = file_meta
    save_json(doc_info, json_path)

    logger.info("Registered PDF: {name}", name=file_name)
    return file_meta

def register_new_pdfs(username: str, dataset_name: str) -> int:
    """
    Scan the documents directory and register new PDF files into documents.json.
    Skips files already registered (matching by name + size).
    Uses batch read/write: loads documents.json once, adds all new files, saves once.

    Args:
        username (str): The username of the user.
        dataset_name (str): The name of the dataset.

    Returns:
        int: Number of newly registered files.
    """
    logger = get_user_logger(username, dataset_name)
    pdf_dir = documents_dir(username, dataset_name)
    pdf_files = sorted(glob(str(pdf_dir / "*.pdf")))

    if not pdf_files:
        logger.info("No PDF files found in {dir}", dir=pdf_dir)
        return 0

    # Batch: load documents.json once
    json_path = documents_json(username, dataset_name)
    doc_info = load_json(json_path) if json_path.exists() else {}

    # Batch: create metadata for new files
    update_time = datetime.now().strftime("%a, %d %b %Y %H:%M")
    new_count = 0

    for pdf_path in pdf_files:
        file_name = os.path.splitext(os.path.basename(pdf_path))[0]
        file_size = os.path.getsize(pdf_path)

        # Check duplicate (same name + same size)
        is_duplicate = any(
            meta["file_name"] == file_name and meta["file_size"] == file_size
            for meta in doc_info.values()
        )
        if is_duplicate:
            continue

        file_id = (
            f"{datetime.now().strftime('%Y%m%dT%H%M%S')}_"
            f"{secrets.token_hex(5)}"
        )

        doc_info[file_id] = asdict(FileData(
            file_id=file_id,
            file_name=file_name,
            file_type="pdf",
            file_path=pdf_path,
            file_size=file_size,
            update_time=update_time,
        ))
        new_count += 1

    # Batch: save once
    if new_count > 0:
        save_json(doc_info, json_path)
        logger.info("Registered {count} new PDF(s) in {dir}", count=new_count, dir=pdf_dir)

    return new_count


if __name__ == "__main__":
    username = "administrator"
    dataset_name = "schwarz"

    print("Preparing PDF files for MinerU processing...")
    new_count = register_new_pdfs(username, dataset_name)
    print(f"Registered {new_count} new PDF(s).")

    all_files = load_document_metadata(username, dataset_name)
    print(f"Total files in dataset: {len(all_files)}")
    for k, v in all_files.items():
        print(f"{k}: {v}")
