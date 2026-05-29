"""
MinerU parser API - Loads configuration from config.py
"""
import os
import time
import requests
import zipfile
from src.core.logger import get_user_logger
from src.core.config import config
from src.core.paths import (
    parse_dir, parse_zip, ensure_dir
)
from src.core.states import ParseStatus, CleanStatus
from src.service.document.load_document import (
    updata_document_metadata, load_document_metadata, parse_path_info
)


def get_api_token():
    """
    Get MinerU API token from config
    
    Raises:
        ValueError: If API key is missing
    """
    cfg = config.get_parser_config()
    return cfg['api_key']


# Configuration
BATCH_URL = "https://mineru.net/api/v4/file-urls/batch"
STATUS_URL = "https://mineru.net/api/v4/extract-results/batch/{batch_id}"
MODEL_VERSION = "vlm"
REQUEST_TIMEOUT = 30  # seconds
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds


def get_headers():
    """Get request headers with API token"""
    return {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {get_api_token()}"
    }


def mineru_parse(file_data):
    """
    Upload file to MinerU for parsing.
    
    Args:
        file_data: Dictionary containing file metadata
        
    Returns:
        int: 0 if already parsed/in progress, 1 if upload successful, -1 if failed
    """
    username, dataset_name = parse_path_info(file_data["file_path"])
    logger = get_user_logger(username, dataset_name)
    
    # Check if already parsed or in progress
    if file_data["parsing_status"] not in (ParseStatus.NOT_PARSED, ParseStatus.FAILED):
        logger.info(
            "File already parsed or in progress: {name}, status: {status}",
            name=file_data['file_name'],
            status=file_data['parsing_status']
        )
        return 0
    
    data = {
        "files": [{
            "name": file_data["file_name"],
            "data_id": file_data["file_id"],
        }],
        "model_version": MODEL_VERSION
    }
    file_path = [file_data['file_path']]
    
    try:
        # Request upload URL with retry
        for attempt in range(MAX_RETRIES):
            try:
                response = requests.post(
                    BATCH_URL, 
                    headers=get_headers(), 
                    json=data,
                    timeout=REQUEST_TIMEOUT
                )
                response.raise_for_status()
                break
            except requests.RequestException as e:
                if attempt < MAX_RETRIES - 1:
                    logger.warning(
                        "Upload request failed (attempt {attempt}/{max}), retrying in {delay}s: {error}",
                        attempt=attempt + 1,
                        max=MAX_RETRIES,
                        delay=RETRY_DELAY,
                        error=str(e)
                    )
                    time.sleep(RETRY_DELAY)
                else:
                    raise
        
        result = response.json()
        logger.info("Upload request successful for: {name}", name=file_data['file_name'])
        
        if result["code"] == 0:
            batch_id = result["data"]["batch_id"]
            urls = result["data"]["file_urls"]
            
            # Upload file to presigned URL
            for i in range(len(urls)):
                with open(file_path[i], 'rb') as f:
                    res_upload = requests.put(urls[i], data=f, timeout=REQUEST_TIMEOUT)
                    
                    if res_upload.status_code == 200:
                        logger.info("File uploaded successfully: {name}", name=file_data['file_name'])
                        
                        # Update metadata using helper function
                        metadata = file_data.copy()
                        metadata["parsing_status"] = ParseStatus.PROCESSING
                        metadata["batch_id"] = batch_id
                        updata_document_metadata(username, dataset_name, metadata, info=False)
                        return 1
                    else:
                        logger.error(
                            "File upload failed: {name}, status: {status}",
                            name=file_data['file_name'],
                            status=res_upload.status_code
                        )
                        return -1
        else:
            logger.error(
                "Apply upload URL failed: {reason}",
                reason=result.get('msg', 'Unknown error')
            )
            return -1
            
    except requests.RequestException as e:
        logger.exception("Network error during upload: {name}", name=file_data['file_name'])
        return -1
    except Exception as e:
        logger.exception("Unexpected error during upload: {name}", name=file_data['file_name'])
        return -1

def mineru_state(file_data):
    """
    Check MinerU parsing status, download, unzip, and clean up zip file.
    
    Args:
        file_data: Dictionary containing file metadata
        
    Returns:
        int: 0 if already downloaded, 1 if download successful, -1 if failed
    """
    username, dataset_name = parse_path_info(file_data["file_path"])
    logger = get_user_logger(username, dataset_name)
    
    # Check if already downloaded
    if file_data.get("parsing_status") == ParseStatus.DOWNLOADED:
        logger.info(
            "File already parsed and downloaded: {name}",
            name=file_data['file_name']
        )
        return 1
    
    time.sleep(0.5)
    batch_id = file_data.get('batch_id')
    if not batch_id:
        logger.warning(
            "No batch_id found for {name}, cannot check status",
            name=file_data['file_name']
        )
        return -1

    url = STATUS_URL.format(batch_id=batch_id)
    
    try:
        # Check status with retry
        for attempt in range(MAX_RETRIES):
            try:
                res = requests.get(url, headers=get_headers(), timeout=REQUEST_TIMEOUT)
                res.raise_for_status()
                break
            except requests.RequestException as e:
                if attempt < MAX_RETRIES - 1:
                    logger.warning(
                        "Status check failed (attempt {attempt}/{max}), retrying in {delay}s: {error}",
                        attempt=attempt + 1,
                        max=MAX_RETRIES,
                        delay=RETRY_DELAY,
                        error=str(e)
                    )
                    time.sleep(RETRY_DELAY)
                else:
                    raise
        
        data = res.json()["data"]['extract_result'][0]
        status = data['state']
        
        logger.info(
            "File parsing status: {status} for {name}",
            status=status,
            name=file_data['file_name']
        )

        # Update metadata with current status
        metadata = file_data.copy()
        metadata["parsing_status"] = ParseStatus.PROCESSING

        # Only download if done and full_zip_url exists
        if status == "done" and data.get('full_zip_url'):
            zip_url = data['full_zip_url']

            # Ensure download folder exists using paths module
            download_dir = parse_dir(username, dataset_name, file_data["file_id"])
            ensure_dir(download_dir)

            # Save zip file
            base_name = os.path.splitext(file_data["file_name"])[0]
            zip_path = parse_zip(username, dataset_name, file_data["file_id"], base_name)

            # Download with retry
            for attempt in range(MAX_RETRIES):
                try:
                    response = requests.get(zip_url, timeout=REQUEST_TIMEOUT)
                    response.raise_for_status()
                    
                    with open(zip_path, 'wb') as f:
                        f.write(response.content)
                    logger.info(
                        "Downloaded zip for {name} to {path}",
                        name=file_data['file_name'],
                        path=str(zip_path)
                    )
                    break
                except requests.RequestException as e:
                    if attempt < MAX_RETRIES - 1:
                        logger.warning(
                            "Download failed (attempt {attempt}/{max}), retrying in {delay}s: {error}",
                            attempt=attempt + 1,
                            max=MAX_RETRIES,
                            delay=RETRY_DELAY,
                            error=str(e)
                        )
                        time.sleep(RETRY_DELAY)
                    else:
                        logger.error(
                            "Failed to download zip for {name} after {max} attempts",
                            name=file_data['file_name'],
                            max=MAX_RETRIES
                        )
                        return -1

            # Extract the zip
            try:
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(download_dir)
                logger.info(
                    "Extracted {name} into {dir}",
                    name=file_data['file_name'],
                    dir=str(download_dir)
                )

                # Delete the zip file
                os.remove(zip_path)
                logger.info("Deleted zip file: {path}", path=str(zip_path))

                # Update status to Download
                metadata["parsing_status"] = ParseStatus.DOWNLOADED
                metadata["clean_state"] = CleanStatus.PARSED
                updata_document_metadata(username, dataset_name, metadata, info=False)
                return 1
                
            except zipfile.BadZipFile:
                logger.error(
                    "Corrupted zip file for {name}, deleting",
                    name=file_data['file_name']
                )
                if os.path.exists(zip_path):
                    os.remove(zip_path)
                return -1
            except Exception as e:
                logger.exception(
                    "Error extracting zip for {name}",
                    name=file_data['file_name']
                )
                return -1
        else:
            # Status not done yet, just update metadata
            updata_document_metadata(username, dataset_name, metadata)
            return 0

    except requests.RequestException:
        logger.exception(
            "Network error while checking status for {name}",
            name=file_data['file_name']
        )
        return -1
    except Exception:
        logger.exception(
            "Unexpected error while processing {name}",
            name=file_data['file_name']
        )
        return -1


def parse_doc(file_data, poll_interval=15, poll_timeout=3600):
    """
    Upload, poll, and download a single PDF from MinerU in one call.

    Polls the MinerU server repeatedly until parsing completes or fails.
    Blocks until done, making it safe to submit to a ThreadPoolExecutor
    for parallel or background execution.

    Args:
        file_data: File metadata dict conforming to FileData schema.
        poll_interval: Seconds between polling attempts (default 30).
        poll_timeout: Max total seconds to keep polling (default 3600 = 1hr).

    Returns:
        int: 1 on success, 0 if already processed, -1 on failure or timeout.
    """
    username, dataset_name = parse_path_info(file_data["file_path"])
    logger = get_user_logger(username, dataset_name)
    logger.info("Start parsing: {name}", name=file_data['file_name'])

    # Step 1: Upload — mineru_parse handles state check internally
    ret = mineru_parse(file_data)
    if ret == -1:
        return -1

    # Step 2: Multi-round polling
    max_rounds = poll_timeout // poll_interval
    for _round in range(max_rounds):
        time.sleep(poll_interval)

        fresh = load_document_metadata(username, dataset_name)
        current = fresh.get(file_data["file_id"], file_data)

        ret = mineru_state(current)
        if ret == 1:
            return 1
        if ret == -1:
            return -1

    logger.error("Polling timeout for {name}", name=file_data['file_name'])
    return -1
