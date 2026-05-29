import re
from pathlib import Path
from src.core.paths import *
from src.core.utils import load_json, save_json
from src.llm.chat.response import llm_response
from src.service.document.load_document import parse_path_info, update_document_metadata

FIG_PATTERN = re.compile(
    r'^(fig|figure)\s*\.?\s*\d+\b',
    re.IGNORECASE
)
IMG_PATTERN = re.compile(r'!\[\]\((.*?)\)')


def identify_figure(chunk: str, prev_chunk: str = None, next_chunk: str = None) -> bool:
    """
    Check if a markdown chunk is a figure or figure caption.

    Uses content patterns and positional context (previous/next chunk)
    to determine if the chunk is a figure image or caption.

    Args:
        chunk (str): Markdown chunk text.
        prev_chunk (str): Previous chunk content for positional context.
        next_chunk (str): Next chunk content for positional context.

    Returns:
        bool: True if the chunk is identified as a figure.
    """
    # Pure image link
    if re.match(r'^\s*!\[\]\(images/[^)]+\)\s*$', chunk):
        return True

    # Image + text in same chunk
    if '![](images/' in chunk:
        if len(chunk) < 150:
            return True
        if re.search(r'(?i)(fig\.|figure)\s+\d+', chunk):
            return True

    # Previous chunk is image + current starts with Fig. N. format
    if prev_chunk and '![](images/' in prev_chunk:
        if re.match(r'(?i)^\s*(fig\.|figure)\s+\d+[\.:]\s+', chunk) and len(chunk) < 1000:
            return True
        if re.match(r'(?i)^\s*(fig\.|figure)\s+\d+[\.:]\s+', chunk):
            has_subfigs = len(re.findall(r'\([a-hA-H]\)', chunk)) >= 2
            if has_subfigs:
                return True
        if re.match(r'^\s*[\(（][a-zA-Z][\)）]\s*', chunk):
            return True

    return False

def parse_figure_block(content: str, base_dir: str) -> list[dict]:
    """
    Parse a block of text containing figure information.

    Args:
        content (str): The text block containing figure information.
        base_dir (str): The base directory for image paths.

    Returns:
        list[dict]: A list of dictionaries, each representing a figure.
            Each dictionary contains the following keys:
                - "figure": A list of image paths.
                - "description": The description of the figure.
    """

    results = []

    pending_images = []
    current_caption = ""

    for line in content.splitlines():
        line = line.strip()
        if not line:
            continue

        # image → accumulate
        img_match = IMG_PATTERN.search(line)
        if img_match:
            pending_images.append(f"{base_dir}/{img_match.group(1)}")
            continue

        # caption → bind images to THIS caption
        if FIG_PATTERN.match(line):
            if pending_images:
                results.append({
                    "figure": pending_images,
                    "description": line.strip()
                })
                pending_images = []
                current_caption = line.strip()
            continue

        # caption continuation
        if current_caption:
            results[-1]["description"] += " " + line

    # images without caption
    if pending_images:
        results.append({
            "figure": pending_images,
            "description": ""
        })

    return results


def process_figures(file_data: dict, reidentify = False)->dict:
    """
    Process figures for a given file.

    Args:
        file_data (dict): The metadata of the file.

    Returns:
        dict: The processed figures.
    """ 
    if  reidentify == False:
        if "figure_state" not in file_data:
            file_data['figure_state']='None'

        elif file_data["figure_state"] == "done" or file_data["figure_state"] == "Not_Found":
            return file_data

    username, dataset_name = parse_path_info(file_data["file_path"])

    base_path = f"users/{username}/{dataset_name}/parse/{file_data['file_id']}"

    label_path = clean_label_cleaned_json(username, dataset_name, file_data['file_id'])

    figures_path = clean_figures_json(username, dataset_name, file_data['file_id'])

    label_structure = load_json(label_path)

    figures_pairs = []  
    for chunk in label_structure:
        if chunk.get("category") == "figure_url":
            content = chunk.get("content", "")
            if content.strip():
                figures = parse_figure_block(content, base_path)
                figures_pairs.extend(figures)

    if figures_pairs:
        file_data["figure_state"] = 'done' 
        update_document_metadata(username, dataset_name, file_data, info=False)
        save_json(figures_pairs, figures_path)
    else:
        file_data["figure_state"] = 'Not_Found'
        update_document_metadata(username, dataset_name, file_data, info=False)
    return file_data
    
                