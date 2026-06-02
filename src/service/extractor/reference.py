import re
from typing import List, Dict
from tqdm import tqdm
from pathlib import Path
from src.core.paths import *
from src.core.utils import load_json, save_json
from src.service.doimeta.fetcher import get_reference_info
from src.service.document.load_document import parse_path_info, update_document_metadata


def split_references(reference_text: str, characters: int = 20) -> List[str]:
    """
    Split a reference block into individual references.
    Supports numbering styles:
        [1] ... , (1) ... , 1. ...

    Only keeps references with at least `characters` characters.

    Args:
        reference_text (str): Raw reference text.
        characters (int): Minimum characters to keep a reference.

    Returns:
        List[str]: Clean reference content strings.
    """
    if not reference_text:
        return []

    reference_text = reference_text.strip()

    # Remove headers like:
    # "REFERENCES", "# REFERENCES", "References and Notes"
    reference_text = re.sub(
        r'^\s*#?\s*references?(?:\s+and\s+notes)?\s*',
        '',
        reference_text,
        flags=re.IGNORECASE
    )

    # Normalize superscript numbering: $^{1}$ → [1]
    reference_text = re.sub(r'\$\^\{(\d+)\}\$', r'[\1]', reference_text)

    # Remove inline LaTeX math fragments: $...$
    reference_text = re.sub(r'\$.*?\$', '', reference_text)

    # Ensure numbering starts on new line (fix inline 5. 6.)
    reference_text = re.sub(r'\s+(\d{1,3}\.)', r'\n\1', reference_text)

    # Unified split rule, anchored to line start to avoid splitting at
    # years like (2020) or numbers like 127. inside reference content
    split_pattern = r'(?:(?<=\n)|(?<=^))(?=\[\d+\]|\(\d+\)|\d{1,3}\.)'
    parts = re.split(split_pattern, reference_text)

    references = []

    for part in parts:
        part = part.strip()
        if not part:
            continue

        match = re.match(
            r'^\[(\d+)\]\s*(.*)|'
            r'^\((\d+)\)\s*(.*)|'
            r'^(\d{1,3})\.\s*(.*)',
            part,
            re.DOTALL
        )

        if not match:
            continue

        groups = match.groups()

        if groups[0]:          # [1]
            content = groups[1]
        elif groups[2]:        # (1)
            content = groups[3]
        else:                  # 1.
            content = groups[5]

        content_clean = content.strip()

        if len(content_clean) >= characters:
            references.append(content_clean)

    return references


def process_references(file_data: dict, reidentify = True) -> dict:
    """
    Process references for a given file.

    Args:
        file_data (dict): The metadata of the file.

    Returns:
        dict: Updated file_data.
    """
    if reidentify == False:
        if "reference_state" not in file_data:
            file_data['reference_state']='None'

        elif file_data["reference_state"] == "done" or file_data["reference_state"] == "Not_Found":
            return file_data

    username, dataset_name = parse_path_info(file_data["file_path"])

    label_path = clean_label_cleaned_json(username, dataset_name, file_data['file_id'])

    references_path = clean_references_json(username, dataset_name, file_data['file_id'])

    label_structure = load_json(label_path)
    raw_references: List[str] = []

    # Extract raw reference strings
    for chunk in label_structure:
        if chunk.get("category") == "reference":
            content = chunk.get("content", "")
            if content.strip():
                raw_references.extend(split_references(content))

    processed_references: Dict[str, Dict] = {}
    with tqdm(
        total=len(raw_references),
        desc=f"agent ref_info [{file_data['file_name'][:8]}..]",
        unit="ref",
        ncols=100,
        position=0,
        leave=False,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
    ) as pbar:
        for ref in raw_references:
            ref_info = get_reference_info(ref)
            # doi = ref_info.get("doi", ref)  # fallback to raw ref if DOI missing
            processed_references[ref] = ref_info
            pbar.update(1)

    if processed_references:
        file_data["reference_state"] = 'done' 
        update_document_metadata(username, dataset_name, file_data, info=False)
        save_json(processed_references, references_path)
    else:
        file_data["reference_state"] = 'Not_Found'
        update_document_metadata(username, dataset_name, file_data, info=False)
    return file_data


REF_HEADER = re.compile(
    r'^#\s*(?:references|bibliography)(?:\s+and\s+notes?)?\s*:?\s*$',
    re.IGNORECASE
)

REF_ENTRY = re.compile(r'^(?:\[\d+\]|\(\d+\)|\d+\.)\s+')




def _has_multiple_refs(chunk: str, min_count: int = 3) -> bool:
    """
    Check if chunk contains multiple reference entries across multiple lines.

    Detects [N] entries at line start (with optional leading symbols like *, †, ‡).
    Requires entries to be on separate lines to distinguish from body paragraphs
    that contain inline citation markers like [1][2].

    Args:
        chunk (str): Markdown chunk text.
        min_count (int): Minimum number of reference entries.

    Returns:
        bool: True if chunk contains >= min_count [N] entries on separate lines.
    """
    pattern = r'^\s*[\*†‡§¶#]*\s*\[\d+\]\s+'
    matches = re.findall(pattern, chunk, re.MULTILINE)
    if len(matches) < min_count:
        return False
    return chunk.count('\n') >= min_count - 1


def identify_references(chunk: str) -> bool:
    """
    Check if a markdown chunk is a reference section.

    Detection priority:
    1. REF_HEADER: chunk starts with "# References" / "# Bibliography"
    2. REF_ENTRY: chunk starts with [N], (N), or N. numbering
    3. _has_multiple_refs: chunk contains >=3 [N] entries on separate lines
       (noise-tolerant: allows leading symbols like *, †, ‡ before the first entry)

    Args:
        chunk (str): Markdown chunk text.

    Returns:
        bool: True if the chunk is identified as a reference section.
    """
    if REF_HEADER.match(chunk):
        return True
    if REF_ENTRY.match(chunk):
        return True
    return _has_multiple_refs(chunk)

