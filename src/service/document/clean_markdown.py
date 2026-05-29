import difflib
import os
import re
from tqdm import tqdm
from pathlib import Path
from src.core.logger import get_user_logger
from src.core.states import CleanStatus, DoiStatus
from src.llm.chat.response import llm_response
from src.service.document.load_document import parse_path_info, updata_document_metadata
from src.core.utils import load_json, save_json
from src.core.paths import (
    parse_full_md, clean_dir, clean_label_structure_json,
    clean_label_cleaned_json, clean_document_md, clean_doi_json
)

def load_markdown(file_data: dict) -> str:
    """
    Load a parsed markdown (.md) file and return its content as a string.

    Args:
        file_data: File metadata. See `FileData` (src/core/states.py)

    Returns:
        str: Content of the markdown file.
    """

    username, dataset_name = parse_path_info(file_data["file_path"])  
    # MinerU default output path
    md_path = parse_full_md(username, dataset_name, file_data["file_id"])
    
    if not md_path.exists():
        raise FileNotFoundError(f"Markdown file not found: {md_path}")

    if not str(md_path).lower().endswith(".md"):
        raise ValueError(f"Not a markdown file: {md_path}")

    with open(md_path, "r", encoding="utf-8") as f:
        content = f.read()

    return content
    
def chunk_markdown_by_blank_lines(md_text: str) -> list[str]:
    """
    Chunk markdown content by blank lines (lines without any words).

    Args:
        md_text (str): The markdown content as a string.

    Returns:
        list[str]: List of markdown chunks.
    """
    chunks = []
    buffer = []

    for line in md_text.splitlines():
        # Line without any words (blank or whitespace)
        if line.strip() == "":
            if buffer:
                chunks.append("\n".join(buffer).strip())
                buffer = []
        else:
            buffer.append(line)

    # Add last chunk
    if buffer:
        chunks.append("\n".join(buffer).strip())

    return chunks


def clean_markdown_content(md_text: str) -> str:
    """
    Remove MinerU HTML artifacts (<details> blocks) and clean up remaining noise.

    Args:
        md_text (str): Raw markdown content.

    Returns:
        str: Cleaned markdown text.
    """
    md_text = re.sub(r'<details>.*?</details>', '', md_text, flags=re.DOTALL)
    md_text = re.sub(r'<summary>.*?</summary>', '', md_text, flags=re.DOTALL)
    md_text = re.sub(r'\n{3,}', '\n\n', md_text)
    return md_text.strip()


def _normalize(text: str) -> str:
    """
    Remove markdown heading prefix and all punctuation, returning lowercase alphanumeric text.

    Args:
        text (str): Raw text to normalize.

    Returns:
        str: Normalized text with only letters, digits, and whitespace.
    """
    text = re.sub(r'^#+\s*', '', text)
    text = re.sub(r'[^\w\s]', '', text, flags=re.UNICODE)
    return re.sub(r'\s+', ' ', text).strip().lower()


def _identify_title(chunk: str, title: str) -> bool:
    """
    Check if a markdown chunk matches the paper title using difflib similarity.

    Args:
        chunk (str): Markdown chunk text.
        title (str): Paper title from doi.json.

    Returns:
        bool: True if similarity ratio exceeds threshold (0.85).
    """
    if len(chunk) < len(title) - 10 or len(chunk) > len(title) + 10:
        return False
    norm_chunk = _normalize(chunk)
    norm_title = _normalize(title)
    return difflib.SequenceMatcher(None, norm_title, norm_chunk).ratio() > 0.85


def identify_main_section(file_data: dict) -> dict:
    """
    Identify title and references section in parsed markdown.

    Finds the title (via doi.json), discards preceding noise, locates the
    references section (via REF_HEADER regex or REF_ENTRY fallback),
    and discards content after the references section.

    Args:
        file_data (dict): File metadata. See `FileData` (src/core/states.py)

    Returns:
        dict: Updated file_data.
    """
    username, dataset_name = parse_path_info(file_data["file_path"])
    logger = get_user_logger(username, dataset_name)

    content_markdown = load_markdown(file_data)
    content_markdown = clean_markdown_content(content_markdown)
    chunks = chunk_markdown_by_blank_lines(content_markdown)

    doi_path = clean_doi_json(username, dataset_name, file_data['file_id'])
    metadata = load_json(doi_path)
    title = metadata.get("title")

    title_idx = 0
    if title:
        for i, chunk in enumerate(chunks):
            if _identify_title(chunk, title):
                title_idx = i
                logger.info("Found title at chunk {}", i)
                break
        else:
            logger.warning("Title not found in markdown chunks, keeping all chunks")

    chunks = chunks[title_idx:]

    from src.service.extractor.reference import identify_references, has_multiple_references, REF_HEADER, SECTION_HEADER

    ref_start = None
    for i, chunk in enumerate(chunks):
        if REF_HEADER.match(chunk):
            ref_start = i
            break

    if ref_start is None:
        for i in range(5, len(chunks)):
            if identify_references(chunks[i]):
                ref_start = i
                break
    
    # Additional check: detect chunks with multiple reference entries (even with noise at start)
    if ref_start is None:
        for i, chunk in enumerate(chunks):
            if has_multiple_references(chunk):
                ref_start = i
                break

    ref_end = len(chunks)
    if ref_start is not None:
        for i in range(ref_start + 1, len(chunks)):
            if SECTION_HEADER.match(chunks[i]):
                ref_end = i
                break

    label_structure = []
    for i, chunk in enumerate(chunks[:ref_end]):
        if i == 0:
            cat = "title"
        elif ref_start is not None and i >= ref_start:
            cat = "references"
        else:
            cat = "other"
        label_structure.append({"category": cat, "content": chunk})

    out_path = clean_label_structure_json(username, dataset_name, file_data['file_id'])
    save_json(label_structure, out_path)
    logger.info("Saved label_structure.json with {} chunks", len(label_structure))

    return file_data


def identify_detail(file_data: dict) -> dict:
    """
    Identify detailed section categories: abstract, figure, table, main_letter.

    Reads label_structure.json, iterates chunks with tqdm. Uses LLM for abstract
    and main_letter detection, rule-based patterns for figure and table.

    Args:
        file_data (dict): File metadata. See `FileData` (src/core/states.py)

    Returns:
        dict: Updated file_data.
    """
    username, dataset_name = parse_path_info(file_data["file_path"])
    logger = get_user_logger(username, dataset_name)

    label_path = clean_label_structure_json(username, dataset_name, file_data['file_id'])
    label_structure = load_json(label_path)

    doi_path = clean_doi_json(username, dataset_name, file_data['file_id'])
    metadata = load_json(doi_path)
    doi_abstract = metadata.get("abstract")

    from src.service.extractor.abstract import identify_abstract
    from src.service.extractor.figure import identify_figure
    from src.service.extractor.table import identify_table
    from src.service.extractor.equation import identify_equation
    from src.service.extractor.main_letter import identify_main_letter

    abstract_found = False

    with tqdm(
        total=len(label_structure),
        desc=f"Identifying detail [{metadata.get("bibkey", "unknown")}]",
        unit="chunk",
        ncols=100,
        position=0,
        leave=False,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} "
                "[{elapsed}<{remaining}, {rate_fmt}]"
    ) as pbar:
        for idx, item in enumerate(label_structure):
            if item["category"] != "other":
                pbar.update(1)
                continue

            content = item["content"]
            prev_content = label_structure[idx - 1]["content"] if idx > 0 else None
            next_content = label_structure[idx + 1]["content"] if idx < len(label_structure) - 1 else None

            # 1. Abstract (LLM, find once)
            if not abstract_found:
                if identify_abstract(content):
                    item["category"] = "abstract"
                    chunk_content = content
                    if doi_abstract:    
                        ratio = difflib.SequenceMatcher(
                            None,
                            _normalize(chunk_content),
                            _normalize(doi_abstract)
                        ).ratio()
                        if ratio > 0.7:
                            item["content"] = doi_abstract
                        else:
                            metadata["abstract"] = chunk_content
                            save_json(metadata, doi_path, info=False)
                    else:
                        metadata["abstract"] = chunk_content
                        save_json(metadata, doi_path, info=False)
                        # logger.info("No abstract in doi.json, wrote identified abstract")
                    abstract_found = True
                    pbar.update(1)
                    continue
            # 2. Figure (rule + positional context)
            if identify_figure(content, prev_content, next_content):
                item["category"] = "figure"
                pbar.update(1)
                continue

            # 3. Table (rule + positional context)
            if identify_table(content, next_content):
                item["category"] = "table"
                pbar.update(1)
                continue

            # 4. Equation (rule)
            if identify_equation(content):
                item["category"] = "equation"
                pbar.update(1)
                continue

            # 5. Main_letter (rule + LLM)
            if identify_main_letter(content):
                item["category"] = "main_letter"

            pbar.update(1)

    save_json(label_structure, label_path)
    logger.info("Saved label_structure.json with detail categories identified")
    return file_data


def combine_label_structure(file_data: dict) -> dict:
    """
    Two-stage merge: first merge figure/table/reference, then merge main_letter.

    Stage 1: Merge figure/table/reference while preserving order.
    Stage 2: Merge main_letter with sentence-aware rules.

    Args:
        file_data (dict): File metadata. See `FileData` (src/core/states.py)

    Returns:
        dict: Updated file_data with clean_state set to COMPLETED.
    """
    username, dataset_name = parse_path_info(file_data["file_path"])
    logger = get_user_logger(username, dataset_name)

    # if file_data["DOI_state"] != DoiStatus.DOCUMENT_UPDATED:
    #     logger.info("Need to update DOI first")
    #     return file_data

    # if file_data.get("clean_state") in {CleanStatus.COMPLETED, CleanStatus.FAILED}:
    #     logger.info("Already {state}", state=file_data['clean_state'])
    #     return file_data

    label_path = clean_label_structure_json(username, dataset_name, file_data['file_id'])
    label_structure = load_json(label_path)

    # ========== Stage 1: Merge figure/table/reference ==========
    stage1_result = []
    buffer = None
    ref_entries = []

    def flush_stage1():
        nonlocal buffer
        if buffer is None:
            return
        content = "\n\n".join(buffer["content"]).strip()
        if content:
            stage1_result.append({
                "category": buffer["category"],
                "content": content
            })
        buffer = None

    for chunk in label_structure:
        cat = chunk.get("category")
        content = chunk.get("content", "")
        if not content or cat == "other":
            continue

        if cat == "references":
            ref_entries.append(content)
            continue

        if cat in {"figure", "table"}:
            if buffer and buffer["category"] == cat:
                buffer["content"].append(content)
            else:
                flush_stage1()
                buffer = {"category": cat, "content": [content]}
            continue

        # Other categories: flush buffer and add as-is
        flush_stage1()
        stage1_result.append({"category": cat, "content": content})

    flush_stage1()

    if ref_entries:
        stage1_result.append({
            "category": "reference",
            "content": "\n".join(ref_entries)
        })

    # ========== Stage 2: Merge main_letter ==========
    final_result = []
    ml_buffer = []
    merging = False
    pending_items = []  # Store non-main_letter items encountered during merge

    def flush_stage2():
        nonlocal ml_buffer, merging, pending_items
        if not ml_buffer:
            return
        content = " ".join(ml_buffer).strip()
        if content:
            final_result.append({"category": "main_letter", "content": content})
        # Append pending items (figure, etc.) that were in between
        final_result.extend(pending_items)
        ml_buffer = []
        merging = False
        pending_items = []

    for item in stage1_result:
        cat = item["category"]
        content = item["content"]

        # Stop conditions: equation, table, or # heading
        if cat in {"equation", "table"}:
            flush_stage2()
            final_result.append(item)
            continue

        if cat == "main_letter":
            ends_with_dot = bool(re.search(r'\.(\d+([,\-−–]\d+)*)?\s*$', content.rstrip()))
            starts_with_hash = content.strip().startswith("#")

            if starts_with_hash:
                flush_stage2()
                final_result.append({"category": "main_letter", "content": content})
                continue

            if ends_with_dot:
                if merging:
                    ml_buffer.append(content)
                    flush_stage2()
                else:
                    final_result.append({"category": "main_letter", "content": content})
                continue

            # Not ending with dot, not starting with #
            if merging:
                ml_buffer.append(content)
            else:
                merging = True
                ml_buffer = [content]
            continue

        # Other categories (title, abstract, figure, reference): don't stop merge
        if merging:
            pending_items.append(item)
        else:
            final_result.append(item)

    flush_stage2()

    new_label_path = clean_label_cleaned_json(username, dataset_name, file_data['file_id'])
    save_json(final_result, new_label_path)
    # file_data["clean_state"] = CleanStatus.COMPLETED
    # updata_document_metadata(username, dataset_name, file_data)
    return file_data


def combile_doc_json(file_data: dict) -> dict:
    """
    Generate cleaned document.md (placeholder for Phase 2).

    Args:
        file_data (dict): File metadata.

    Returns:
        dict: Updated file_data.
    """
    return file_data