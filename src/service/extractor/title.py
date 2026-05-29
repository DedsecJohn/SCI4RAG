
from tqdm import tqdm
from src.core.logger import get_user_logger
from src.core.paths import *
from src.core.utils import save_json
from src.core.states import DoiStatus
from src.llm.chat.response import llm_response
from src.service.doimeta.fetcher import get_title_info
from src.service.document.load_document import parse_path_info, update_document_metadata
from src.service.document.clean_markdown import chunk_markdown_by_blank_lines, load_markdown


_TITLE_SYSTEM_PROMPT = (
    "You are an expert in academic paper structure analysis.\n"
    "Classify the given text into ONE of the following categories ONLY:\n\n"
    "title or other\n\n"
    "Category definitions:\n"
    "title: The main title of the paper, must more than 4 words long."
    "Examples: # Enhanced thermal stability of nanograined metals below a critical grain size' \n"
    "other: Anything else — including author names, affiliations, "
    "acknowledgments, references, figure captions, tables, or any non-abstract content.\n\n"
    "Rules:\n"
    "1. Return ONLY the category name in lowercase.\n"
    "2. Do NOT add explanations or extra text.\n"
    "3. If uncertain, return 'other'.\n"
)


# 1. First identify main section-> clean_state = "identified_main_section"
def identify_title(file_data: dict) -> dict:
    """ 
    Identify title from markdown chunks by querying CrossRef API.

    Args:
        file_data: File metadata. See `FileData` (src/core/states.py)

    Returns: 
        dict: Updated file_data with DOI_state.
    """
    username, dataset_name = parse_path_info(file_data["file_path"])
    logger = get_user_logger(username, dataset_name)

    # ---- load markdown ----
    content_markdown = load_markdown(file_data)
    chunks = chunk_markdown_by_blank_lines(content_markdown)

    # ---- load path info ----
    doi_path = clean_doi_json(username, dataset_name, file_data['file_id'])

    with tqdm(
        total=len(chunks),
        desc=f"Identifying title [{file_data['file_name'][:5]}..]",
        unit="chunk",
        ncols=100,
        position=0,
        leave=False,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} "
                "[{elapsed}<{remaining}, {rate_fmt}]"
    ) as pbar:
        for chunk in chunks:
            if len(chunk) <= 4: continue
            label = llm_response(query=chunk, system_prompt=_TITLE_SYSTEM_PROMPT, temperature=0.1)["content"].strip().lower()
            if label == "title":
                title_info = get_title_info(chunk)
                if title_info and title_info.get('doi'):
                    file_data["DOI_state"] = DoiStatus.METADATA_FETCHED
                    file_data["doi"] = title_info['doi']
                    file_data["file_name"] = title_info['title']
                    update_document_metadata(username, dataset_name, file_data, info=False)
                    save_json(title_info, doi_path)
                return file_data
            pbar.update(1)
