
from tqdm import tqdm
from src.core.logger import get_user_logger
from src.core.paths import *
from src.core.utils import save_json
from src.core.states import DoiStatus
from src.service.doimeta.fetcher import get_reference_info
from src.service.document.load_document import parse_path_info, updata_document_metadata
from src.service.document.clean_markdown import chunk_identify_main_section, chunk_markdown_by_blank_lines, load_markdown

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

    # ---- initial categories ----     
    BASE_CATEGORIES = {
        "title",
        "other",
    }
    CATEGORIES = set(BASE_CATEGORIES)

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
        for i, chunk in enumerate(chunks):
            label = chunk_identify_main_section(query=chunk, CATEGORIES=CATEGORIES)
            if label == "title":
                title_info = get_reference_info(chunk)
                if title_info and title_info.get('doi'):
                    file_data["DOI_state"] = DoiStatus.METADATA_FETCHED
                    file_data["doi"] = title_info['doi']
                    file_data["file_name"] = title_info['title']
                    updata_document_metadata(username, dataset_name, file_data, info=False)
                    save_json(title_info, doi_path)
                return file_data
            pbar.update(1)
