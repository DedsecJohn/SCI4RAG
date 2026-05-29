
from pathlib import Path
from langchain_core.messages import HumanMessage, SystemMessage
from src.core.paths import *
from src.core.utils import load_json, save_json
from src.llm.chat.response import llm_response
from src.service.document.load_document import parse_path_info, update_document_metadata


def construct_kg_doc(file_data: dict, reidentify = False) -> dict:
    """
    Construct a knowledge graph from a document.

    Args:
        file_data: File metadata. See `FileData` (src/core/states.py)
        reidentify (bool): Whether to force re-identification. Defaults to False.

    Returns: 
        file_data (dict): Updated file_data with knowledge_graph.
    """
    username, dataset_name = parse_path_info(file_data["file_path"])
    label_path = clean_document_md(username, dataset_name, file_data['file_id'])