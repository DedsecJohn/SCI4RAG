
from pathlib import Path
from langchain_core.messages import HumanMessage, SystemMessage
from src.service.generator.llm_response import llm_response
from src.service.document.load_document import load_json, save_json, parse_path_info, updata_document_metadata


def construct_kg_doc(file_data: dict, reidentify = False) -> dict:
    """
    Construct a knowledge graph from a document.

    Args:
        file_data (dict): Metadata dictionary containing:
            - file_name
            - file_type
            - file_path
            - file_id
            - file_size
            - update_time
            - knowledge_graph (optional)

    Returns: 
        file_data (dict): Updated file_data with knowledge_graph.
    """
    username, dataset_name = parse_path_info(file_data["file_path"])
    label_path = Path(
    f"users/{username}/{dataset_name}/data_clean/{file_data['file_id']}/document.md"
    )