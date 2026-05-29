import json
from pathlib import Path
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from src.core.logger import get_user_logger, get_logger
from src.core.paths import *


class Triplet(BaseModel):
    source: str = Field(description="Atomic head entity (Strictly 1-3 words, e.g., 'Schwarz Crystal', NOT a sentence)")
    target: str = Field(description="Atomic tail entity (Strictly 1-3 words, e.g., 'Copper', NOT a description)")
    relation: str = Field(description="Concise relational verb (e.g., 'exhibits', 'consists of')")


class KnowledgeGraph(BaseModel):
    triplets: list[Triplet] = Field(description="List of extracted knowledge triplets")


def process_tree_node(node: dict, path_title: str, chain, parser, all_triplets: list):
    """
    Function: Recursively traverses the document tree and uses the LLM to extract entity triplets from each section.

    Input:
    - node (dict): The current tree node being processed.
    - path_title (str): The hierarchical path of the current section (e.g., 'Introduction > Background').
    - chain (Runnable): The LangChain execution chain (Prompt + LLM + Parser).
    - parser (JsonOutputParser): The parser to format LLM output.
    - all_triplets (list): The global list to store extracted triplets.

    Output:
    - None (Extracted triplets are appended directly to the 'all_triplets' list).
    """
    current_title = f"{path_title} > {node['title']}" if path_title else node['title']
    content = node.get("content", "").strip()

    if content:
        get_logger().info("Extracting data from: {title}", title=current_title)
        chunk_size = 3000
        chunks = [content[i:i + chunk_size] for i in range(0, len(content), chunk_size)]

        for chunk in chunks:
            try:
                result = chain.invoke({
                    "section_title": current_title,
                    "text": chunk,
                    "format_instructions": parser.get_format_instructions()
                })
                all_triplets.extend(result.get("triplets", []))
            except Exception as e:
                get_logger().error("Failed to extract from chunk: {error}", error=str(e))

    for child in node.get("children", []):
        process_tree_node(child, current_title, chain, parser, all_triplets)


def run_step2(username: str, dataset_name: str, file_id: str, api_key: str):
    """
    Function: Initializes the LLM and runs the context-aware extraction process on the document tree.

    Input:
    - username (str): The username of the workspace.
    - dataset_name (str): The name of the dataset.
    - file_id (str): The unique identifier for the file directory.
    - api_key (str): API key for the language model.

    Output:
    - None (Saves the extracted triplets to '02_raw.json').
    """
    logger = get_user_logger(username, dataset_name)
    tree_path = graph_tree_json(username, dataset_name, file_id)
    raw_path = graph_raw_json(username, dataset_name, file_id)

    with open(tree_path, "r", encoding="utf-8") as f:
        document_tree = json.load(f)

    llm = ChatOpenAI(temperature=0.0, model="deepseek-chat", api_key=api_key, base_url="https://api.deepseek.com")
    parser = JsonOutputParser(pydantic_object=KnowledgeGraph)

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a data extraction tool.
        SECTION PATH: [{section_title}]

        RULES:
        1. ATOMIC ENTITIES ONLY: Entities MUST be concise, strictly 1 to 3 words.
        2. NO DESCRIPTIONS: NEVER extract full sentences or long phrases as entities.
           - Bad Entity: "Temperatures Close To The Melting Point" -> Good Entity: "Melting Point"
           - Bad Entity: "Ideal Strength Close To Limit" -> Good Entity: "Ideal Strength"

        {format_instructions}"""),
        ("human", "Text:\n{text}")
    ])

    all_raw_triplets = []
    process_tree_node(document_tree, "", prompt | llm | parser, parser, all_raw_triplets)

    ensure_parent_dir(raw_path)
    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(all_raw_triplets, f, ensure_ascii=False, indent=4)

    logger.success("Step 2: Extracted {count} raw triplets", count=len(all_raw_triplets))