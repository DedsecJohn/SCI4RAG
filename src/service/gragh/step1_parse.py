import json
import re
from pathlib import Path
from src.core.logger import get_user_logger
from src.core.paths import *


class SectionNode:
    def __init__(self, level: int, title: str):
        self.level = level
        self.title = title
        self.content = ""
        self.children = []

    def to_dict(self):
        return {
            "level": self.level,
            "title": self.title,
            "content": self.content.strip(),
            "children": [child.to_dict() for child in self.children]
        }


def parse_markdown_to_tree(md_text: str) -> SectionNode:
    """
    Function: Parses a Markdown string into a hierarchical tree structure based on heading levels.

    Input:
    - md_text (str): The raw Markdown text to parse.

    Output:
    - root (SectionNode): The root node of the parsed document tree.
    """
    md_text = md_text.lstrip('\ufeff')
    root = SectionNode(level=0, title="Document Root")
    stack = [root]
    heading_pattern = re.compile(r'^(#+)\s+(.*)')

    for line in md_text.split('\n'):
        match = heading_pattern.match(line)
        if match:
            level = len(match.group(1))
            title = match.group(2).strip()
            new_node = SectionNode(level=level, title=title)

            while stack and stack[-1].level >= level:
                stack.pop()

            if stack:
                stack[-1].children.append(new_node)
            stack.append(new_node)
        else:
            if line.strip() or stack[-1].content:
                stack[-1].content += line + "\n"

    return root


def run_step1(username: str, dataset_name: str, file_id: str):
    """
    Function: Reads a Markdown file, extracts its hierarchical structure, and saves it as a JSON file.

    Input:
    - username (str): The username of the workspace.
    - dataset_name (str): The name of the dataset.
    - file_id (str): The unique identifier for the file directory.

    Output:
    - None (Saves the output directly to '01_tree.json').
    """
    md_path = clean_document_md(username, dataset_name, file_id)
    tree_path = graph_tree_json(username, dataset_name, file_id)

    with open(md_path, "r", encoding="utf-8") as f:
        tree_dict = parse_markdown_to_tree(f.read()).to_dict()

    ensure_parent_dir(tree_path)
    with open(tree_path, "w", encoding="utf-8") as f:
        json.dump(tree_dict, f, ensure_ascii=False, indent=4)

    get_user_logger(username, dataset_name).success("Step 1: Document tree parsed successfully")