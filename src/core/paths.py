"""
Centralized path management for SCI4RAG project.

This module provides a unified interface for all file and directory paths
used throughout the project. All paths follow the structure:
    users/{username}/{dataset_name}/{function_dir}/{file_id}/

Directory structure:
    - documents/: Original PDF files
    - parse/: Raw parsing results from MinerU
    - data_clean/: Cleaned and structured data
    - vector/: Vector database storage
    - graph/: Knowledge graph data
    - logs/: Log files
"""

import os
from pathlib import Path
from typing import Optional, Tuple


# ============================================================================
# Base Paths
# ============================================================================

def user_base(username: str, dataset_name: str) -> Path:
    """Get the base directory for a user's dataset."""
    return Path("users") / username / dataset_name


def user_dir(username: str) -> Path:
    """Get the base directory for a user."""
    return Path("users") / username


def project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).resolve().parent.parent.parent


def parse_path_info(file_path: str) -> Tuple[str, str]:
    """
    Parse username and dataset_name from a file path.
    
    Expected format: users/{username}/{dataset_name}/documents/{filename}
    
    Args:
        file_path: Path string to parse
        
    Returns:
        Tuple of (username, dataset_name)
        
    Raises:
        ValueError: If path format is unexpected
    """
    parts = os.path.normpath(file_path).split(os.sep)
    if len(parts) < 5 or parts[0] != "users" or parts[-2] != "documents":
        raise ValueError(f"Unexpected path format: {file_path}")
    return parts[1], parts[2]


# ============================================================================
# Documents - Original PDF files and metadata
# ============================================================================

def documents_dir(username: str, dataset_name: str) -> Path:
    """Get the directory containing original PDF documents."""
    return user_base(username, dataset_name) / "documents"


def documents_json(username: str, dataset_name: str) -> Path:
    """Get the path to documents.json metadata file."""
    return user_base(username, dataset_name) / "documents.json"


def document_pdf(username: str, dataset_name: str, filename: str) -> Path:
    """Get the path to a specific PDF document."""
    return documents_dir(username, dataset_name) / filename


# ============================================================================
# Parse - Raw parsing results from MinerU
# ============================================================================

def parse_dir(username: str, dataset_name: str, file_id: str) -> Path:
    """Get the directory for parsed results of a specific file."""
    return user_base(username, dataset_name) / "parse" / file_id


def parse_layout_json(username: str, dataset_name: str, file_id: str) -> Path:
    """Get the path to layout.json from MinerU parsing."""
    return parse_dir(username, dataset_name, file_id) / "layout.json"


def parse_full_md(username: str, dataset_name: str, file_id: str) -> Path:
    """Get the path to full.md from MinerU parsing."""
    return parse_dir(username, dataset_name, file_id) / "full.md"


def parse_images_dir(username: str, dataset_name: str, file_id: str) -> Path:
    """Get the directory containing extracted images."""
    return parse_dir(username, dataset_name, file_id) / "images"


def parse_zip(username: str, dataset_name: str, file_id: str, base_name: str) -> Path:
    """Get the path to the downloaded zip file from MinerU."""
    return parse_dir(username, dataset_name, file_id) / f"{base_name}.zip"


# ============================================================================
# Data Clean - Cleaned and structured data
# ============================================================================

def clean_dir(username: str, dataset_name: str, file_id: str) -> Path:
    """Get the directory for cleaned data of a specific file."""
    return user_base(username, dataset_name) / "data_clean" / file_id


def clean_doi_json(username: str, dataset_name: str, file_id: str) -> Path:
    """Get the path to doi.json containing DOI and metadata."""
    return clean_dir(username, dataset_name, file_id) / "doi.json"


def clean_label_structure_json(username: str, dataset_name: str, file_id: str) -> Path:
    """Get the path to label_structure.json containing document structure."""
    return clean_dir(username, dataset_name, file_id) / "label_structure.json"


def clean_label_cleaned_json(username: str, dataset_name: str, file_id: str) -> Path:
    """Get the path to label_structure_cleaned.json."""
    return clean_dir(username, dataset_name, file_id) / "label_structure_cleaned.json"


def clean_document_md(username: str, dataset_name: str, file_id: str) -> Path:
    """Get the path to cleaned document.md."""
    return clean_dir(username, dataset_name, file_id) / "document.md"


def clean_references_json(username: str, dataset_name: str, file_id: str) -> Path:
    """Get the path to references.json containing extracted references."""
    return clean_dir(username, dataset_name, file_id) / "references.json"


def clean_figures_json(username: str, dataset_name: str, file_id: str) -> Path:
    """Get the path to figures.json containing figure information."""
    return clean_dir(username, dataset_name, file_id) / "figures.json"


def clean_equation_json(username: str, dataset_name: str, file_id: str) -> Path:
    """Get the path to equation.json containing extracted equations."""
    return clean_dir(username, dataset_name, file_id) / "equation.json"


def clean_table_json(username: str, dataset_name: str, file_id: str) -> Path:
    """Get the path to table.json containing extracted tables."""
    return clean_dir(username, dataset_name, file_id) / "table.json"


def clean_keyword_json(username: str, dataset_name: str, file_id: str) -> Path:
    """Get the path to keyword.json containing extracted keywords."""
    return clean_dir(username, dataset_name, file_id) / "keyword.json"


# ============================================================================
# Vector - Vector database storage
# ============================================================================

def vector_dir(username: str, dataset_name: str) -> Path:
    """Get the directory for vector database storage."""
    return user_base(username, dataset_name) / "vector"


# ============================================================================
# Graph - Knowledge graph data
# ============================================================================

def graph_dir(username: str, dataset_name: str) -> Path:
    """Get the directory for knowledge graph data."""
    return user_base(username, dataset_name) / "graph"


def graph_document_md(username: str, dataset_name: str) -> Path:
    """Get the path to document.md in graph directory."""
    return graph_dir(username, dataset_name) / "document.md"


def graph_tree_json(username: str, dataset_name: str) -> Path:
    """Get the path to 01_tree.json (parsed tree structure)."""
    return graph_dir(username, dataset_name) / "01_tree.json"


def graph_raw_json(username: str, dataset_name: str) -> Path:
    """Get the path to 02_raw.json (raw extracted triplets)."""
    return graph_dir(username, dataset_name) / "02_raw.json"


def graph_final_json(username: str, dataset_name: str) -> Path:
    """Get the path to 03_final.json (aligned and enriched triplets)."""
    return graph_dir(username, dataset_name) / "03_final.json"


def graph_citation_gexf(username: str, dataset_name: str) -> Path:
    """Get the path to citation_graph.gexf."""
    return graph_dir(username, dataset_name) / "citation_graph.gexf"


# ============================================================================
# Logs - Log files
# ============================================================================

def logs_dir(username: Optional[str] = None, dataset_name: Optional[str] = None) -> Path:
    """
    Get the logs directory.
    
    Args:
        username: If None, returns global logs directory
        dataset_name: If provided (with username), returns dataset-specific logs
        
    Returns:
        Path to logs directory:
        - logs/ (global)
        - users/{username}/logs/ (user-level)
        - users/{username}/{dataset_name}/logs/ (dataset-level)
    """
    if username is None:
        return Path("logs")
    elif dataset_name is None:
        return user_dir(username) / "logs"
    else:
        return user_base(username, dataset_name) / "logs"


def logs_activity(username: Optional[str] = None, dataset_name: Optional[str] = None) -> Path:
    """Get the path to activity.log."""
    return logs_dir(username, dataset_name) / "activity.log"


def logs_error(username: Optional[str] = None, dataset_name: Optional[str] = None) -> Path:
    """Get the path to error.log."""
    return logs_dir(username, dataset_name) / "error.log"


def logs_sci4rag() -> Path:
    """Get the path to global sci4rag.log."""
    return Path("logs") / "sci4rag.log"


# ============================================================================
# Evaluate - Evaluation benchmarks and results
# ============================================================================

def evaluate_dir() -> Path:
    """Get the evaluate directory."""
    return Path("evaluate")


def evaluate_benchmark_dir() -> Path:
    """Get the benchmark directory."""
    return evaluate_dir() / "benchmark"


def evaluate_benchmark(benchmark_name: str) -> Path:
    """Get the path to a specific benchmark JSON file."""
    return evaluate_benchmark_dir() / f"{benchmark_name}.json"


def evaluate_result(benchmark_name: str) -> Path:
    """Get the path to evaluation result JSON file."""
    return evaluate_dir() / f"evaluator_result_{benchmark_name}.json"


# ============================================================================
# Config - Configuration files
# ============================================================================

def config_json() -> Path:
    """Get the path to config.json."""
    return Path("config.json")


def env_file() -> Path:
    """Get the path to .env file."""
    return Path(".env")


def apikey_file() -> Path:
    """Get the path to apikey file."""
    return Path("apikey")


# ============================================================================
# Helper Functions
# ============================================================================

def ensure_dir(path: Path) -> Path:
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        path: Path to directory
        
    Returns:
        The same path (for chaining)
    """
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_parent_dir(file_path: Path) -> Path:
    """
    Ensure the parent directory of a file exists.
    
    Args:
        file_path: Path to file
        
    Returns:
        The same file path (for chaining)
    """
    file_path.parent.mkdir(parents=True, exist_ok=True)
    return file_path
