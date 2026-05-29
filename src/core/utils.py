"""
Utility functions for SCI4RAG project.

This module provides helper functions for common operations like
JSON file I/O, file existence checks, similarity calculations, etc.
"""

import json
import numpy as np
from pathlib import Path
from typing import Any, List, Optional, Union
from src.core.logger import get_logger


def load_json(path: Union[str, Path], default: Optional[dict] = None) -> dict:
    """
    Load JSON data from a file.
    
    Args:
        path: Path to JSON file
        default: Default value to return if file doesn't exist
        
    Returns:
        Loaded JSON data, or default value if file doesn't exist
    """
    if not Path(path).exists():
        return default if default is not None else {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: dict, path: Union[str, Path], indent: int = 2, info: bool = True) -> None:
    """
    Save data to a JSON file.
    
    Args:
        path: Path to JSON file
        data: Data to save
        indent: JSON indentation level (default: 2)
        info: Whether to print save confirmation (default: True)
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)
    if info:
        print(f"JSON saved to: {path}")


def exists(path: Union[str, Path]) -> bool:
    """
    Check if a path exists.
    
    Args:
        path: Path to check
        
    Returns:
        True if path exists, False otherwise
    """
    return Path(path).exists()


def read_text(path: Union[str, Path], encoding: str = "utf-8") -> str:
    """
    Read text content from a file.
    
    Args:
        path: Path to text file
        encoding: File encoding (default: utf-8)
        
    Returns:
        File content as string
        
    Raises:
        FileNotFoundError: If file doesn't exist
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    with open(path, "r", encoding=encoding) as f:
        return f.read()


def write_text(path: Union[str, Path], content: str, encoding: str = "utf-8") -> None:
    """
    Write text content to a file.
    
    Args:
        path: Path to text file
        content: Content to write
        encoding: File encoding (default: utf-8)
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding=encoding) as f:
        f.write(content)


def list_files(directory: Union[str, Path], pattern: str = "*", recursive: bool = False) -> List[Path]:
    """
    List files in a directory matching a pattern.
    
    Args:
        directory: Directory to search
        pattern: Glob pattern (default: "*")
        recursive: Whether to search recursively (default: False)
        
    Returns:
        List of matching file paths
    """
    directory = Path(directory)
    if not directory.exists():
        return []
    
    if recursive:
        return list(directory.rglob(pattern))
    else:
        return list(directory.glob(pattern))


def cosine_similarity(vec_a, vec_b) -> float:
    import numpy as np
    dot = np.dot(vec_a, vec_b)
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot / (norm_a * norm_b))


def cosine_similarity_matrix(
    vec1: "np.ndarray",
    vec2: "np.ndarray",
) -> "np.ndarray":
    import numpy as np
    dot_product = np.dot(vec1, vec2.T)
    norm_vec1 = np.linalg.norm(vec1, axis=1, keepdims=True)
    norm_vec2 = np.linalg.norm(vec2, axis=1, keepdims=True)
    similarity_matrix = dot_product / (norm_vec1 * norm_vec2.T + 1e-10)
    return similarity_matrix
