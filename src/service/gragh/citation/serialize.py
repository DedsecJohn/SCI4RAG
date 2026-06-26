"""
Serialization of feature products (§5).

Writes column-oriented parquet for edge/node tables (with embedded provenance
metadata), the F6 embedding matrix as ``.npy``, and the report as JSON. Every
product carries a ``metadata`` block for full-chain traceability (§5.4).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.core.paths import ensure_dir


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def _write_parquet_with_metadata(df: pd.DataFrame, path: Path, metadata: dict) -> None:
    """Write a DataFrame to parquet, embedding ``metadata`` in the file schema."""
    table = pa.Table.from_pandas(df, preserve_index=False)
    existing = dict(table.schema.metadata or {})
    existing[b"feature_metadata"] = json.dumps(
        metadata, ensure_ascii=False, default=_json_default
    ).encode("utf-8")
    table = table.replace_schema_metadata(existing)
    pq.write_table(table, str(path))


def write_edge_features(df: pd.DataFrame, path, metadata: dict) -> Path:
    """Write edge_features.parquet (one row per candidate pair, §5.1)."""
    path = Path(path)
    ensure_dir(path.parent)
    _write_parquet_with_metadata(df, path, metadata)
    return path


def write_node_features(df: pd.DataFrame, path, metadata: dict) -> Path:
    """Write node_features.parquet (one row per local paper, §5.2)."""
    path = Path(path)
    ensure_dir(path.parent)
    _write_parquet_with_metadata(df, path, metadata)
    return path


def write_embeddings(embeddings: Optional[np.ndarray], path) -> Optional[Path]:
    """Write a single embedding matrix to ``path`` (.npy), skipping None (§5.2)."""
    if embeddings is None:
        return None
    path = Path(path)
    ensure_dir(path.parent)
    np.save(str(path), embeddings)
    return path


def write_report(report: dict, path) -> Path:
    """Write feature_report.json (§5.3)."""
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=_json_default)
    return path
