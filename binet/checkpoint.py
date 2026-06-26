"""
Checkpoint / resume support (FR-5.4, AC-4).

The crawler periodically dumps its full state (queue, nodes, edges, visited
set, counters) to ``checkpoint.json`` so a Ctrl-C'd run can ``--resume`` from
the most recent checkpoint without re-crawling already-processed nodes.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger("binet.checkpoint")


def save_checkpoint(path: Path, state: dict) -> None:
    """
    Atomically persist the crawler state to ``path``.

    Writes to a temp file first then renames, so an interrupted write never
    corrupts an existing checkpoint.

    Args:
        path: Destination checkpoint file.
        state: Serializable state dict.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
    tmp.replace(path)
    logger.debug("Checkpoint saved to %s", path)


def load_checkpoint(path: Path) -> Optional[dict]:
    """
    Load a checkpoint if it exists.

    Args:
        path: Checkpoint file path.

    Returns:
        The state dict, or None if no checkpoint exists / it is unreadable.
    """
    path = Path(path)
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Failed to read checkpoint %s: %s", path, exc)
        return None


def clear_checkpoint(path: Path) -> None:
    """
    Remove the checkpoint file (called after a successful completion, §4.2).

    Args:
        path: Checkpoint file path.
    """
    path = Path(path)
    if path.exists():
        try:
            path.unlink()
            logger.debug("Checkpoint cleared: %s", path)
        except OSError as exc:
            logger.warning("Failed to clear checkpoint %s: %s", path, exc)
