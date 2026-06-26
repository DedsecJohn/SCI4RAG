from dataclasses import dataclass, field
from typing import Optional


class ParseStatus:
    """MinerU PDF parsing status constants."""
    NOT_PARSED = "Not Parsed"
    PROCESSING = "processing"
    DOWNLOADED = "Download"    # Result downloaded to local storage
    FAILED = "failed"


class DoiStatus:
    """DOI extraction and metadata status constants."""
    NOT_DOI = "Not_DOI"
    DOI_EXTRACTED = "Doi_Getted"
    METADATA_FETCHED = "Doi_Info_Getted"
    UPDATED = "Doi_Updated"


class CleanStatus:
    """Document cleaning pipeline status constants."""
    NOT_CLEANED = "Not Cleaned"
    STRUCTURED = "Structured"
    EXTRACTED = "Extracted"
    COMPLETED = "Completed"
    FAILED = "Failed"


@dataclass
class FileData:
    """
    Metadata schema for a single file in the pipeline.
    Each dict passed as ``file_data`` throughout the codebase
    is expected to conform to this structure.
    """

    # ── File identity ──────────────────────────────────────────────
    file_id: str                                    # Unique ID: 20260521T143045_a1b2c3d4e5
    file_name: str                                  # File stem (no extension)
    file_type: str = "pdf"                          # File extension type
    file_path: str = ""                             # Full path: users/{user}/{ds}/documents/{file}.pdf
    file_size: int = 0                              # File size in bytes
    update_time: str = ""                           # Last update: "Sat, 21 May 2026 14:30"

    # ── MinerU parsing phase ───────────────────────────────────────
    parsing_status: str = ParseStatus.NOT_PARSED    # ParseStatus: parsing progress
    batch_id: Optional[str] = None                  # MinerU batch ID (after upload)

    # ── DOI / title identification ─────────────────────────────────
    DOI_state: str = DoiStatus.NOT_DOI              # DoiStatus: DOI extraction progress
    doi: Optional[str] = None                       # DOI value: "10.1103/PhysRevLett.127.136101"

    # ── Document cleaning ──────────────────────────────────────────
    clean_state: str = CleanStatus.NOT_CLEANED       # CleanStatus: cleaning progress

    # ── Detail extraction (equations, references, figures) ─────────
    equation_state: Optional[str] = None            # ExtractStatus
    reference_state: Optional[str] = None           # ExtractStatus
    citation_state: Optional[str] = None            # ExtractStatus: in-text citations
    figure_state: Optional[str] = None              # ExtractStatus

    # ── Vectorization ──────────────────────────────────────────────
    vector_status: str = ""                         # VectorStatus: embedding progress

    # ── Extra metadata ─────────────────────────────────────────────
    bibjson: dict = field(default_factory=dict)     # Crossref / DOI metadata
