"""
Abstract data-source contract (§5.2, MUST).

Defines a unified interface so that the references source and the cited-by
source are independently replaceable. A concrete source that does not support
a direction MUST raise ``NotSupportedError`` (it MUST NOT silently return
wrong-direction data).
"""

from __future__ import annotations

from typing import List, Optional, Protocol, runtime_checkable

from binet.models import PaperMeta


@runtime_checkable
class CitationSource(Protocol):
    """Protocol every data source implements."""

    #: Short identifier used in logs and the ``data_sources`` metadata field.
    name: str

    def fetch_metadata(self, doi: str) -> Optional[PaperMeta]:
        """
        Return title / authors / year for a DOI.

        Args:
            doi: Normalized DOI.

        Returns:
            A ``PaperMeta`` on success, or None on failure.
        """
        ...

    def fetch_references(self, doi: str) -> List[str]:
        """
        Backward direction: DOIs that this paper cites.

        Args:
            doi: Normalized DOI.

        Returns:
            A list of normalized target DOIs (may be empty).

        Raises:
            NotSupportedError: If the source cannot provide references.
        """
        ...

    def fetch_citations(self, doi: str) -> List[str]:
        """
        Forward direction: DOIs of papers that cite this paper (cited-by).

        Args:
            doi: Normalized DOI.

        Returns:
            A list of normalized source DOIs (may be empty).

        Raises:
            NotSupportedError: If the source cannot provide cited-by.
        """
        ...


class BaseSource:
    """
    Convenience base class providing a shared ``http`` client and a counter for
    edges dropped because a reference target lacked a DOI (FR-4.3).
    """

    name: str = "base"

    def __init__(self, http):
        self.http = http
        # Count of references/citations dropped due to missing DOI (FR-4.3).
        self.dropped_no_doi = 0
