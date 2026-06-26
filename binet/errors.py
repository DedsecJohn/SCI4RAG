"""
Custom exceptions for binet.
"""


class BinetError(Exception):
    """Base class for all binet-specific exceptions."""


class NotSupportedError(BinetError):
    """
    Raised when a data source is asked to provide a direction it does not
    support (e.g. CrossRef cited-by).

    Per §5.2 of the requirement document, unsupported directions MUST raise
    this error explicitly and MUST NOT silently return wrong-direction data.
    """


class DeterministicFailure(BinetError):
    """
    Raised for deterministic, non-retryable failures (e.g. HTTP 404).

    These failures are recorded in the ``failed`` list and the request is
    abandoned without retry (FR-5.2).
    """

    def __init__(self, message: str, doi: str = "", reason: str = ""):
        super().__init__(message)
        self.doi = doi
        self.reason = reason or message
