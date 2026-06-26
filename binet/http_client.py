"""
Unified HTTP client with exponential-backoff retry and rate control (FR-5).

Behaviour contract:
    - Exponential backoff on transient errors (FR-5.1): 1s / 2s / 4s (+ jitter).
    - HTTP 404 and other deterministic failures are NOT retried (FR-5.2); they
      raise ``DeterministicFailure`` so callers can record them in ``failed``.
    - Random per-request delay for rate control (FR-5.3), default 0.5-1.0s.
    - Polite-pool contact email injected into the User-Agent header (§2).
"""

from __future__ import annotations

import time
import random
import logging
from typing import Optional, Tuple

import requests

from binet import __version__
from binet.errors import DeterministicFailure

logger = logging.getLogger("binet.http")

# HTTP status codes that warrant a retry (transient).
_RETRYABLE_STATUS = {429, 500, 502, 503, 504}
# HTTP status codes that are deterministic failures (do not retry, FR-5.2).
_DETERMINISTIC_STATUS = {400, 401, 403, 404, 410}


class HttpClient:
    """A thin requests wrapper with retry + rate limiting."""

    def __init__(
        self,
        email: str,
        delay_range: Tuple[float, float] = (0.5, 1.0),
        max_retries: int = 3,
        timeout: int = 20,
    ):
        self.email = email
        self.delay_range = delay_range
        self.max_retries = max_retries
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": f"binet/{__version__} (mailto:{email})",
                "Accept": "application/json",
            }
        )

    # ------------------------------------------------------------------ #

    def _sleep_rate_limit(self) -> None:
        """Random delay before each request to respect source rate limits."""
        lo, hi = self.delay_range
        if hi > 0:
            time.sleep(random.uniform(lo, hi))

    def get_json(self, url: str, params: Optional[dict] = None) -> Optional[dict]:
        """
        Perform a GET request and return the parsed JSON body.

        Args:
            url: Request URL.
            params: Query parameters.

        Returns:
            Parsed JSON dict on success.

        Raises:
            DeterministicFailure: On 404 and other non-retryable statuses.

        Returns None only if all retries are exhausted on transient errors.
        """
        last_exc: Optional[Exception] = None

        for attempt in range(self.max_retries):
            self._sleep_rate_limit()
            try:
                resp = self.session.get(url, params=params, timeout=self.timeout)

                if resp.status_code in _DETERMINISTIC_STATUS:
                    raise DeterministicFailure(
                        f"HTTP {resp.status_code} for {url}",
                        reason=f"http_{resp.status_code}",
                    )

                if resp.status_code in _RETRYABLE_STATUS:
                    wait = 2 ** attempt + random.uniform(0, 1)
                    logger.warning(
                        "HTTP %s for %s, retry %d/%d in %.1fs",
                        resp.status_code, url, attempt + 1, self.max_retries, wait,
                    )
                    time.sleep(wait)
                    continue

                resp.raise_for_status()
                return resp.json()

            except DeterministicFailure:
                raise
            except (requests.exceptions.ConnectionError,
                    requests.exceptions.Timeout) as exc:
                last_exc = exc
                if attempt < self.max_retries - 1:
                    wait = 2 ** attempt + random.uniform(0, 1)
                    logger.warning(
                        "%s for %s, retry %d/%d in %.1fs",
                        type(exc).__name__, url, attempt + 1, self.max_retries, wait,
                    )
                    time.sleep(wait)
                    continue
            except requests.exceptions.HTTPError as exc:
                # Non-classified HTTP error: treat as transient if retryable.
                last_exc = exc
                status = getattr(exc.response, "status_code", None)
                if status in _RETRYABLE_STATUS and attempt < self.max_retries - 1:
                    wait = 2 ** attempt + random.uniform(0, 1)
                    time.sleep(wait)
                    continue
                logger.warning("HTTPError for %s: %s", url, exc)
                return None
            except ValueError as exc:
                # JSON decode error.
                logger.warning("Invalid JSON from %s: %s", url, exc)
                return None

        logger.warning(
            "All %d retries exhausted for %s (%s)",
            self.max_retries, url, last_exc,
        )
        return None
