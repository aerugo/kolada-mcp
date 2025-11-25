"""Kolada API client service.

This module provides a clean, composable HTTP client for the Kolada API
with automatic pagination, retry logic, and proper error handling.
"""

import asyncio
import logging
from collections.abc import Callable
from typing import Any, Protocol

import httpx

from kolada_mcp.config import settings
from kolada_mcp.models.types import (
    KoladaApiResponse,
    KoladaDataPoint,
    KoladaKpi,
    KoladaMunicipality,
)

logger = logging.getLogger(__name__)


class HttpClientProtocol(Protocol):
    """Protocol for HTTP client dependency injection."""

    async def get(
        self, url: str, **kwargs: Any
    ) -> httpx.Response: ...


class KoladaClient:
    """Client for interacting with the Kolada API.

    This client provides methods to fetch KPIs, municipalities, and data
    from the Kolada API. It handles pagination automatically and implements
    retry logic with exponential backoff for resilience.

    Attributes:
        base_url: The base URL for the Kolada API
        page_size: Number of items per page for pagination
        max_retries: Maximum number of retry attempts
        retry_base_delay: Base delay between retries (exponential backoff)
    """

    def __init__(
        self,
        base_url: str = settings.kolada_base_url,
        page_size: int = settings.kolada_page_size,
        max_retries: int = settings.max_retries,
        retry_base_delay: float = settings.retry_base_delay,
        http_client: HttpClientProtocol | None = None,
    ) -> None:
        """Initialize the Kolada client.

        Args:
            base_url: The base URL for the Kolada API
            page_size: Number of items per page for pagination
            max_retries: Maximum number of retry attempts
            retry_base_delay: Base delay between retries
            http_client: Optional HTTP client for dependency injection (testing)
        """
        self.base_url = base_url.rstrip("/")
        self.page_size = page_size
        self.max_retries = max_retries
        self.retry_base_delay = retry_base_delay
        self._http_client = http_client

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create an HTTP client."""
        return httpx.AsyncClient(timeout=30.0)

    async def _fetch_with_retry(
        self, url: str, client: httpx.AsyncClient
    ) -> KoladaApiResponse:
        """Fetch data from URL with retry logic.

        Args:
            url: The URL to fetch
            client: The HTTP client to use

        Returns:
            The parsed JSON response

        Raises:
            httpx.HTTPStatusError: If all retries fail
            httpx.RequestError: If a network error occurs
        """
        last_error: Exception | None = None

        for attempt in range(self.max_retries):
            try:
                response = await client.get(url)
                response.raise_for_status()
                return response.json()
            except httpx.HTTPStatusError as e:
                last_error = e
                if e.response.status_code >= 500:
                    # Server error, retry with backoff
                    delay = self.retry_base_delay * (2**attempt)
                    logger.warning(
                        f"[Kolada MCP] Server error {e.response.status_code}, "
                        f"retrying in {delay}s (attempt {attempt + 1}/{self.max_retries})"
                    )
                    await asyncio.sleep(delay)
                else:
                    # Client error, don't retry
                    raise
            except httpx.RequestError as e:
                last_error = e
                delay = self.retry_base_delay * (2**attempt)
                logger.warning(
                    f"[Kolada MCP] Request error: {e}, "
                    f"retrying in {delay}s (attempt {attempt + 1}/{self.max_retries})"
                )
                await asyncio.sleep(delay)

        # All retries exhausted
        if last_error:
            raise last_error
        raise RuntimeError("Unexpected error in fetch_with_retry")

    async def _fetch_paginated(
        self, url: str, client: httpx.AsyncClient
    ) -> list[Any]:
        """Fetch all pages from a paginated endpoint.

        Args:
            url: The initial URL to fetch
            client: The HTTP client to use

        Returns:
            Combined list of all values from all pages
        """
        all_values: list[Any] = []
        visited_urls: set[str] = set()
        current_url: str | None = url

        while current_url and current_url not in visited_urls:
            visited_urls.add(current_url)
            data = await self._fetch_with_retry(current_url, client)
            values = data.get("values", [])
            all_values.extend(values)

            # Check for next page
            current_url = data.get("next_page")
            if current_url:
                logger.debug(f"[Kolada MCP] Fetching next page: {current_url}")

        return all_values

    async def fetch_kpis(self) -> list[KoladaKpi]:
        """Fetch all KPIs from the Kolada API.

        Returns:
            List of all KPIs
        """
        url = f"{self.base_url}/kpi"
        async with httpx.AsyncClient(timeout=60.0) as client:
            values = await self._fetch_paginated(url, client)
            return values

    async def fetch_municipalities(self) -> list[KoladaMunicipality]:
        """Fetch all municipalities from the Kolada API.

        Returns:
            List of all municipalities
        """
        url = f"{self.base_url}/municipality"
        async with httpx.AsyncClient(timeout=60.0) as client:
            values = await self._fetch_paginated(url, client)
            return values

    def _build_data_url(
        self,
        kpi_id: str,
        municipality_ids: list[str],
        years: list[int] | None = None,
    ) -> str:
        """Build the URL for fetching KPI data.

        Args:
            kpi_id: The KPI identifier
            municipality_ids: List of municipality IDs
            years: Optional list of years to fetch

        Returns:
            The constructed URL
        """
        municipalities = ",".join(municipality_ids)
        url = f"{self.base_url}/data/kpi/{kpi_id}/municipality/{municipalities}"

        if years:
            years_str = ",".join(str(y) for y in years)
            url = f"{url}/year/{years_str}"

        return url

    async def fetch_data(
        self,
        kpi_id: str,
        municipality_ids: list[str],
        years: list[int] | None = None,
    ) -> list[KoladaDataPoint]:
        """Fetch KPI data for specified municipalities and years.

        Args:
            kpi_id: The KPI identifier
            municipality_ids: List of municipality IDs
            years: Optional list of years to fetch

        Returns:
            List of data points
        """
        url = self._build_data_url(kpi_id, municipality_ids, years)
        async with httpx.AsyncClient(timeout=60.0) as client:
            values = await self._fetch_paginated(url, client)
            return values

    async def fetch_data_from_url(self, url: str) -> KoladaApiResponse:
        """Fetch data from a pre-built URL.

        This is useful for fetching data with custom URL patterns.

        Args:
            url: The full URL to fetch

        Returns:
            The API response
        """
        async with httpx.AsyncClient(timeout=60.0) as client:
            return await self._fetch_with_retry(url, client)
