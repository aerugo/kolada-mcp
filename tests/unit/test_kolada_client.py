"""Unit tests for Kolada API client."""

from typing import Any
from unittest.mock import AsyncMock, patch

import httpx
import pytest
import respx

from kolada_mcp.services.kolada_client import KoladaClient


class TestKoladaClient:
    """Tests for KoladaClient."""

    @pytest.fixture
    def client(self) -> KoladaClient:
        """Create a KoladaClient instance for testing."""
        return KoladaClient(
            base_url="https://api.kolada.se/v2",
            page_size=100,
            max_retries=3,
            retry_base_delay=0.1,
        )

    @pytest.mark.asyncio
    @respx.mock
    async def test_fetch_kpis_success(self, client: KoladaClient) -> None:
        """Test successful KPI fetching."""
        response_data = {
            "count": 2,
            "values": [
                {"id": "N00945", "title": "KPI 1", "operating_area": "Test"},
                {"id": "N01951", "title": "KPI 2", "operating_area": "Test"},
            ],
        }
        respx.get("https://api.kolada.se/v2/kpi").mock(
            return_value=httpx.Response(200, json=response_data)
        )

        result = await client.fetch_kpis()

        assert len(result) == 2
        assert result[0]["id"] == "N00945"
        assert result[1]["id"] == "N01951"

    @pytest.mark.asyncio
    @respx.mock
    async def test_fetch_kpis_pagination(self, client: KoladaClient) -> None:
        """Test KPI fetching with pagination."""
        page1_data = {
            "count": 3,
            "next_page": "https://api.kolada.se/v2/kpi?page=2",
            "values": [
                {"id": "N00001", "title": "KPI 1"},
            ],
        }
        page2_data = {
            "count": 3,
            "values": [
                {"id": "N00002", "title": "KPI 2"},
                {"id": "N00003", "title": "KPI 3"},
            ],
        }
        respx.get("https://api.kolada.se/v2/kpi").mock(
            return_value=httpx.Response(200, json=page1_data)
        )
        respx.get("https://api.kolada.se/v2/kpi?page=2").mock(
            return_value=httpx.Response(200, json=page2_data)
        )

        result = await client.fetch_kpis()

        assert len(result) == 3
        assert result[0]["id"] == "N00001"
        assert result[2]["id"] == "N00003"

    @pytest.mark.asyncio
    @respx.mock
    async def test_fetch_municipalities_success(self, client: KoladaClient) -> None:
        """Test successful municipality fetching."""
        response_data = {
            "count": 3,
            "values": [
                {"id": "0180", "title": "Stockholm", "type": "K"},
                {"id": "1480", "title": "Göteborg", "type": "K"},
                {"id": "01", "title": "Region Stockholm", "type": "R"},
            ],
        }
        respx.get("https://api.kolada.se/v2/municipality").mock(
            return_value=httpx.Response(200, json=response_data)
        )

        result = await client.fetch_municipalities()

        assert len(result) == 3
        assert result[0]["id"] == "0180"
        assert result[0]["type"] == "K"
        assert result[2]["type"] == "R"

    @pytest.mark.asyncio
    @respx.mock
    async def test_fetch_data_success(self, client: KoladaClient) -> None:
        """Test successful data fetching."""
        response_data = {
            "count": 2,
            "values": [
                {
                    "kpi": "N00945",
                    "municipality": "0180",
                    "period": 2023,
                    "values": [{"gender": "T", "value": 984748, "count": 1}],
                },
                {
                    "kpi": "N00945",
                    "municipality": "1480",
                    "period": 2023,
                    "values": [{"gender": "T", "value": 590580, "count": 1}],
                },
            ],
        }
        respx.get("https://api.kolada.se/v2/data/kpi/N00945/municipality/0180,1480/year/2023").mock(
            return_value=httpx.Response(200, json=response_data)
        )

        result = await client.fetch_data(
            kpi_id="N00945",
            municipality_ids=["0180", "1480"],
            years=[2023],
        )

        assert len(result) == 2
        assert result[0]["kpi"] == "N00945"
        assert result[0]["municipality"] == "0180"
        assert result[0]["values"][0]["value"] == 984748

    @pytest.mark.asyncio
    @respx.mock
    async def test_fetch_data_multiple_years(self, client: KoladaClient) -> None:
        """Test data fetching with multiple years."""
        response_data = {
            "count": 2,
            "values": [
                {
                    "kpi": "N00945",
                    "municipality": "0180",
                    "period": 2022,
                    "values": [{"gender": "T", "value": 978770, "count": 1}],
                },
                {
                    "kpi": "N00945",
                    "municipality": "0180",
                    "period": 2023,
                    "values": [{"gender": "T", "value": 984748, "count": 1}],
                },
            ],
        }
        respx.get("https://api.kolada.se/v2/data/kpi/N00945/municipality/0180/year/2022,2023").mock(
            return_value=httpx.Response(200, json=response_data)
        )

        result = await client.fetch_data(
            kpi_id="N00945",
            municipality_ids=["0180"],
            years=[2022, 2023],
        )

        assert len(result) == 2
        assert result[0]["period"] == 2022
        assert result[1]["period"] == 2023

    @pytest.mark.asyncio
    @respx.mock
    async def test_retry_on_server_error(self, client: KoladaClient) -> None:
        """Test retry logic on server error."""
        route = respx.get("https://api.kolada.se/v2/kpi")
        route.side_effect = [
            httpx.Response(503),
            httpx.Response(503),
            httpx.Response(200, json={"count": 1, "values": [{"id": "N00001", "title": "KPI"}]}),
        ]

        result = await client.fetch_kpis()

        assert len(result) == 1
        assert route.call_count == 3

    @pytest.mark.asyncio
    @respx.mock
    async def test_raises_after_max_retries(self, client: KoladaClient) -> None:
        """Test that exception is raised after max retries."""
        route = respx.get("https://api.kolada.se/v2/kpi")
        route.side_effect = [
            httpx.Response(503),
            httpx.Response(503),
            httpx.Response(503),
        ]

        with pytest.raises(httpx.HTTPStatusError):
            await client.fetch_kpis()

        assert route.call_count == 3

    @pytest.mark.asyncio
    @respx.mock
    async def test_handles_empty_response(self, client: KoladaClient) -> None:
        """Test handling of empty response."""
        respx.get("https://api.kolada.se/v2/kpi").mock(
            return_value=httpx.Response(200, json={"count": 0, "values": []})
        )

        result = await client.fetch_kpis()

        assert len(result) == 0

    @pytest.mark.asyncio
    @respx.mock
    async def test_url_building_for_data(self, client: KoladaClient) -> None:
        """Test correct URL building for data fetching."""
        respx.get("https://api.kolada.se/v2/data/kpi/N00945/municipality/0180/year/2023").mock(
            return_value=httpx.Response(200, json={"count": 0, "values": []})
        )

        await client.fetch_data(
            kpi_id="N00945",
            municipality_ids=["0180"],
            years=[2023],
        )

        # Verify the URL was called correctly
        assert respx.calls.last.request.url.path == "/v2/data/kpi/N00945/municipality/0180/year/2023"

    @pytest.mark.asyncio
    @respx.mock
    async def test_prevents_infinite_pagination_loop(self, client: KoladaClient) -> None:
        """Test that client prevents infinite pagination loops."""
        # Response that references itself
        response_data = {
            "count": 1,
            "next_page": "https://api.kolada.se/v2/kpi",
            "values": [{"id": "N00001", "title": "KPI"}],
        }
        respx.get("https://api.kolada.se/v2/kpi").mock(
            return_value=httpx.Response(200, json=response_data)
        )

        result = await client.fetch_kpis()

        # Should only fetch once and not loop infinitely
        assert len(result) == 1

    @pytest.mark.asyncio
    @respx.mock
    async def test_fetch_data_without_years(self, client: KoladaClient) -> None:
        """Test data fetching without specifying years."""
        response_data = {
            "count": 1,
            "values": [
                {
                    "kpi": "N00945",
                    "municipality": "0180",
                    "period": 2023,
                    "values": [{"gender": "T", "value": 984748, "count": 1}],
                },
            ],
        }
        respx.get("https://api.kolada.se/v2/data/kpi/N00945/municipality/0180").mock(
            return_value=httpx.Response(200, json=response_data)
        )

        result = await client.fetch_data(
            kpi_id="N00945",
            municipality_ids=["0180"],
        )

        assert len(result) == 1


class TestKoladaClientURLBuilding:
    """Tests for URL building in KoladaClient."""

    @pytest.fixture
    def client(self) -> KoladaClient:
        """Create a KoladaClient instance for testing."""
        return KoladaClient(base_url="https://api.kolada.se/v2")

    def test_build_data_url_single_municipality_single_year(
        self, client: KoladaClient
    ) -> None:
        """Test URL building for single municipality and year."""
        url = client._build_data_url("N00945", ["0180"], [2023])
        assert url == "https://api.kolada.se/v2/data/kpi/N00945/municipality/0180/year/2023"

    def test_build_data_url_multiple_municipalities(
        self, client: KoladaClient
    ) -> None:
        """Test URL building for multiple municipalities."""
        url = client._build_data_url("N00945", ["0180", "1480", "1280"], [2023])
        assert url == "https://api.kolada.se/v2/data/kpi/N00945/municipality/0180,1480,1280/year/2023"

    def test_build_data_url_multiple_years(self, client: KoladaClient) -> None:
        """Test URL building for multiple years."""
        url = client._build_data_url("N00945", ["0180"], [2021, 2022, 2023])
        assert url == "https://api.kolada.se/v2/data/kpi/N00945/municipality/0180/year/2021,2022,2023"

    def test_build_data_url_no_years(self, client: KoladaClient) -> None:
        """Test URL building without years."""
        url = client._build_data_url("N00945", ["0180"], None)
        assert url == "https://api.kolada.se/v2/data/kpi/N00945/municipality/0180"
