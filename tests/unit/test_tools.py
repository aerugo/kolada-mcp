"""Unit tests for MCP tools."""

from unittest.mock import AsyncMock, MagicMock, patch
from typing import Any

import numpy as np
import pytest

from kolada_mcp.models.types import (
    KoladaKpi,
    KoladaMunicipality,
    OperatingAreaSummary,
    ServerContext,
)


@pytest.fixture
def mock_context() -> ServerContext:
    """Create a mock server context."""
    kpis: list[KoladaKpi] = [
        {"id": "N00945", "title": "Invånare totalt", "operating_area": "Befolkning"},
        {"id": "N01951", "title": "Förvärvsarbetande", "operating_area": "Arbetsmarknad"},
        {"id": "U09400", "title": "Elever kunskapskrav", "operating_area": "Utbildning"},
    ]
    municipalities: list[KoladaMunicipality] = [
        {"id": "0180", "title": "Stockholm", "type": "K"},
        {"id": "1480", "title": "Göteborg", "type": "K"},
        {"id": "01", "title": "Region Stockholm", "type": "R"},
    ]
    operating_areas: list[OperatingAreaSummary] = [
        {"operating_area": "Befolkning", "kpi_count": 100},
        {"operating_area": "Arbetsmarknad", "kpi_count": 150},
        {"operating_area": "Utbildning", "kpi_count": 200},
    ]

    # Create mock sentence model
    mock_model = MagicMock()
    mock_model.encode.return_value = np.random.randn(1, 768).astype(np.float32)

    # Create embeddings
    embeddings = np.random.randn(3, 768).astype(np.float32)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / norms

    return ServerContext(
        kpis=kpis,
        kpi_map={kpi["id"]: kpi for kpi in kpis},
        operating_areas_summary=operating_areas,
        municipalities=municipalities,
        municipality_map={m["id"]: m for m in municipalities},
        embeddings=embeddings,
        embedding_ids=["N00945", "N01951", "U09400"],
        sentence_model=mock_model,
    )


class TestMetadataTools:
    """Tests for metadata tools."""

    @pytest.mark.asyncio
    async def test_list_operating_areas(self, mock_context: ServerContext) -> None:
        """Test listing operating areas."""
        from kolada_mcp.tools.metadata import list_operating_areas

        result = await list_operating_areas(mock_context)

        assert "areas" in result
        assert len(result["areas"]) == 3
        assert result["areas"][0]["operating_area"] == "Befolkning"

    @pytest.mark.asyncio
    async def test_get_kpis_by_operating_area(self, mock_context: ServerContext) -> None:
        """Test getting KPIs by operating area."""
        from kolada_mcp.tools.metadata import get_kpis_by_operating_area

        result = await get_kpis_by_operating_area(mock_context, "Befolkning")

        assert "kpis" in result
        assert len(result["kpis"]) == 1
        assert result["kpis"][0]["id"] == "N00945"

    @pytest.mark.asyncio
    async def test_get_kpis_by_operating_area_case_insensitive(
        self, mock_context: ServerContext
    ) -> None:
        """Test that operating area matching is case insensitive."""
        from kolada_mcp.tools.metadata import get_kpis_by_operating_area

        result = await get_kpis_by_operating_area(mock_context, "befolkning")

        assert len(result["kpis"]) == 1

    @pytest.mark.asyncio
    async def test_get_kpi_metadata(self, mock_context: ServerContext) -> None:
        """Test getting KPI metadata."""
        from kolada_mcp.tools.metadata import get_kpi_metadata

        result = await get_kpi_metadata(mock_context, "N00945")

        assert result["id"] == "N00945"
        assert result["title"] == "Invånare totalt"

    @pytest.mark.asyncio
    async def test_get_kpi_metadata_not_found(self, mock_context: ServerContext) -> None:
        """Test getting metadata for non-existent KPI."""
        from kolada_mcp.tools.metadata import get_kpi_metadata

        result = await get_kpi_metadata(mock_context, "INVALID")

        assert "error" in result

    @pytest.mark.asyncio
    async def test_search_kpis(self, mock_context: ServerContext) -> None:
        """Test semantic search for KPIs."""
        from kolada_mcp.tools.metadata import search_kpis

        result = await search_kpis(mock_context, "befolkning", limit=2)

        assert "results" in result
        assert len(result["results"]) <= 2
        assert all("id" in r and "title" in r and "score" in r for r in result["results"])


class TestMunicipalityTools:
    """Tests for municipality tools."""

    @pytest.mark.asyncio
    async def test_list_municipalities(self, mock_context: ServerContext) -> None:
        """Test listing all municipalities."""
        from kolada_mcp.tools.municipality import list_municipalities

        result = await list_municipalities(mock_context)

        assert "municipalities" in result
        assert len(result["municipalities"]) == 3

    @pytest.mark.asyncio
    async def test_list_municipalities_filter_by_type(
        self, mock_context: ServerContext
    ) -> None:
        """Test filtering municipalities by type."""
        from kolada_mcp.tools.municipality import list_municipalities

        result = await list_municipalities(mock_context, municipality_type="K")

        assert len(result["municipalities"]) == 2
        assert all(m["type"] == "K" for m in result["municipalities"])

    @pytest.mark.asyncio
    async def test_list_municipalities_filter_region(
        self, mock_context: ServerContext
    ) -> None:
        """Test filtering to only regions."""
        from kolada_mcp.tools.municipality import list_municipalities

        result = await list_municipalities(mock_context, municipality_type="R")

        assert len(result["municipalities"]) == 1
        assert result["municipalities"][0]["id"] == "01"


class TestDataTools:
    """Tests for data tools."""

    @pytest.mark.asyncio
    async def test_fetch_kolada_data(self, mock_context: ServerContext) -> None:
        """Test fetching Kolada data."""
        from kolada_mcp.tools.data import fetch_kolada_data

        mock_response = {
            "count": 1,
            "values": [
                {
                    "kpi": "N00945",
                    "municipality": "0180",
                    "period": 2023,
                    "values": [{"gender": "T", "value": 984748, "count": 1}],
                }
            ],
        }

        with patch(
            "kolada_mcp.tools.data.KoladaClient.fetch_data",
            new_callable=AsyncMock,
            return_value=mock_response["values"],
        ):
            result = await fetch_kolada_data(
                mock_context,
                kpi_id="N00945",
                municipality_ids="0180",
            )

            assert "data" in result
            assert len(result["data"]) == 1
            assert result["data"][0]["kpi_id"] == "N00945"

    @pytest.mark.asyncio
    async def test_fetch_kolada_data_multiple_municipalities(
        self, mock_context: ServerContext
    ) -> None:
        """Test fetching data for multiple municipalities."""
        from kolada_mcp.tools.data import fetch_kolada_data

        mock_response_values = [
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
        ]

        with patch(
            "kolada_mcp.tools.data.KoladaClient.fetch_data",
            new_callable=AsyncMock,
            return_value=mock_response_values,
        ):
            result = await fetch_kolada_data(
                mock_context,
                kpi_id="N00945",
                municipality_ids="0180,1480",
            )

            assert len(result["data"]) == 2

    @pytest.mark.asyncio
    async def test_fetch_kolada_data_invalid_municipality(
        self, mock_context: ServerContext
    ) -> None:
        """Test fetching data for invalid municipality."""
        from kolada_mcp.tools.data import fetch_kolada_data

        result = await fetch_kolada_data(
            mock_context,
            kpi_id="N00945",
            municipality_ids="INVALID",
        )

        assert "error" in result


class TestComparisonTools:
    """Tests for comparison tools."""

    @pytest.mark.asyncio
    async def test_compare_kpis_single_year(self, mock_context: ServerContext) -> None:
        """Test comparing two KPIs for a single year."""
        from kolada_mcp.tools.comparison import compare_kpis

        mock_data_1 = [
            {
                "kpi": "N00945",
                "municipality": "0180",
                "period": 2023,
                "values": [{"gender": "T", "value": 100, "count": 1}],
            }
        ]
        mock_data_2 = [
            {
                "kpi": "N01951",
                "municipality": "0180",
                "period": 2023,
                "values": [{"gender": "T", "value": 75, "count": 1}],
            }
        ]

        with patch(
            "kolada_mcp.tools.comparison.KoladaClient.fetch_data",
            new_callable=AsyncMock,
            side_effect=[mock_data_1, mock_data_2],
        ):
            result = await compare_kpis(
                mock_context,
                kpi_id_1="N00945",
                kpi_id_2="N01951",
                years="2023",
                municipality_ids="0180",
            )

            assert "comparison" in result
            assert len(result["comparison"]) == 1
            assert result["comparison"][0]["difference"] == 25
