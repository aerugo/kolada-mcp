"""Unit tests for data models."""

from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from kolada_mcp.models.types import (
    KoladaKpi,
    KoladaMunicipality,
    OperatingAreaSummary,
    ServerContext,
)


class TestKoladaKpi:
    """Tests for KoladaKpi TypedDict."""

    def test_create_kpi_with_required_fields(self) -> None:
        """Test creating a KPI with only required fields."""
        kpi: KoladaKpi = {
            "id": "N00945",
            "title": "Invånare totalt, antal",
        }
        assert kpi["id"] == "N00945"
        assert kpi["title"] == "Invånare totalt, antal"

    def test_create_kpi_with_all_fields(self) -> None:
        """Test creating a KPI with all fields."""
        kpi: KoladaKpi = {
            "id": "N00945",
            "title": "Invånare totalt, antal",
            "description": "Total antal invånare i kommunen",
            "operating_area": "Befolkning",
        }
        assert kpi["id"] == "N00945"
        assert kpi["title"] == "Invånare totalt, antal"
        assert kpi["description"] == "Total antal invånare i kommunen"
        assert kpi["operating_area"] == "Befolkning"

    def test_kpi_optional_fields_can_be_none(self) -> None:
        """Test that optional fields can be None."""
        kpi: KoladaKpi = {
            "id": "N00945",
            "title": "Invånare totalt, antal",
            "description": None,
            "operating_area": None,
        }
        assert kpi["description"] is None
        assert kpi["operating_area"] is None


class TestKoladaMunicipality:
    """Tests for KoladaMunicipality TypedDict."""

    def test_create_municipality(self) -> None:
        """Test creating a municipality."""
        municipality: KoladaMunicipality = {
            "id": "0180",
            "title": "Stockholm",
            "type": "K",
        }
        assert municipality["id"] == "0180"
        assert municipality["title"] == "Stockholm"
        assert municipality["type"] == "K"

    def test_municipality_type_kommun(self) -> None:
        """Test municipality type K (kommun)."""
        municipality: KoladaMunicipality = {
            "id": "0180",
            "title": "Stockholm",
            "type": "K",
        }
        assert municipality["type"] == "K"

    def test_municipality_type_region(self) -> None:
        """Test municipality type R (region)."""
        municipality: KoladaMunicipality = {
            "id": "01",
            "title": "Region Stockholm",
            "type": "R",
        }
        assert municipality["type"] == "R"

    def test_municipality_type_landsting(self) -> None:
        """Test municipality type L (landsting/county council)."""
        municipality: KoladaMunicipality = {
            "id": "01L",
            "title": "Stockholms läns landsting",
            "type": "L",
        }
        assert municipality["type"] == "L"


class TestOperatingAreaSummary:
    """Tests for OperatingAreaSummary TypedDict."""

    def test_create_operating_area_summary(self) -> None:
        """Test creating an operating area summary."""
        summary: OperatingAreaSummary = {
            "operating_area": "Befolkning",
            "kpi_count": 150,
        }
        assert summary["operating_area"] == "Befolkning"
        assert summary["kpi_count"] == 150


class TestServerContext:
    """Tests for ServerContext dataclass."""

    def test_create_server_context(self) -> None:
        """Test creating a server context with all required fields."""
        kpis: list[KoladaKpi] = [
            {"id": "N00945", "title": "Test KPI"}
        ]
        municipalities: list[KoladaMunicipality] = [
            {"id": "0180", "title": "Stockholm", "type": "K"}
        ]
        operating_areas: list[OperatingAreaSummary] = [
            {"operating_area": "Test", "kpi_count": 1}
        ]
        mock_model = MagicMock()
        embeddings = np.array([[0.1, 0.2, 0.3]])
        embedding_ids = ["N00945"]

        ctx = ServerContext(
            kpis=kpis,
            kpi_map={"N00945": kpis[0]},
            operating_areas_summary=operating_areas,
            municipalities=municipalities,
            municipality_map={"0180": municipalities[0]},
            embeddings=embeddings,
            embedding_ids=embedding_ids,
            sentence_model=mock_model,
        )

        assert ctx.kpis == kpis
        assert ctx.kpi_map["N00945"] == kpis[0]
        assert ctx.municipalities == municipalities
        assert ctx.municipality_map["0180"] == municipalities[0]
        assert ctx.operating_areas_summary == operating_areas
        assert np.array_equal(ctx.embeddings, embeddings)
        assert ctx.embedding_ids == embedding_ids
        assert ctx.sentence_model == mock_model

    def test_server_context_kpi_lookup(self) -> None:
        """Test looking up KPI by ID in server context."""
        kpis: list[KoladaKpi] = [
            {"id": "N00945", "title": "KPI 1"},
            {"id": "N01951", "title": "KPI 2"},
        ]
        ctx = ServerContext(
            kpis=kpis,
            kpi_map={kpi["id"]: kpi for kpi in kpis},
            operating_areas_summary=[],
            municipalities=[],
            municipality_map={},
            embeddings=np.array([]),
            embedding_ids=[],
            sentence_model=MagicMock(),
        )

        assert ctx.kpi_map["N00945"]["title"] == "KPI 1"
        assert ctx.kpi_map["N01951"]["title"] == "KPI 2"
        assert ctx.kpi_map.get("INVALID") is None

    def test_server_context_municipality_lookup(self) -> None:
        """Test looking up municipality by ID in server context."""
        municipalities: list[KoladaMunicipality] = [
            {"id": "0180", "title": "Stockholm", "type": "K"},
            {"id": "01", "title": "Region Stockholm", "type": "R"},
        ]
        ctx = ServerContext(
            kpis=[],
            kpi_map={},
            operating_areas_summary=[],
            municipalities=municipalities,
            municipality_map={m["id"]: m for m in municipalities},
            embeddings=np.array([]),
            embedding_ids=[],
            sentence_model=MagicMock(),
        )

        assert ctx.municipality_map["0180"]["title"] == "Stockholm"
        assert ctx.municipality_map["01"]["type"] == "R"
        assert ctx.municipality_map.get("INVALID") is None

    def test_server_context_empty_state(self) -> None:
        """Test server context with empty/minimal state."""
        ctx = ServerContext(
            kpis=[],
            kpi_map={},
            operating_areas_summary=[],
            municipalities=[],
            municipality_map={},
            embeddings=np.array([]),
            embedding_ids=[],
            sentence_model=MagicMock(),
        )

        assert len(ctx.kpis) == 0
        assert len(ctx.municipalities) == 0
        assert len(ctx.kpi_map) == 0
