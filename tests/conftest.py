"""Shared pytest fixtures for Kolada MCP Server tests."""

from collections.abc import AsyncGenerator, Generator
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import httpx
import numpy as np
import pytest
import respx

from kolada_mcp.models.types import (
    KoladaKpi,
    KoladaMunicipality,
    OperatingAreaSummary,
    ServerContext,
)
from kolada_mcp.services.kolada_client import KoladaClient


# Sample test data
SAMPLE_KPIS: list[KoladaKpi] = [
    {
        "id": "N00945",
        "title": "Invånare totalt, antal",
        "description": "Total antal invånare i kommunen",
        "operating_area": "Befolkning",
    },
    {
        "id": "N01951",
        "title": "Andel förvärvsarbetande invånare 20-64 år, %",
        "description": "Andel av befolkningen som förvärvsarbetar",
        "operating_area": "Arbetsmarknad",
    },
    {
        "id": "U09400",
        "title": "Elever i åk 9 som uppnått kunskapskraven, andel (%)",
        "description": "Andel elever som uppnått kunskapskraven i alla ämnen",
        "operating_area": "Utbildning",
    },
]

SAMPLE_MUNICIPALITIES: list[KoladaMunicipality] = [
    {"id": "0180", "title": "Stockholm", "type": "K"},
    {"id": "1480", "title": "Göteborg", "type": "K"},
    {"id": "1280", "title": "Malmö", "type": "K"},
    {"id": "01", "title": "Region Stockholm", "type": "R"},
    {"id": "14", "title": "Västra Götalandsregionen", "type": "R"},
]

SAMPLE_OPERATING_AREAS: list[OperatingAreaSummary] = [
    {"operating_area": "Befolkning", "kpi_count": 150},
    {"operating_area": "Arbetsmarknad", "kpi_count": 200},
    {"operating_area": "Utbildning", "kpi_count": 300},
]


@pytest.fixture
def sample_kpis() -> list[KoladaKpi]:
    """Return sample KPI data."""
    return SAMPLE_KPIS.copy()


@pytest.fixture
def sample_municipalities() -> list[KoladaMunicipality]:
    """Return sample municipality data."""
    return SAMPLE_MUNICIPALITIES.copy()


@pytest.fixture
def sample_operating_areas() -> list[OperatingAreaSummary]:
    """Return sample operating area summaries."""
    return SAMPLE_OPERATING_AREAS.copy()


@pytest.fixture
def sample_embeddings() -> tuple[np.ndarray, list[str]]:
    """Return sample embeddings data."""
    # Create fake embeddings (3 KPIs, 768-dimensional vectors)
    embeddings = np.random.randn(3, 768).astype(np.float32)
    # Normalize embeddings
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / norms
    kpi_ids = ["N00945", "N01951", "U09400"]
    return embeddings, kpi_ids


@pytest.fixture
def mock_server_context(
    sample_kpis: list[KoladaKpi],
    sample_municipalities: list[KoladaMunicipality],
    sample_operating_areas: list[OperatingAreaSummary],
    sample_embeddings: tuple[np.ndarray, list[str]],
) -> ServerContext:
    """Return a mock server context with sample data."""
    embeddings, kpi_ids = sample_embeddings

    # Create mock sentence model
    mock_model = MagicMock()
    mock_model.encode.return_value = np.random.randn(768).astype(np.float32)

    return ServerContext(
        kpis=sample_kpis,
        kpi_map={kpi["id"]: kpi for kpi in sample_kpis},
        operating_areas_summary=sample_operating_areas,
        municipalities=sample_municipalities,
        municipality_map={m["id"]: m for m in sample_municipalities},
        embeddings=embeddings,
        embedding_ids=kpi_ids,
        sentence_model=mock_model,
    )


@pytest.fixture
def mock_kolada_client() -> AsyncMock:
    """Return a mock Kolada API client."""
    client = AsyncMock(spec=KoladaClient)
    return client


@pytest.fixture
def respx_mock() -> Generator[respx.MockRouter, None, None]:
    """Return a respx mock router for HTTP testing."""
    with respx.mock(assert_all_called=False) as router:
        yield router


@pytest.fixture
async def http_client() -> AsyncGenerator[httpx.AsyncClient, None]:
    """Return an httpx async client for testing."""
    async with httpx.AsyncClient() as client:
        yield client


# API Response fixtures
@pytest.fixture
def kolada_kpi_response() -> dict[str, Any]:
    """Return a sample Kolada KPI API response."""
    return {
        "count": 3,
        "values": [
            {
                "id": "N00945",
                "title": "Invånare totalt, antal",
                "description": "Total antal invånare i kommunen",
                "operating_area": "Befolkning",
            },
            {
                "id": "N01951",
                "title": "Andel förvärvsarbetande invånare 20-64 år, %",
                "description": "Andel av befolkningen som förvärvsarbetar",
                "operating_area": "Arbetsmarknad",
            },
            {
                "id": "U09400",
                "title": "Elever i åk 9 som uppnått kunskapskraven, andel (%)",
                "description": "Andel elever som uppnått kunskapskraven i alla ämnen",
                "operating_area": "Utbildning",
            },
        ],
    }


@pytest.fixture
def kolada_municipality_response() -> dict[str, Any]:
    """Return a sample Kolada municipality API response."""
    return {
        "count": 5,
        "values": [
            {"id": "0180", "title": "Stockholm", "type": "K"},
            {"id": "1480", "title": "Göteborg", "type": "K"},
            {"id": "1280", "title": "Malmö", "type": "K"},
            {"id": "01", "title": "Region Stockholm", "type": "R"},
            {"id": "14", "title": "Västra Götalandsregionen", "type": "R"},
        ],
    }


@pytest.fixture
def kolada_data_response() -> dict[str, Any]:
    """Return a sample Kolada data API response."""
    return {
        "count": 3,
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
            {
                "kpi": "N00945",
                "municipality": "1280",
                "period": 2023,
                "values": [{"gender": "T", "value": 357377, "count": 1}],
            },
        ],
    }
