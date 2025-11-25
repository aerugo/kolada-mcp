"""Data models for Kolada MCP Server.

This module defines the core data structures used throughout the application.
We use TypedDict for API response types and dataclasses for internal state,
following the composition-over-inheritance principle.
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, TypedDict

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


class KoladaKpi(TypedDict, total=False):
    """Represents a Key Performance Indicator from the Kolada API.

    The Kolada database contains approximately 6,500 KPIs covering various
    aspects of Swedish municipal and regional governance.

    Attributes:
        id: Unique identifier (e.g., "N00945")
        title: Human-readable title
        description: Optional detailed description
        operating_area: Optional category/theme (e.g., "Befolkning", "Utbildning")
    """

    id: str
    title: str
    description: str | None
    operating_area: str | None


class KoladaMunicipality(TypedDict):
    """Represents a Swedish municipality or region from the Kolada API.

    Attributes:
        id: Unique identifier (e.g., "0180" for Stockholm)
        title: Municipality/region name
        type: Classification type:
            - "K" = Kommun (municipality)
            - "R" = Region
            - "L" = Landsting (county council, historical)
    """

    id: str
    title: str
    type: str


class OperatingAreaSummary(TypedDict):
    """Summary of KPIs grouped by operating area.

    Attributes:
        operating_area: The category name
        kpi_count: Number of KPIs in this category
    """

    operating_area: str
    kpi_count: int


class KoladaDataValue(TypedDict):
    """A single data value from the Kolada API.

    Attributes:
        gender: Gender category ("T" = Total, "M" = Male, "K" = Female)
        value: The numeric value (can be None if no data)
        count: Number of data points aggregated
    """

    gender: str
    value: float | None
    count: int


class KoladaDataPoint(TypedDict):
    """A data point from the Kolada API containing KPI values.

    Attributes:
        kpi: KPI identifier
        municipality: Municipality identifier
        period: Year of the data
        values: List of values by gender
    """

    kpi: str
    municipality: str
    period: int
    values: list[KoladaDataValue]


class KoladaApiResponse(TypedDict, total=False):
    """Generic response from the Kolada API.

    Attributes:
        count: Total number of items
        values: List of response items
        next_page: URL for the next page of results (if paginated)
    """

    count: int
    values: list[Any]
    next_page: str | None


@dataclass
class ServerContext:
    """Server-wide context containing cached data loaded at startup.

    This context is initialized during server startup and provides
    efficient access to KPI metadata, municipality data, and semantic
    search capabilities.

    Attributes:
        kpis: List of all KPIs
        kpi_map: Dictionary mapping KPI IDs to KPI objects
        operating_areas_summary: Summary of KPIs by operating area
        municipalities: List of all municipalities
        municipality_map: Dictionary mapping municipality IDs to objects
        embeddings: Pre-computed KPI title embeddings for semantic search
        embedding_ids: KPI IDs corresponding to embeddings
        sentence_model: Loaded sentence transformer model
    """

    kpis: list[KoladaKpi]
    kpi_map: dict[str, KoladaKpi]
    operating_areas_summary: list[OperatingAreaSummary]
    municipalities: list[KoladaMunicipality]
    municipality_map: dict[str, KoladaMunicipality]
    embeddings: NDArray[np.float32]
    embedding_ids: list[str]
    sentence_model: "SentenceTransformer"

    def get_kpi(self, kpi_id: str) -> KoladaKpi | None:
        """Look up a KPI by its ID.

        Args:
            kpi_id: The KPI identifier to look up

        Returns:
            The KPI object if found, None otherwise
        """
        return self.kpi_map.get(kpi_id)

    def get_municipality(self, municipality_id: str) -> KoladaMunicipality | None:
        """Look up a municipality by its ID.

        Args:
            municipality_id: The municipality identifier to look up

        Returns:
            The municipality object if found, None otherwise
        """
        return self.municipality_map.get(municipality_id)

    def filter_municipalities_by_type(
        self, municipality_type: str
    ) -> list[KoladaMunicipality]:
        """Filter municipalities by type.

        Args:
            municipality_type: Type to filter by ("K", "R", "L", or "" for all)

        Returns:
            List of municipalities matching the type
        """
        if not municipality_type:
            return self.municipalities
        return [m for m in self.municipalities if m["type"] == municipality_type]


@dataclass
class FlatDataRow:
    """A flattened data row for DataFrame compatibility.

    This structure is used when converting Kolada API responses to
    tabular format suitable for Polars DataFrames.

    Attributes:
        kpi_id: KPI identifier
        municipality_id: Municipality identifier
        municipality_name: Municipality name
        period: Year
        gender: Gender category
        value: The numeric value
        count: Number of data points
    """

    kpi_id: str
    municipality_id: str
    municipality_name: str
    period: int
    gender: str
    value: float | None
    count: int
