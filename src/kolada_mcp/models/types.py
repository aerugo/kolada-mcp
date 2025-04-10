from typing import Required, TypedDict

import numpy as np
import numpy.typing as npt
from sentence_transformers import SentenceTransformer


class KoladaKpi(TypedDict, total=False):
    """
    Represents a single Key Performance Indicator (KPI) from Kolada.
    Kolada (Kommun- och landstingsdatabasen) provides ~6,500 KPIs
    covering Swedish municipalities and regions across various sectors
    like economy, schools, healthcare, environment, etc.
    """

    id: Required[str]  # The unique identifier for the KPI (e.g., "N00945")
    title: str  # The human-readable name of the KPI (e.g., "Population size")
    description: str  # A longer explanation of the KPI.
    operating_area: str  # The thematic category/categories (e.g., "Demographics", "Economy,Environment")


class KoladaMunicipality(TypedDict, total=False):
    """
    Represents a single municipality (or region) from Kolada.
    Each entry typically has:
      - 'id': Municipality ID (e.g., "0180" for Stockholm),
      - 'title': The human-readable municipality name (e.g., "Stockholm"),
      - 'type': "K" (kommun), "R" (region), "L" (landsting), etc.
    """

    id: str
    title: str
    type: str


class KoladaLifespanContext(TypedDict):
    """
    Data cached in the server's memory at startup ('lifespan_context').
    This avoids repeatedly fetching static metadata from the Kolada API.
    """

    # KPI data
    kpi_cache: list[KoladaKpi]  # A list of all KPI metadata objects.
    kpi_map: dict[str, KoladaKpi]  # Mapping from KPI ID -> KPI object
    operating_areas_summary: list[dict[str, str | int]]

    # Municipality data
    municipality_cache: list[KoladaMunicipality]
    municipality_map: dict[str, KoladaMunicipality]

    # Vector search additions
    sentence_model: SentenceTransformer  # The loaded embedding model
    kpi_embeddings: npt.NDArray[np.float32]  # The embeddings for all KPIs
    kpi_ids: list[str]  # KPI IDs in the same order as rows in kpi_embeddings
