"""Kolada MCP Server implementation.

This module provides the FastMCP server with all registered tools
and lifecycle management.
"""

import asyncio
import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

from fastmcp import FastMCP

from kolada_mcp.config import settings
from kolada_mcp.models.types import (
    KoladaKpi,
    KoladaMunicipality,
    OperatingAreaSummary,
    ServerContext,
)
from kolada_mcp.services.kolada_client import KoladaClient
from kolada_mcp.services.embeddings import EmbeddingsService, load_sentence_model
from kolada_mcp.services.data_processing import DataProcessor

logger = logging.getLogger(__name__)

# Global cached context for fast restarts
_cached_ctx: ServerContext | None = None
_init_lock = asyncio.Lock()


async def _load_kpis(client: KoladaClient) -> list[KoladaKpi]:
    """Load all KPIs from the Kolada API."""
    logger.info("[Kolada MCP] Loading KPIs from Kolada API...")
    kpis = await client.fetch_kpis()
    logger.info(f"[Kolada MCP] Loaded {len(kpis)} KPIs")
    return kpis


async def _load_municipalities(client: KoladaClient) -> list[KoladaMunicipality]:
    """Load all municipalities from the Kolada API."""
    logger.info("[Kolada MCP] Loading municipalities from Kolada API...")
    municipalities = await client.fetch_municipalities()
    logger.info(f"[Kolada MCP] Loaded {len(municipalities)} municipalities")
    return municipalities


async def _create_context() -> ServerContext:
    """Create the server context with all cached data."""
    global _cached_ctx

    async with _init_lock:
        # Check if already initialized
        if _cached_ctx is not None:
            logger.info("[Kolada MCP] Using cached context")
            return _cached_ctx

        logger.info("[Kolada MCP] Initializing server context...")

        client = KoladaClient()

        try:
            # Load KPIs and municipalities concurrently
            kpis, municipalities = await asyncio.gather(
                _load_kpis(client),
                _load_municipalities(client),
            )
        except Exception as e:
            logger.error(f"[Kolada MCP] Failed to load data from API: {e}")
            # Return minimal context on failure
            return ServerContext(
                kpis=[],
                kpi_map={},
                operating_areas_summary=[],
                municipalities=[],
                municipality_map={},
                embeddings=None,  # type: ignore
                embedding_ids=[],
                sentence_model=None,  # type: ignore
            )

        # Create indexes
        kpi_map = {kpi["id"]: kpi for kpi in kpis}
        municipality_map = {m["id"]: m for m in municipalities}

        # Compute operating areas summary
        operating_areas_summary = DataProcessor.compute_operating_areas_summary(kpis)
        logger.info(f"[Kolada MCP] Found {len(operating_areas_summary)} operating areas")

        # Load sentence model and embeddings
        try:
            logger.info("[Kolada MCP] Loading sentence transformer model...")
            sentence_model = await asyncio.to_thread(load_sentence_model)

            embeddings_service = EmbeddingsService(
                model=sentence_model,
                cache_path=settings.embeddings_cache_path,
            )
            embeddings, embedding_ids = await embeddings_service.load_or_create_embeddings(kpis)
            logger.info(f"[Kolada MCP] Loaded embeddings for {len(embedding_ids)} KPIs")
        except Exception as e:
            logger.error(f"[Kolada MCP] Failed to load embeddings: {e}")
            sentence_model = None  # type: ignore
            embeddings = None  # type: ignore
            embedding_ids = []

        ctx = ServerContext(
            kpis=kpis,
            kpi_map=kpi_map,
            operating_areas_summary=operating_areas_summary,
            municipalities=municipalities,
            municipality_map=municipality_map,
            embeddings=embeddings,
            embedding_ids=embedding_ids,
            sentence_model=sentence_model,
        )

        _cached_ctx = ctx
        logger.info("[Kolada MCP] Server context initialized successfully")
        return ctx


@asynccontextmanager
async def app_lifespan(app: FastMCP) -> AsyncGenerator[ServerContext, None]:
    """Lifespan context manager for the MCP server.

    This handles initialization at startup and cleanup at shutdown.
    """
    logger.info("[Kolada MCP] Starting server...")
    ctx = await _create_context()
    yield ctx
    logger.info("[Kolada MCP] Server shutting down")


# Create the FastMCP server instance
mcp = FastMCP(
    "KoladaServer",
    description="MCP server for Sweden's Kolada municipal statistics API",
    lifespan=app_lifespan,
)


# Register tools
@mcp.tool()
async def list_operating_areas(ctx: ServerContext) -> dict[str, Any]:
    """List all operating areas (KPI categories) with counts.

    Returns thematic categories of KPIs available in the Kolada database.
    """
    from kolada_mcp.tools.metadata import list_operating_areas as _list_operating_areas
    return await _list_operating_areas(ctx)


@mcp.tool()
async def get_kpis_by_operating_area(
    ctx: ServerContext, operating_area: str
) -> dict[str, Any]:
    """Get all KPIs within a specific operating area.

    Args:
        operating_area: The operating area/category to filter by (e.g., "Befolkning")
    """
    from kolada_mcp.tools.metadata import get_kpis_by_operating_area as _get_kpis
    return await _get_kpis(ctx, operating_area)


@mcp.tool()
async def get_kpi_metadata(ctx: ServerContext, kpi_id: str) -> dict[str, Any]:
    """Get detailed metadata for a specific KPI.

    Args:
        kpi_id: The unique KPI identifier (e.g., "N00945")
    """
    from kolada_mcp.tools.metadata import get_kpi_metadata as _get_metadata
    return await _get_metadata(ctx, kpi_id)


@mcp.tool()
async def search_kpis(
    ctx: ServerContext, query: str, limit: int = 20
) -> dict[str, Any]:
    """Search for KPIs using semantic similarity.

    Uses embeddings to find KPIs semantically similar to the query.

    Args:
        query: Natural language search query
        limit: Maximum number of results (default: 20)
    """
    from kolada_mcp.tools.metadata import search_kpis as _search
    return await _search(ctx, query, limit)


@mcp.tool()
async def list_municipalities(
    ctx: ServerContext, municipality_type: str = ""
) -> dict[str, Any]:
    """List all municipalities or filter by type.

    Args:
        municipality_type: "K" for kommun, "R" for region, "L" for landsting, "" for all
    """
    from kolada_mcp.tools.municipality import list_municipalities as _list
    return await _list(ctx, municipality_type)


@mcp.tool()
async def filter_municipalities_by_kpi(
    ctx: ServerContext,
    kpi_id: str,
    cutoff_value: float,
    comparison: str = "above",
    year: int | None = None,
    municipality_type: str = "",
    gender: str = "T",
) -> dict[str, Any]:
    """Filter municipalities based on KPI value threshold.

    Args:
        kpi_id: The KPI identifier
        cutoff_value: The threshold value for filtering
        comparison: "above" or "below" the cutoff
        year: Specific year (default: latest available)
        municipality_type: "K", "R", "L", or "" for all
        gender: "T" (total), "M" (male), "K" (female)
    """
    from kolada_mcp.tools.municipality import filter_municipalities_by_kpi as _filter
    return await _filter(ctx, kpi_id, cutoff_value, comparison, year, municipality_type, gender)


@mcp.tool()
async def fetch_kolada_data(
    ctx: ServerContext,
    kpi_id: str,
    municipality_ids: str,
    years: str | None = None,
    municipality_type: str = "",
    gender: str | None = None,
) -> dict[str, Any]:
    """Fetch raw KPI data for specified municipalities.

    Args:
        kpi_id: The KPI identifier
        municipality_ids: Comma-separated municipality IDs (e.g., "0180,1480")
        years: Optional comma-separated years (e.g., "2022,2023")
        municipality_type: Optional filter by type
        gender: Optional filter by gender ("T", "M", "K")
    """
    from kolada_mcp.tools.data import fetch_kolada_data as _fetch
    return await _fetch(ctx, kpi_id, municipality_ids, years, municipality_type, gender)


@mcp.tool()
async def analyze_kpi_across_municipalities(
    ctx: ServerContext,
    kpi_id: str,
    year: int | None = None,
    municipality_type: str = "",
    gender: str = "T",
    sort_direction: str = "desc",
    limit: int | None = None,
    municipality_ids: str | None = None,
) -> dict[str, Any]:
    """Analyze a KPI across multiple municipalities with rankings and statistics.

    Args:
        kpi_id: The KPI identifier
        year: Specific year (default: latest available)
        municipality_type: "K", "R", "L", or "" for all
        gender: "T" (total), "M" (male), "K" (female)
        sort_direction: "asc" or "desc"
        limit: Maximum number of results
        municipality_ids: Optional specific municipalities to analyze
    """
    from kolada_mcp.tools.data import analyze_kpi_across_municipalities as _analyze
    return await _analyze(
        ctx, kpi_id, year, municipality_type, gender, sort_direction, limit, municipality_ids
    )


@mcp.tool()
async def compare_kpis(
    ctx: ServerContext,
    kpi_id_1: str,
    kpi_id_2: str,
    years: str,
    municipality_ids: str | None = None,
    municipality_type: str = "",
    gender: str = "T",
) -> dict[str, Any]:
    """Compare two KPIs across municipalities.

    Single year: calculates difference. Multiple years: calculates correlation.

    Args:
        kpi_id_1: First KPI identifier
        kpi_id_2: Second KPI identifier
        years: Comma-separated years (e.g., "2023" or "2020,2021,2022,2023")
        municipality_ids: Optional specific municipalities
        municipality_type: "K", "R", "L", or "" for all
        gender: "T" (total), "M" (male), "K" (female)
    """
    from kolada_mcp.tools.comparison import compare_kpis as _compare
    return await _compare(ctx, kpi_id_1, kpi_id_2, years, municipality_ids, municipality_type, gender)


def get_app() -> FastMCP:
    """Get the FastMCP application instance."""
    return mcp


# Health check endpoints for HTTP mode
def add_health_endpoints(app: Any) -> None:
    """Add health check endpoints to the ASGI app."""
    from starlette.responses import JSONResponse
    from starlette.routing import Route

    async def health_live(request: Any) -> JSONResponse:
        """Liveness probe - server is running."""
        return JSONResponse({"status": "ok"})

    async def health_ready(request: Any) -> JSONResponse:
        """Readiness probe - server is ready to handle requests."""
        if _cached_ctx is None:
            return JSONResponse({"status": "initializing"}, status_code=503)
        if not _cached_ctx.kpis:
            return JSONResponse({"status": "no_data"}, status_code=503)
        return JSONResponse({
            "status": "ready",
            "kpis_loaded": len(_cached_ctx.kpis),
            "municipalities_loaded": len(_cached_ctx.municipalities),
        })

    # Add routes to the app
    if hasattr(app, "routes"):
        app.routes.extend([
            Route("/health/live", health_live),
            Route("/health/ready", health_ready),
        ])
