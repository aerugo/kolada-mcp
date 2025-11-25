"""Data tools for Kolada MCP.

These tools provide access to KPI data retrieval and analysis.
"""

import logging
from typing import Any

from kolada_mcp.models.types import ServerContext
from kolada_mcp.services.kolada_client import KoladaClient
from kolada_mcp.services.data_processing import DataProcessor

logger = logging.getLogger(__name__)


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
        ctx: The server context containing cached data
        kpi_id: The KPI identifier
        municipality_ids: Comma-separated list of municipality IDs
        years: Optional comma-separated list of years
        municipality_type: Optional filter by municipality type
        gender: Optional filter by gender ("T", "M", "K")

    Returns:
        Dictionary with 'data' key containing flattened data rows
    """
    # Validate KPI exists
    kpi = ctx.get_kpi(kpi_id)
    if kpi is None:
        return {"error": f"KPI with id '{kpi_id}' not found"}

    # Parse municipality IDs
    mun_ids = [m.strip() for m in municipality_ids.split(",")]

    # Validate municipalities exist
    invalid_ids = [m for m in mun_ids if m not in ctx.municipality_map]
    if invalid_ids:
        return {"error": f"Invalid municipality IDs: {', '.join(invalid_ids)}"}

    # Apply municipality type filter if specified
    if municipality_type:
        mun_ids = [
            m for m in mun_ids
            if ctx.municipality_map[m].get("type") == municipality_type
        ]
        if not mun_ids:
            return {"error": f"No municipalities match type '{municipality_type}'"}

    # Parse years
    year_list: list[int] | None = None
    if years:
        year_list = [int(y.strip()) for y in years.split(",")]

    # Fetch data
    client = KoladaClient()
    try:
        data_points = await client.fetch_data(
            kpi_id=kpi_id,
            municipality_ids=mun_ids,
            years=year_list,
        )
    except Exception as e:
        logger.error(f"[Kolada MCP] Error fetching data: {e}")
        return {"error": f"Failed to fetch data: {str(e)}"}

    # Flatten data
    rows = DataProcessor.flatten_data_points(data_points, ctx.municipality_map)

    # Apply gender filter
    if gender:
        rows = DataProcessor.filter_by_gender(rows, gender)

    return {
        "data": rows,
        "count": len(rows),
        "kpi_id": kpi_id,
        "kpi_title": kpi.get("title"),
    }


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
    """Analyze a KPI across multiple municipalities.

    Provides comparative analysis with rankings and summary statistics.

    Args:
        ctx: The server context containing cached data
        kpi_id: The KPI identifier
        year: Specific year (default: latest available)
        municipality_type: Filter by type ("K", "R", "L", or "")
        gender: Gender category ("T", "M", "K")
        sort_direction: "asc" or "desc"
        limit: Maximum number of results
        municipality_ids: Optional specific municipalities to analyze

    Returns:
        Dictionary with analysis results including rankings and statistics
    """
    # Validate KPI
    kpi = ctx.get_kpi(kpi_id)
    if kpi is None:
        return {"error": f"KPI with id '{kpi_id}' not found"}

    # Determine which municipalities to query
    if municipality_ids:
        mun_ids = [m.strip() for m in municipality_ids.split(",")]
        # Validate
        invalid_ids = [m for m in mun_ids if m not in ctx.municipality_map]
        if invalid_ids:
            return {"error": f"Invalid municipality IDs: {', '.join(invalid_ids)}"}
    else:
        municipalities = ctx.filter_municipalities_by_type(municipality_type)
        mun_ids = [m["id"] for m in municipalities]

    if not mun_ids:
        return {"error": "No municipalities to analyze"}

    # Fetch data
    client = KoladaClient()
    years = [year] if year else None

    try:
        data_points = await client.fetch_data(
            kpi_id=kpi_id,
            municipality_ids=mun_ids,
            years=years,
        )
    except Exception as e:
        logger.error(f"[Kolada MCP] Error fetching data: {e}")
        return {"error": f"Failed to fetch data: {str(e)}"}

    # Process data
    rows = DataProcessor.flatten_data_points(data_points, ctx.municipality_map)
    rows = DataProcessor.filter_by_gender(rows, gender)

    # Use latest period if no year specified
    if year is None:
        rows = DataProcessor.get_latest_period_per_municipality(rows)

    # Get values for statistics
    values = [r["value"] for r in rows if r.get("value") is not None]
    statistics = DataProcessor.compute_statistics(values)

    # Rank municipalities
    ranked = DataProcessor.rank_municipalities(rows, sort_direction, limit)

    # Add rank to results
    for i, row in enumerate(ranked, 1):
        row["rank"] = i

    return {
        "results": ranked,
        "count": len(ranked),
        "total_municipalities": len(mun_ids),
        "statistics": statistics,
        "kpi_id": kpi_id,
        "kpi_title": kpi.get("title"),
        "year": year or "latest",
        "gender": gender,
        "sort_direction": sort_direction,
    }
