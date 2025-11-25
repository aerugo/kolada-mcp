"""Municipality tools for Kolada MCP.

These tools provide access to municipality data and filtering capabilities.
"""

import logging
from typing import Any

from kolada_mcp.models.types import ServerContext
from kolada_mcp.services.kolada_client import KoladaClient
from kolada_mcp.services.data_processing import DataProcessor

logger = logging.getLogger(__name__)


async def list_municipalities(
    ctx: ServerContext, municipality_type: str = ""
) -> dict[str, Any]:
    """List all municipalities or filter by type.

    Args:
        ctx: The server context containing cached data
        municipality_type: Filter by type:
            - "K" = Kommun (municipality)
            - "R" = Region
            - "L" = Landsting (county council)
            - "" = All types (default)

    Returns:
        Dictionary with 'municipalities' key containing the list
    """
    municipalities = ctx.filter_municipalities_by_type(municipality_type)

    # Return sorted list with basic info
    result = sorted(
        [{"id": m["id"], "title": m["title"], "type": m["type"]} for m in municipalities],
        key=lambda x: x["title"],
    )

    return {"municipalities": result, "count": len(result)}


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

    This tool identifies municipalities where a specific KPI value
    is above or below a specified cutoff.

    Args:
        ctx: The server context containing cached data
        kpi_id: The KPI identifier
        cutoff_value: The threshold value for filtering
        comparison: "above" or "below" the cutoff
        year: Specific year to filter (default: latest available)
        municipality_type: Filter by municipality type ("K", "R", "L", or "")
        gender: Gender category ("T" = Total, "M" = Male, "K" = Female)

    Returns:
        Dictionary with 'municipalities' key containing matching municipalities
    """
    # Validate KPI exists
    kpi = ctx.get_kpi(kpi_id)
    if kpi is None:
        return {"error": f"KPI with id '{kpi_id}' not found"}

    # Get municipalities to query
    municipalities = ctx.filter_municipalities_by_type(municipality_type)
    if not municipalities:
        return {"error": f"No municipalities found for type '{municipality_type}'"}

    municipality_ids = [m["id"] for m in municipalities]

    # Fetch data
    client = KoladaClient()
    years = [year] if year else None

    try:
        data_points = await client.fetch_data(
            kpi_id=kpi_id,
            municipality_ids=municipality_ids,
            years=years,
        )
    except Exception as e:
        logger.error(f"[Kolada MCP] Error fetching data: {e}")
        return {"error": f"Failed to fetch data: {str(e)}"}

    # Flatten and process data
    rows = DataProcessor.flatten_data_points(data_points, ctx.municipality_map)
    rows = DataProcessor.filter_by_gender(rows, gender)

    # If no year specified, use latest per municipality
    if year is None:
        rows = DataProcessor.get_latest_period_per_municipality(rows)

    # Apply cutoff filter
    results = []
    for row in rows:
        value = row.get("value")
        if value is None:
            continue

        passes_filter = (
            (comparison == "above" and value >= cutoff_value) or
            (comparison == "below" and value <= cutoff_value)
        )

        if passes_filter:
            results.append({
                "municipality_id": row["municipality_id"],
                "municipality_name": row["municipality_name"],
                "value": value,
                "period": row["period"],
                "cutoff": cutoff_value,
                "difference": value - cutoff_value,
            })

    # Sort by municipality ID
    results.sort(key=lambda x: x["municipality_id"])

    return {
        "municipalities": results,
        "count": len(results),
        "kpi_id": kpi_id,
        "kpi_title": kpi.get("title"),
        "comparison": comparison,
        "cutoff_value": cutoff_value,
    }
