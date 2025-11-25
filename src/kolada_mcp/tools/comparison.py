"""Comparison tools for Kolada MCP.

These tools enable comparison and correlation analysis between KPIs.
"""

import logging
import statistics
from typing import Any

from kolada_mcp.models.types import ServerContext
from kolada_mcp.services.kolada_client import KoladaClient
from kolada_mcp.services.data_processing import DataProcessor

logger = logging.getLogger(__name__)


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

    For single year: calculates the difference between KPI values.
    For multiple years: calculates correlation between time series.

    Args:
        ctx: The server context containing cached data
        kpi_id_1: First KPI identifier
        kpi_id_2: Second KPI identifier
        years: Comma-separated list of years
        municipality_ids: Optional specific municipalities
        municipality_type: Filter by type ("K", "R", "L", or "")
        gender: Gender category ("T", "M", "K")

    Returns:
        Dictionary with comparison results
    """
    # Validate KPIs
    kpi_1 = ctx.get_kpi(kpi_id_1)
    kpi_2 = ctx.get_kpi(kpi_id_2)

    if kpi_1 is None:
        return {"error": f"KPI with id '{kpi_id_1}' not found"}
    if kpi_2 is None:
        return {"error": f"KPI with id '{kpi_id_2}' not found"}

    # Parse years
    year_list = [int(y.strip()) for y in years.split(",")]

    # Determine municipalities
    if municipality_ids:
        mun_ids = [m.strip() for m in municipality_ids.split(",")]
        invalid_ids = [m for m in mun_ids if m not in ctx.municipality_map]
        if invalid_ids:
            return {"error": f"Invalid municipality IDs: {', '.join(invalid_ids)}"}
    else:
        municipalities = ctx.filter_municipalities_by_type(municipality_type)
        mun_ids = [m["id"] for m in municipalities]

    if not mun_ids:
        return {"error": "No municipalities to compare"}

    # Fetch data for both KPIs
    client = KoladaClient()

    try:
        data_1 = await client.fetch_data(
            kpi_id=kpi_id_1,
            municipality_ids=mun_ids,
            years=year_list,
        )
        data_2 = await client.fetch_data(
            kpi_id=kpi_id_2,
            municipality_ids=mun_ids,
            years=year_list,
        )
    except Exception as e:
        logger.error(f"[Kolada MCP] Error fetching data: {e}")
        return {"error": f"Failed to fetch data: {str(e)}"}

    # Process data
    rows_1 = DataProcessor.flatten_data_points(data_1, ctx.municipality_map)
    rows_2 = DataProcessor.flatten_data_points(data_2, ctx.municipality_map)

    rows_1 = DataProcessor.filter_by_gender(rows_1, gender)
    rows_2 = DataProcessor.filter_by_gender(rows_2, gender)

    # Index data by municipality and year
    def index_by_mun_year(rows: list[dict]) -> dict[tuple[str, int], float | None]:
        result: dict[tuple[str, int], float | None] = {}
        for row in rows:
            key = (row["municipality_id"], row["period"])
            result[key] = row.get("value")
        return result

    indexed_1 = index_by_mun_year(rows_1)
    indexed_2 = index_by_mun_year(rows_2)

    # Perform comparison based on number of years
    if len(year_list) == 1:
        # Single year: compute difference
        return _compare_single_year(
            ctx, kpi_1, kpi_2, year_list[0], indexed_1, indexed_2, mun_ids
        )
    else:
        # Multiple years: compute correlation
        return _compare_correlation(
            ctx, kpi_1, kpi_2, year_list, indexed_1, indexed_2, mun_ids
        )


def _compare_single_year(
    ctx: ServerContext,
    kpi_1: dict,
    kpi_2: dict,
    year: int,
    indexed_1: dict[tuple[str, int], float | None],
    indexed_2: dict[tuple[str, int], float | None],
    mun_ids: list[str],
) -> dict[str, Any]:
    """Compare two KPIs for a single year."""
    comparison = []

    for mun_id in mun_ids:
        key = (mun_id, year)
        val_1 = indexed_1.get(key)
        val_2 = indexed_2.get(key)

        if val_1 is not None and val_2 is not None:
            municipality = ctx.get_municipality(mun_id)
            comparison.append({
                "municipality_id": mun_id,
                "municipality_name": municipality["title"] if municipality else mun_id,
                "kpi_1_id": kpi_1["id"],
                "kpi_1_value": val_1,
                "kpi_2_id": kpi_2["id"],
                "kpi_2_value": val_2,
                "difference": val_1 - val_2,
                "year": year,
            })

    return {
        "comparison": comparison,
        "count": len(comparison),
        "year": year,
        "kpi_1": {"id": kpi_1["id"], "title": kpi_1.get("title")},
        "kpi_2": {"id": kpi_2["id"], "title": kpi_2.get("title")},
        "analysis_type": "difference",
    }


def _compare_correlation(
    ctx: ServerContext,
    kpi_1: dict,
    kpi_2: dict,
    years: list[int],
    indexed_1: dict[tuple[str, int], float | None],
    indexed_2: dict[tuple[str, int], float | None],
    mun_ids: list[str],
) -> dict[str, Any]:
    """Compare two KPIs using correlation over multiple years."""
    comparison = []

    for mun_id in mun_ids:
        # Collect values for both KPIs across years
        values_1 = []
        values_2 = []
        common_years = []

        for year in years:
            key = (mun_id, year)
            val_1 = indexed_1.get(key)
            val_2 = indexed_2.get(key)

            if val_1 is not None and val_2 is not None:
                values_1.append(val_1)
                values_2.append(val_2)
                common_years.append(year)

        # Need at least 2 points for correlation
        if len(values_1) >= 2:
            try:
                correlation = statistics.correlation(values_1, values_2)
            except statistics.StatisticsError:
                correlation = None

            municipality = ctx.get_municipality(mun_id)
            comparison.append({
                "municipality_id": mun_id,
                "municipality_name": municipality["title"] if municipality else mun_id,
                "correlation": correlation,
                "years_analyzed": common_years,
                "data_points": len(values_1),
            })

    return {
        "comparison": comparison,
        "count": len(comparison),
        "years": years,
        "kpi_1": {"id": kpi_1["id"], "title": kpi_1.get("title")},
        "kpi_2": {"id": kpi_2["id"], "title": kpi_2.get("title")},
        "analysis_type": "correlation",
    }
