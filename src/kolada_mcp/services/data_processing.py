import sys
from typing import Any

import polars as pl

from kolada_mcp.models.types import KoladaKpi, KoladaMunicipality
from kolada_mcp.utils.statistics import (
    calculate_summary_stats,
    rank_and_slice_municipalities,
)


def group_kpis_by_operating_area(
    kpis: list[KoladaKpi],
) -> dict[str, list[KoladaKpi]]:
    """Groups KPIs by their 'operating_area' field."""
    grouped: dict[str, list[KoladaKpi]] = {}
    for kpi in kpis:
        operating_area_field: str = kpi.get("operating_area", "Unknown")
        areas: list[str] = [a.strip() for a in operating_area_field.split(",")]
        for area in areas:
            if area:
                if area not in grouped:
                    grouped[area] = []
                grouped[area].append(kpi)
    return grouped


def get_operating_areas_summary(
    kpis: list[KoladaKpi],
) -> list[dict[str, str | int]]:
    """Generates a summary list of operating areas and their KPI counts."""
    grouped: dict[str, list[KoladaKpi]] = group_kpis_by_operating_area(kpis)
    areas_with_counts: list[dict[str, str | int]] = []
    for area in sorted(grouped.keys()):
        area_dict: dict[str, str | int] = {
            "operating_area": area,
            "kpi_count": len(grouped[area]),
        }
        areas_with_counts.append(area_dict)
    return areas_with_counts


def fetch_and_group_data_by_municipality(
    data: dict[str, Any],
    gender: str,
) -> dict[str, dict[str, float]]:
    """
    From a Kolada data response containing multiple years, extracts numeric values
    per (municipality, year) for the specified gender. Returns a structure
    { "0180": { "2020": val, "2021": val}, ... }.
    """
    raw_rows: list[dict[str, Any]] = []

    values_data: list[dict[str, Any]] = data.get("values", [])
    for item in values_data:
        municipality_id: str | None = item.get("municipality")
        raw_period: int | str | None = item.get("period")
        if not municipality_id or raw_period is None:
            print(
                f"Warning: Skipping due to missing municipality_id or period: {item}",
                file=sys.stderr,
            )

        period_str: str = str(raw_period)

        for subval in item.get("values", []):
            row_gender: str | None = subval.get("gender")
            val: Any = subval.get("value")

            row: dict[str, Any] = {
                "municipality": municipality_id,
                "period": period_str,
                "gender": row_gender,
                "value": val,
            }
            raw_rows.append(row)

    df: pl.DataFrame = pl.from_dicts(raw_rows)
    if df.is_empty():
        empty_dict: dict[str, dict[str, float]] = {}
        return empty_dict

    df_filtered: pl.DataFrame = df.filter(
        (pl.col("gender") == gender) & (pl.col("value").is_not_null())
    ).drop(["gender"])

    if df_filtered.is_empty():
        empty_dict: dict[str, dict[str, float]] = {}
        return empty_dict

    df_cast: pl.DataFrame = df_filtered.with_columns(
        value=pl.col("value").cast(pl.Float64, strict=False)
    )

    municipality_dict: dict[str, dict[str, float]] = {}
    for row_data in df_cast.to_dicts():
        m_id: str = row_data["municipality"]
        p_str: str = row_data["period"]
        v_val: float = row_data["value"]
        if m_id not in municipality_dict:
            municipality_dict[m_id] = {}
        municipality_dict[m_id][p_str] = v_val

    return municipality_dict


def parse_years_param(year_str: str) -> list[str]:
    """
    Parses a comma-separated string of years into a list (e.g. "2020,2021" -> ["2020","2021"]).
    If empty or invalid, returns an empty list.
    """
    if not year_str:
        return []
    parts: list[str] = [y.strip() for y in year_str.split(",") if y.strip()]
    return parts


def build_flat_list_of_municipalities(
    municipality_data: dict[str, dict[str, float]],
    municipality_map: dict[str, KoladaMunicipality],
    years: list[str]
) -> list[dict[str, Any]]:
    """
    Returns a flat list of municipality results without any ranking.
    Each entry includes municipality_id, municipality_name, and all available year values.
    """
    flat_list: list[dict[str, Any]] = []
    for m_id, year_vals in municipality_data.items():
        entry = {
            "municipality_id": m_id,
            "municipality_name": municipality_map.get(m_id, {}).get("title", f"Kommun {m_id}"),
            "data": {year: year_vals.get(year) for year in years if year in year_vals}
        }
        flat_list.append(entry)
    return flat_list


def build_flat_list_of_municipalities_with_delta(
    municipality_data: dict[str, dict[str, float]],
    municipality_map: dict[str, KoladaMunicipality],
    years: list[str]
) -> list[dict[str, Any]]:
    """
    Returns a flat list of municipality results including delta calculations.
    Each entry includes municipality_id, municipality_name, the raw data per year,
    and, if available, the latest and earliest years with their values as well as
    the computed delta (latest_value - earliest_value).
    """
    flat_list = []
    for m_id, year_vals in municipality_data.items():
        # Only include requested years that are available
        available_years = [year for year in years if year in year_vals]
        available_years.sort()  # ascending order

        entry = {
            "municipality_id": m_id,
            "municipality_name": municipality_map.get(m_id, {}).get("title", f"Kommun {m_id}"),
            "data": {year: year_vals[year] for year in available_years},
        }
        if available_years:
            entry["latest_year"] = available_years[-1]
            entry["latest_value"] = year_vals[available_years[-1]]
            if len(available_years) >= 2:
                entry["earliest_year"] = available_years[0]
                entry["earliest_value"] = year_vals[available_years[0]]
                entry["delta_value"] = year_vals[available_years[-1]] - year_vals[available_years[0]]
        flat_list.append(entry)
    return flat_list


def process_kpi_data(
    municipality_data: dict[str, dict[str, float]],
    municipality_map: dict[str, KoladaMunicipality],
    years: list[str],
    sort_order: str,
    limit: int,
    kpi_metadata: dict[str, Any],
    gender: str,
    only_return_rate: bool,
) -> dict[str, Any]:
    """
    This function processes the KPI data for a given municipality and returns
    a summary of the results. It handles both single-year and multi-year data.
    It also calculates summary statistics and ranks the municipalities based
    on the specified sort order.
    """

    sorted_years: list[str] = sorted(years)
    is_multi_year: bool = len(sorted_years) > 1

    print(
        f"[Kolada MCP] Unified KPI processing. Requested years: {sorted_years}",
        file=sys.stderr,
    )
    print(
        f"[Kolada MCP] Processing data for {len(municipality_data)} municipalities.",
        file=sys.stderr,
    )

    full_municipality_list: list[dict[str, Any]] = []
    latest_values_list: list[float] = []
    delta_list: list[dict[str, Any]] = []
    delta_values: list[float] = []

    for m_id, yearly_values in municipality_data.items():
        available_years: list[str] = [y for y in sorted_years if y in yearly_values]
        if not available_years:
            continue

        earliest_year: str = available_years[0]
        latest_year: str = available_years[-1]
        earliest_val: float = yearly_values[earliest_year]
        latest_val: float = yearly_values[latest_year]

        m_name: str = municipality_map.get(m_id, {}).get("title", f"Kommun {m_id}")
        entry: dict[str, Any] = {
            "municipality_id": m_id,
            "municipality_name": m_name,
            "latest_year": latest_year,
            "latest_value": latest_val,
            "years_in_data": available_years,
        }
        full_municipality_list.append(entry)
        latest_values_list.append(latest_val)

        # Multi-year delta
        if len(available_years) >= 2:
            delta_value: float = latest_val - earliest_val
            entry["earliest_year"] = earliest_year
            entry["earliest_value"] = earliest_val
            entry["delta_value"] = delta_value
            delta_list.append(entry)
            delta_values.append(delta_value)

    if not full_municipality_list:
        return {
            "error": f"No data available for the specified parameters (Years: {years}, Gender: {gender}).",
            "kpi_info": kpi_metadata,
            "selected_gender": gender,
            "selected_years": years,
            "municipalities_count": 0,
            "summary_stats": {},
            "top_municipalities": [],
            "bottom_municipalities": [],
            "median_municipalities": [],
        }

    if only_return_rate:
        delta_top, delta_bottom, delta_median = rank_and_slice_municipalities(
            delta_list, "delta_value", sort_order, limit
        )
        delta_stats: dict[str, float | int | None] = calculate_summary_stats(
            delta_values
        )
        delta_summary_stats: dict[str, float | int | None] = {
            "min_delta": delta_stats["min"],
            "max_delta": delta_stats["max"],
            "mean_delta": delta_stats["mean"],
            "median_delta": delta_stats["median"],
            "count": delta_stats["count"],
        }
        return {
            "kpi_info": kpi_metadata,
            "summary_stats": delta_summary_stats,
            "top_municipalities": [],
            "bottom_municipalities": [],
            "median_municipalities": [],
            "municipalities_count": len(delta_list),
            "selected_gender": gender,
            "selected_years": years,
            "sort_order": sort_order,
            "limit": limit,
            "multi_year_delta": is_multi_year,
            "only_return_rate": True,
            "delta_municipalities": delta_list,
            "top_delta_municipalities": delta_top,
            "bottom_delta_municipalities": delta_bottom,
            "median_delta_municipalities": delta_median,
        }

    top_main, bottom_main, median_main = rank_and_slice_municipalities(
        full_municipality_list, "latest_value", sort_order, limit
    )
    main_stats: dict[str, float | int | None] = calculate_summary_stats(
        latest_values_list
    )
    summary_stats: dict[str, float | int | None] = {
        "min_latest": main_stats["min"],
        "max_latest": main_stats["max"],
        "mean_latest": main_stats["mean"],
        "median_latest": main_stats["median"],
        "count": main_stats["count"],
    }

    delta_top, delta_bottom, delta_median = rank_and_slice_municipalities(
        delta_list, "delta_value", sort_order, limit
    )
    delta_stats: dict[str, float | int | None] = calculate_summary_stats(delta_values)
    delta_summary_stats: dict[str, float | int | None] = {
        "min_delta": delta_stats["min"],
        "max_delta": delta_stats["max"],
        "mean_delta": delta_stats["mean"],
        "median_delta": delta_stats["median"],
        "count": delta_stats["count"],
    }

    return {
        "kpi_info": kpi_metadata,
        "summary_stats": summary_stats,
        "top_municipalities": top_main,
        "bottom_municipalities": bottom_main,
        "median_municipalities": median_main,
        "municipalities_count": len(full_municipality_list),
        "selected_gender": gender,
        "selected_years": years,
        "sort_order": sort_order,
        "limit": limit,
        "multi_year_delta": is_multi_year,
        "only_return_rate": False,
        "top_delta_municipalities": delta_top,
        "bottom_delta_municipalities": delta_bottom,
        "median_delta_municipalities": delta_median,
    }
