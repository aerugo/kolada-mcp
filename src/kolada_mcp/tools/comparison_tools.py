import statistics
from typing import Any

from mcp.server.fastmcp.server import Context

from kolada_mcp.config import BASE_URL
from kolada_mcp.models.types import KoladaKpi, KoladaLifespanContext, KoladaMunicipality
from kolada_mcp.services.api import fetch_data_from_kolada
from kolada_mcp.services.data_processing import (
    fetch_and_group_data_by_municipality,
    parse_years_param,
)
from kolada_mcp.tools.metadata_tools import get_kpi_metadata  # type: ignore[Context]
from kolada_mcp.utils.context import safe_get_lifespan_context  # type: ignore[Context]


async def compare_kpis(
    kpi1_id: str,
    kpi2_id: str,
    year: str,
    ctx: Context,  # type: ignore[Context]
    gender: str = "T",
    municipality_type: str = "K",
    municipality_ids: str | None = None,
) -> dict[str, Any]:
    """
    **Purpose:** Compares two different Kolada Key Performance Indicators (KPIs)
    across Swedish municipalities for one or more specified years. It calculates
    either the difference or the correlation between the KPIs, depending on
    whether a single year or multiple years are provided.

    If municipality_ids is provided, only those specific municipalities will be compared
    and no ranking (top, bottom, or median) is computed. A flat list of results is returned instead.

    **Use Cases:**
    *   "How does KPI [A] correlate with KPI [B] across municipalities in year [YYYY]?"
    *   "Compare KPI [X] and KPI [Y] for the years [YYYY1],[YYYY2]. Which municipalities show the strongest positive/negative correlation?"
    *   "For year [YYYY], which municipalities have the largest difference between KPI [A] and KPI [B]?"
    *   "Analyze the relationship between [topic 1, e.g., unemployment] and [topic 2, e.g., education level] across municipalities over the period [YYYY1]-[YYYY2]." (Requires finding relevant KPI IDs first using `search_kpis`).

    **Arguments:**
    *   `kpi1_id` (str): The unique identifier of the first Kolada KPI (e.g., "N00945"). Use `search_kpis` or `get_kpis_by_operating_area` if you don't have the ID. **Required.**
    *   `kpi2_id` (str): The unique identifier of the second Kolada KPI. **Required.**
    *   `year` (str): Specifies the year(s) for the comparison.
        *   **Single Year:** Provide a single year (e.g., "2022"). The tool will calculate the *difference* (KPI2 - KPI1) for each municipality.
        *   **Multiple Years:** Provide a comma-separated list of years (e.g., "2020,2021,2022"). The tool will calculate the *Pearson correlation* between the time series of the two KPIs *within each municipality* that has data for at least two overlapping years.
        **Required.**
    *   `ctx` (Context): The server context (automatically injected by the MCP framework). You do not need to provide this.
    *   `gender` (str, optional): Filters the data by gender before comparison.
        *   "T": Total (default)
        *   "M": Men
        *   "K": Women
    *   `municipality_type` (str, optional): Filters the comparison to include only municipalities of a specific type.
        *   "K": Kommun (Municipality, default)
        *   "R": Region
        *   "L": Landsting (County Council - older term, often equivalent to Region)
        The tool will only include municipalities matching this type in the analysis.

    **Core Logic:**
    1.  Retrieves metadata for both `kpi1_id` and `kpi2_id` from the server cache.
    2.  Parses the `year` parameter to determine if it's a single-year or multi-year analysis.
    3.  Fetches the actual data values **from the live Kolada API** for BOTH `kpi1_id` and `kpi2_id` for the specified `year`(s).
    4.  Filters the fetched data based on the requested `gender` and `municipality_type`.
    5.  Identifies municipalities that have data for *both* KPIs for the relevant year(s).
    6.  **If Single Year:**
        *   Calculates the difference (`kpi2_value - kpi1_value`) for each common municipality.
        *   Calculates the overall Pearson correlation between the two KPIs across all common municipalities for that single year.
        *   Ranks municipalities based on the calculated difference.
    7.  **If Multiple Years:**
        *   For each common municipality with at least two overlapping data points, calculates the Pearson correlation between the time series of KPI1 and KPI2.
        *   Calculates the overall Pearson correlation using *all* available (municipality, year) data points combined.
        *   Ranks municipalities based on their individual time-series correlations.
    8.  Constructs and returns a detailed dictionary with the results.

    **Return Value:**
    A dictionary containing:
    *   `kpi1_info` (dict): Metadata (id, title, description, area) for the first KPI.
    *   `kpi2_info` (dict): Metadata for the second KPI.
    *   `selected_years` (list[str]): The list of years used in the analysis.
    *   `gender` (str): The gender filter used.
    *   `municipality_type` (str): The municipality type filter used.
    *   `multi_year` (bool): True if multiple years were analyzed, False otherwise.
    *   `overall_correlation` (float | None): The Pearson correlation coefficient calculated across all data points (either all municipalities in a single year, or all municipality-year pairs in a multi-year analysis). Can be `None` if insufficient data exists.
    *   **If `multi_year` is False (Single Year Analysis):**
        *   `municipality_differences` (list[dict]): A list of dictionaries, one per municipality with data, containing `municipality_id`, `municipality_name`, `kpi1_value`, `kpi2_value`, and `difference`. Sorted by difference.
        *   `top_difference_municipalities` (list[dict]): Top N municipalities with the largest positive difference (KPI2 > KPI1).
        *   `bottom_difference_municipalities` (list[dict]): Top N municipalities with the largest negative difference (KPI1 > KPI2).
        *   `median_difference_municipalities` (list[dict]): N municipalities around the median difference.
    *   **If `multi_year` is True (Multi-Year Analysis):**
        *   `municipality_correlations` (list[dict]): A list of dictionaries, one per municipality with sufficient data, containing `municipality_id`, `municipality_name`, `correlation` (the within-municipality time-series correlation), `years_used`, and `n_years`. Sorted by correlation.
        *   `top_correlation_municipalities` (list[dict]): Top N municipalities with the highest positive correlation.
        *   `bottom_correlation_municipalities` (list[dict]): Top N municipalities with the lowest (most negative) correlation.
        *   `median_correlation_municipalities` (list[dict]): N municipalities around the median correlation.
    *   `error` (str, optional): If an error occurred (e.g., API fetch failed, no overlapping data found), this key will contain an error message.

    **Important Notes:**
    *   This tool makes **live calls to the Kolada API** (potentially two separate calls for the data), which might take some time.
    *   Ensure you provide valid `kpi1_id` and `kpi2_id`.
    *   The analysis depends on data availability in Kolada for the selected KPIs, years, gender, and municipalities. Lack of overlapping data can lead to empty results or `None` for correlations.
    *   For multi-year analysis, a municipality is only included in the `municipality_correlations` list if it has data for *both* KPIs in at least *two* common years within the specified range.
    """
    kpi1_meta: KoladaKpi | dict[str, str] = await get_kpi_metadata(kpi1_id, ctx)
    kpi2_meta: KoladaKpi | dict[str, str] = await get_kpi_metadata(kpi2_id, ctx)

    kpi1_info: dict[str, str] = {
        "id": kpi1_id,
        "title": kpi1_meta.get("title", ""),
        "description": kpi1_meta.get("description", ""),
        "operating_area": kpi1_meta.get("operating_area", ""),
    }
    kpi2_info: dict[str, str] = {
        "id": kpi2_id,
        "title": kpi2_meta.get("title", ""),
        "description": kpi2_meta.get("description", ""),
        "operating_area": kpi2_meta.get("operating_area", ""),
    }

    year_list: list[str] = parse_years_param(year)
    is_multi_year: bool = len(year_list) > 1

    lifespan_ctx: KoladaLifespanContext | None = safe_get_lifespan_context(ctx)
    if not lifespan_ctx:
        await ctx.error(
            "compare_kpis: Server context invalid or missing lifespan context."
        )
        return {
            "error": "Server context invalid.",
            "kpi1_info": kpi1_info,
            "kpi2_info": kpi2_info,
        }

    municipality_map: dict[str, KoladaMunicipality] = lifespan_ctx["municipality_map"]

    from tools.url_builders import build_kolada_url_for_kpi
    url1: str = build_kolada_url_for_kpi(BASE_URL, kpi1_id, municipality_ids, year)

    data_kpi1: dict[str, Any] = await fetch_data_from_kolada(url1)
    if "error" in data_kpi1:
        await ctx.error(
            f"compare_kpis: Error fetching data for KPI1 '{kpi1_id}' at '{url1}': {data_kpi1['error']}"
        )
        return {
            "error": data_kpi1["error"],
            "kpi1_info": kpi1_info,
            "kpi2_info": kpi2_info,
        }

    municipality_data1: dict[str, dict[str, float]] = (
        fetch_and_group_data_by_municipality(data_kpi1, gender)
    )

    url2: str = build_kolada_url_for_kpi(BASE_URL, kpi2_id, municipality_ids, year)

    data_kpi2: dict[str, Any] = await fetch_data_from_kolada(url2)
    if "error" in data_kpi2:
        await ctx.error(
            f"compare_kpis: Error fetching data for KPI2 '{kpi2_id}' at '{url2}': {data_kpi2['error']}"
        )
        return {
            "error": data_kpi2["error"],
            "kpi1_info": kpi1_info,
            "kpi2_info": kpi2_info,
        }

    municipality_data2: dict[str, dict[str, float]] = (
        fetch_and_group_data_by_municipality(data_kpi2, gender)
    )

    def filter_muni_type(
        data_dict: dict[str, dict[str, float]],
    ) -> dict[str, dict[str, float]]:
        result_dict: dict[str, dict[str, float]] = {}
        for m_id, year_values in data_dict.items():
            muni_obj: KoladaMunicipality | None = municipality_map.get(m_id)
            if muni_obj and muni_obj.get("type") == municipality_type:
                result_dict[m_id] = year_values
        return result_dict

    municipality_data1 = filter_muni_type(municipality_data1)
    municipality_data2 = filter_muni_type(municipality_data2)

    result: dict[str, Any] = {
        "kpi1_info": kpi1_info,
        "kpi2_info": kpi2_info,
        "selected_years": year_list,
        "gender": gender,
        "municipality_type": municipality_type,
        "multi_year": is_multi_year,
    }

    async def compute_pearson_correlation(
        x_vals: list[float], y_vals: list[float]
    ) -> float | None:
        if len(x_vals) < 2 or len(y_vals) < 2:
            return None
        try:
            return statistics.correlation(x_vals, y_vals)
        except (ValueError, statistics.StatisticsError) as exc:
            await ctx.warning(f"Failed to compute correlation: {exc}")
            return None

    if not is_multi_year:
        if not year_list:
            await ctx.warning("compare_kpis: No valid single year specified.")
            return {
                **result,
                "error": "No valid year specified for single-year analysis.",
            }

        single_year: str = year_list[0]
        x_vals: list[float] = []
        y_vals: list[float] = []
        cross_section_data: list[dict[str, Any]] = []

        for m_id, values_1 in municipality_data1.items():
            values_2: dict[str, float] | None = municipality_data2.get(m_id)
            if not values_2:
                continue
            if single_year in values_1 and single_year in values_2:
                k1_val: float = values_1[single_year]
                k2_val: float = values_2[single_year]
                x_vals.append(k1_val)
                y_vals.append(k2_val)

                cross_section_data.append(
                    {
                        "municipality_id": m_id,
                        "municipality_name": municipality_map.get(m_id, {}).get(
                            "title", f"Municipality {m_id}"
                        ),
                        "kpi1_value": k1_val,
                        "kpi2_value": k2_val,
                        "difference": k2_val - k1_val,
                    }
                )

        if not cross_section_data:
            await ctx.warning(
                f"compare_kpis: No overlapping data found for year {single_year}."
            )
            return {
                **result,
                "error": f"No overlapping data for single year {single_year}.",
            }

        # If municipality_ids is provided, skip ranking and return flat list
        if municipality_ids:
            return {
                **result,
                "flat_results": cross_section_data,
            }

        overall_corr: float | None = await compute_pearson_correlation(x_vals, y_vals)
        result["overall_correlation"] = overall_corr

        cross_section_data.sort(key=lambda item: item["difference"])
        n_muni: int = len(cross_section_data)
        slice_limit: int = min(10, n_muni)
        median_start: int = max(0, (n_muni - 1) // 2 - (slice_limit // 2))
        median_end: int = min(median_start + slice_limit, n_muni)

        result["municipality_differences"] = cross_section_data
        result["top_difference_municipalities"] = list(
            reversed(cross_section_data[-slice_limit:])
        )
        result["bottom_difference_municipalities"] = cross_section_data[:slice_limit]
        result["median_difference_municipalities"] = cross_section_data[
            median_start:median_end
        ]

        return result

    big_x: list[float] = []
    big_y: list[float] = []
    municipality_correlations: list[dict[str, Any]] = []

    for m_id, values_1 in municipality_data1.items():
        values_2: dict[str, float] | None = municipality_data2.get(m_id)
        if not values_2:
            continue

        intersection_years: list[str] = sorted(
            set(values_1.keys()) & set(values_2.keys())
        )
        if not intersection_years:
            continue

        ts_x: list[float] = []
        ts_y: list[float] = []
        for y in intersection_years:
            ts_x.append(values_1[y])
            ts_y.append(values_2[y])
            big_x.append(values_1[y])
            big_y.append(values_2[y])

        muni_corr: float | None = await compute_pearson_correlation(ts_x, ts_y)
        if muni_corr is not None:
            municipality_correlations.append(
                {
                    "municipality_id": m_id,
                    "municipality_name": municipality_map.get(m_id, {}).get(
                        "title", f"Municipality {m_id}"
                    ),
                    "correlation": muni_corr,
                    "years_used": intersection_years,
                    "n_years": len(intersection_years),
                }
            )

    # If municipality_ids is provided, skip ranking and return flat list
    if municipality_ids:
        return {
            **result,
            "flat_correlation_results": municipality_correlations,
        }

    overall_corr: float | None = await compute_pearson_correlation(big_x, big_y)
    result["overall_correlation"] = overall_corr

    municipality_correlations.sort(key=lambda item: item["correlation"])
    n_corr: int = len(municipality_correlations)
    if n_corr == 0:
        await ctx.warning(
            "compare_kpis: No municipality had at least 2 overlapping years for both KPIs."
        )
        return {
            **result,
            "error": "No municipality had 2+ overlapping data points to compute correlation.",
        }

    slice_limit: int = min(10, n_corr)
    median_start: int = max(0, (n_corr - 1) // 2 - (slice_limit // 2))
    median_end: int = min(median_start + slice_limit, n_corr)

    result["municipality_correlations"] = municipality_correlations
    result["top_correlation_municipalities"] = list(
        reversed(municipality_correlations[-slice_limit:])
    )
    result["bottom_correlation_municipalities"] = municipality_correlations[
        :slice_limit
    ]
    result["median_correlation_municipalities"] = municipality_correlations[
        median_start:median_end
    ]

    return result
