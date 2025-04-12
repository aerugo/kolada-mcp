import sys
from typing import Any

from mcp.server.fastmcp.server import Context

from kolada_mcp.config import BASE_URL
from kolada_mcp.models.types import KoladaKpi, KoladaLifespanContext, KoladaMunicipality
from kolada_mcp.services.api import fetch_data_from_kolada
from kolada_mcp.services.data_processing import (
    build_flat_list_of_municipalities_with_delta,
    fetch_and_group_data_by_municipality,
    parse_years_param,
    process_kpi_data,  # type: ignore[Context]
)
from kolada_mcp.tools.metadata_tools import get_kpi_metadata  # type: ignore[Context]
from kolada_mcp.utils.context import safe_get_lifespan_context  # type: ignore[Context]


async def fetch_kolada_data(
    kpi_id: str,
    municipality_id: str,  # This can be comma-separated
    ctx: Context,  # type: ignore[Context]
    year: str | None = None,
    municipality_type: str = "K",
) -> dict[str, Any]:
    """
    **Purpose:** Fetches the raw, specific statistical data points for a single
    Kolada Key Performance Indicator (KPI) within a single designated Swedish
    municipality or region. It allows specifying particular years or retrieving
    all available historical data points for that specific KPI/municipality pair.

    The municipality_id parameter can be a comma-separated string to fetch data for
    multiple municipalities in a single request.

    **Use Cases:**
    *   "What was the exact value for KPI [ID or name] in [Municipality Name or ID] for the year [YYYY]?"
    *   "Get all historical data points available for KPI [ID] in [Municipality ID]."
    *   "Retrieve the data for [KPI ID] in [Municipality ID] specifically for the years [YYYY1],[YYYY2]."
    *   (Used internally by other tools, but can be called directly if a very specific raw data point is needed).

    **Arguments:**
    *   `kpi_id` (str): The unique identifier of the Kolada KPI whose data is needed (e.g., "N00945"). Use `search_kpis` or `get_kpis_by_operating_area` if you don't have the ID. **Required.**
    *   `municipality_id` (str): The official unique identifier of the specific Swedish municipality or region (e.g., "0180" for Stockholm, "1480" for Göteborg). The server cache contains valid IDs. **Required.**
    *   `ctx` (Context): The server context (automatically injected by the MCP framework). You do not need to provide this.
    *   `year` (str | None, optional): Specifies the year(s) for which to fetch data.
        *   `None` (default): Fetches data for *all* available years for this KPI/municipality combination.
        *   Single Year (e.g., "2023"): Fetches data only for that specific year.
        *   Multiple Years (e.g., "2020,2021,2022"): Fetches data for the specified years.
    *   `municipality_type` (str, optional): Ensures the requested `municipality_id` actually corresponds to the expected type ("K", "R", or "L"). If the ID exists but its type in the server cache doesn't match this parameter, an error is returned *before* calling the Kolada API. Default is "K" (Kommun/Municipality).

    **Core Logic:**
    1.  Retrieves the cached Kolada context (`lifespan_ctx`).
    2.  Validates that `kpi_id` and `municipality_id` are provided.
    3.  Looks up the `municipality_id` in the cached `municipality_map`. If not found, returns an error.
    4.  Checks if the cached type of the `municipality_id` matches the `municipality_type` parameter. If they don't match, returns an error.
    5.  Constructs the specific Kolada API URL targeting the `/v2/data/kpi/{kpi_id}/municipality/{municipality_id}` endpoint. If `year` is provided, it appends `/year/{year}` to the URL.
    6.  Calls the internal `_fetch_data_from_kolada` helper function to make the **live call to the Kolada API**, handling potential errors and pagination (though pagination is less common for this specific endpoint).
    7.  If the API call returns an error, the error dictionary is returned immediately.
    8.  If the API call is successful, it iterates through the returned data points (in the `values` list of the response). For each data point, it attempts to add a `municipality_name` field by looking up the municipality ID (which should be the one requested) in the cached `municipality_map`.
    9.  Returns the dictionary received from Kolada (potentially augmented with `municipality_name` fields).

    **Return Value:**
    A dictionary representing the response from the Kolada API, typically structured as:
    *   `count` (int): The number of main data entries returned (usually 1, representing the municipality).
    *   `values` (list[dict]): A list containing the primary data structure(s). For this endpoint, it's usually a list with one dictionary representing the requested municipality. This dictionary typically contains:
        *   `kpi` (str): The KPI ID.
        *   `municipality` (str): The Municipality ID.
        *   `period` (int): The year for the data points within the 'values' sub-list.
        *   `municipality_name` (str): The human-readable name added by this tool from the cache.
        *   `values` (list[dict]): A sub-list containing the actual data points, often broken down by gender. Each dict here usually has:
            *   `gender` (str): "T", "M", or "K".
            *   `status` (str | None): Data status flag (e.g., None, "B", "M").
            *   `value` (float | int | None): The actual statistical value.
    *   (Other potential keys from Kolada API like `value_types`, etc.)
    *   `error` (str, optional): If an error occurred (e.g., invalid ID provided *to Kolada*, API down, type mismatch detected *before* API call), this key will contain an error message instead of `count` and `values`.

    **Important Notes:**
    *   This tool makes a **live call to the Kolada API** for each execution.
    *   It requires **specific, valid** `kpi_id` and `municipality_id`. Use other tools like `search_kpis` or `analyze_kpi_across_municipalities` if you need to explore or compare data more broadly.
    *   The exact structure of the returned `values` list and its sub-lists depends on the specific KPI and how Kolada structures its data (e.g., whether it includes gender breakdowns).
    *   The `municipality_type` check happens *before* the API call, preventing unnecessary requests if the provided ID is known to be of the wrong type based on the server's cache.
    """
    lifespan_ctx: KoladaLifespanContext | None = safe_get_lifespan_context(ctx)
    if not lifespan_ctx:
        return {"error": "Server context structure invalid or incomplete."}

    if not kpi_id or not municipality_id:
        return {"error": "kpi_id and municipality_id are required."}

    municipality_map: dict[str, KoladaMunicipality] = lifespan_ctx["municipality_map"]
    muni_ids = [mid.strip() for mid in municipality_id.split(",") if mid.strip()]
    if not muni_ids:
        return {"error": "No valid municipality ID provided."}
    for mid in muni_ids:
        if mid not in municipality_map:
            return {"error": f"Municipality ID '{mid}' not found in system."}

    for mid in muni_ids:
        actual_type = municipality_map[mid].get("type", None)
        if actual_type != municipality_type:
            return {
                "error": f"Municipality '{mid}' is type '{actual_type}', but user requested type '{municipality_type}'."
            }

    muni_ids_clean = ",".join(muni_ids)
    url: str = f"{BASE_URL}/data/kpi/{kpi_id}/municipality/{muni_ids_clean}"
    if year:
        url += f"/year/{year}"

    resp_data: dict[str, Any] = await fetch_data_from_kolada(url)
    if "error" in resp_data:
        return resp_data

    values_list: list[dict[str, Any]] = resp_data.get("values", [])
    for item in values_list:
        m_id: str = item.get("municipality", "Unknown")
        if m_id in municipality_map:
            item["municipality_name"] = municipality_map[m_id].get(
                "title", f"Kommun {m_id}"
            )
        else:
            item["municipality_name"] = f"Kommun {m_id}"

    return resp_data


async def analyze_kpi_across_municipalities(
    kpi_id: str,
    ctx: Context,  # type: ignore[Context]
    year: str,
    sort_order: str = "desc",
    limit: int = 10,
    gender: str = "T",
    only_return_rate: bool = False,
    municipality_type: str = "K",
    municipality_ids: str | None = None,
) -> dict[str, Any]:
    """
    **Purpose:** Analyzes a single Kolada Key Performance Indicator (KPI) across
    all relevant Swedish municipalities for one or more specified years. It
    provides overall summary statistics (min, max, mean, median) and lists
    of municipalities ranking highest, lowest, and around the median for the
    KPI's value. If multiple years are provided, it also calculates and ranks
    the change (delta) in the KPI value over the specified period for each
    municipality.

    If municipality_ids is provided, only those specific municipalities will be analyzed
    and no ranking (top, bottom, or median) is computed. A flat list of results is returned instead.

    **Use Cases:**
    *   "Which municipalities had the highest [KPI description, e.g., population] in year [YYYY]?"
    *   "Show the lowest performing municipalities for KPI [ID or name] in [YYYY], sorted ascending."
    *   "What are the average, minimum, and maximum values for [KPI description] across all municipalities in [YYYY]?"
    *   "Analyze KPI [ID or name] for the years [YYYY1],[YYYY2]. Which municipalities showed the largest increase?"
    *   "Get only the rate of change statistics for [KPI description] between [YYYY1] and [YYYY2]."
    *   "Compare regions (type R) based on their values for KPI [ID] in [YYYY]."

    **Arguments:**
    *   `kpi_id` (str): The unique identifier of the Kolada KPI to analyze (e.g., "N00945"). Use `search_kpis` or `get_kpis_by_operating_area` if you don't have the ID. **Required.**
    *   `ctx` (Context): The server context (automatically injected by the MCP framework). You do not need to provide this.
    *   `year` (str): Specifies the year(s) for the analysis.
        *   **Single Year:** Provide a single year (e.g., "2022"). Analysis focuses on the values in that year.
        *   **Multiple Years:** Provide a comma-separated list of years (e.g., "2020,2021,2022"). Analysis includes both the latest value within the range and the *change (delta)* between the earliest and latest available year within the range for each municipality.
        **Required.**
    *   `sort_order` (str, optional): Determines the sorting direction for rankings.
        *   "desc": Descending order (highest values first, default).
        *   "asc": Ascending order (lowest values first).
    *   `limit` (int, optional): The maximum number of municipalities to include in the 'top', 'bottom', and 'median' ranking lists (default is 10).
    *   `gender` (str, optional): Filters the data by gender before analysis.
        *   "T": Total (default)
        *   "M": Men
        *   "K": Women
    *   `only_return_rate` (bool, optional): If True **and** multiple years are specified, the returned results will *only* include statistics and rankings related to the *change (delta)* over the period. The statistics and rankings based on the absolute latest value will be omitted. Default is False. Has no effect if only a single year is provided.
    *   `municipality_type` (str, optional): Filters the analysis to include only municipalities of a specific type.
        *   "K": Kommun (Municipality, default)
        *   "R": Region
        *   "L": Landsting (County Council - older term, often equivalent to Region)
        The tool will only include municipalities matching this type in the analysis.

    **Core Logic:**
    1.  Retrieves metadata (title, description, etc.) for the specified `kpi_id` from the server cache.
    2.  Parses the `year` parameter into a list of years.
    3.  Constructs the appropriate URL and fetches the actual data values **from the live Kolada API** for the given `kpi_id` and `year`(s) across all municipalities.
    4.  Processes the raw API response: filters by the specified `gender`, extracts values, and groups them into a structure like `{ municipality_id: { year: value } }`.
    5.  Filters this grouped data to include only municipalities matching the specified `municipality_type`.
    6.  Performs the main analysis (`_process_kpi_data`):
        *   For each included municipality, identifies the value for the latest available year within the requested `year` range (`latest_value`).
        *   If multiple years were requested and data exists for at least two years for a municipality within that range, calculates the `delta_value` (`latest_value` - `earliest_value` in range).
        *   Calculates overall summary statistics (min, max, mean, median, count) across all included municipalities based on their `latest_value`.
        *   If delta values were calculated, calculates overall summary statistics for the `delta_value` across relevant municipalities.
        *   Ranks municipalities based on `latest_value` according to `sort_order` and extracts top, bottom, and median lists based on `limit`.
        *   If delta values were calculated, ranks municipalities based on `delta_value` and extracts top, bottom, and median lists for the delta.
    7.  Constructs and returns a detailed dictionary based on the analysis results and the `only_return_rate` flag.

    **Return Value:**
    A dictionary containing:
    *   `kpi_info` (dict): Metadata (id, title, description, area) for the analyzed KPI.
    *   `selected_years` (list[str]): The list of years used in the analysis.
    *   `selected_gender` (str): The gender filter used.
    *   `sort_order` (str): The sorting order used for rankings.
    *   `limit` (int): The limit used for the size of ranking lists.
    *   `multi_year_delta` (bool): True if multiple years were specified AND delta calculations were possible for at least one municipality.
    *   `only_return_rate` (bool): Reflects the value of the input parameter.
    *   `municipalities_count` (int): The number of municipalities included in the analysis after all filtering (gender, type, data availability).
    *   `summary_stats` (dict): Overall statistics (`min_latest`, `max_latest`, `mean_latest`, `median_latest`, `count`) based on the latest available value for each municipality. **Omitted if `only_return_rate` is True and `multi_year_delta` is True.**
    *   `top_municipalities` (list[dict]): List of municipalities (up to `limit`) with the highest `latest_value` (or lowest if `sort_order`="asc"). Each entry contains `municipality_id`, `municipality_name`, `latest_year`, `latest_value`, potentially `earliest_year`, `earliest_value`, `delta_value`. **Omitted if `only_return_rate` is True and `multi_year_delta` is True.**
    *   `bottom_municipalities` (list[dict]): List of municipalities (up to `limit`) with the lowest `latest_value` (or highest if `sort_order`="asc"). **Omitted if `only_return_rate` is True and `multi_year_delta` is True.**
    *   `median_municipalities` (list[dict]): List of municipalities (up to `limit`) around the median `latest_value`. **Omitted if `only_return_rate` is True and `multi_year_delta` is True.**
    *   `delta_summary_stats` (dict): Overall statistics (`min_delta`, `max_delta`, `mean_delta`, `median_delta`, `count`) based on the calculated change (delta) over the period. **Included only if `multi_year_delta` is True.**
    *   `top_delta_municipalities` (list[dict]): List of municipalities (up to `limit`) with the highest `delta_value` (largest increase, or decrease if `sort_order`="asc"). **Included only if `multi_year_delta` is True.**
    *   `bottom_delta_municipalities` (list[dict]): List of municipalities (up to `limit`) with the lowest `delta_value` (largest decrease, or increase if `sort_order`="asc"). **Included only if `multi_year_delta` is True.**
    *   `median_delta_municipalities` (list[dict]): List of municipalities (up to `limit`) around the median `delta_value`. **Included only if `multi_year_delta` is True.**
    *   `error` (str, optional): If an error occurred (e.g., API fetch failed, no data found for the parameters), this key will contain an error message.

    **Important Notes:**
    *   This tool makes a **live call to the Kolada API** to fetch the raw data, which might take some time depending on the KPI and number of years requested.
    *   The results depend heavily on data availability within Kolada for the specific KPI, years, gender, and municipality type. If no data matching the criteria is found, the counts will be zero and rankings empty.
    *   The `delta_value` and associated statistics/rankings are only calculated and returned (`multi_year_delta`=True) if multiple years are specified *and* at least one municipality has data for two or more years *within the requested range*.
    *   Using `only_return_rate=True` can simplify the output when you are specifically interested in the *change* over time rather than the absolute latest values.
    """
    kpi_metadata_result: KoladaKpi | dict[str, str] = await get_kpi_metadata(
        kpi_id, ctx
    )
    kpi_metadata: dict[str, Any] = {
        "id": kpi_id,
        "title": kpi_metadata_result.get("title", ""),
        "description": kpi_metadata_result.get("description", ""),
        "operating_area": kpi_metadata_result.get("operating_area", ""),
    }

    print(
        f"[Kolada MCP] Analyzing KPI {kpi_id} ({kpi_metadata['title']}) across municipalities.",
        file=sys.stderr,
    )

    lifespan_ctx: KoladaLifespanContext | None = safe_get_lifespan_context(ctx)
    if not lifespan_ctx:
        return {
            "error": "Server context structure invalid or incomplete.",
            "kpi_info": kpi_metadata,
        }

    municipality_map: dict[str, KoladaMunicipality] = lifespan_ctx["municipality_map"]
    year_list: list[str] = parse_years_param(year)

    from tools.url_builders import build_kolada_url_for_kpi

    url: str = build_kolada_url_for_kpi(BASE_URL, kpi_id, municipality_ids, year)

    kolada_data: dict[str, Any] = await fetch_data_from_kolada(url)
    if "error" in kolada_data:
        return {"error": kolada_data["error"], "kpi_info": kpi_metadata}

    print(
        f"[Kolada MCP] Fetched data for {len(kolada_data.get('values', []))} entries.",
        file=sys.stderr,
    )
    print(
        f"[Kolada MCP] Sample data: {list(kolada_data.get('values', [])[:5])}",
        file=sys.stderr,
    )

    municipality_data: dict[str, dict[str, float]] = (
        fetch_and_group_data_by_municipality(kolada_data, gender)
    )
    print(
        f"[Kolada MCP] Fetched data for {len(municipality_data)} municipalities.",
        file=sys.stderr,
    )

    filtered_municipality_data: dict[str, dict[str, float]] = {}
    for m_id, yearly_vals in municipality_data.items():
        if m_id in municipality_map:
            actual_type: str = municipality_map[m_id].get("type", "")
            # If municipality_type is provided, filter by it; otherwise, include all types.
            if not municipality_type or actual_type == municipality_type:
                filtered_municipality_data[m_id] = yearly_vals

    # If user specified municipality_ids, skip ranking and return flat list
    if municipality_ids:
        result_list = build_flat_list_of_municipalities_with_delta(
            filtered_municipality_data, municipality_map, year_list
        )
        return {
            "kpi_info": kpi_metadata,
            "selected_years": year_list,
            "selected_gender": gender,
            "only_return_rate": only_return_rate,
            "municipalities_count": len(result_list),
            "municipalities_data": result_list,
        }
    else:
        return process_kpi_data(
            municipality_data=filtered_municipality_data,
            municipality_map=municipality_map,
            years=year_list,
            sort_order=sort_order,
            limit=limit,
            kpi_metadata=kpi_metadata,
            gender=gender,
            only_return_rate=only_return_rate,
        )
