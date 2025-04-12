from typing import Any

from mcp.server.fastmcp.server import Context

from kolada_mcp.models.types import KoladaLifespanContext, KoladaMunicipality
from kolada_mcp.tools.data_tools import fetch_kolada_data  # type: ignore[Context]
from kolada_mcp.utils.context import safe_get_lifespan_context  # type: ignore[Context]


async def list_municipalities(
    ctx: Context,  # type: ignore[Context]
    municipality_type: str = "K",
) -> list[dict[str, str]]:
    """
    **Purpose:** Returns a list of all municipalities or regions in the system,
    filtered by type. This is useful for getting valid municipality IDs and names
    that can be used with other tools.

    **Use Cases:**
    *   "List all municipalities (kommuner) in the system."
    *   "Get the IDs and names of all regions."
    *   "What are the valid municipality IDs I can use with other tools?"

    **Arguments:**
    *   `ctx` (Context): The server context (automatically injected by the MCP framework). You do not need to provide this.
    *   `municipality_type` (str, optional): Filter municipalities by their type.
        *   "K": Kommun (Municipality, default)
        *   "R": Region
        *   "L": Landsting (County Council - older term, often equivalent to Region)
        *   "": Empty string means "no filtering" - returns all types

    **Return Value:**
    A list of dictionaries, each containing:
    *   `id` (str): The municipality ID (e.g., "0180" for Stockholm)
    *   `name` (str): The municipality name (e.g., "Stockholm")
    *   If the context is invalid, returns a list with a single error entry.

    **Important Notes:**
    *   This tool accesses the server's cache, not the live Kolada API.
    *   The returned list is sorted by municipality ID.
    """
    lifespan_ctx: KoladaLifespanContext | None = safe_get_lifespan_context(ctx)
    if not lifespan_ctx:
        return [{"error": "Server context invalid or incomplete."}]

    municipality_map: dict[str, KoladaMunicipality] = lifespan_ctx.get(
        "municipality_map", {}
    )
    result: list[dict[str, str]] = []
    for m_id, muni in municipality_map.items():
        # If municipality_type is provided (default "K"), match only those; allow empty string to mean "no filtering"
        if municipality_type and muni.get("type") != municipality_type:
            continue
        result.append({"id": m_id, "name": muni.get("title", f"Municipality {m_id}")})

    # Sort by municipality ID
    result.sort(key=lambda x: x["id"])
    return result


async def filter_municipalities_by_kpi(
    ctx: Context,  # type: ignore[Context]
    kpi_id: str,
    cutoff: float,
    operator: str = "above",
    year: str | None = None,
    municipality_type: str = "K",
    gender: str = "T",
) -> list[dict[str, Any]]:
    """
    **Purpose:** Returns a list of municipalities for which the specified KPI value is
    either above or below a given cutoff.

    **Use Cases:**
    *   "Return all municipalities with KPI X above 50.0."
    *   "Return all municipalities with KPI Y below 75.0."

    **Arguments:**
    *   `ctx` (Context): The server context.
    *   `kpi_id` (str): The KPI identifier.
    *   `cutoff` (float): The threshold value to compare KPI values against.
    *   `operator` (str, optional): Either "above" or "below". Defaults to "above".
    *   `year` (str, optional): The specific year to consider. If omitted, the latest available period is used.
    *   `municipality_type` (str, optional): Filter municipalities by type (default "K").
    *   `gender` (str, optional): The gender category for KPI values (default "T").

    **Return Value:**
    A list of dictionaries, each containing:
      - `municipality_id` (str)
      - `municipality_name` (str)
      - `period` (str or int): The year of the KPI data.
      - `value` (float): The KPI value.
      - `cutoff` (float): The provided cutoff value.
      - `difference` (float): The difference (value - cutoff).
    """
    lifespan_ctx: Any = safe_get_lifespan_context(ctx)
    if not lifespan_ctx:
        return [{"error": "Server context invalid or incomplete."}]

    municipality_map: dict[str, dict[str, Any]] = lifespan_ctx.get(
        "municipality_map", {}
    )
    # Filter municipality IDs by provided type (or include all if empty)
    filtered_muni_ids = [
        m_id
        for m_id, muni in municipality_map.items()
        if not municipality_type or muni.get("type") == municipality_type
    ]
    if not filtered_muni_ids:
        return [{"error": f"No municipalities found for type: {municipality_type}"}]

    muni_ids_str = ",".join(filtered_muni_ids)

    # Fetch KPI data for these municipalities.
    data_response = await fetch_kolada_data(
        kpi_id=kpi_id, municipality_id=muni_ids_str, ctx=ctx, year=year
    )
    if "error" in data_response:
        return [data_response]

    values_list = data_response.get("values", [])

    # Organize results by municipality.
    muni_data: dict[str, dict[str, Any]] = {}
    if year:
        # Use only records matching the specified year.
        for rec in values_list:
            if str(rec.get("period")) == year:
                muni_data[rec.get("municipality")] = rec
    else:
        # For each municipality, choose the record with the highest (latest) period.
        for rec in values_list:
            m_id = rec.get("municipality")
            if m_id is None:
                continue
            period = rec.get("period")
            if period is None:
                continue
            cur = muni_data.get(m_id)
            if cur is None or rec.get("period") > cur.get("period"):
                muni_data[m_id] = rec

    results: list[dict[str, Any]] = []
    for m_id, rec in muni_data.items():
        # Extract KPI value for the chosen gender.
        val = None
        for d in rec.get("values", []):
            if d.get("gender") == gender:
                val = d.get("value")
                break
        if val is None:
            continue
        try:
            val_float = float(val)
        except (ValueError, TypeError):
            continue

        include = False
        if operator == "above" and (val_float > cutoff):
            include = True
        elif operator == "below" and (val_float < cutoff):
            include = True
        if include:
            diff = val_float - cutoff
            results.append(
                {
                    "municipality_id": m_id,
                    "municipality_name": municipality_map.get(m_id, {}).get(
                        "title", f"Municipality {m_id}"
                    ),
                    "period": rec.get("period"),
                    "value": val_float,
                    "cutoff": cutoff,
                    "difference": diff,
                }
            )

    results.sort(key=lambda x: x["municipality_id"])
    return results
