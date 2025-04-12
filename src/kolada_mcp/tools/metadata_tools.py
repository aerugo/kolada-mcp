import sys

import numpy as np
from mcp.server.fastmcp.server import Context

from kolada_mcp.models.types import KoladaKpi, KoladaLifespanContext
from kolada_mcp.utils.context import safe_get_lifespan_context  # type: ignore[Context]


async def list_operating_areas(ctx: Context) -> list[dict[str, str | int]]:  # type: ignore[Context]
    """
    **Step 1: Discover KPI Categories.**
    Retrieves a summary of all available 'operating areas' (thematic categories)
    for Kolada KPIs, along with the number of KPIs in each area.
    Use this tool first to understand the available categories before filtering KPIs.
    The data is sourced from the server's cache, populated at startup.

    Example Output:
    [
        {'operating_area': 'Demographics', 'kpi_count': 50},
        {'operating_area': 'Economy', 'kpi_count': 120},
        ...
    ]
    """
    lifespan_ctx: KoladaLifespanContext | None = safe_get_lifespan_context(ctx)
    if not lifespan_ctx:
        error_list: list[dict[str, str]] = [{"error": "Server context invalid."}]
        raise RuntimeError(
            "Server context invalid. Unable to retrieve operating areas."
            f" Context: {error_list}"
        )

    summary: list[dict[str, str | int]] = lifespan_ctx.get(
        "operating_areas_summary", []
    )
    if not summary:
        print("Warning: Operating areas summary is empty in context.", file=sys.stderr)
        empty_list: list[dict[str, str | int]] = []
        return empty_list
    return summary


async def get_kpis_by_operating_area(
    operating_area: str,
    ctx: Context,  # type: ignore[Context]
) -> list[KoladaKpi]:
    """
    **Step 2: Filter KPIs by Category.**
    Retrieves a list of Kolada KPIs that belong to the specified 'operating_area'.
    Use this tool *after* identifying a relevant area using `list_operating_areas`.
    Provide the exact operating area name obtained from `list_operating_areas`.
    The data is sourced from the server's cache. Note that some KPIs might belong
    to multiple areas; this tool checks if the *specified* area is associated
    with the KPI.

    Args:
        operating_area: The exact name of the operating area to filter by.
        ctx: The server context (injected automatically).

    Returns:
        A list of KoladaKpi objects matching the area, or an empty list.
    """
    lifespan_ctx: KoladaLifespanContext | None = safe_get_lifespan_context(ctx)
    if not lifespan_ctx:
        empty_list: list[KoladaKpi] = []
        return empty_list

    kpi_list: list[KoladaKpi] = lifespan_ctx.get("kpi_cache", [])
    if not kpi_list:
        print("Warning: KPI cache is empty in context.", file=sys.stderr)
        empty_list: list[KoladaKpi] = []
        return empty_list

    target_area_lower: str = operating_area.lower().strip()
    matches: list[KoladaKpi] = []

    for kpi in kpi_list:
        area_field: str = kpi.get("operating_area", "").lower()
        kpi_areas: set[str] = {a.strip() for a in area_field.split(",")}
        if target_area_lower in kpi_areas:
            matches.append(kpi)

    if not matches:
        print(
            f"Info: No KPIs found for operating area '{operating_area}'.",
            file=sys.stderr,
        )
    return matches


async def get_kpi_metadata(
    kpi_id: str,
    ctx: Context,  # type: ignore[Context]
) -> KoladaKpi | dict[str, str]:
    """
    Retrieves the cached metadata (title, description, operating area) for a
    *specific* Kolada KPI using its unique ID (e.g., "N00945").
    Kolada KPIs (Key Performance Indicators) represent various metrics for
    Swedish municipalities and regions.
    Use this tool when you have identified a specific KPI ID (e.g., from
    `get_kpis_by_operating_area` or `search_kpis`) and need its details.
    This tool accesses the server's cache, not the live Kolada API.

    Args:
        kpi_id: The unique identifier of the KPI.
        ctx: The server context (injected automatically).

    Returns:
        A KoladaKpi object if found, or an error dictionary.
    """
    lifespan_ctx: KoladaLifespanContext | None = safe_get_lifespan_context(ctx)
    if not lifespan_ctx:
        return {"error": "Server context structure invalid or incomplete."}

    kpi_map: dict[str, KoladaKpi] = lifespan_ctx.get("kpi_map", {})
    kpi_obj: KoladaKpi | None = kpi_map.get(kpi_id)

    if not kpi_obj:
        print(
            f"Info: KPI metadata request failed for ID '{kpi_id}'. Not found in cache.",
            file=sys.stderr,
        )
        return {"error": f"No KPI metadata found in cache for ID: {kpi_id}"}
    return kpi_obj


async def search_kpis(
    keyword: str,
    ctx: Context,  # type: ignore[Context]
    limit: int = 20,
) -> list[KoladaKpi]:
    """
    **Purpose:** Performs a semantic search for Kolada Key Performance Indicators (KPIs)
    based on a user-provided keyword or phrase. Instead of simple text matching,
    it uses vector embeddings to find KPIs whose titles are semantically related
    to the search term, even if the exact words don't match. This is useful for
    discovering relevant KPIs when the exact ID, title, or operating area is unknown.

    **Use Cases:**
    *   "Find KPIs related to 'school results'."
    *   "Search for indicators about 'environmental quality'."
    *   "Are there any KPIs measuring 'elderly care satisfaction'?"
    *   "Look up KPIs for 'unemployment rates'."
    *   (Used as a preliminary step before using tools like `analyze_kpi_across_municipalities` or `fetch_kolada_data` if the KPI ID is not known).

    **Arguments:**
    *   `keyword` (str): The search term or phrase describing the topic of interest. The tool will find KPIs with semantically similar titles. **Required.**
    *   `ctx` (Context): The server context (automatically injected by the MCP framework). You do not need to provide this.
    *   `limit` (int, optional): The maximum number of matching KPIs to return, ordered by relevance (highest relevance first). Default is 20.

    **Core Logic:**
    1.  Accesses the pre-loaded data from the server's lifespan context (`lifespan_ctx`), specifically:
        *   The `SentenceTransformer` model (e.g., `KBLab/sentence-bert-swedish-cased`).
        *   The pre-computed `kpi_embeddings` (a NumPy array where each row is the vector embedding of a KPI title).
        *   The list of `kpi_ids` corresponding to the rows in the embeddings array.
        *   The `kpi_map` (dictionary mapping KPI IDs to their full metadata objects).
    2.  Checks if embeddings are available. If not (e.g., failed during startup), returns an empty list.
    3.  **Embeds the User Query:** Takes the input `keyword` string and uses the loaded SentenceTransformer model to convert it into a numerical vector representation (embedding). This captures the semantic meaning of the keyword.
    4.  **Calculates Similarity:** Computes the cosine similarity between the user's query vector and *all* the pre-computed KPI title vectors stored in `kpi_embeddings`. Since the embeddings are pre-normalized during startup, this is efficiently done using a matrix-vector dot product (`embeddings @ query_vec`).
    5.  **Sorts by Relevance:** Sorts the results based on the calculated similarity scores in descending order. The indices of the most similar KPI embeddings are identified.
    6.  **Selects Top N:** Takes the top `limit` indices from the sorted list.
    7.  **Retrieves KPI Metadata:** Uses the top indices to look up the corresponding `kpi_ids` and then retrieves the full `KoladaKpi` metadata objects for those IDs from the `kpi_map`.
    8.  Returns the list of found `KoladaKpi` objects.

    **Return Value:**
    *   A list of `KoladaKpi` dictionaries (containing `id`, `title`, `description`, `operating_area`).
    *   The list is sorted by semantic relevance to the `keyword`, with the most relevant KPI appearing first.
    *   The list contains at most `limit` items.
    *   Returns an empty list (`[]`) if no relevant KPIs are found or if the embeddings cache is unavailable.

    **Important Notes:**
    *   This tool operates entirely on **cached data** loaded at server startup. It does **not** call the live Kolada API.
    *   The search is **semantic**, meaning it looks for related concepts, not just exact word matches. A search for "cars" might find KPIs about "vehicle traffic".
    *   The quality of the search results depends on the chosen SentenceTransformer model and the clarity/informativeness of the cached KPI titles.
    *   It searches primarily based on **KPI titles**. While descriptions are part of the metadata, the embeddings used for the search are generated *only* from the titles for efficiency.
    *   The default `limit` is 20, but can be adjusted if more or fewer results are needed.
    """
    lifespan_ctx: KoladaLifespanContext | None = safe_get_lifespan_context(ctx)
    if not lifespan_ctx:
        empty_list: list[KoladaKpi] = []
        return empty_list

    # --- Vector-based approach (while keeping the original docstring) ---
    model = lifespan_ctx["sentence_model"]
    embeddings = lifespan_ctx["kpi_embeddings"]
    kpi_ids = lifespan_ctx["kpi_ids"]
    kpi_map = lifespan_ctx["kpi_map"]

    if embeddings.shape[0] == 0:
        print(
            "[Kolada MCP] No KPI embeddings found; returning empty list.",
            file=sys.stderr,
        )
        empty_list: list[KoladaKpi] = []
        return empty_list

    # 1) Embed user query
    query_vector = model.encode([keyword], normalize_embeddings=True)  # type: ignore[encode]
    query_vec = query_vector[0]

    # 2) Compute dot products with normalized embeddings
    #    (We assume embeddings is already normalized)
    sims = embeddings @ query_vec

    # 3) Sort descending by similarity
    indices_sorted = np.argsort(-sims)
    top_indices = indices_sorted[:limit]

    results: list[KoladaKpi] = []
    for idx in top_indices:
        if kpi_ids[idx] in kpi_map:
            results.append(kpi_map[kpi_ids[idx]])

    return results
