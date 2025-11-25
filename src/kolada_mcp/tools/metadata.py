"""Metadata tools for Kolada MCP.

These tools provide access to KPI metadata, operating areas,
and semantic search capabilities.
"""

import logging
from typing import Any

import numpy as np

from kolada_mcp.models.types import ServerContext

logger = logging.getLogger(__name__)


async def list_operating_areas(ctx: ServerContext) -> dict[str, Any]:
    """List all operating areas (KPI categories) with counts.

    This tool retrieves the thematic categories of KPIs available
    in the Kolada database.

    Args:
        ctx: The server context containing cached data

    Returns:
        Dictionary with 'areas' key containing list of operating area summaries
    """
    return {"areas": ctx.operating_areas_summary}


async def get_kpis_by_operating_area(
    ctx: ServerContext, operating_area: str
) -> dict[str, Any]:
    """Get all KPIs within a specific operating area.

    Args:
        ctx: The server context containing cached data
        operating_area: The operating area/category to filter by

    Returns:
        Dictionary with 'kpis' key containing matching KPIs
    """
    operating_area_lower = operating_area.lower()

    matching_kpis = []
    for kpi in ctx.kpis:
        kpi_area = kpi.get("operating_area") or ""
        # Handle comma-separated areas
        areas = [a.strip().lower() for a in kpi_area.split(",")]
        if operating_area_lower in areas:
            matching_kpis.append({
                "id": kpi["id"],
                "title": kpi["title"],
                "operating_area": kpi.get("operating_area"),
            })

    return {"kpis": matching_kpis, "count": len(matching_kpis)}


async def get_kpi_metadata(ctx: ServerContext, kpi_id: str) -> dict[str, Any]:
    """Get detailed metadata for a specific KPI.

    Args:
        ctx: The server context containing cached data
        kpi_id: The unique KPI identifier (e.g., "N00945")

    Returns:
        The KPI metadata or an error dictionary if not found
    """
    kpi = ctx.get_kpi(kpi_id)

    if kpi is None:
        return {"error": f"KPI with id '{kpi_id}' not found"}

    return dict(kpi)


async def search_kpis(
    ctx: ServerContext, query: str, limit: int = 20
) -> dict[str, Any]:
    """Search for KPIs using semantic similarity.

    This tool uses pre-computed embeddings to find KPIs that are
    semantically similar to the search query.

    Args:
        ctx: The server context containing cached data
        query: Natural language search query
        limit: Maximum number of results to return (default: 20)

    Returns:
        Dictionary with 'results' key containing matching KPIs with scores
    """
    if ctx.embeddings is None or len(ctx.embedding_ids) == 0:
        return {"error": "Embeddings not available", "results": []}

    # Encode query
    query_embedding = ctx.sentence_model.encode([query], normalize_embeddings=True)
    query_embedding = query_embedding.flatten()

    # Normalize query embedding
    norm = np.linalg.norm(query_embedding)
    if norm > 0:
        query_embedding = query_embedding / norm

    # Compute cosine similarity
    similarities = np.dot(ctx.embeddings, query_embedding)

    # Get top-k indices
    k = min(limit, len(ctx.embedding_ids))
    top_indices = np.argsort(similarities)[-k:][::-1]

    results = []
    for idx in top_indices:
        kpi_id = ctx.embedding_ids[idx]
        kpi = ctx.get_kpi(kpi_id)
        if kpi:
            results.append({
                "id": kpi_id,
                "title": kpi.get("title", ""),
                "operating_area": kpi.get("operating_area"),
                "score": float(similarities[idx]),
            })

    return {"results": results, "query": query}
