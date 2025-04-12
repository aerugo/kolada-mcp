import json
import sys
import traceback
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, cast

import httpx
from mcp.server.fastmcp import FastMCP
from sentence_transformers import SentenceTransformer

from kolada_mcp.config import BASE_URL, KPI_PER_PAGE
from kolada_mcp.models.types import KoladaKpi, KoladaLifespanContext, KoladaMunicipality
from kolada_mcp.services.api import fetch_data_from_kolada
from kolada_mcp.services.data_processing import get_operating_areas_summary
from kolada_mcp.services.embeddings import load_or_create_embeddings


@asynccontextmanager
async def app_lifespan(server: FastMCP) -> AsyncIterator[KoladaLifespanContext]:
    """
    Initializes the Kolada MCP Server at startup. Includes stderr logging.
    Yields the dictionary that becomes ctx.request_context.lifespan_context.
    """
    print("[Kolada MCP Lifespan] Starting lifespan setup...", file=sys.stderr)

    kpi_list: list[KoladaKpi] = []
    municipality_list: list[KoladaMunicipality] = []

    print(
        "[Kolada MCP] Initializing: Fetching all KPI metadata from Kolada API...",
        file=sys.stderr,
    )
    async with httpx.AsyncClient() as client:
        next_url: str | None = f"{BASE_URL}/kpi?per_page={KPI_PER_PAGE}"
        while next_url:
            print(f"[Kolada MCP] Fetching page: {next_url}", file=sys.stderr)
            try:
                resp = await client.get(next_url, timeout=180.0)
                resp.raise_for_status()
                data: dict[str, Any] = resp.json()
            except (
                httpx.RequestError,
                httpx.HTTPStatusError,
                json.JSONDecodeError,
            ) as e:
                print(
                    f"[Kolada MCP] CRITICAL ERROR fetching Kolada KPIs: {e}",
                    file=sys.stderr,
                )
                print(f"Failed URL: {next_url}", file=sys.stderr)
                raise RuntimeError(f"Failed to initialize Kolada KPI cache: {e}") from e

            values: list[KoladaKpi] = data.get("values", [])
            kpi_list.extend(values)
            next_url = data.get("next_page")

    print(
        f"[Kolada MCP] Fetched {len(kpi_list)} total KPIs from Kolada.", file=sys.stderr
    )

    print("[Kolada MCP] Fetching municipality data...", file=sys.stderr)
    try:
        muni_resp: dict[str, Any] = await fetch_data_from_kolada(
            f"{BASE_URL}/municipality"
        )
        if "error" in muni_resp:
            raise RuntimeError(
                f"Failed to initialize municipality cache: {muni_resp['error']}"
            )
        muni_values: list[Any] = muni_resp.get("values", [])
        municipality_list = cast(list[KoladaMunicipality], muni_values)
        print(
            f"[Kolada MCP] Fetched {len(municipality_list)} municipalities/regions.",
            file=sys.stderr,
        )
    except Exception as e:
        print(
            f"[Kolada MCP] CRITICAL ERROR fetching municipality data: {e}",
            file=sys.stderr,
        )
        raise RuntimeError(f"Failed to initialize municipality cache: {e}") from e

    kpi_map: dict[str, KoladaKpi] = {}
    for kpi_obj in kpi_list:
        k_id: str | None = kpi_obj.get("id")
        kpi_map[k_id] = kpi_obj

    municipality_map: dict[str, KoladaMunicipality] = {}
    for m_obj in municipality_list:
        m_id: str | None = m_obj.get("id")
        if m_id is not None:
            municipality_map[m_id] = m_obj

    operating_areas_summary: list[dict[str, str | int]] = get_operating_areas_summary(
        kpi_list
    )
    print(
        f"[Kolada MCP] Identified {len(operating_areas_summary)} unique operating areas.",
        file=sys.stderr,
    )

    # ----------------------------------------------------------------
    # Load or create embeddings for the simpler vector-based approach
    # ----------------------------------------------------------------
    print("[Kolada MCP] Loading SentenceTransformer model...", file=sys.stderr)
    sentence_model: SentenceTransformer = SentenceTransformer(
        "KBLab/sentence-bert-swedish-cased"  # type: ignore
    )
    print("[Kolada MCP] Model loaded.", file=sys.stderr)

    all_kpis: list[KoladaKpi] = [k for k in kpi_list if "id" in k]
    embeddings, kpi_ids_list = await load_or_create_embeddings(all_kpis, sentence_model)

    # Create the final context data
    context_data: KoladaLifespanContext = {
        "kpi_cache": kpi_list,
        "kpi_map": kpi_map,
        "operating_areas_summary": operating_areas_summary,
        "municipality_cache": municipality_list,
        "municipality_map": municipality_map,
        "sentence_model": sentence_model,
        "kpi_embeddings": embeddings,
        "kpi_ids": kpi_ids_list,
    }

    print("[Kolada MCP] Initialization complete. All data cached.", file=sys.stderr)
    print(
        f"[Kolada MCP Lifespan] Yielding context with {len(kpi_list)} KPIs and {len(municipality_list)} municipalities...",
        file=sys.stderr,
    )
    try:
        yield context_data
        print(
            "[Kolada MCP Lifespan] Post-yield (server shutting down)...",
            file=sys.stderr,
        )
    except Exception as e:
        print(
            f"[Kolada MCP Lifespan] Exception DURING yield/server run?: {e}",
            file=sys.stderr,
        )
        traceback.print_exc(file=sys.stderr)
        raise
    finally:
        print(
            "[Kolada MCP Lifespan] Entering finally block (shutdown).", file=sys.stderr
        )
        print("[Kolada MCP] Shutting down.", file=sys.stderr)
