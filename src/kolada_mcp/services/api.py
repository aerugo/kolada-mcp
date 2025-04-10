import json
import sys
import traceback
from typing import Any

import httpx


async def fetch_data_from_kolada(url: str) -> dict[str, Any]:
    """
    Helper function to fetch data from Kolada with consistent error handling.
    Now includes pagination support: if 'next_page' is present, we keep fetching
    subsequent pages and merge 'values' into one combined list.
    """
    combined_values: list[dict[str, Any]] = []
    visited_urls: set[str] = set()

    this_url: str | None = url
    async with httpx.AsyncClient() as client:
        while this_url and this_url not in visited_urls:
            visited_urls.add(this_url)
            print(f"[Kolada MCP] Fetching page: {this_url}", file=sys.stderr)
            try:
                resp = await client.get(this_url, timeout=60.0)
                resp.raise_for_status()
                data: dict[str, Any] = resp.json()
            except (
                httpx.RequestError,
                httpx.HTTPStatusError,
                json.JSONDecodeError,
            ) as ex:
                error_msg: str = f"Error accessing Kolada API: {ex}"
                print(f"[Kolada MCP] {error_msg}", file=sys.stderr)
                traceback.print_exc(file=sys.stderr)
                return {"error": error_msg, "details": str(ex), "endpoint": this_url}

            if "error" in data:
                return data

            page_values: list[dict[str, Any]] = data.get("values", [])
            combined_values.extend(page_values)

            next_url: str | None = data.get("next_page")
            if not next_url:
                this_url = None
            else:
                this_url = next_url

    return {
        "count": len(combined_values),
        "values": combined_values,
    }
