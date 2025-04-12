import sys
import traceback
from mcp.server.fastmcp import FastMCP

from lifespan.context import app_lifespan
from prompts.entry_prompt import kolada_entry_point
from tools.comparison_tools import compare_kpis  # type: ignore[Context]
from tools.data_tools import (
    analyze_kpi_across_municipalities,  # type: ignore[Context]
    fetch_kolada_data,  # type: ignore[Context]
)
from tools.municipality_tools import list_municipalities, filter_municipalities_by_kpi  # type: ignore[Context]
from tools.metadata_tools import (
    get_kpi_metadata,  # type: ignore[Context]
    get_kpis_by_operating_area,  # type: ignore[Context]
    list_operating_areas,  # type: ignore[Context]
    search_kpis,  # type: ignore[Context]
)

# Instantiate FastMCP
mcp: FastMCP = FastMCP("KoladaServer", lifespan=app_lifespan)

# Register all tool functions
mcp.tool()(list_operating_areas)  # type: ignore[Context]
mcp.tool()(get_kpis_by_operating_area)  # type: ignore[Context]
mcp.tool()(get_kpi_metadata)  # type: ignore[Context]
mcp.tool()(search_kpis)  # type: ignore[Context]
mcp.tool()(fetch_kolada_data)  # type: ignore[Context]
mcp.tool()(analyze_kpi_across_municipalities)  # type: ignore[Context]
mcp.tool()(compare_kpis)  # type: ignore[Context]
mcp.tool()(list_municipalities)  # type: ignore[Context]
mcp.tool()(filter_municipalities_by_kpi)  # type: ignore[Context]

# Register the prompt
mcp.prompt()(kolada_entry_point)

def main():
    print("[Kolada MCP Main] Script starting...", file=sys.stderr)
    try:
        print(
            "[Kolada MCP Main] Calling mcp.run(transport='stdio')...", file=sys.stderr
        )
        mcp.run(transport="stdio")
        print("[Kolada MCP Main] mcp.run() finished unexpectedly.", file=sys.stderr)
    except Exception as e:
        print(
            f"[Kolada MCP Main] EXCEPTION caught around mcp.run(): {e}", file=sys.stderr
        )
        traceback.print_exc(file=sys.stderr)
    finally:
        print("[Kolada MCP Main] Script exiting.", file=sys.stderr)

if __name__ == "__main__":
    main()
