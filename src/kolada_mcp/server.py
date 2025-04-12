import sys
import traceback
from mcp.server.fastmcp import FastMCP

from kolada_mcp.lifespan.context import app_lifespan
from kolada_mcp.prompts.entry_prompt import kolada_entry_point
from kolada_mcp.tools.comparison_tools import compare_kpis  # type: ignore[Context]
from kolada_mcp.tools.data_tools import (
    analyze_kpi_across_municipalities,  # type: ignore[Context]
    fetch_kolada_data,  # type: ignore[Context]
)
from kolada_mcp.tools.municipality_tools import list_municipalities, filter_municipalities_by_kpi  # type: ignore[Context]
from kolada_mcp.tools.metadata_tools import (
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
mcp.tool()(get_kpi_metadata)      # type: ignore[Context]
mcp.tool()(search_kpis)           # type: ignore[Context]
mcp.tool()(fetch_kolada_data)     # type: ignore[Context]
mcp.tool()(analyze_kpi_across_municipalities)  # type: ignore[Context]
mcp.tool()(compare_kpis)          # type: ignore[Context]
mcp.tool()(list_municipalities)   # type: ignore[Context]
mcp.tool()(filter_municipalities_by_kpi)  # type: ignore[Context]

# Register the prompt
mcp.prompt()(kolada_entry_point)

def main():
    """
    Runs the Kolada MCP server on stdio.
    This call is synchronous—no 'await' needed.
    """
    print("[Kolada MCP] Starting server on stdio...", file=sys.stderr)
    try:
        mcp.run("stdio")
        print("[Kolada MCP] Finished cleanly.", file=sys.stderr)
    except Exception as e:
        print(f"[Kolada MCP] EXCEPTION: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
    finally:
        print("[Kolada MCP] Exiting.", file=sys.stderr)

if __name__ == "__main__":
    main()
