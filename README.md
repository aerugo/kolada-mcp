# Kolada MCP Server

[![https://modelcontextprotocol.io](https://badge.mcpx.dev?type=server 'MCP Server')](https://modelcontextprotocol.io)

**Kolada MCP Server** enables seamless integration between Large Language Models (LLMs) and [Kolada](https://www.kolada.se/), Sweden’s comprehensive municipal and regional statistical database. It provides structured access to thousands of Key Performance Indicators (KPIs), facilitating rich, data-driven analysis, comparisons, and explorations of public sector statistics.

## Overview

Kolada MCP Server acts as intelligent middleware between LLM-based applications and the Kolada database, allowing easy querying and analyzing of data related to Swedish municipalities and regions. With semantic search capabilities and robust analysis tools, Kolada MCP significantly simplifies navigating and interpreting the vast array of KPIs in Kolada.

## Example Queries

Ask Kolada MCP Server complex questions requiring data analysis:
- Where in Sweden should a family move to find affordable housing, good schools, and healthcare?
- Investigate the connection between unemployment and mental illness in Västernorrland.
- Identify municipalities with the highest increase in preschool quality over the last five years.
- Create a dashboard visualizing municipalities with the best and worst public transportation.

## Features
- **Semantic Search**: Natural language queries for KPIs.
- **Category Filtering**: Access KPIs grouped by thematic areas.
- **Municipal & Regional Data Retrieval**: Fetch KPI data or historical time series.
- **Multi-Year Comparative Analysis**: Evaluate KPI performance changes across municipalities.
- **Cross-KPI Correlation**: Analyze relationships between KPIs.

## Available Tools
1. **list_operating_areas**: Retrieve available KPI categories.
2. **get_kpis_by_operating_area**: List KPIs under a category.
3. **search_kpis**: Discover KPIs using semantic search.
4. **get_kpi_metadata**: Access detailed KPI metadata.
5. **fetch_kolada_data**: Retrieve KPI values.
6. **analyze_kpi_across_municipalities**: In-depth municipal KPI analysis.
7. **compare_kpis**: Evaluate KPI correlations.
8. **list_municipalities**: List municipality IDs and names.

## Quick Start
Kolada MCP Server includes pre-cached KPI metadata. Delete `kpi_embeddings.npz` to refresh.

## Installation
Use `uv` to install Kolada MCP dependencies:

```bash
uv sync
```

## Running Locally for Development
Start the server locally:

```bash
uv run mcp dev server.py
```
Open [MCP Inspector](https://github.com/modelcontextprotocol/inspector) at `http://localhost:5173` to test and debug.

## Claude Desktop Integration

Edit your `claude_desktop_config.json` to add Kolada MCP Server:

### Docker Image (Local Build)
```json
"KoladaDocker": {
  "args": [
    "run",
    "-i",
    "--rm",
    "--name",
    "kolada-mcp-managed",
    "kolada-mcp:local"
  ],
  "command": "docker",
  "env": {}
}
```

### Prebuilt Container via PyPI
```json
"KoladaPyPI": {
  "args": ["kolada-mcp"],
  "command": "/Users/hugi/.cargo/bin/uvx"
}
```

### Local UV Execution (without Docker)
Replace `[path to kolada-mcp]` with your local directory:
```json
"KoladaLocal": {
  "args": [
    "--directory",
    "[path to kolada-mcp]/src/kolada_mcp",
    "run",
    "kolada-mcp"
  ],
  "command": "uv"
}
```

Restart Claude Desktop after updating.

## Contributing
Contributions are welcome! Submit issues, enhancements, or PRs on GitHub.

## Disclaimer
Kolada MCP Server is independently developed, not endorsed by or affiliated with RKA or other organizations.

## License
Kolada MCP Server is licensed under the [Apache License 2.0](LICENSE).