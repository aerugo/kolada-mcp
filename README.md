# Kolada MCP Server

A clean, standalone MCP (Model Context Protocol) server for Sweden's Kolada municipal statistics API.

## Overview

This server provides AI applications with access to Sweden's comprehensive municipal and regional statistics database. It enables natural language queries against thousands of Key Performance Indicators (KPIs) covering various aspects of Swedish public sector data.

## Features

- **9 MCP Tools** for comprehensive data access
- **Semantic Search** using Swedish BERT embeddings
- **No External Dependencies** - fully standalone (no Mima or Redis)
- **Containerized** with Docker support
- **Modern Python 3.11+** with type hints and async/await

## Available Tools

| Tool | Description |
|------|-------------|
| `list_operating_areas` | List all KPI categories with counts |
| `get_kpis_by_operating_area` | Get KPIs within a specific category |
| `search_kpis` | Semantic search for KPIs using natural language |
| `get_kpi_metadata` | Get detailed metadata for a specific KPI |
| `fetch_kolada_data` | Fetch raw KPI data for municipalities |
| `analyze_kpi_across_municipalities` | Comparative analysis with rankings |
| `compare_kpis` | Compare two KPIs (difference or correlation) |
| `list_municipalities` | List municipalities/regions |
| `filter_municipalities_by_kpi` | Filter by KPI threshold |

## Installation

### Using pip

```bash
pip install -e .
```

### Using Docker

```bash
docker-compose up -d kolada-mcp
```

## Usage

### Stdio Mode (Default)

```bash
kolada-mcp
# or
python -m kolada_mcp
```

### HTTP Mode

```bash
MCP_TRANSPORT=http PORT=8001 kolada-mcp
```

### Claude Desktop Integration

Add to your Claude Desktop configuration:

```json
{
  "mcpServers": {
    "kolada": {
      "command": "kolada-mcp"
    }
  }
}
```

## Configuration

Environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `MCP_TRANSPORT` | `stdio` | Transport mode (`stdio` or `http`) |
| `PORT` | `8001` | HTTP server port |
| `LOG_LEVEL` | `INFO` | Logging level |

## Development

### Setup

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install with dev dependencies
pip install -e ".[dev]"
```

### Running Tests

```bash
pytest
```

### Code Quality

```bash
# Linting
ruff check src tests

# Type checking
mypy src
```

## Architecture

```
src/kolada_mcp/
├── __init__.py          # Package init
├── __main__.py          # Entry point
├── config.py            # Settings (Pydantic)
├── server.py            # FastMCP server
├── models/
│   └── types.py         # Data models
├── services/
│   ├── kolada_client.py # API client
│   ├── embeddings.py    # Semantic search
│   └── data_processing.py
└── tools/
    ├── metadata.py      # KPI metadata tools
    ├── data.py          # Data fetching tools
    ├── municipality.py  # Municipality tools
    └── comparison.py    # Comparison tools
```

## License

Apache-2.0
