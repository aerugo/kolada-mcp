"""Main entry point for Kolada MCP Server."""

import logging
import sys

import uvicorn

from kolada_mcp.config import settings
from kolada_mcp.server import mcp


def setup_logging() -> None:
    """Configure logging for the application."""
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        stream=sys.stderr,
    )


def main() -> None:
    """Run the Kolada MCP server."""
    setup_logging()
    logger = logging.getLogger(__name__)

    if settings.mcp_transport == "stdio":
        logger.info("[Kolada MCP] Starting in stdio mode")
        mcp.run()
    else:
        logger.info(f"[Kolada MCP] Starting HTTP server on port {settings.port}")
        uvicorn.run(
            "kolada_mcp.server:mcp",
            host="0.0.0.0",
            port=settings.port,
            log_level=settings.log_level.lower(),
        )


if __name__ == "__main__":
    main()
