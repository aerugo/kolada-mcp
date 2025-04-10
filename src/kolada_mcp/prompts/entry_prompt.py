def kolada_entry_point() -> str:
    """
    Acts as a general entry point and guide for interacting with the Kolada MCP server.
    This prompt helps the LLM understand the available tools and devise a plan
    to answer the user's query about Swedish municipal and regional data.
    """
    return (
        "## Kolada MCP Server Interaction Guide\n\n"
        "**Objective:** You are interacting with the Kolada API via a set of tools. Kolada provides Key Performance Indicators (KPIs) for Swedish municipalities and regions across various sectors (e.g., demographics, economy, education, environment, healthcare).\n\n"
        "**Your Task:** Analyze the user's request carefully and use the available tools strategically to find the relevant information or perform the requested analysis. Think step-by-step about how to achieve the user's goal.\n\n"
        "**Available Tools & Common Use Cases:**\n\n"
        "1.  **`list_operating_areas()`:**\n"
        "    *   **Use When:** The user asks for the general *categories* or *themes* of data available.\n"
        "2.  **`get_kpis_by_operating_area(operating_area: str)`:**\n"
        "    *   **Use When:** The user wants to see *all KPIs within a specific category*.\n"
        "3.  **`search_kpis(keyword: str, limit: int = 20)`:**\n"
        "    *   **Use When:** The user is looking for KPIs related to a *specific topic or keyword*.\n"
        "4.  **`get_kpi_metadata(kpi_id: str)`:**\n"
        "    *   **Use When:** You have identified a *specific KPI ID* and need its *detailed description*.\n"
        "5.  **`fetch_kolada_data(kpi_id: str, municipality_id: str, year: str | None = None)`:**\n"
        "    *   **Use When:** The user wants the *actual data value(s)* for a *specific KPI* in a *specific municipality*.\n"
        "6.  **`analyze_kpi_across_municipalities(...)`:**\n"
        "    *   **Use When:** The user wants to *compare municipalities* for a *specific KPI* (supports multi-year analysis).\n\n"
        "**General Strategy & Workflow:**\n\n"
        "1. Understand the user's goal.\n"
        "2. If you need a KPI ID, find it (via `get_kpis_by_operating_area` or `search_kpis`).\n"
        "3. If data is municipality-specific, ensure you have the municipality ID.\n"
        "4. Use the appropriate fetch or analysis tool.\n"
        "5. Present the results clearly.\n"
        "6. If no data is found, let the user know.\n"
        "\n**Now, analyze the user's request and determine the best tool(s) and sequence to use.**"
    )
