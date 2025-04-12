def build_kolada_url_for_kpi(
    base_url: str,
    kpi_id: str,
    municipality_ids: str | None,
    year: str | None
) -> str:
    """
    Constructs the Kolada API endpoint based on provided parameters.
    If municipality_ids is provided (comma-separated), use the built-in query.
    """
    if municipality_ids:
        url = f"{base_url}/data/kpi/{kpi_id}/municipality/{municipality_ids}"
        if year:
            url += f"/year/{year}"
    else:
        if year:
            url = f"{base_url}/data/kpi/{kpi_id}/year/{year}"
        else:
            url = f"{base_url}/data/kpi/{kpi_id}"
    return url
