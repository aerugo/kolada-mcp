"""Sample data for tests."""

from typing import Any

# Sample Kolada API responses
SAMPLE_KPI_API_RESPONSE: dict[str, Any] = {
    "count": 6500,
    "values": [
        {
            "id": "N00945",
            "title": "Invånare totalt, antal",
            "description": "Total antal invånare i kommunen per den 31 december.",
            "operating_area": "Befolkning",
            "perspective": "Volymer",
            "prel_publication_date": None,
            "publication_date": "2024-02-15",
            "ou_publication_date": None,
            "auspices": "SCB",
            "is_divided_by_gender": 1,
            "municipality_type": "K,R",
            "has_ou_data": False,
        },
        {
            "id": "N01951",
            "title": "Andel förvärvsarbetande invånare 20-64 år, %",
            "description": "Andel av befolkningen i åldern 20-64 år som förvärvsarbetar.",
            "operating_area": "Arbetsmarknad",
            "perspective": "Andel",
            "prel_publication_date": None,
            "publication_date": "2024-03-01",
            "ou_publication_date": None,
            "auspices": "SCB",
            "is_divided_by_gender": 1,
            "municipality_type": "K",
            "has_ou_data": False,
        },
    ],
}

SAMPLE_MUNICIPALITY_API_RESPONSE: dict[str, Any] = {
    "count": 310,
    "values": [
        {"id": "0114", "title": "Upplands Väsby", "type": "K"},
        {"id": "0115", "title": "Vallentuna", "type": "K"},
        {"id": "0117", "title": "Österåker", "type": "K"},
        {"id": "0120", "title": "Värmdö", "type": "K"},
        {"id": "0123", "title": "Järfälla", "type": "K"},
        {"id": "0180", "title": "Stockholm", "type": "K"},
        {"id": "01", "title": "Region Stockholm", "type": "R"},
        {"id": "14", "title": "Västra Götalandsregionen", "type": "R"},
    ],
}

SAMPLE_DATA_API_RESPONSE: dict[str, Any] = {
    "count": 6,
    "values": [
        {
            "kpi": "N00945",
            "municipality": "0180",
            "period": 2023,
            "values": [
                {"gender": "T", "value": 984748, "count": 1},
                {"gender": "M", "value": 492000, "count": 1},
                {"gender": "K", "value": 492748, "count": 1},
            ],
        },
        {
            "kpi": "N00945",
            "municipality": "0180",
            "period": 2022,
            "values": [
                {"gender": "T", "value": 978770, "count": 1},
                {"gender": "M", "value": 489000, "count": 1},
                {"gender": "K", "value": 489770, "count": 1},
            ],
        },
        {
            "kpi": "N00945",
            "municipality": "1480",
            "period": 2023,
            "values": [
                {"gender": "T", "value": 590580, "count": 1},
            ],
        },
    ],
}

# Paginated response example
SAMPLE_PAGINATED_RESPONSE_PAGE1: dict[str, Any] = {
    "count": 100,
    "next_page": "https://api.kolada.se/v2/kpi?page=2",
    "values": [
        {"id": f"N{i:05d}", "title": f"KPI {i}", "operating_area": "Test"}
        for i in range(50)
    ],
}

SAMPLE_PAGINATED_RESPONSE_PAGE2: dict[str, Any] = {
    "count": 100,
    "values": [
        {"id": f"N{i:05d}", "title": f"KPI {i}", "operating_area": "Test"}
        for i in range(50, 100)
    ],
}
