"""Data processing utilities for Kolada data.

This module provides functions for transforming and processing
Kolada API data into formats suitable for analysis and display.
"""

from collections import defaultdict
from typing import Any

from kolada_mcp.models.types import (
    FlatDataRow,
    KoladaDataPoint,
    KoladaKpi,
    KoladaMunicipality,
    OperatingAreaSummary,
)


class DataProcessor:
    """Processor for transforming Kolada data.

    This class provides static methods for data transformation,
    following the composition pattern for easy testing and reuse.
    """

    @staticmethod
    def flatten_data_points(
        data_points: list[KoladaDataPoint],
        municipality_map: dict[str, KoladaMunicipality],
    ) -> list[dict[str, Any]]:
        """Flatten nested data points into tabular format.

        Converts the nested Kolada API response into a flat structure
        suitable for DataFrame operations.

        Args:
            data_points: List of data points from Kolada API
            municipality_map: Mapping of municipality IDs to names

        Returns:
            List of flattened dictionaries
        """
        rows: list[dict[str, Any]] = []

        for point in data_points:
            municipality = municipality_map.get(point["municipality"])
            municipality_name = municipality["title"] if municipality else point["municipality"]

            for value_entry in point.get("values", []):
                rows.append({
                    "kpi_id": point["kpi"],
                    "municipality_id": point["municipality"],
                    "municipality_name": municipality_name,
                    "period": point["period"],
                    "gender": value_entry.get("gender", "T"),
                    "value": value_entry.get("value"),
                    "count": value_entry.get("count", 1),
                })

        return rows

    @staticmethod
    def compute_operating_areas_summary(
        kpis: list[KoladaKpi],
    ) -> list[OperatingAreaSummary]:
        """Compute summary of KPIs by operating area.

        Args:
            kpis: List of all KPIs

        Returns:
            List of operating area summaries with counts
        """
        area_counts: dict[str, int] = defaultdict(int)

        for kpi in kpis:
            # Handle comma-separated operating areas
            operating_area = kpi.get("operating_area") or "Okänd"
            areas = [a.strip() for a in operating_area.split(",")]
            for area in areas:
                if area:
                    area_counts[area] += 1

        summaries = [
            OperatingAreaSummary(operating_area=area, kpi_count=count)
            for area, count in sorted(area_counts.items())
        ]

        return summaries

    @staticmethod
    def filter_by_gender(
        rows: list[dict[str, Any]], gender: str | None = None
    ) -> list[dict[str, Any]]:
        """Filter data rows by gender.

        Args:
            rows: List of flattened data rows
            gender: Gender to filter by ("T", "M", "K") or None for all

        Returns:
            Filtered list of rows
        """
        if not gender:
            return rows
        return [r for r in rows if r.get("gender") == gender]

    @staticmethod
    def filter_by_municipality_type(
        rows: list[dict[str, Any]],
        municipality_map: dict[str, KoladaMunicipality],
        municipality_type: str | None = None,
    ) -> list[dict[str, Any]]:
        """Filter data rows by municipality type.

        Args:
            rows: List of flattened data rows
            municipality_map: Mapping of municipality IDs to objects
            municipality_type: Type to filter by ("K", "R", "L") or None for all

        Returns:
            Filtered list of rows
        """
        if not municipality_type:
            return rows

        return [
            r for r in rows
            if municipality_map.get(r["municipality_id"], {}).get("type") == municipality_type
        ]

    @staticmethod
    def get_latest_period_per_municipality(
        rows: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Get only the latest period for each municipality.

        Args:
            rows: List of flattened data rows

        Returns:
            List with only the latest period per municipality
        """
        # Group by municipality and find max period
        latest_periods: dict[str, int] = {}
        for row in rows:
            mun_id = row["municipality_id"]
            period = row["period"]
            if mun_id not in latest_periods or period > latest_periods[mun_id]:
                latest_periods[mun_id] = period

        # Filter to only latest periods
        return [
            r for r in rows
            if r["period"] == latest_periods.get(r["municipality_id"])
        ]

    @staticmethod
    def compute_statistics(values: list[float]) -> dict[str, float | None]:
        """Compute basic statistics for a list of values.

        Args:
            values: List of numeric values

        Returns:
            Dictionary with min, max, mean, median statistics
        """
        if not values:
            return {"min": None, "max": None, "mean": None, "median": None}

        sorted_values = sorted(values)
        n = len(sorted_values)

        mean = sum(values) / n
        median = (
            sorted_values[n // 2]
            if n % 2 == 1
            else (sorted_values[n // 2 - 1] + sorted_values[n // 2]) / 2
        )

        return {
            "min": min(values),
            "max": max(values),
            "mean": mean,
            "median": median,
        }

    @staticmethod
    def rank_municipalities(
        rows: list[dict[str, Any]],
        sort_direction: str = "desc",
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """Rank municipalities by value.

        Args:
            rows: List of flattened data rows with 'value' field
            sort_direction: "asc" for ascending, "desc" for descending
            limit: Maximum number of results to return

        Returns:
            Sorted and optionally limited list of rows
        """
        # Filter out None values
        valid_rows = [r for r in rows if r.get("value") is not None]

        reverse = sort_direction == "desc"
        sorted_rows = sorted(valid_rows, key=lambda r: r["value"], reverse=reverse)

        if limit:
            sorted_rows = sorted_rows[:limit]

        return sorted_rows
