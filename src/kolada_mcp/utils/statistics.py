import sys
import statistics
from typing import Any

def calculate_summary_stats(
    values: list[float], prefix: str = ""
) -> dict[str, float | int | None]:
    """
    Given a list of float values, computes min, max, mean, median, and count.
    Uses an optional prefix for keys (e.g., "" vs "delta_").
    """
    summary_stats: dict[str, float | int | None] = {
        f"{prefix}min": None,
        f"{prefix}max": None,
        f"{prefix}mean": None,
        f"{prefix}median": None,
        "count": len(values),
    }

    if values:
        try:
            summary_stats[f"{prefix}min"] = min(values)
            summary_stats[f"{prefix}max"] = max(values)
            summary_stats[f"{prefix}mean"] = statistics.mean(values)
            summary_stats[f"{prefix}median"] = statistics.median(values)
        except statistics.StatisticsError as stat_err:
            print(
                f"Warning: Could not calculate statistics: {stat_err}", file=sys.stderr
            )

    return summary_stats


def rank_and_slice_municipalities(
    data: list[dict[str, Any]],
    sort_key: str,
    sort_order: str,
    limit: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Sorts a list of dict items by sort_key (e.g. 'value' or 'delta_value'),
    slices out top, bottom, and median sub-lists, and returns them.
    """
    is_descending: bool = sort_order.lower() == "desc"
    sorted_data: list[dict[str, Any]] = sorted(
        data,
        key=lambda x: (x.get(sort_key, 0.0), x.get("municipality_id", "")),
        reverse=is_descending,
    )
    count_data: int = len(sorted_data)
    if count_data == 0:
        empty_list: list[dict[str, Any]] = []
        return empty_list, empty_list, empty_list

    safe_limit: int = max(1, min(limit, count_data))

    top_municipalities: list[dict[str, Any]] = sorted_data[:safe_limit]
    bottom_municipalities: list[dict[str, Any]] = sorted_data[-safe_limit:]
    bottom_municipalities.reverse()

    median_municipalities: list[dict[str, Any]] = []
    if count_data > 0:
        n: int = count_data
        median_rank_index_lower: int = (n - 1) // 2
        start_offset: int = safe_limit // 2
        median_start_index: int = max(0, median_rank_index_lower - start_offset)
        median_start_index = min(median_start_index, n - safe_limit)
        median_start_index = max(0, median_start_index)
        median_end_index: int = median_start_index + safe_limit
        median_municipalities = sorted_data[median_start_index:median_end_index]

    return top_municipalities, bottom_municipalities, median_municipalities
