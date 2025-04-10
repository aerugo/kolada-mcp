import os
import sys
from typing import Any

import numpy as np
import numpy.typing as npt
from sentence_transformers import SentenceTransformer

from config import EMBEDDINGS_CACHE_FILE
from models.types import KoladaKpi


async def load_or_create_embeddings(
    all_kpis: list[KoladaKpi], model: SentenceTransformer
) -> tuple[npt.NDArray[np.float32], list[str]]:
    """
    Loads existing embeddings from cache file or creates new ones if needed.
    Returns the embeddings array and the list of KPI IDs.
    """
    kpi_ids_list: list[str] = []
    titles_list: list[str] = []

    for kpi_obj in all_kpis:
        k_id = kpi_obj["id"]
        title_str: str = kpi_obj.get("title", "")
        kpi_ids_list.append(k_id)
        titles_list.append(title_str)

    # Attempt to load cached .npz
    existing_embeddings: npt.NDArray[np.float32] | None = None
    loaded_ids = []
    if os.path.isfile(EMBEDDINGS_CACHE_FILE):
        print(
            f"[Kolada MCP] Found embeddings cache at {EMBEDDINGS_CACHE_FILE}",
            file=sys.stderr,
        )
        try:
            cache_data: dict[str, Any] = dict(
                np.load(EMBEDDINGS_CACHE_FILE, allow_pickle=True)
            )
            existing_embeddings = cache_data.get("embeddings", None)
            loaded_ids_arr: npt.NDArray[np.str_] = cache_data.get("kpi_ids", [])
            loaded_ids = loaded_ids_arr.tolist()
        except Exception as ex:
            print(f"[Kolada MCP] Failed to load .npz cache: {ex}", file=sys.stderr)
        if existing_embeddings is None or existing_embeddings.size == 0:
            print(
                "[Kolada MCP] WARNING: No valid embeddings found in cache.",
                file=sys.stderr,
            )
            existing_embeddings = None

    # Check if we can reuse the loaded embeddings
    embeddings: npt.NDArray[np.float32] | None = None
    if (
        existing_embeddings is not None
        and existing_embeddings.size > 0
        and len(loaded_ids) == len(kpi_ids_list)
        and set(loaded_ids) == set(kpi_ids_list)
    ):
        print("[Kolada MCP] Using existing cached embeddings.", file=sys.stderr)
        embeddings = existing_embeddings
    else:
        print(
            "[Kolada MCP] Generating new embeddings for all KPI titles...",
            file=sys.stderr,
        )
        embeddings = model.encode(  # type: ignore[encode]
            titles_list, show_progress_bar=True, normalize_embeddings=True
        )

        # Save them
        try:
            np.savez(
                EMBEDDINGS_CACHE_FILE,
                embeddings=embeddings,
                kpi_ids=np.array(kpi_ids_list),
            )
            print("[Kolada MCP] Embeddings saved to disk.", file=sys.stderr)
        except Exception as ex:
            print(
                f"[Kolada MCP] WARNING: Failed to save embeddings: {ex}",
                file=sys.stderr,
            )

    return embeddings, kpi_ids_list
