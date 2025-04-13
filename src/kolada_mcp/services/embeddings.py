import os
import sys
from typing import Any

import numpy as np
import numpy.typing as npt
from kolada_mcp.config import EMBEDDINGS_CACHE_FILE
from kolada_mcp.models.types import KoladaKpi
from sentence_transformers import SentenceTransformer


async def load_or_create_embeddings(
    all_kpis: list[KoladaKpi], model: SentenceTransformer
) -> tuple[npt.NDArray[np.float32], list[str]]:
    """
    Loads or creates sentence embeddings for KPI titles. Uses a cache file
    so that if new KPIs appear, only the missing ones are embedded.

    Args:
        all_kpis: A list of KoladaKpi objects, each containing an 'id' key
            and optionally a 'title' key.
        model: A SentenceTransformer model to generate embeddings.

    Returns:
        A tuple consisting of:
          1. A numpy array of shape (len(all_kpis), embedding_dim), where
             each row corresponds to a KPI's embedding in the order of
             all_kpis.
          2. A list of KPI IDs in the order that matches the embeddings array.
    """
    # Separate out IDs and titles from the incoming KPI list
    kpi_ids_list: list[str] = []
    titles_list: list[str] = []
    for kpi_obj in all_kpis:
        k_id = kpi_obj["id"]
        title_str: str = kpi_obj.get("title", "")
        kpi_ids_list.append(k_id)
        titles_list.append(title_str)

    # Attempt to load existing cache
    existing_embeddings: npt.NDArray[np.float32] | None = None
    loaded_ids: list[str] = []

    # console log current path
    print(f"[Kolada MCP] Current working directory: {os.getcwd()}", file=sys.stderr)
    # console log cache file path
    print(f"[Kolada MCP] Cache file path: {EMBEDDINGS_CACHE_FILE}", file=sys.stderr)
    # Check if the cache file exists
    # and if it is a valid .npz file
    if os.path.exists(EMBEDDINGS_CACHE_FILE):
        print(
            f"[Kolada MCP] Found embeddings cache at {EMBEDDINGS_CACHE_FILE}",
            file=sys.stderr,
        )
        if not EMBEDDINGS_CACHE_FILE.endswith(".npz"):
            print(
                f"[Kolada MCP] WARNING: Cache file is not a .npz file: {EMBEDDINGS_CACHE_FILE}",
                file=sys.stderr,
            )
            return np.array([], dtype=np.float32), []
    if not os.path.isfile(EMBEDDINGS_CACHE_FILE):
        print(
            f"[Kolada MCP] WARNING: Cache file does not exist: {EMBEDDINGS_CACHE_FILE}",
            file=sys.stderr,
        )
        return np.array([], dtype=np.float32), []

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
            loaded_ids = [str(id) for id in loaded_ids_arr]
        except Exception as ex:
            print(f"[Kolada MCP] Failed to load .npz cache: {ex}", file=sys.stderr)
            existing_embeddings = None

        if existing_embeddings is None or existing_embeddings.size == 0:
            print(
                "[Kolada MCP] WARNING: No valid embeddings found in cache.",
                file=sys.stderr,
            )
            existing_embeddings = None

    # If we have valid cached embeddings, build a map of kpi_id -> embedding
    embedding_map: dict[str, npt.NDArray[np.float32]] = {}
    if existing_embeddings is not None:
        # Each row in existing_embeddings matches the corresponding ID in loaded_ids
        for idx, old_id in enumerate(loaded_ids):
            embedding_map[old_id] = existing_embeddings[idx]

    # Determine which KPI IDs need new embeddings
    missing_indices: list[int] = []
    missing_ids: list[str] = []
    missing_titles: list[str] = []

    for i, k_id in enumerate(kpi_ids_list):
        if k_id not in embedding_map:
            missing_ids.append(k_id)
            missing_titles.append(titles_list[i])
            missing_indices.append(i)

    # Create a final array of shape [len(all_kpis), embedding_dim]
    # If everything is missing, we will fill it after generating new embeddings
    # If partial missing, we copy from existing and generate new for the rest.
    if len(embedding_map) > 0:
        # We can infer embedding dimension from the first existing embedding
        embedding_dim = embedding_map[loaded_ids[0]].shape[0]
    else:
        # We'll infer from newly generated embeddings if no cache is usable
        embedding_dim = None

    final_embeddings: npt.NDArray[np.float32] | None = None
    if len(kpi_ids_list) > 0:
        # If we know the embedding dimension from existing data, initialize
        # the final array with zeros. Otherwise, we'll fill it after generating.
        final_embeddings = (
            np.zeros((len(kpi_ids_list), embedding_dim), dtype=np.float32)
            if embedding_dim
            else None
        )

    # Copy already existing embeddings into final_embeddings
    if final_embeddings is not None and embedding_map:
        for i, k_id in enumerate(kpi_ids_list):
            if k_id in embedding_map:
                final_embeddings[i] = embedding_map[k_id]

    # Generate embeddings for missing IDs
    if missing_ids:
        print(
            f"[Kolada MCP] Generating embeddings for {len(missing_ids)} new KPIs...",
            file=sys.stderr,
        )
        new_embeds = model.encode(  # type: ignore[encde]
            missing_titles,
            show_progress_bar=True,
            normalize_embeddings=True,
        )

        # If we had no existing dimension, set up final_embeddings now
        if final_embeddings is None:
            embedding_dim = new_embeds.shape[1]  # type: ignore[embed]
            final_embeddings = np.zeros((len(kpi_ids_list), embedding_dim), dtype=np.float32)  # type: ignore[embed]

        # Place the newly generated embeddings into final_embeddings
        for j, idx in enumerate(missing_indices):
            final_embeddings[idx] = new_embeds[j]

    # If we have absolutely no final_embeddings (edge case: no KPIs at all),
    # return an empty array and empty list
    if final_embeddings is None:
        return np.array([], dtype=np.float32), []

    # Save the combined result back to disk
    try:
        np.savez(
            EMBEDDINGS_CACHE_FILE,
            embeddings=final_embeddings,
            kpi_ids=np.array(kpi_ids_list),
        )
        print("[Kolada MCP] Embeddings (updated) saved to disk.", file=sys.stderr)
    except Exception as ex:
        print(
            f"[Kolada MCP] WARNING: Failed to save embeddings: {ex}",
            file=sys.stderr,
        )

    return final_embeddings, kpi_ids_list
