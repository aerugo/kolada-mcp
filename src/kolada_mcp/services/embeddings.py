import os
import sys
from typing import Any

import numpy as np
import numpy.typing as npt
from kolada_mcp.models.types import KoladaKpi
from sentence_transformers import SentenceTransformer


async def load_or_create_embeddings(
    all_kpis: list[KoladaKpi], model: SentenceTransformer
) -> tuple[npt.NDArray[np.float32], list[str]]:
    """
    Loads or creates sentence embeddings for KPI titles. Uses a cache file
    so that if new KPIs appear, only the missing ones are embedded.

    This version will look for the embeddings cache in two specific places:
      1. ./kpi_embeddings.npz
      2. mcp-servers/kolada-mcp/src/kolada_mcp/kpi_embeddings.npz

    If found in either location, that file is used. Otherwise, the cache is
    treated as missing and a new one will be created and saved (to whichever
    cache path you prefer—below it saves back to the path from which it
    loaded if found; otherwise, it defaults to the first path in the list).

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

    # --- Step 1: Determine which cache file to use ---
    possible_cache_paths = [
        os.getcwd(),
        os.path.join(
            os.getcwd(),
            "src",
            "kolada_mcp",
        ),
    ]

    selected_cache_file = None
    print(
        "[Kolada MCP] Checking for embeddings cache in the following paths:",
        file=sys.stderr,
    )
    for path in possible_cache_paths:
        test_path = os.path.join(path, "kpi_embeddings.npz")
        if os.path.isfile(test_path):
            selected_cache_file = test_path
            break

    if selected_cache_file:
        print(
            f"[Kolada MCP] Found embeddings cache at {selected_cache_file}",
            file=sys.stderr,
        )
    else:
        print(
            "[Kolada MCP] No embeddings cache found in either location.",
            file=sys.stderr,
        )

    if selected_cache_file is not None:
        cache_file = selected_cache_file
    else:
        cache_file = possible_cache_paths[0]
        print(
            f"[Kolada MCP] Will create a new cache file at: {cache_file}",
            file=sys.stderr,
        )

    # --- Step 2: Prepare lists of IDs and titles ---
    kpi_ids_list: list[str] = []
    titles_list: list[str] = []
    for kpi_obj in all_kpis:
        k_id = kpi_obj["id"]
        title_str: str = kpi_obj.get("title", "")
        kpi_ids_list.append(k_id)
        titles_list.append(title_str)

    # --- Step 3: Try loading existing embeddings (if a file was found) ---
    existing_embeddings: npt.NDArray[np.float32] | None = None
    loaded_ids: list[str] = []

    # Print out debug info
    print(f"[Kolada MCP] Current working directory: {os.getcwd()}", file=sys.stderr)
    print(
        f"[Kolada MCP] Files in current directory: {os.listdir(os.getcwd())}",
        file=sys.stderr,
    )

    if selected_cache_file and os.path.isfile(selected_cache_file):
        try:
            if not selected_cache_file.endswith(".npz"):
                print(
                    f"[Kolada MCP] WARNING: Cache file is not a .npz file: {selected_cache_file}",
                    file=sys.stderr,
                )
            else:
                cache_data: dict[str, Any] = dict(
                    np.load(selected_cache_file, allow_pickle=True)
                )
                existing_embeddings = cache_data.get("embeddings", None)
                loaded_ids_arr: npt.NDArray[np.str_] = cache_data.get("kpi_ids", [])
                loaded_ids = [str(id) for id in loaded_ids_arr]

            if existing_embeddings is None or existing_embeddings.size == 0:
                print(
                    "[Kolada MCP] WARNING: No valid embeddings found in cache.",
                    file=sys.stderr,
                )
                existing_embeddings = None

        except Exception as ex:
            print(f"[Kolada MCP] Failed to load .npz cache: {ex}", file=sys.stderr)
            existing_embeddings = None

    # --- Step 4: Build a map of existing embeddings ---
    embedding_map: dict[str, npt.NDArray[np.float32]] = {}
    if existing_embeddings is not None:
        for idx, old_id in enumerate(loaded_ids):
            embedding_map[old_id] = existing_embeddings[idx]

    # --- Step 5: Determine which KPIs need new embeddings ---
    missing_indices: list[int] = []
    missing_ids: list[str] = []
    missing_titles: list[str] = []

    for i, k_id in enumerate(kpi_ids_list):
        if k_id not in embedding_map:
            missing_ids.append(k_id)
            missing_titles.append(titles_list[i])
            missing_indices.append(i)

    # --- Step 6: Prepare final_embeddings array ---
    embedding_dim = None
    if len(embedding_map) > 0:
        # Infer dimension from existing data
        embedding_dim = next(iter(embedding_map.values())).shape[0]

    final_embeddings: npt.NDArray[np.float32] | None = None
    if kpi_ids_list:
        if embedding_dim:
            final_embeddings = np.zeros(
                (len(kpi_ids_list), embedding_dim), dtype=np.float32
            )

    # --- Step 7: Copy any existing embeddings into the final array ---
    if final_embeddings is not None and embedding_map:
        for i, k_id in enumerate(kpi_ids_list):
            if k_id in embedding_map:
                final_embeddings[i] = embedding_map[k_id]

    # --- Step 8: Generate new embeddings (if needed) ---
    if missing_ids:
        print(
            f"[Kolada MCP] Generating embeddings for {len(missing_ids)} new KPIs...",
            file=sys.stderr,
        )
        new_embeds = model.encode(  # type: ignore
            missing_titles,
            show_progress_bar=True,
            normalize_embeddings=True,
        )

        if final_embeddings is None:
            embedding_dim = new_embeds.shape[1]  # type: ignore
            final_embeddings = np.zeros(
                (len(kpi_ids_list), embedding_dim), dtype=np.float32  # type: ignore
            )

        # Place new embeddings in final
        for j, idx in enumerate(missing_indices):
            final_embeddings[idx] = new_embeds[j]

    # If we have absolutely no final_embeddings (e.g., empty KPI list):
    if final_embeddings is None:
        return np.array([], dtype=np.float32), []

    # --- Step 9: Save updated embeddings back to disk ---
    try:
        np.savez(
            cache_file,
            embeddings=final_embeddings,
            kpi_ids=np.array(kpi_ids_list),
        )
        print(
            f"[Kolada MCP] Embeddings saved (updated) to {cache_file}",
            file=sys.stderr,
        )
    except Exception as ex:
        print(f"[Kolada MCP] WARNING: Failed to save embeddings: {ex}", file=sys.stderr)

    return final_embeddings, kpi_ids_list
