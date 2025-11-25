"""Embeddings service for semantic search.

This module provides functionality for creating and managing KPI embeddings
for semantic search capabilities.
"""

import asyncio
import logging
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

from kolada_mcp.config import settings

logger = logging.getLogger(__name__)


class SentenceModelProtocol(Protocol):
    """Protocol for sentence transformer model dependency injection."""

    def encode(
        self, sentences: list[str], normalize_embeddings: bool = True, **kwargs: Any
    ) -> NDArray[np.float32]: ...


class EmbeddingsService:
    """Service for managing KPI embeddings and semantic search.

    This service handles:
    - Creating embeddings for KPI titles using sentence transformers
    - Caching embeddings to disk for fast startup
    - Semantic search using cosine similarity

    Attributes:
        model: The sentence transformer model
        cache_path: Path to the embeddings cache file
        embeddings: The current embeddings matrix
        embedding_ids: KPI IDs corresponding to embeddings
    """

    EMBEDDING_DIM = 768  # Default dimension for sentence-bert models

    def __init__(
        self,
        model: SentenceModelProtocol,
        cache_path: Path = settings.embeddings_cache_path,
    ) -> None:
        """Initialize the embeddings service.

        Args:
            model: The sentence transformer model to use
            cache_path: Path to store/load cached embeddings
        """
        self.model = model
        self.cache_path = cache_path
        self.embeddings: NDArray[np.float32] | None = None
        self.embedding_ids: list[str] = []

    def set_embeddings(
        self, embeddings: NDArray[np.float32], kpi_ids: list[str]
    ) -> None:
        """Set the current embeddings and IDs.

        Args:
            embeddings: The embeddings matrix
            kpi_ids: KPI IDs corresponding to each row
        """
        self.embeddings = embeddings
        self.embedding_ids = kpi_ids

    def create_embeddings(
        self, kpis: list[dict[str, Any]]
    ) -> tuple[NDArray[np.float32], list[str]]:
        """Create embeddings for a list of KPIs.

        Args:
            kpis: List of KPI dictionaries with 'id' and 'title' keys

        Returns:
            Tuple of (embeddings matrix, list of KPI IDs)
        """
        if not kpis:
            return np.array([], dtype=np.float32).reshape(0, self.EMBEDDING_DIM), []

        titles = [kpi.get("title", "") for kpi in kpis]
        kpi_ids = [kpi["id"] for kpi in kpis]

        logger.info(f"[Kolada MCP] Creating embeddings for {len(kpis)} KPIs")

        # Encode titles with normalization
        embeddings = self.model.encode(titles, normalize_embeddings=True)

        # Ensure normalized (in case model doesn't support normalize_embeddings)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        # Avoid division by zero
        norms = np.where(norms == 0, 1, norms)
        embeddings = embeddings / norms

        return embeddings.astype(np.float32), kpi_ids

    def update_embeddings(
        self,
        kpis: list[dict[str, Any]],
        existing_embeddings: NDArray[np.float32],
        existing_ids: list[str],
    ) -> tuple[NDArray[np.float32], list[str]]:
        """Update embeddings with new KPIs.

        Only creates embeddings for KPIs not already in existing_ids.

        Args:
            kpis: Full list of KPIs
            existing_embeddings: Current embeddings matrix
            existing_ids: Current KPI IDs

        Returns:
            Tuple of (updated embeddings matrix, updated KPI IDs)
        """
        existing_set = set(existing_ids)
        new_kpis = [kpi for kpi in kpis if kpi["id"] not in existing_set]

        if not new_kpis:
            return existing_embeddings, existing_ids

        logger.info(f"[Kolada MCP] Creating embeddings for {len(new_kpis)} new KPIs")

        new_embeddings, new_ids = self.create_embeddings(new_kpis)

        # Combine existing and new
        if existing_embeddings.size == 0:
            combined_embeddings = new_embeddings
        else:
            combined_embeddings = np.vstack([existing_embeddings, new_embeddings])

        combined_ids = existing_ids + new_ids

        return combined_embeddings, combined_ids

    def search(
        self, query: str, top_k: int = 20
    ) -> list[dict[str, Any]]:
        """Search for KPIs similar to the query.

        Args:
            query: The search query
            top_k: Number of results to return

        Returns:
            List of results with 'id' and 'score' keys
        """
        if self.embeddings is None or len(self.embedding_ids) == 0:
            logger.warning("[Kolada MCP] No embeddings available for search")
            return []

        # Encode query
        query_embedding = self.model.encode([query], normalize_embeddings=True)
        query_embedding = query_embedding.flatten()

        # Normalize query embedding
        norm = np.linalg.norm(query_embedding)
        if norm > 0:
            query_embedding = query_embedding / norm

        # Compute cosine similarity (embeddings are already normalized)
        similarities = np.dot(self.embeddings, query_embedding)

        # Get top-k indices
        k = min(top_k, len(self.embedding_ids))
        top_indices = np.argsort(similarities)[-k:][::-1]

        results = [
            {"id": self.embedding_ids[idx], "score": float(similarities[idx])}
            for idx in top_indices
        ]

        return results

    def save_cache(
        self, embeddings: NDArray[np.float32], kpi_ids: list[str]
    ) -> None:
        """Save embeddings to cache file.

        Args:
            embeddings: The embeddings matrix
            kpi_ids: KPI IDs corresponding to each row
        """
        logger.info(f"[Kolada MCP] Saving embeddings cache to {self.cache_path}")
        np.savez(
            self.cache_path,
            embeddings=embeddings,
            kpi_ids=np.array(kpi_ids, dtype=object),
        )

    def load_cache(self) -> tuple[NDArray[np.float32], list[str]] | None:
        """Load embeddings from cache file.

        Returns:
            Tuple of (embeddings, kpi_ids) if cache exists, None otherwise
        """
        if not self.cache_path.exists():
            logger.info("[Kolada MCP] No embeddings cache found")
            return None

        logger.info(f"[Kolada MCP] Loading embeddings cache from {self.cache_path}")
        data = np.load(self.cache_path, allow_pickle=True)
        embeddings = data["embeddings"]
        kpi_ids = data["kpi_ids"].tolist()

        return embeddings, kpi_ids

    async def load_or_create_embeddings(
        self, kpis: list[dict[str, Any]]
    ) -> tuple[NDArray[np.float32], list[str]]:
        """Load embeddings from cache or create new ones.

        This method runs the CPU-intensive operations in a thread pool
        to avoid blocking the async event loop.

        Args:
            kpis: List of KPIs to create embeddings for

        Returns:
            Tuple of (embeddings matrix, list of KPI IDs)
        """
        # Try to load from cache first
        cached = await asyncio.to_thread(self.load_cache)

        if cached is not None:
            cached_embeddings, cached_ids = cached

            # Check if we need to update with new KPIs
            if len(cached_ids) >= len(kpis):
                self.set_embeddings(cached_embeddings, cached_ids)
                return cached_embeddings, cached_ids

            # Update with new KPIs
            embeddings, kpi_ids = await asyncio.to_thread(
                self.update_embeddings, kpis, cached_embeddings, cached_ids
            )
        else:
            # Create new embeddings
            embeddings, kpi_ids = await asyncio.to_thread(
                self.create_embeddings, kpis
            )

        # Save to cache
        await asyncio.to_thread(self.save_cache, embeddings, kpi_ids)

        self.set_embeddings(embeddings, kpi_ids)
        return embeddings, kpi_ids


def load_sentence_model(model_name: str = settings.sentence_model_name) -> Any:
    """Load the sentence transformer model.

    Args:
        model_name: Name of the model to load

    Returns:
        The loaded sentence transformer model
    """
    from sentence_transformers import SentenceTransformer

    logger.info(f"[Kolada MCP] Loading sentence model: {model_name}")
    return SentenceTransformer(model_name)
