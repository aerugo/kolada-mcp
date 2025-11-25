"""Unit tests for embeddings service."""

from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock
import tempfile

import numpy as np
import pytest

from kolada_mcp.services.embeddings import EmbeddingsService


class TestEmbeddingsService:
    """Tests for EmbeddingsService."""

    @pytest.fixture
    def mock_sentence_model(self) -> MagicMock:
        """Create a mock sentence transformer model."""
        model = MagicMock()
        # Return normalized vectors
        def mock_encode(texts: list[str], **kwargs) -> np.ndarray:
            embeddings = np.random.randn(len(texts), 768).astype(np.float32)
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            return embeddings / norms
        model.encode = mock_encode
        return model

    @pytest.fixture
    def sample_kpis(self) -> list[dict]:
        """Return sample KPIs for testing."""
        return [
            {"id": "N00945", "title": "Invånare totalt, antal"},
            {"id": "N01951", "title": "Andel förvärvsarbetande invånare"},
            {"id": "U09400", "title": "Elever som uppnått kunskapskraven"},
        ]

    @pytest.fixture
    def embeddings_service(self, mock_sentence_model: MagicMock) -> EmbeddingsService:
        """Create an EmbeddingsService instance for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "test_embeddings.npz"
            service = EmbeddingsService(
                model=mock_sentence_model,
                cache_path=cache_path,
            )
            yield service

    def test_create_embeddings(
        self, embeddings_service: EmbeddingsService, sample_kpis: list[dict]
    ) -> None:
        """Test creating embeddings for KPIs."""
        embeddings, kpi_ids = embeddings_service.create_embeddings(sample_kpis)

        assert embeddings.shape == (3, 768)
        assert len(kpi_ids) == 3
        assert kpi_ids == ["N00945", "N01951", "U09400"]

    def test_embeddings_are_normalized(
        self, embeddings_service: EmbeddingsService, sample_kpis: list[dict]
    ) -> None:
        """Test that embeddings are normalized."""
        embeddings, _ = embeddings_service.create_embeddings(sample_kpis)

        # Check that all embeddings have unit norm
        norms = np.linalg.norm(embeddings, axis=1)
        np.testing.assert_array_almost_equal(norms, np.ones(3), decimal=5)

    def test_search_similar(
        self, embeddings_service: EmbeddingsService, sample_kpis: list[dict]
    ) -> None:
        """Test semantic search functionality."""
        embeddings, kpi_ids = embeddings_service.create_embeddings(sample_kpis)
        embeddings_service.set_embeddings(embeddings, kpi_ids)

        # Search for similar KPIs
        results = embeddings_service.search("befolkning", top_k=2)

        assert len(results) == 2
        assert all("id" in r and "score" in r for r in results)
        # Scores should be between -1 and 1 (cosine similarity)
        assert all(-1 <= r["score"] <= 1 for r in results)

    def test_search_returns_top_k_results(
        self, embeddings_service: EmbeddingsService, sample_kpis: list[dict]
    ) -> None:
        """Test that search returns exactly top_k results."""
        embeddings, kpi_ids = embeddings_service.create_embeddings(sample_kpis)
        embeddings_service.set_embeddings(embeddings, kpi_ids)

        results = embeddings_service.search("test query", top_k=1)
        assert len(results) == 1

        results = embeddings_service.search("test query", top_k=3)
        assert len(results) == 3

    def test_search_with_more_k_than_embeddings(
        self, embeddings_service: EmbeddingsService, sample_kpis: list[dict]
    ) -> None:
        """Test search when top_k exceeds available embeddings."""
        embeddings, kpi_ids = embeddings_service.create_embeddings(sample_kpis)
        embeddings_service.set_embeddings(embeddings, kpi_ids)

        # Request more results than available
        results = embeddings_service.search("test query", top_k=10)
        assert len(results) == 3  # Should return all available

    def test_save_and_load_cache(
        self, mock_sentence_model: MagicMock, sample_kpis: list[dict]
    ) -> None:
        """Test saving and loading embeddings from cache."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "test_cache.npz"

            # Create service and generate embeddings
            service1 = EmbeddingsService(
                model=mock_sentence_model,
                cache_path=cache_path,
            )
            embeddings1, kpi_ids1 = service1.create_embeddings(sample_kpis)
            service1.save_cache(embeddings1, kpi_ids1)

            assert cache_path.exists()

            # Create new service and load from cache
            service2 = EmbeddingsService(
                model=mock_sentence_model,
                cache_path=cache_path,
            )
            embeddings2, kpi_ids2 = service2.load_cache()

            np.testing.assert_array_almost_equal(embeddings1, embeddings2)
            assert kpi_ids1 == kpi_ids2

    def test_load_cache_returns_none_when_missing(
        self, embeddings_service: EmbeddingsService
    ) -> None:
        """Test that load_cache returns None when cache doesn't exist."""
        result = embeddings_service.load_cache()
        assert result is None

    def test_update_embeddings_for_new_kpis(
        self, embeddings_service: EmbeddingsService, sample_kpis: list[dict]
    ) -> None:
        """Test updating embeddings when new KPIs are added."""
        # Create initial embeddings
        embeddings1, kpi_ids1 = embeddings_service.create_embeddings(sample_kpis[:2])
        embeddings_service.set_embeddings(embeddings1, kpi_ids1)

        # Add new KPI
        new_kpis = sample_kpis[:2] + [sample_kpis[2]]
        embeddings2, kpi_ids2 = embeddings_service.update_embeddings(
            new_kpis, embeddings1, kpi_ids1
        )

        assert embeddings2.shape[0] == 3
        assert len(kpi_ids2) == 3
        assert "U09400" in kpi_ids2

    def test_empty_kpis_returns_empty_embeddings(
        self, embeddings_service: EmbeddingsService
    ) -> None:
        """Test that empty KPI list returns empty embeddings."""
        embeddings, kpi_ids = embeddings_service.create_embeddings([])

        assert embeddings.shape == (0, 768)
        assert len(kpi_ids) == 0


class TestEmbeddingsServiceAsync:
    """Async tests for EmbeddingsService."""

    @pytest.fixture
    def mock_sentence_model(self) -> MagicMock:
        """Create a mock sentence transformer model."""
        model = MagicMock()
        def mock_encode(texts: list[str], **kwargs) -> np.ndarray:
            embeddings = np.random.randn(len(texts), 768).astype(np.float32)
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            return embeddings / norms
        model.encode = mock_encode
        return model

    @pytest.fixture
    def sample_kpis(self) -> list[dict]:
        """Return sample KPIs for testing."""
        return [
            {"id": "N00945", "title": "Invånare totalt, antal"},
            {"id": "N01951", "title": "Andel förvärvsarbetande invånare"},
        ]

    @pytest.mark.asyncio
    async def test_load_or_create_embeddings_creates_new(
        self, mock_sentence_model: MagicMock, sample_kpis: list[dict]
    ) -> None:
        """Test async load_or_create creates new embeddings when no cache."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "test_cache.npz"
            service = EmbeddingsService(
                model=mock_sentence_model,
                cache_path=cache_path,
            )

            embeddings, kpi_ids = await service.load_or_create_embeddings(sample_kpis)

            assert embeddings.shape == (2, 768)
            assert len(kpi_ids) == 2

    @pytest.mark.asyncio
    async def test_load_or_create_embeddings_uses_cache(
        self, mock_sentence_model: MagicMock, sample_kpis: list[dict]
    ) -> None:
        """Test async load_or_create uses cache when available."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "test_cache.npz"
            service = EmbeddingsService(
                model=mock_sentence_model,
                cache_path=cache_path,
            )

            # Create and save embeddings
            embeddings1, kpi_ids1 = service.create_embeddings(sample_kpis)
            service.save_cache(embeddings1, kpi_ids1)

            # Load from cache
            embeddings2, kpi_ids2 = await service.load_or_create_embeddings(sample_kpis)

            np.testing.assert_array_almost_equal(embeddings1, embeddings2)
            assert kpi_ids1 == kpi_ids2
