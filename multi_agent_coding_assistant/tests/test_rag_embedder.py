"""Tests for the Embedder abstraction and DeterministicEmbedder."""

from __future__ import annotations

import pytest

from rag.embedder import DeterministicEmbedder, Embedder


def test_deterministic_embedder_dimension() -> None:
    emb = DeterministicEmbedder(dimension=64)
    assert emb.dimension == 64


def test_deterministic_embedder_deterministic() -> None:
    emb = DeterministicEmbedder(dimension=64)
    v1 = emb.embed_query("authentication login")
    v2 = emb.embed_query("authentication login")
    assert v1 == v2


def test_deterministic_embedder_documents() -> None:
    emb = DeterministicEmbedder(dimension=64)
    vectors = emb.embed_documents(["a b", "c d"])
    assert len(vectors) == 2
    assert all(len(v) == 64 for v in vectors)


def test_deterministic_embedder_normalized() -> None:
    emb = DeterministicEmbedder(dimension=64)
    v = emb.embed_query("some text here")
    norm = sum(x * x for x in v) ** 0.5
    assert abs(norm - 1.0) < 1e-6


def test_embedder_is_abstract() -> None:
    with pytest.raises(TypeError):
        Embedder()  # type: ignore[abstract]
