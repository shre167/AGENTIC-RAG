import pytest

from src.embeddings import EmbeddingService


def test_allows_titan_model_on_capgemini_openai_endpoint():
    service = EmbeddingService(
        api_key="test-key",
        model="amazon.titan-embed-text-v2:0",
        base_url="https://openai.generative.engine.capgemini.com/v1",
    )

    assert service.model == "amazon.titan-embed-text-v2:0"
    assert service.base_url == "https://openai.generative.engine.capgemini.com/v1"
