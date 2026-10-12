import pytest

from rag import config


@pytest.fixture(autouse=True)
def no_api_token(monkeypatch):
    """The developer's .env may set RAG_API_TOKEN: tests start without a login and turn it on themselves."""
    monkeypatch.setattr(config, "API_TOKEN", "")
