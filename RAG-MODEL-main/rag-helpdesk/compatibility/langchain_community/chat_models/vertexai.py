"""Compatibility shim for RAGAS / legacy imports.

Vertex AI chat models were removed from langchain-community in favor of
langchain-google-vertexai. RAGAS 0.4.3 still imports:

    from langchain_community.chat_models.vertexai import ChatVertexAI

Install this module into the active environment with:
    python scripts/install_vertexai_shim.py
"""

from langchain_google_vertexai import ChatVertexAI

__all__ = ["ChatVertexAI"]
