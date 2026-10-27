import os
import importlib


def test_doc_loader_does_not_try_remote_tokenizer_by_default(monkeypatch):
    monkeypatch.delenv("CHUNK_TOKENIZER", raising=False)
    import src.document_loader as document_loader
    importlib.reload(document_loader)

    assert document_loader.TOKENIZER_NAME == ""
    assert document_loader._get_tokenizer() is None
