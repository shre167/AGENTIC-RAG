"""
Persistent chat history storage for RAG queries and responses.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def chat_history_path(vectorstore_path: Path) -> Path:
    """Return path to chat history file."""
    return vectorstore_path / "chat_history.jsonl"


def save_chat_entry(
    vectorstore_path: Path,
    query: str,
    answer: str,
    metadata: dict[str, Any] | None = None
) -> None:
    """Save a chat entry (query + answer + metadata) to persistent storage."""
    path = chat_history_path(vectorstore_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "query": query,
        "answer": answer,
        "metadata": metadata or {},
    }
    
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def load_chat_history(
    vectorstore_path: Path,
    limit: int | None = None
) -> list[dict[str, Any]]:
    """Load chat history from persistent storage.
    
    Args:
        vectorstore_path: Path to vector store directory
        limit: Maximum number of recent entries to return (None = all)
    
    Returns:
        List of chat entries, most recent first
    """
    path = chat_history_path(vectorstore_path)
    
    if not path.exists():
        return []
    
    entries = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                entries.append(entry)
            except json.JSONDecodeError:
                continue
    
    # Reverse to get most recent first
    entries.reverse()
    
    if limit is not None:
        entries = entries[:limit]
    
    return entries


def clear_chat_history(vectorstore_path: Path) -> None:
    """Clear all chat history."""
    path = chat_history_path(vectorstore_path)
    if path.exists():
        path.unlink()
