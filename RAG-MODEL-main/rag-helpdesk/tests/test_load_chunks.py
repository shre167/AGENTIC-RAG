import sys
from pathlib import Path
sys.path.insert(0, r"c:\Users\shrey\Downloads\RAG-MODEL-main\RAG-MODEL-main\rag-helpdesk")
from src.config import KNOWLEDGE_BASE_DIR, VECTORSTORE_DIR
from app import _load_chunks, _pipeline

pipeline = _pipeline()
print("Calling _load_chunks...")
chunks = _load_chunks(pipeline)
print(f"Loaded {len(chunks)} chunks from _load_chunks")
if chunks:
    books = set(c.get("book_title") or c.get("filename") or c.get("book_name") for c in chunks)
    print("Books in chunks:", books)
