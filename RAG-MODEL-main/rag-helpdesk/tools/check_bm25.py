import sys
from pathlib import Path
import pickle
sys.path.insert(0, r"c:\Users\shrey\Downloads\RAG-MODEL-main\RAG-MODEL-main\rag-helpdesk")
from src.config import VECTORSTORE_DIR

pkl = Path(VECTORSTORE_DIR) / "bm25_index.pkl"
if pkl.exists():
    with open(pkl, "rb") as f:
        data = pickle.load(f)
    print("BM25 keys/attributes:", dir(data))
    chunks = getattr(data, "_chunks", [])
    print("BM25 chunks count:", len(chunks))
    books = set()
    for c in chunks:
        books.add(c.get("book_title") or c.get("filename") or c.get("book_name"))
    print("Books in BM25:", books)
else:
    print("No bm25_index.pkl")
