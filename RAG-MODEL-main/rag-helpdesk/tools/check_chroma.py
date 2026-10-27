import sys
from pathlib import Path
sys.path.insert(0, r"c:\Users\shrey\Downloads\RAG-MODEL-main\RAG-MODEL-main\rag-helpdesk")
import chromadb
from src.config import VECTORSTORE_DIR

client = chromadb.PersistentClient(path=str(VECTORSTORE_DIR))
col = client.get_collection("knowledge_base")
data = col.get(include=["metadatas"])
print("Total items in Chroma:", len(data["ids"]))

books = set()
chapters_by_book = {}
counts_by_book = {}
for m in data["metadatas"]:
    b = m.get("book_title") or m.get("filename") or m.get("book_name") or "UNKNOWN"
    books.add(b)
    ch = m.get("chapter_num") or m.get("chapter") or 0
    chapters_by_book.setdefault(b, set()).add(ch)
    counts_by_book[b] = counts_by_book.get(b, 0) + 1

print("Books in Chroma:", books)
for b, count in counts_by_book.items():
    chs = chapters_by_book[b]
    print(f"Book: {b} | Chunks: {count} | Chapters: {sorted(list(chs))[:15]} | Total unique chs: {len(chs)}")

if data["metadatas"]:
    print("Sample metadata keys:", list(data["metadatas"][0].keys()))
    print("Sample metadata 0:", data["metadatas"][0])
    print("Sample metadata last:", data["metadatas"][-1])
