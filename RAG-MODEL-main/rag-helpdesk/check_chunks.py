import sys
sys.path.insert(0, '.')
from src.vector_store import ChromaVectorStore
from pathlib import Path

c = ChromaVectorStore()
print('Chroma chunks:', c.get_collection_count())
data = c.collection.get(include=['metadatas'])
books = {}
for m in data.get('metadatas', []):
    book = m.get('book_title', m.get('book', 'Unknown'))
    books[book] = books.get(book, 0) + 1
for b, n in sorted(books.items(), key=lambda x: -x[1]):
    print(b + ': ' + str(n))
