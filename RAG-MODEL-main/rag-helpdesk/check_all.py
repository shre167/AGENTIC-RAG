import sys
sys.path.insert(0, '.')
from pathlib import Path
from src.vector_store import ChromaVectorStore

dirs = [
    ('vectorstore', 'Main'),
    ('vectorstore_3072_partial', 'Backup 1'),
    ('vectorstore_mixed_backup', 'Backup 2'),
    ('vectorstore_old', 'Old'),
    ('../chroma_db', 'Parent chroma_db'),
]

total = 0

for dir_path, label in dirs:
    print(f'\n=== {label} ===')
    try:
        c = ChromaVectorStore(persist_directory=Path(dir_path))
        cnt = c.get_collection_count()
        print(f'Chunks: {cnt}')
        total += cnt
        
        if cnt > 0:
            data = c.collection.get(include=['metadatas'])
            books = {}
            for m in data.get('metadatas', []):
                book = m.get('book_title', m.get('book', ''))
                if book:
                    books[book] = books.get(book, 0) + 1
            
            for b, n in sorted(books.items(), key=lambda x: -x[1]):
                print(f'  {b}: {n}')
    except Exception as e:
        print(f'Error: {str(e)[:50]}')

print(f'\n=== TOTAL: {total} chunks ===')
