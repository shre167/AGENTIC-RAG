import sys
sys.path.insert(0, 'rag-helpdesk')

from pathlib import Path
from rag_helpdesk.src.vector_store import ChromaVectorStore

dirs = [
    ('rag-helpdesk/vectorstore', 'Main - vectorstore'),
    ('rag-helpdesk/vectorstore_3072_partial', 'Backup1 - vectorstore_3072_partial'),
    ('rag-helpdesk/vectorstore_mixed_backup', 'Backup2 - vectorstore_mixed_backup'),
    ('rag-helpdesk/vectorstore_old', 'Old - vectorstore_old'),
    ('chroma_db', 'Parent folder - chroma_db'),
]

total_chunks = 0

for dir_path, label in dirs:
    print(f'\n=== {label} ===')
    try:
        c = ChromaVectorStore(persist_directory=Path(dir_path))
        cnt = c.get_collection_count()
        print(f'Chunks: {cnt}')
        total_chunks += cnt
        
        if cnt > 0:
            data = c.collection.get(include=['metadatas'])
            books = {}
            for m in data.get('metadatas', []):
                book = m.get('book_title', m.get('book', 'Unknown'))
                books[book] = books.get(book, 0) + 1
            
            for b, n in sorted(books.items(), key=lambda x: -x[1]):
                print(f'  {b}: {n}')
    except Exception as e:
        print(f'Error: {e}')

print(f'\n=== TOTAL CHUNKS: {total_chunks} ===')
