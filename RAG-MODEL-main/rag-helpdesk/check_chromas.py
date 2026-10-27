import sys
sys.path.insert(0, '.')

# Check each vectorstore directory
dirs = [
    ("vectorstore", "Main"),
    ("vectorstore_3072_partial", "Backup 1"),
    ("vectorstore_mixed_backup", "Backup 2"),
]

for dirname, label in dirs:
    print(f"\n{'='*60}")
    print(f"Checking {label}: {dirname}")
    print('='*60)
    
    try:
        from pathlib import Path
        from src.vector_store import ChromaVectorStore
        
        c = ChromaVectorStore(persist_directory=Path(dirname))
        cnt = c.get_collection_count()
        print(f"Total chunks: {cnt}")
        
        if cnt > 0:
            data = c.collection.get(include=['metadatas'])
            books = {}
            for m in data.get('metadatas', []):
                book = m.get('book_title', m.get('book', 'Unknown'))
                if book and book != 'Unknown':
                    books[book] = books.get(book, 0) + 1
            
            print(f"\nBooks found ({len(books)}):")
            for book, c in sorted(books.items(), key=lambda x: -x[1]):
                print(f"  {book}: {c} chunks")
        else:
            print("  (empty)")
            
    except Exception as e:
        print(f"Error: {e}")

print("\n" + "="*60)
print("SUMMARY")
print("="*60)
