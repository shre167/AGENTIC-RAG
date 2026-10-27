import chromadb

for p in ['vectorstore', 'vectorstore_3072_partial', 'vectorstore_mixed_backup', 'vectorstore_old']:
    try:
        c = chromadb.PersistentClient(path='c:/Users/shrey/Downloads/RAG-MODEL-main/RAG-MODEL-main/rag-helpdesk/' + p)
        col = c.get_collection('knowledge_base')
        data = col.get(include=['metadatas'])
        books = set((m.get('book_title') or m.get('filename') or m.get('book_name')) for m in data['metadatas'])
        print(f"{p}: count={len(data['ids'])}, books={books}")
    except Exception as e:
        print(f"{p}: error {e}")
