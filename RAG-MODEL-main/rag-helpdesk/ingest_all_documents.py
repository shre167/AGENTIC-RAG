#!/usr/bin/env python3
"""
COMPLETE DOCUMENT INGESTION SCRIPT
1. Checks current Chroma chunks
2. Finds all PDF/text documents
3. Creates embeddings for missing documents
4. Ingest into PostgreSQL
"""
import sys
import os
from pathlib import Path

# Add to path
sys.path.insert(0, str(Path(__file__).parent))

def analyze_current_state():
    """Check what we have in Chroma and knowledge base."""
    print("=" * 70)
    print("ANALYZING CURRENT STATE")
    print("=" * 70)
    
    # Check Chroma
    try:
        from src.vector_store import ChromaVectorStore
        from src.config import VECTORSTORE_DIR
        
        print("1. Checking Chroma vector store...")
        chroma = ChromaVectorStore(persist_directory=VECTORSTORE_DIR)
        chroma_count = chroma.get_collection_count()
        print(f"   Current chunks in Chroma: {chroma_count}")
        
        if chroma_count > 0:
            data = chroma.collection.get(limit=100, include=["metadatas"])
            books = {}
            for m in data.get("metadatas", []):
                book = m.get("book_title", m.get("book", "Unknown"))
                filename = m.get("filename", "Unknown")
                if book and book != "Unknown":
                    books[book] = books.get(book, 0) + 1
            
            if books:
                print(f"   Books currently chunked ({len(books)}):")
                for book, cnt in sorted(books.items(), key=lambda x: -x[1]):
                    print(f"     {book}: {cnt} chunks")
        
    except Exception as e:
        print(f"   Error reading Chroma: {e}")
        chroma_count = 0
    
    # Check knowledge base
    print("\n2. Checking knowledge_base directory...")
    from src.document_loader import list_knowledge_files
    
    kb_path = Path("knowledge_base")
    all_files = list_knowledge_files(kb_path)
    
    print(f"   Total files in knowledge_base: {len(all_files)}")
    print("   File types:")
    
    file_types = {}
    for f in all_files:
        ext = f.suffix.lower()
        file_types[ext] = file_types.get(ext, 0) + 1
    
    for ext, count in sorted(file_types.items(), key=lambda x: -x[1]):
        print(f"     {ext}: {count} files")
    
    # List some files
    print("\n3. Sample files:")
    for i, f in enumerate(sorted(all_files)[:10]):
        print(f"   {i+1:2d}. {f.name}")
    
    if len(all_files) > 10:
        print(f"   ... and {len(all_files) - 10} more files")
    
    return {
        "chroma_chunks": chroma_count,
        "total_files": len(all_files),
        "files": all_files
    }

def setup_postgresql():
    """Configure PostgreSQL connection."""
    print("\n" + "=" * 70)
    print("SETTING UP POSTGRESQL")
    print("=" * 70)
    
    # Update environment to use PostgreSQL
    os.environ["USE_POSTGRES"] = "true"
    
    # Get or prompt for database URL
    db_url = os.getenv("DATABASE_URL", "")
    if not db_url:
        print("Please enter your PostgreSQL connection details:")
        password = input("PostgreSQL password for 'postgres' user: ").strip()
        db_url = f"postgresql://postgres:{password}@localhost:5432/rag_helpdesk"
        os.environ["DATABASE_URL"] = db_url
    
    print(f"Database URL: {db_url.split(':')[0]}://{db_url.split(':')[1]}:{db_url.split(':')[2]}//...")
    
    # Test connection
    try:
        from src.postgres_vector_store_simple import PostgresVectorStore
        
        print("Testing PostgreSQL connection...")
        store = PostgresVectorStore()
        postgres_count = store.get_collection_count()
        print(f"Chunks in PostgreSQL: {postgres_count}")
        
        return store
        
    except Exception as e:
        print(f"❌ PostgreSQL connection failed: {e}")
        print("\nPlease ensure:")
        print("  1. PostgreSQL is installed")
        print("  2. Database 'rag_helpdesk' exists")
        print("  3. PostgreSQL service is running")
        return None

def ingest_all_documents():
    """Main ingestion function."""
    print("\n" + "=" * 70)
    print("INGESTING ALL DOCUMENTS")
    print("=" * 70)
    
    # Import what we need
    from src.rag_pipeline import RAGPipeline
    from src.config import KNOWLEDGE_BASE_DIR, VECTORSTORE_DIR
    from pathlib import Path
    
    print("Initializing RAG pipeline...")
    pipeline = RAGPipeline(
        knowledge_base_path=Path(KNOWLEDGE_BASE_DIR),
        vectorstore_path=Path(VECTORSTORE_DIR),
    )
    
    print("Starting full ingestion of all documents...")
    try:
        result = pipeline.ingest_documents()
        
        print("\n✅ INGESTION COMPLETE!")
        print(f"Files processed: {result.get('files_processed', 0)}")
        print(f"Chunks indexed: {result.get('chunks_indexed', 0)}")
        print(f"Chunks skipped (already exist): {result.get('chunks_skipped', 0)}")
        print(f"Total chunks: {result.get('chunks_total', 0)}")
        
        return result
        
    except Exception as e:
        print(f"❌ Ingestion failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Main orchestration function."""
    print("\n📚 RAG HELPDESK - COMPLETE DOCUMENT INGESTION")
    print("This will:")
    print("  1. Analyze current state")
    print("  2. Set up PostgreSQL (if needed)")
    print("  3. Create embeddings and chunks for ALL documents")
    print("  4. Store everything in PostgreSQL")
    print()
    
    response = input("Proceed with complete ingestion? (yes/no): ").strip().lower()
    if response != "yes":
        print("Cancelled.")
        return
    
    # Step 1: Analyze
    state = analyze_current_state()
    
    # Step 2: Setup PostgreSQL
    store = setup_postgresql()
    if not store:
        print("PostgreSQL setup failed. Exiting.")
        return
    
    # Step 3: Ingest
    result = ingest_all_documents()
    
    # Step 4: Verify
    if result:
        print("\n" + "=" * 70)
        print("VERIFICATION")
        print("=" * 70)
        
        final_count = store.get_collection_count()
        print(f"Final chunk count in PostgreSQL: {final_count}")
        
        print("\n✅ All done! Your RAG system is now fully populated in PostgreSQL.")
        print("\nTo use it:")
        print("  1. Update .env: USE_POSTGRES=true")
        print("  2. Restart app: streamlit run app.py")
        print("  3. Go to Database page to verify all chunks are there")

if __name__ == "__main__":
    main()
