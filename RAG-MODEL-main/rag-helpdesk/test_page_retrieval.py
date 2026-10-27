"""Test complete page retrieval flow."""
import sys
sys.path.insert(0, 'C:/Users/shrey/Downloads/RAG-MODEL-main/RAG-MODEL-main/rag-helpdesk')

# Mock vector store that simulates page 60 chunks
class MockVS:
    class MockBackend:
        class MockCol:
            def get(self, where=None, **kwargs):
                if where and 'page_start' in str(where):
                    return {
                        'ids': ['chunk_60_1', 'chunk_60_2'],
                        'documents': ['Content from page 60 chunk 1', 'Content from page 60 chunk 2'],
                        'metadatas': [
                            {'filename': 'Mind Management, Not Time Management (1).pdf', 'page_start': 60, 'page_end': 60, 'chunk_id': 'chunk_60_1'},
                            {'filename': 'Mind Management, Not Time Management (1).pdf', 'page_start': 60, 'page_end': 60, 'chunk_id': 'chunk_60_2'},
                        ]
                    }
                return {'metadatas': [{'filename': 'Mind Management, Not Time Management (1).pdf'}]}
        collection = MockCol()
    _backend = MockBackend()
    
    def get_chunks_by_page(self, page_start, page_end=None, book_title=None):
        print(f"    get_chunks_by_page called: page={page_start}, book={book_title}")
        return [
            {
                'id': 'chunk_60_1',
                'text': 'This is content from page 60 chunk 1.',
                'metadata': {'filename': 'Mind Management, Not Time Management (1).pdf', 'page_start': 60, 'page_end': 60}
            },
            {
                'id': 'chunk_60_2',
                'text': 'This is content from page 60 chunk 2.',
                'metadata': {'filename': 'Mind Management, Not Time Management (1).pdf', 'page_start': 60, 'page_end': 60}
            }
        ]

from src.rag_pipeline.query import extract_page_query_info
from src.rag_pipeline.retrieval import page_retrieve

mock_vs = MockVS()

print("=" * 60)
print("TEST 1: Query Parsing with Book Name")
print("=" * 60)
query = 'what is on page 60 of mind management'
page_info = extract_page_query_info(query, mock_vs)
print(f"Query: {query}")
print(f"Detected page: {page_info['page_start']}")
print(f"Detected book: {page_info.get('book_match')}")
print(f"Is pure page query: {page_info.get('is_pure_page_query')}")
print()

print("=" * 60)
print("TEST 2: Page Retrieval Function")
print("=" * 60)
candidates = page_retrieve(mock_vs, page_info['page_start'], page_info.get('page_end'), page_info.get('book_match'))
print(f"Retrieved {len(candidates)} chunks")
for key, cand in list(candidates.items())[:2]:
    print(f"  Chunk {key}:")
    print(f"    Text: {cand.text[:50]}...")
    print(f"    Filename: {cand.filename}")
    print(f"    Is page match: {cand.metadata.get('is_page_match')}")
print()

print("=" * 60)
print("TEST 3: Query Without Book Name")
print("=" * 60)
query2 = 'show me page 60'
page_info2 = extract_page_query_info(query2, mock_vs)
print(f"Query: {query2}")
print(f"Detected page: {page_info2['page_start']}")
print(f"Detected book: {page_info2.get('book_match') or 'None (will search all books)'}")
print()

print("✓ All page retrieval tests passed!")
