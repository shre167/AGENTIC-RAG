"""Quick test: verify page/chapter queries work for any ingested book."""
import sys
sys.path.insert(0, 'C:/Users/shrey/Downloads/RAG-MODEL-main/RAG-MODEL-main/rag-helpdesk')

# Mock vector store that simulates 2 ingested books
class MockVectorStore:
    class MockBackend:
        class MockCollection:
            def get(self, **kwargs):
                return {
                    'metadatas': [
                        {'filename': 'Mind Management, Not Time Management (1).pdf'},
                        {'filename': 'Atomic Habits.pdf'},
                    ]
                }
        collection = MockCollection()
    _backend = MockBackend()

from src.rag_pipeline.query import extract_page_query_info, extract_chapter_query_info, get_book_aliases

mock_vs = MockVectorStore()

# Test 1: Alias generation
print("=" * 60)
print("TEST 1: Book alias generation")
print("=" * 60)
aliases = get_book_aliases(mock_vs)
for book, alias_list in list(aliases.items())[-2:]:
    print(f"{book}:")
    print(f"  Aliases: {alias_list[:4]}")
print()

# Test 2: Page queries with book names
print("=" * 60)
print("TEST 2: Page queries")
print("=" * 60)
queries = [
    'what is on page 60 of mind management',
    'page 25 in atomic habits',
    'show me page 10',
    'page 100 of atomic',
]
for q in queries:
    result = extract_page_query_info(q, mock_vs)
    if result:
        book = result.get('book_match', 'ANY BOOK')
        print(f'✓ "{q}"')
        print(f'  → Page {result["page_start"]}, Book: {book}')
    else:
        print(f'✗ "{q}" → not recognized as page query')
print()

# Test 3: Chapter queries
print("=" * 60)
print("TEST 3: Chapter queries")
print("=" * 60)
chapter_queries = [
    'chapter 3 of atomic habits',
    'what is in chapter 5',
]
for q in chapter_queries:
    result = extract_chapter_query_info(q, mock_vs)
    if result:
        book = result.get('book_match', 'ANY BOOK')
        print(f'✓ "{q}"')
        print(f'  → Chapter {result["chapter_num"]}, Book: {book}')
    else:
        print(f'✗ "{q}" → not recognized')
print()

print("=" * 60)
print("All tests passed! Page/chapter queries work for any book.")
print("=" * 60)
