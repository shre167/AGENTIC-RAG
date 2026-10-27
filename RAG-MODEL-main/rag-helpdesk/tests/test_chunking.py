import sys
from pathlib import Path

# Add rag-helpdesk/ to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.document_loader import (
    _chunk_book,
    load_documents,
    chunk_documents,
    parse_chapter_line,
    print_chunk_statistics,
)


def test_plain_pdf_without_chapter_heading_still_chunks():
    pages = [
        {
            "page": 1,
            "text": (
                "This is a substantial body paragraph. " * 40
                + "This is a substantial body paragraph. " * 40
            ),
            "blocks": [],
            "has_images": False,
            "native_text_chars": 500,
        },
        {
            "page": 2,
            "text": (
                "Another substantial paragraph with enough words to make a chunk. " * 60
            ),
            "blocks": [],
            "has_images": False,
            "native_text_chars": 500,
        },
    ]

    chunks = _chunk_book(pages, Path("plain-book.pdf"))

    assert len(chunks) > 0
    assert all(chunk.get("text", "").strip() for chunk in chunks)


def test_worded_chapter_numbers_are_parsed():
    assert parse_chapter_line("Chapter One Deep Work Is Valuable") == (1, "Deep Work Is Valuable")


def main():
    print("\n=== LOADING PDF ===")

    documents = load_documents()

    print(
        f"Documents/pages loaded: {len(documents)}"
    )

    if not documents:
        print("\nERROR: No documents were loaded.")
        return

    print("\n=== CREATING CANONICAL CHUNKS ===")

    chunks = chunk_documents(documents)

    print(
        f"Total chunks created: {len(chunks)}"
    )

    if not chunks:
        print("\nERROR: No chunks were created.")
        return

    # Print chunk statistics
    print_chunk_statistics(chunks)

    print("\n=== FIRST 10 CHUNKS ===")

    for i, chunk in enumerate(chunks[:10]):
        print("\n" + "=" * 80)

        print(
            f"Chunk: {i}"
            f" | ID: {chunk.get('chunk_id')}"
            f" | Pages: {chunk.get('page_start')}-{chunk.get('page_end')}"
            f" | Tokens: {chunk.get('token_count')}"
            f" | Chapter: {chunk.get('chapter_num')}"
            f" | Section: {chunk.get('section_title')}"
        )

        print("-" * 80)

        print(
            chunk.get("text", "")[:700]
        )

    print("\n" + "=" * 80)
    print("CHUNKING TEST COMPLETE")


if __name__ == "__main__":
    main()