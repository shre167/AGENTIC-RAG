# src/document_loader.py

from __future__ import annotations

import hashlib
import os
import re
import unicodedata
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable

import pymupdf as fitz  # PyMuPDF (replaces legacy `import fitz`)


def _configure_tesseract_path() -> None:
    """Point pytesseract at the installed Windows Tesseract binary if present."""
    try:
        import pytesseract
    except Exception:
        return

    candidates = []

    env_tess = os.getenv("TESSERACT_PATH")
    if env_tess:
        candidates.append(Path(env_tess))

    candidates.extend(
        [
            Path(r"C:\Users\shrsamal\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"),
            Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe"),
            Path(r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"),
        ]
    )

    for exe in candidates:
        if exe and exe.exists():
            exe_dir = str(exe.parent)
            current_path = os.environ.get("PATH", "")
            if exe_dir not in current_path.split(os.pathsep):
                os.environ["PATH"] = exe_dir + os.pathsep + current_path
            pytesseract.pytesseract.tesseract_cmd = str(exe)
            return


_configure_tesseract_path()


BASE_DIR = Path(__file__).resolve().parent.parent
KNOWLEDGE_BASE_DIR = BASE_DIR / "knowledge_base"

TARGET_TOKENS = int(os.getenv("CHUNK_TARGET_TOKENS", "350"))
MAX_TOKENS = int(os.getenv("CHUNK_MAX_TOKENS", "450"))
MIN_TOKENS = int(os.getenv("CHUNK_MIN_TOKENS", "80"))

# Only used when one individual paragraph is too large.
OVERSIZED_PARAGRAPH_OVERLAP = int(
    os.getenv("OVERSIZED_PARAGRAPH_OVERLAP", "40")
)

MERGE_SLACK = int(os.getenv("CHUNK_MERGE_SLACK", "30"))


def get_chunking_config() -> dict[str, int]:
    """Token-based chunking settings used by the canonical chunker."""
    return {
        "target_tokens": TARGET_TOKENS,
        "max_tokens": MAX_TOKENS,
        "min_tokens": MIN_TOKENS,
        "paragraph_overlap_tokens": OVERSIZED_PARAGRAPH_OVERLAP,
    }


# OCR_MODE is intentionally NOT cached at module level â€”
# read it dynamically so callers can set the env var after import.
def _get_ocr_mode() -> str:
    return os.getenv("OCR_MODE", "auto").strip().lower()


# Keep a module-level alias for code that reads it directly (e.g. _ocr_pages).
# All new code should call _get_ocr_mode() instead.
OCR_MODE = os.getenv("OCR_MODE", "auto").strip().lower()

# Windows multiprocessing + OCR can be fragile.
# Default to 1 worker on Windows for reliability.
DEFAULT_OCR_WORKERS = 1 if os.name == "nt" else max(
    1, min(4, (os.cpu_count() or 2) - 1)
)

OCR_WORKERS = int(
    os.getenv("OCR_WORKERS", str(DEFAULT_OCR_WORKERS))
)

OCR_LANGUAGE = os.getenv("OCR_LANGUAGE", "eng")

OCR_MIN_TEXT_CHARS = int(
    os.getenv("OCR_MIN_TEXT_CHARS", "30")
)

# Optional tokenizer.
# Do not default to a remote Hugging Face model because some environments
# (corporate proxies, self-signed certs, or no internet access) fail while
# downloading model metadata. If a local tokenizer is needed, set
# CHUNK_TOKENIZER to a cached local model name; otherwise keep it empty to
# skip remote downloads and use the conservative fallback token counting path.
TOKENIZER_NAME = os.getenv("CHUNK_TOKENIZER", "").strip()


# ============================================================
# REGEX
# ============================================================

_CONTROL_RE = re.compile(
    r"[\x00-\x08\x0b\x0c\x0e-\x1f\u200b\u200c\u200d\ufeff\u00ad]"
)

_PAGE_NUMBER_RE = re.compile(
    r"^(?:page\s+)?(?:\d{1,4}|[ivxlcdm]{1,8})$",
    re.IGNORECASE,
)

_CHAPTER_NUMBER_TOKEN = (
    r"(?:\d{1,3}|[ivxlcdm]{1,8}|one|two|three|four|five|six|seven|eight|nine|ten|"
    r"eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty)"
)

_CHAPTER_RE = re.compile(
    r"^(?:chapter|ch\.)\s+"
    rf"({_CHAPTER_NUMBER_TOKEN})"
    r"\s*"
    r"[:.\-\u2013\u2014]?\s*"
    r"(.*)$",
    re.IGNORECASE,
)

_BARE_CHAPTER_RE = re.compile(
    r"^(?:chapter|ch\.)\s+"
    rf"({_CHAPTER_NUMBER_TOKEN})"
    r"\s*$",
    re.IGNORECASE,
)

# Matches "C H A P T E R N" (spaced-letter format used in book body pages)
_SPACED_CHAPTER_RE = re.compile(
    r"^C\s+H\s+A\s+P\s+T\s+E\s+R\s+"
    r"(\d{1,3}|[ivxlcdm]{1,8})"
    r"\s*$",
    re.IGNORECASE,
)


def _normalize_chapter_line(line: str) -> str:
    """Normalize spaced-letter chapter headings to standard form."""

    match = _SPACED_CHAPTER_RE.match(line.strip())

    if match:
        return f"CHAPTER {match.group(1).strip()}"

    return line


_SECTION_NUMBER_RE = re.compile(
    r"^(?:\d+(?:\.\d+)*|[IVXLC]+(?:\.[IVXLC]+)*)[.)]?\s+.+$",
    re.IGNORECASE,
)

_TOC_DOT_LEADER_RE = re.compile(r"\.{2,}\s*\d+\s*$")
_TOC_PAGE_NUMBER_RE = re.compile(r"\s+\d{1,3}\s*$")

_SENTENCE_SPLIT_RE = re.compile(
    r'(?:(?<=[.!?])|(?<=[.!?][\u201d"\u2019)]))\s+'
    r'(?=[A-Z0-9\u201c"\u2018\'(])'
)

_BACK_MATTER_RE = re.compile(
    r"^(?:acknowledg(?:e)?ments|about the author|notes|endnotes|"
    r"references|works cited|bibliography|index)$",
    re.IGNORECASE,
)

_COMMON_ABBREVIATIONS = {
    "mr.", "mrs.", "ms.", "dr.", "prof.", "sr.", "jr.", "etc.",
    "e.g.", "i.e.", "vs.", "fig.", "no.", "approx.", "inc.",
    "jan.", "feb.", "mar.", "apr.", "jun.", "jul.", "aug.",
    "sep.", "sept.", "oct.", "nov.", "dec.",
}


# ============================================================
# TOKENIZER
# ============================================================

_TOKENIZER = None
_TOKENIZER_FAILED = False


def _get_tokenizer():
    """
    Load the tokenizer lazily.

    This prevents importing/loading transformers just to use
    basic PDF utilities.
    """
    global _TOKENIZER
    global _TOKENIZER_FAILED

    if _TOKENIZER is not None:
        return _TOKENIZER

    if _TOKENIZER_FAILED:
        return None

    if not TOKENIZER_NAME:
        _TOKENIZER_FAILED = True
        return None

    try:
        from transformers import AutoTokenizer

        _TOKENIZER = AutoTokenizer.from_pretrained(
            TOKENIZER_NAME,
            local_files_only=False,
        )

        return _TOKENIZER

    except Exception as exc:
        _TOKENIZER_FAILED = True

        print(
            f"[TOKENIZER] Warning: could not load {TOKENIZER_NAME}: {exc}"
        )

        return None


def count_tokens(text: str) -> int:
    """
    Count tokens using the actual tokenizer when available.

    Falls back to a conservative word/punctuation estimate.
    """
    if not text:
        return 0

    tokenizer = _get_tokenizer()

    if tokenizer is not None:
        try:
            backend_tokenizer = getattr(
                tokenizer,
                "backend_tokenizer",
                None,
            )

            if backend_tokenizer is not None:
                return len(
                    backend_tokenizer.encode(
                        text,
                        add_special_tokens=False,
                    ).ids
                )

            return len(tokenizer._tokenize(text))
        except Exception:
            pass

    pieces = re.findall(
        r"\w+|[^\w\s]",
        text,
        flags=re.UNICODE,
    )

    return max(
        1,
        int(len(pieces) * 1.15),
    )


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_line(text: str) -> str:
    if not text:
        return ""

    text = unicodedata.normalize("NFKC", text)

    text = _CONTROL_RE.sub("", text)

    text = text.replace("\u00a0", " ")
    text = text.replace("\u2010", "-")
    text = text.replace("\u2011", "-")

    text = re.sub(r"[ \t]+", " ", text)

    return text.strip()


def clean_text(text: str) -> str:
    if not text:
        return ""

    text = unicodedata.normalize("NFKC", text)

    text = _CONTROL_RE.sub("", text)

    text = text.replace("\u00a0", " ")

    # Normalize whitespace.
    text = re.sub(r"\s+", " ", text)

    # Remove spaces before punctuation.
    text = re.sub(
        r"(?<=\w) +([,.;:!?])",
        r"\1",
        text,
    )

    # Normalize quote/bracket spacing.
    text = re.sub(
        r"([\u201c\u2018(\[])\s+",
        r"\1",
        text,
    )

    text = re.sub(
        r"\s+([\u201d)\]])",
        r"\1",
        text,
    )

    return text.strip()


def normalize_hyphenation(
    previous: str,
    current: str,
    vocabulary: set[str] | None = None,
) -> str:
    """
    Join PDF line/page hyphenation conservatively.

    Example:
        manage-
        ment

    -> management
    """

    if not previous:
        return current

    if not current:
        return previous

    if not previous.endswith("-"):
        return f"{previous} {current}"

    left = previous[:-1].rstrip()
    right = current.lstrip()

    if not left or not right:
        return f"{left} {right}".strip()

    candidate = left + right

    # If we have a vocabulary, only join when the combined word
    # looks legitimate.
    if vocabulary:
        candidate_lower = candidate.lower()

        if candidate_lower in vocabulary:
            return candidate

        # Common English suffixes where PDF hyphenation is safe.
        if right.lower().startswith(
            (
                "ing",
                "ed",
                "er",
                "est",
                "ly",
                "tion",
                "sion",
                "ment",
                "ness",
                "ful",
                "less",
                "able",
                "ible",
            )
        ):
            return candidate

        return f"{left} {right}"
    # Without vocabulary, join only when both sides look like
    # normal alphabetic word fragments.
    if re.fullmatch(r"[A-Za-z]{2,}", left) and re.fullmatch(
        r"[A-Za-z]{2,}",
        right,
    ):
        return candidate

    return f"{left} {right}"


def build_vocabulary(documents: list[dict[str, Any]]) -> set[str]:
    """
    Build a lightweight vocabulary from extracted text.

    This is used only for conservative dehyphenation.
    """
    vocabulary: set[str] = set()

    for doc in documents:
        text = doc.get("text", "")

        for word in re.findall(
            r"\b[A-Za-z]{3,}\b",
            text,
        ):
            vocabulary.add(word.lower())

    return vocabulary


# ============================================================
# SENTENCE SPLITTING
# ============================================================

def split_sentences(text: str) -> list[str]:
    if not text:
        return []

    # Protect common abbreviations.
    protected = text

    replacements: dict[str, str] = {}

    for index, abbreviation in enumerate(_COMMON_ABBREVIATIONS):
        marker = f"__ABBR_{index}__"
        protected = protected.replace(
            abbreviation,
            marker,
        )
        replacements[marker] = abbreviation

    parts = _SENTENCE_SPLIT_RE.split(protected)

    result: list[str] = []

    for part in parts:
        part = part.strip()

        for marker, abbreviation in replacements.items():
            part = part.replace(
                marker,
                abbreviation,
            )

        if part:
            result.append(part)

    return result


# ============================================================
# PAGE / BLOCK EXTRACTION
# ============================================================

def _extract_page_blocks(
    page: fitz.Page,
) -> list[dict[str, Any]]:
    """
    Extract text blocks while preserving approximate vertical
    ordering and image presence.
    """

    blocks: list[dict[str, Any]] = []

    try:
        raw_blocks = page.get_text("blocks")
    except Exception:
        raw_blocks = []

    for raw in raw_blocks:
        if len(raw) < 5:
            continue

        x0, y0, x1, y1, text = raw[:5]

        if not isinstance(text, str):
            continue

        text = text.strip()

        if not text:
            continue

        blocks.append(
            {
                "text": text,
                "x0": float(x0),
                "y0": float(y0),
                "x1": float(x1),
                "y1": float(y1),
            }
        )

    blocks.sort(
        key=lambda item: (
            item["y0"],
            item["x0"],
        )
    )

    return blocks


def _page_has_images(page: fitz.Page) -> bool:
    try:
        return bool(page.get_images(full=True))
    except Exception:
        return False


def _extract_page_native(
    pdf_path: str,
    page_index: int,
) -> dict[str, Any]:
    """
    Worker-safe native PDF extraction.
    """

    document = fitz.open(pdf_path)

    try:
        page = document[page_index]

        blocks = _extract_page_blocks(page)

        text_parts = [
            clean_line(block["text"])
            for block in blocks
            if clean_line(block["text"])
        ]

        text = "\n".join(text_parts)

        return {
            "page": page_index + 1,
            "text": text,
            "blocks": blocks,
            "has_images": _page_has_images(page),
            "native_text_chars": len(text),
        }

    finally:
        document.close()


# ============================================================
# OCR
# ============================================================

def _ocr_single_page(
    args: tuple[str, int, str],
) -> dict[str, Any]:
    """
    OCR one PDF page.

    Kept as a top-level function so it is safe with
    ProcessPoolExecutor on Windows.
    """

    pdf_path, page_index, language = args

    try:
        import pymupdf as fitz
        import pytesseract
        from PIL import Image
        import io

        document = fitz.open(pdf_path)

        try:
            page = document[page_index]

            pixmap = page.get_pixmap(
                matrix=fitz.Matrix(2.0, 2.0),
                alpha=False,
            )

            image_bytes = pixmap.tobytes("png")

        finally:
            document.close()

        image = Image.open(
            io.BytesIO(image_bytes)
        )

        text = pytesseract.image_to_string(
            image,
            lang=language,
        )

        text = clean_text(text)

        return {
            "page": page_index + 1,
            "text": text,
            "success": bool(text),
        }

    except Exception as exc:
        return {
            "page": page_index + 1,
            "text": "",
            "success": False,
            "error": str(exc),
        }


def _ocr_pages(
    pdf_path: str,
    page_indexes: list[int],
) -> dict[int, str]:
    """
    OCR only pages where native extraction failed.

    On Windows the default is sequential processing to avoid
    multiprocessing/spawn failures.
    """

    if not page_indexes:
        return {}

    if OCR_MODE == "off":
        return {}

    if OCR_MODE not in {"auto", "on"}:
        return {}

    try:
        import pytesseract  # noqa: F401
        from PIL import Image  # noqa: F401
    except Exception as exc:
        print(
            f"[OCR] OCR unavailable: {exc}"
        )
        return {}

    workers = max(1, OCR_WORKERS)

    print(
        f"[OCR] {len(page_indexes)} page(s) to OCR "
        f"with {workers} worker(s)..."
    )

    tasks = [
        (
            pdf_path,
            page_index,
            OCR_LANGUAGE,
        )
        for page_index in page_indexes
    ]

    results: dict[int, str] = {}

    # Sequential on Windows by default.
    if workers == 1:
        for task in tasks:
            result = _ocr_single_page(task)

            if result.get("success"):
                results[result["page"]] = result["text"]

            elif result.get("error"):
                print(
                    f"[OCR] Page {result['page']} failed: "
                    f"{result['error']}"
                )

        return results

    # Multiprocessing only when explicitly configured.
    try:
        with ProcessPoolExecutor(
            max_workers=workers
        ) as executor:

            futures = [
                executor.submit(
                    _ocr_single_page,
                    task,
                )
                for task in tasks
            ]

            for future in as_completed(futures):
                result = future.result()

                if result.get("success"):
                    results[result["page"]] = result["text"]

                elif result.get("error"):
                    print(
                        f"[OCR] Page {result['page']} failed: "
                        f"{result['error']}"
                    )

    except Exception as exc:
        print(
            f"[OCR] Parallel OCR failed: {exc}"
        )

    return results


# ============================================================
# ACTIVE FILE SELECTION
# ============================================================

def _get_active_kb_files_raw() -> str:
    """
    IMPORTANT:
    Read ACTIVE_KB_FILES dynamically.

    Do NOT store this in a module-level constant because the
    user may set the environment variable after importing this
    module.
    """

    return os.getenv(
        "ACTIVE_KB_FILES",
        "",
    ).strip()


def _normalise_filename(value: str) -> str:
    value = value.strip()

    # Allow values written as:
    # "book.pdf"
    # 'book.pdf'
    value = value.strip('"').strip("'")

    return value.casefold()


def get_active_files(
    kb_dir: Path | None = None,
) -> list[Path]:
    """
    Return only the knowledge-base files explicitly requested
    through ACTIVE_KB_FILES.

    Matching is done against the exact filename, case-insensitively.
    """
    if kb_dir is None:
        kb_dir = KNOWLEDGE_BASE_DIR

    if not kb_dir.exists():
        print(f"[KB] Knowledge base directory not found: {kb_dir}")
        return []

    supported_exts = {
        ".pdf", ".docx", ".epub", ".mobi", ".txt", ".md", ".rtf", ".rst",
        ".html", ".htm", ".json", ".csv", ".tsv", ".doc", ".xps", ".fb2"
    }
    files = [
        p for p in kb_dir.iterdir()
        if p.is_file() and p.suffix.lower() in supported_exts
    ]

    requested_raw = os.getenv("ACTIVE_KB_FILES", "").strip()

    if not requested_raw:
        print(f"[KB] Found {len(files)} knowledge file(s)")
        return files

    # Try exact match first for filenames with commas
    exact_match = next(
        (p for p in files if p.name.casefold() == requested_raw.casefold()),
        None
    )
    
    if exact_match:
        active_files = [exact_match]
    else:
        # Fall back to delimiter-based matching
        delimiter = ';' if ';' in requested_raw else ','
        requested = {
            name.strip().casefold()
            for name in requested_raw.split(delimiter)
            if name.strip()
        }
        
        active_files = [
            p for p in files
            if p.name.casefold() in requested
        ]

    print(f"[KB] Found {len(active_files)} active knowledge file(s)")
    print(f"[KB] ACTIVE_KB_FILES requested: {requested_raw}")

    if not active_files:
        print("[KB] Available files:")
        for p in files:
            print(f"     - {p.name}")

    return active_files


# ============================================================
# RUNNING HEADER / FOOTER REMOVAL
# ============================================================

def _normalise_for_comparison(text: str) -> str:
    text = clean_text(text).lower()

    # Remove page numbers.
    text = re.sub(r"\bpage\s+\d+\b", "", text)
    text = re.sub(r"\b\d{1,4}\b", "", text)

    return re.sub(
        r"\s+",
        " ",
        text,
    ).strip()


def remove_running_headers_footers(
    pages: list[dict[str, Any]],
) -> list[dict[str, Any]]:

    if len(pages) < 5:
        return pages

    top_counter: Counter[str] = Counter()
    bottom_counter: Counter[str] = Counter()

    page_lines: dict[int, list[str]] = {}

    for page in pages:
        page_number = page["page"]

        lines = [
            clean_line(line)
            for line in page.get("text", "").splitlines()
        ]

        lines = [
            line
            for line in lines
            if line
        ]

        page_lines[page_number] = lines

        if lines:
            top_counter[
                _normalise_for_comparison(lines[0])
            ] += 1

            bottom_counter[
                _normalise_for_comparison(lines[-1])
            ] += 1

    threshold = max(
        3,
        int(len(pages) * 0.12),
    )

    repeated_top = {
        value
        for value, count in top_counter.items()
        if value and count >= threshold
    }

    repeated_bottom = {
        value
        for value, count in bottom_counter.items()
        if value and count >= threshold
    }

    cleaned_pages: list[dict[str, Any]] = []

    for page in pages:

        lines = page_lines[page["page"]][:]

        if lines:
            first = _normalise_for_comparison(lines[0])

            if (
                first in repeated_top
                or _PAGE_NUMBER_RE.fullmatch(lines[0])
            ):
                lines.pop(0)

        if lines:
            last = _normalise_for_comparison(lines[-1])

            if (
                last in repeated_bottom
                or _PAGE_NUMBER_RE.fullmatch(lines[-1])
            ):
                lines.pop()

        page_copy = dict(page)

        page_copy["text"] = "\n".join(lines)

        cleaned_pages.append(page_copy)

    return cleaned_pages


# ============================================================
# TABLE OF CONTENTS DETECTION
# ============================================================

def _is_toc_line(line: str) -> bool:
    line = clean_line(line)

    if not line:
        return False

    if _TOC_DOT_LEADER_RE.search(line):
        return True

    # Typical:
    # "Welcome to the Creative Age 12"
    if _TOC_PAGE_NUMBER_RE.search(line):
        stripped = re.sub(
            r"\s+\d{1,3}\s*$",
            "",
            line,
        )

        if len(stripped.split()) >= 2:
            return True

    return False


def _toc_score(text: str) -> int:
    lines = [
        clean_line(line)
        for line in text.splitlines()
        if clean_line(line)
    ]

    if not lines:
        return 0

    score = 0

    for line in lines:

        if _is_toc_line(line):
            score += 1

        norm = _normalize_chapter_line(line)
        if _CHAPTER_RE.match(norm):
            score += 2

        if "contents" in line.lower():
            score += 4

    return score


def _looks_like_toc_page(page: dict[str, Any]) -> bool:
    text = page.get("text", "")

    lines = [
        clean_line(line)
        for line in text.splitlines()
        if clean_line(line)
    ]

    if not lines:
        return False

    if any(
        re.search(r"\b(?:table of )?contents\b", line, re.IGNORECASE)
        for line in lines
    ):
        return True

    chapter_count = sum(
        1
        for line in lines
        if _CHAPTER_RE.match(_normalize_chapter_line(line))
        or _BARE_CHAPTER_RE.match(_normalize_chapter_line(line))
    )

    if chapter_count >= 2:
        return True

    toc_entries = [
        line
        for line in lines
        if _is_toc_line(line)
        and not _CHAPTER_RE.match(_normalize_chapter_line(line))
        and not _BARE_CHAPTER_RE.match(_normalize_chapter_line(line))
    ]

    dot_leader_entries = sum(
        1
        for line in lines
        if _TOC_DOT_LEADER_RE.search(line)
    )

    return (
        dot_leader_entries >= 2
        or (
            len(toc_entries) >= 4
            and len(toc_entries) / len(lines) >= 0.35
        )
    )


# ============================================================
# CHAPTER PARSING
# ============================================================

def _parse_chapter_number(value: str) -> int | None:
    value = value.strip().lower()

    if value.isdigit():
        return int(value)

    word_map = {
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
        "eight": 8,
        "nine": 9,
        "ten": 10,
        "eleven": 11,
        "twelve": 12,
        "thirteen": 13,
        "fourteen": 14,
        "fifteen": 15,
        "sixteen": 16,
        "seventeen": 17,
        "eighteen": 18,
        "nineteen": 19,
        "twenty": 20,
    }
    if value in word_map:
        return word_map[value]

    roman_map = {
        "i": 1,
        "ii": 2,
        "iii": 3,
        "iv": 4,
        "v": 5,
        "vi": 6,
        "vii": 7,
        "viii": 8,
        "ix": 9,
        "x": 10,
        "xi": 11,
        "xii": 12,
        "xiii": 13,
        "xiv": 14,
        "xv": 15,
        "xvi": 16,
        "xvii": 17,
        "xviii": 18,
        "xix": 19,
        "xx": 20,
    }

    return roman_map.get(value)


def parse_chapter_line(
    line: str,
) -> tuple[int, str | None] | None:

    line = clean_line(line)

    if not line:
        return None

    # Normalize "C H A P T E R N" â†’ "CHAPTER N" before matching.
    line = _normalize_chapter_line(line)

    match = _CHAPTER_RE.match(line)

    if not match:
        return None

    chapter_number = _parse_chapter_number(
        match.group(1)
    )

    if chapter_number is None:
        return None

    title = clean_line(
        match.group(2)
    )

    title = title.rstrip(".:;-â€“â€” ")

    return (
        chapter_number,
        title or None,
    )


def parse_bare_chapter_line(
    line: str,
) -> int | None:

    line = clean_line(line)

    # Normalize "C H A P T E R N" â†’ "CHAPTER N" before matching.
    line = _normalize_chapter_line(line)

    match = _BARE_CHAPTER_RE.match(line)

    if not match:
        return None

    return _parse_chapter_number(
        match.group(1)
    )


def _looks_like_real_chapter_heading(
    lines: list[str],
    index: int,
    expected_chapter: int | None,
) -> tuple[int, str | None] | None:

    line = lines[index]

    parsed = parse_chapter_line(line)

    if parsed:
        chapter_number, title = parsed

        if (
            expected_chapter is not None
            and chapter_number != expected_chapter
        ):
            return None

        # A chapter heading with a useful title is strong evidence.
        if title:
            return (
                chapter_number,
                title,
            )

        # If title is missing, use the next short line as title.
        if index + 1 < len(lines):
            next_line = clean_line(
                lines[index + 1]
            )

            if (
                next_line
                and len(next_line) <= 120
                and not parse_chapter_line(next_line)
                and not _is_toc_line(next_line)
            ):
                return (
                    chapter_number,
                    next_line,
                )

        return (
            chapter_number,
            None,
        )

    bare_number = parse_bare_chapter_line(line)

    if bare_number is not None:

        if (
            expected_chapter is not None
            and bare_number != expected_chapter
        ):
            return None

        if index + 1 < len(lines):
            next_line = clean_line(
                lines[index + 1]
            )

            if (
                next_line
                and len(next_line) <= 120
                and not _is_toc_line(next_line)
                and not parse_chapter_line(next_line)
            ):
                return (
                    bare_number,
                    next_line,
                )

        return (
            bare_number,
            None,
        )

    return None


# ============================================================
# SECTION DETECTION
# ============================================================

def _is_probable_section_heading(
    line: str,
) -> bool:
    """
    Conservatively classify a text fragment as a section heading.

    Returns True only when there is good evidence the line is a
    heading â€” numbered sections or ALL-CAPS short phrases.

    Title-case heuristics are intentionally NOT used here because
    they produce too many false positives with normal prose that
    starts with a capitalized word.
    """

    line = clean_line(line)

    if not line:
        return False

    if len(line) < 3 or len(line) > 120:
        return False

    if _PAGE_NUMBER_RE.fullmatch(line):
        return False

    norm = _normalize_chapter_line(line)

    if _CHAPTER_RE.match(norm):
        return False

    if _BARE_CHAPTER_RE.match(norm):
        return False

    if _is_toc_line(line):
        return False

    words = line.split()

    if len(words) > 12:
        return False

    # Numbered sections are strong candidates.
    if _SECTION_NUMBER_RE.match(line):
        return True

    # ALL CAPS headings â€” the only reliable signal for this book.
    letters = [char for char in line if char.isalpha()]

    if letters:
        uppercase_ratio = sum(
            char.isupper() for char in letters
        ) / len(letters)

        if uppercase_ratio >= 0.90 and len(words) <= 10:
            return True

    return False



# ============================================================
# PARAGRAPH RECONSTRUCTION
# ============================================================

def _is_paragraph_boundary(
    previous_line: str,
    current_line: str,
) -> bool:

    previous_line = previous_line.strip()
    current_line = current_line.strip()

    if not previous_line or not current_line:
        return True

    # A new sentence beginning with a capital letter after a
    # paragraph-like ending is usually safe.
    if previous_line.endswith(
        (".", "!", "?", '"', "'", "â€", "â€™", ")")
    ):
        if current_line[:1].isupper():
            return True

    return False


def reconstruct_paragraphs(
    page: dict[str, Any],
    vocabulary: set[str] | None = None,
) -> list[dict[str, Any]]:
    """
    Convert a page's block list into a list of paragraph dicts.

    Each fitz "blocks" entry is already a semantic paragraph unit.
    Lines within a block are joined with conservative dehyphenation.

    We also detect multi-line ALL-CAPS fragments that form section
    headings (e.g. the block "THE END OF TIME\\nMANAGEMENT" should
    become the heading "THE END OF TIME MANAGEMENT").
    """

    blocks = page.get("blocks", [])

    def is_all_caps_heading_fragment(text: str) -> bool:
        lines = [
            clean_line(line)
            for line in text.splitlines()
            if clean_line(line)
        ]

        if lines and parse_bare_chapter_line(lines[0]) is not None:
            lines = lines[1:]

        if not lines or len(lines) > 5:
            return False

        if sum(len(line.split()) for line in lines) > 12:
            return False

        for line in lines:
            if (
                parse_chapter_line(line)
                or parse_bare_chapter_line(line) is not None
                or len(line.split()) > 8
            ):
                return False

            letters = [char for char in line if char.isalpha()]

            if (
                not letters
                or sum(char.isupper() for char in letters) / len(letters) < 0.90
            ):
                return False

        return True

    merged_blocks: list[dict[str, Any]] = []

    for block in blocks:
        current = dict(block)

        if merged_blocks and is_all_caps_heading_fragment(
            str(merged_blocks[-1].get("text", ""))
        ) and is_all_caps_heading_fragment(
            str(current.get("text", ""))
        ):
            previous = merged_blocks[-1]
            close_blocks = True

            if "y1" in previous and "y0" in current:
                gap = float(current["y0"]) - float(previous["y1"])
                close_blocks = -5 <= gap <= 18

            if close_blocks:
                previous["text"] = (
                    f"{str(previous.get('text', '')).strip()} "
                    f"{str(current.get('text', '')).strip()}"
                )

                if "y1" in current:
                    previous["y1"] = current["y1"]

                continue

        merged_blocks.append(current)

    blocks = merged_blocks
    page_number = page["page"]
    has_images = page.get("has_images", False)
    paragraphs: list[dict[str, Any]] = []

    # ----------------------------------------------------------------
    # If no block list is available, fall back to the text string.
    # ----------------------------------------------------------------
    if not blocks:
        text = page.get("text", "")
        raw_lines = [
            clean_line(line)
            for line in text.splitlines()
            if clean_line(line)
        ]
        if not raw_lines:
            return []
        # Treat the whole page as one paragraph.
        para_text = clean_text(" ".join(raw_lines))
        if para_text:
            paragraphs.append({
                "text": para_text,
                "page_start": page_number,
                "page_end": page_number,
                "has_images": has_images,
            })
        return paragraphs

    for block in blocks:
        raw_block_text = block.get("text", "").strip()
        if not raw_block_text:
            continue

        # Split the block into its constituent lines and clean them.
        block_lines = [
            clean_line(ln)
            for ln in raw_block_text.splitlines()
            if clean_line(ln)
        ]

        if not block_lines:
            continue

        # ----------------------------------------------------------------
        # Detect multi-line ALL-CAPS section headings within a block.
        # Example: "THE END OF TIME\nMANAGEMENT" â†’ one heading.
        # We recognise this pattern when:
        #   â€¢ The block has â‰¤ 5 lines
        #   â€¢ Every line is short (â‰¤ 6 words each)
        #   â€¢ Every line is predominantly ALL CAPS
        # ----------------------------------------------------------------
        def _line_is_all_caps(ln: str) -> bool:
            letters = [c for c in ln if c.isalpha()]
            if not letters:
                return False
            return (
                sum(c.isupper() for c in letters) / len(letters) >= 0.90
                and len(ln.split()) <= 8
            )

        norm_first = _normalize_chapter_line(block_lines[0])
        is_chapter_block = (
            _CHAPTER_RE.match(norm_first)
            or _BARE_CHAPTER_RE.match(norm_first)
        )

        if not is_chapter_block and len(block_lines) <= 5 and all(
            _line_is_all_caps(ln) for ln in block_lines
        ):
            # Join ALL-CAPS lines into a single heading phrase.
            joined = " ".join(block_lines)
            # Verify the whole joined text qualifies as a heading.
            if _is_probable_section_heading(joined):
                paragraphs.append({
                    "text": joined,
                    "page_start": page_number,
                    "page_end": page_number,
                    "is_heading_candidate": True,
                    "has_images": has_images,
                })
                continue

        # ----------------------------------------------------------------
        # Standard block processing.
        # Join lines within the block using dehyphenation.
        # ----------------------------------------------------------------
        if len(block_lines) == 1:
            line = block_lines[0]
            norm = _normalize_chapter_line(line)

            if (
                _CHAPTER_RE.match(norm)
                or _BARE_CHAPTER_RE.match(norm)
                or _is_probable_section_heading(line)
            ):
                paragraphs.append({
                    "text": norm if norm != line else line,
                    "page_start": page_number,
                    "page_end": page_number,
                    "is_heading_candidate": True,
                    "has_images": has_images,
                })
            else:
                cleaned = clean_text(line)
                if cleaned:
                    paragraphs.append({
                        "text": cleaned,
                        "page_start": page_number,
                        "page_end": page_number,
                        "has_images": has_images,
                    })
            continue

        # Multi-line block â€” check if it's a chapter heading block
        # (e.g. "C H A P T E R 2\nCREATIVE SWEET SPOT").
        norm_first = _normalize_chapter_line(block_lines[0])
        if _BARE_CHAPTER_RE.match(norm_first):
            # The next line may be the chapter title.
            chapter_title_line = (
                block_lines[1]
                if len(block_lines) > 1
                else None
            )
            # Emit chapter line; remaining lines are prose.
            paragraphs.append({
                "text": norm_first,
                "page_start": page_number,
                "page_end": page_number,
                "is_heading_candidate": True,
                "has_images": has_images,
            })
            remaining = block_lines[1:]
        else:
            remaining = block_lines

        if not remaining:
            continue

        # Join remaining lines with dehyphenation.
        para_text = remaining[0]
        for ln in remaining[1:]:
            para_text = normalize_hyphenation(
                para_text, ln, vocabulary
            )

        para_text = clean_text(para_text)
        if not para_text:
            continue

        # Check if the joined text is a section heading.
        norm_para = _normalize_chapter_line(para_text)
        if (
            _CHAPTER_RE.match(norm_para)
            or _BARE_CHAPTER_RE.match(norm_para)
            or _is_probable_section_heading(para_text)
        ):
            paragraphs.append({
                "text": norm_para if norm_para != para_text else para_text,
                "page_start": page_number,
                "page_end": page_number,
                "is_heading_candidate": True,
                "has_images": has_images,
            })
        else:
            paragraphs.append({
                "text": para_text,
                "page_start": page_number,
                "page_end": page_number,
                "has_images": has_images,
            })

    return paragraphs



def merge_cross_page_paragraphs(
    paragraphs: list[dict[str, Any]],
    vocabulary: set[str] | None = None,
) -> list[dict[str, Any]]:

    if not paragraphs:
        return []

    merged: list[dict[str, Any]] = []

    for paragraph in paragraphs:

        if not merged:
            merged.append(dict(paragraph))
            continue

        previous = merged[-1]

        # Headings must remain separate.
        if paragraph.get("is_heading_candidate"):
            merged.append(dict(paragraph))
            continue

        if previous.get("is_heading_candidate"):
            merged.append(dict(paragraph))
            continue

        # Merge only when the previous paragraph appears to be
        # an unfinished paragraph.
        previous_text = previous.get(
            "text",
            "",
        ).strip()

        if (
            previous["page_end"] < paragraph["page_start"]
            and previous_text
            and not previous_text.endswith(
                (".", "!", "?", '"', "'", "â€", "â€™", ")")
            )
        ):
            previous["text"] = normalize_hyphenation(
                previous_text,
                paragraph["text"],
                vocabulary,
            )

            previous["page_end"] = paragraph[
                "page_end"
            ]

            previous["has_images"] = bool(
                previous.get("has_images")
                or paragraph.get("has_images")
            )

        else:
            merged.append(dict(paragraph))

    return merged


# ============================================================
# OVERSIZED PARAGRAPH SPLITTING
# ============================================================

def _split_by_words(
    text: str,
    max_tokens: int,
) -> list[str]:

    words = text.split()

    if not words:
        return []

    pieces: list[str] = []
    current: list[str] = []

    for word in words:

        candidate = (
            " ".join(current + [word])
        )

        if (
            current
            and count_tokens(candidate) > max_tokens
        ):
            pieces.append(
                " ".join(current)
            )

            current = [word]

        else:
            current.append(word)

    if current:
        pieces.append(
            " ".join(current)
        )

    return pieces


def split_oversized_paragraph(
    paragraph: dict[str, Any],
) -> list[dict[str, Any]]:

    text = paragraph.get(
        "text",
        "",
    ).strip()

    if not text:
        return []

    if count_tokens(text) <= MAX_TOKENS:
        return [paragraph]

    sentences = split_sentences(text)

    if not sentences:
        sentences = [text]

    pieces: list[str] = []
    current: list[str] = []

    for sentence in sentences:

        candidate = " ".join(
            current + [sentence]
        )

        if (
            current
            and count_tokens(candidate) > MAX_TOKENS
        ):
            pieces.append(
                " ".join(current)
            )

            current = [sentence]

        else:
            current.append(sentence)

    if current:
        pieces.append(
            " ".join(current)
        )

    # A single sentence may itself exceed MAX_TOKENS.
    final_pieces: list[str] = []

    for piece in pieces:

        if count_tokens(piece) <= MAX_TOKENS:
            final_pieces.append(piece)
        else:
            final_pieces.extend(
                _split_by_words(
                    piece,
                    MAX_TOKENS,
                )
            )

    result: list[dict[str, Any]] = []

    for index, piece in enumerate(final_pieces):

        piece = clean_text(piece)

        if not piece:
            continue

        if (
            index > 0
            and OVERSIZED_PARAGRAPH_OVERLAP > 0
        ):
            previous_words = final_pieces[
                index - 1
            ].split()

            overlap_words: list[str] = []

            for word in reversed(
                previous_words
            ):
                candidate_overlap = " ".join(
                    [word] + overlap_words
                )

                if count_tokens(candidate_overlap) > OVERSIZED_PARAGRAPH_OVERLAP:
                    break

                overlap_words.insert(
                    0,
                    word,
                )

            if overlap_words:
                overlapped_piece = (
                    " ".join(overlap_words)
                    + " "
                    + piece
                )

                while (
                    overlap_words
                    and count_tokens(overlapped_piece) > MAX_TOKENS
                ):
                    overlap_words.pop(0)
                    overlapped_piece = (
                        " ".join(overlap_words)
                        + " "
                        + piece
                    )

                if count_tokens(overlapped_piece) <= MAX_TOKENS:
                    piece = overlapped_piece

        item = dict(paragraph)

        item["text"] = piece

        result.append(item)

    return result


# ============================================================
# CHUNK CREATION
# ============================================================

def _flush_chunk(
    chunks: list[dict[str, Any]],
    paragraphs: list[dict[str, Any]],
    book_id: str,
    chapter_num: int | None,
    chapter_title: str | None,
    section_title: str | None,
    book_name: str = "",
    is_front_matter: bool = False,
) -> None:

    if not paragraphs:
        return

    text = clean_text(
        "\n\n".join(
            paragraph["text"]
            for paragraph in paragraphs
            if paragraph.get("text")
        )
    )

    if not text:
        return

    chunk_section_title = next(
        (
            paragraph.get("section_title")
            for paragraph in paragraphs
            if paragraph.get("section_title")
        ),
        section_title,
    )

    page_start = min(
        paragraph["page_start"]
        for paragraph in paragraphs
    )

    page_end = max(
        paragraph["page_end"]
        for paragraph in paragraphs
    )

    has_images = any(
        paragraph.get(
            "has_images",
            False,
        )
        for paragraph in paragraphs
    )

    book_title = _humanize_book_title(book_name)
    chunk_label = _make_source_label(
        book_title,
        chapter_num,
        chapter_title,
        chunk_section_title,
        page_start,
        page_end,
    )

    chunks.append(
        {
            "text": text,
            "book_id": book_id,
            "book_name": book_name,
            "book_title": book_title,
            "source_name": book_name,
            "source_label": chunk_label,
            "chapter_num": chapter_num,
            "chapter_title": chapter_title,
            "section_title": chunk_section_title,
            "page_start": page_start,
            "page_end": page_end,
            "token_count": count_tokens(text),
            "has_images": has_images,
            "is_front_matter": is_front_matter,
        }
    )


def _make_chunk_id(
    chunk: dict[str, Any],
) -> str:

    raw = "|".join(
        [
            str(chunk.get("book_id", "")),
            str(chunk.get("chapter_num", "")),
            str(chunk.get("page_start", "")),
            str(chunk.get("page_end", "")),
            chunk.get("text", ""),
        ]
    )

    digest = hashlib.sha1(
        raw.encode(
            "utf-8",
            errors="ignore",
        )
    ).hexdigest()[:16]

    return (
        f"{chunk.get('book_id', 'book')}:"
        f"{digest}"
    )


def _merge_small_chunks(
    chunks: list[dict[str, Any]],
) -> list[dict[str, Any]]:

    if not chunks:
        return []

    result: list[dict[str, Any]] = []

    for chunk in chunks:

        if not result:
            result.append(chunk)
            continue

        previous = result[-1]

        same_chapter = (
            previous.get("chapter_num")
            == chunk.get("chapter_num")
        )

        same_book = (
            previous.get("book_id")
            == chunk.get("book_id")
        )

        combined_text = (
            previous.get("text", "")
            + "\n\n"
            + chunk.get("text", "")
        )

        combined_tokens = count_tokens(
            combined_text
        )

        # Only merge small chunks.
        if (
            same_chapter
            and same_book
            and previous.get("token_count", 0)
            < MIN_TOKENS
            and combined_tokens <= MAX_TOKENS
        ):
            previous["text"] = clean_text(
                combined_text
            )

            previous["page_end"] = max(
                previous["page_end"],
                chunk["page_end"],
            )

            previous["token_count"] = combined_tokens

            previous["has_images"] = bool(
                previous.get("has_images")
                or chunk.get("has_images")
            )

        else:
            result.append(chunk)

    return result


def _book_id_from_path(
    path: Path,
) -> str:

    stem = path.stem.lower()

    stem = re.sub(
        r"[^a-z0-9]+",
        "_",
        stem,
    )

    stem = stem.strip("_")

    return stem or "book"


def _humanize_book_title(book_name: str | None) -> str:
    """Turn a filename into a clean, generic book title without hardcoded titles."""
    if not book_name:
        return "Untitled document"

    text = Path(str(book_name)).stem
    text = text.replace("_", " ").replace("-", " ")
    text = re.sub(r"^(?:epdf\.pub|pdf|book|document|upload|uploaded|kb)\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return "Untitled document"
    return text


def _make_source_label(
    book_title: str | None,
    chapter_num: int | None,
    chapter_title: str | None,
    section_title: str | None,
    page_start: int | None,
    page_end: int | None,
) -> str:
    """Create a readable source label without falling back to a specific book name."""
    title = _humanize_book_title(book_title) if book_title else "Untitled document"
    parts: list[str] = []

    if title and title.lower() != "untitled document":
        parts.append(title)

    if chapter_num and chapter_num > 0:
        chapter_ref = f"Chapter {chapter_num}"
        if chapter_title and chapter_title.strip():
            chapter_ref = f"{chapter_ref}: {chapter_title.strip()}"
        parts.append(chapter_ref)
    elif chapter_title and chapter_title.strip():
        parts.append(chapter_title.strip())

    if section_title and section_title.strip():
        parts.append(section_title.strip())

    if page_start and page_start > 0:
        if page_end and page_end > page_start:
            parts.append(f"Pages {page_start}-{page_end}")
        else:
            parts.append(f"Page {page_start}")

    if not parts:
        return "Untitled document"
    return " — ".join(parts)


def _chunk_book(
    pages: list[dict[str, Any]],
    book_path: Path,
) -> list[dict[str, Any]]:

    if not pages:
        return []

    book_id = _book_id_from_path(
        book_path
    )

    book_name = book_path.name

    # --------------------------------------------------------
    # Remove TOC/front matter until genuine Chapter 1.
    # --------------------------------------------------------

    start_index: int | None = None

    expected_chapter = 1

    for page_index, page in enumerate(pages):

        if _looks_like_toc_page(page):
            continue

        lines = [
            _normalize_chapter_line(clean_line(line))
            for line in page.get(
                "text",
                "",
            ).splitlines()
            if clean_line(line)
        ]

        for line_index in range(len(lines)):

            candidate = _looks_like_real_chapter_heading(
                lines,
                line_index,
                expected_chapter,
            )

            if candidate is None:
                continue

            chapter_number, title = candidate

            if chapter_number == 1:
                start_index = page_index
                break

        if start_index is not None:
            break

    fallback_to_plain_text = False

    if start_index is None:
        print(
            f"[CHUNK] Warning: could not identify "
            f"the start of Chapter 1 in {book_path.name}"
        )

        # Safer fallback: don't throw away the entire document.
        # Start from the first page containing substantial text.
        for index, page in enumerate(pages):
            if len(page.get("text", "").strip()) >= 100:
                start_index = index
                fallback_to_plain_text = True
                break

    if start_index is None:
        return []

    # --------------------------------------------------------
    # Ingest front-matter pages (everything before Chapter 1)
    # as separate chunks tagged is_front_matter=True so that
    # page-number queries for TOC/preface/dedication pages still
    # return content instead of a hard "page not found" error.
    # --------------------------------------------------------

    front_matter_chunks: list[dict[str, Any]] = []

    if start_index > 0:
        fm_pages = pages[:start_index]
        fm_vocabulary = build_vocabulary(fm_pages)
        fm_paragraphs: list[dict[str, Any]] = []
        for page in fm_pages:
            if len(page.get("text", "").strip()) < 30:
                continue  # skip truly empty pages (blank, cover image-only)
            fm_paragraphs.extend(
                reconstruct_paragraphs(page, fm_vocabulary)
            )
        fm_paragraphs = merge_cross_page_paragraphs(fm_paragraphs, fm_vocabulary)

        fm_current_paras: list[dict[str, Any]] = []
        for para in fm_paragraphs:
            text = clean_text(para.get("text", ""))
            if not text:
                continue
            fm_current_paras.append(dict(para))
            # Flush when we hit the token target
            candidate_tokens = count_tokens(
                clean_text("\n\n".join(p["text"] for p in fm_current_paras))
            )
            if candidate_tokens >= TARGET_TOKENS:
                _flush_chunk(
                    front_matter_chunks,
                    fm_current_paras,
                    book_id,
                    chapter_num=None,
                    chapter_title=None,
                    section_title=None,
                    book_name=book_name,
                    is_front_matter=True,
                )
                fm_current_paras = []
        # Flush remainder
        _flush_chunk(
            front_matter_chunks,
            fm_current_paras,
            book_id,
            chapter_num=None,
            chapter_title=None,
            section_title=None,
            book_name=book_name,
            is_front_matter=True,
        )

    pages = pages[start_index:]
    allow_unstructured_content = fallback_to_plain_text

    # --------------------------------------------------------
    # Build vocabulary for conservative dehyphenation.
    # --------------------------------------------------------

    vocabulary = build_vocabulary(
        pages
    )

    # --------------------------------------------------------
    # Convert pages to paragraphs.
    # --------------------------------------------------------

    paragraphs: list[dict[str, Any]] = []

    for page in pages:
        page_paragraphs = reconstruct_paragraphs(
            page,
            vocabulary,
        )

        paragraphs.extend(
            page_paragraphs
        )

    paragraphs = merge_cross_page_paragraphs(
        paragraphs,
        vocabulary,
    )

    # --------------------------------------------------------
    # Chapter/section state.
    # --------------------------------------------------------

    chunks: list[dict[str, Any]] = []

    current_chunk_paragraphs: list[
        dict[str, Any]
    ] = []

    current_chapter: int | None = None
    current_chapter_title: str | None = None
    current_section: str | None = None

    # --------------------------------------------------------
    # Iterate paragraphs.
    # --------------------------------------------------------

    _skip_para_index: int | None = None  # index of a para consumed by chapter-title look-ahead

    for para_index, paragraph in enumerate(paragraphs):

        # Skip paragraphs already consumed by look-ahead (e.g. the chapter
        # title line that immediately follows a bare "CHAPTER N" line).
        if _skip_para_index is not None and para_index == _skip_para_index:
            _skip_para_index = None
            continue

        text = clean_text(
            paragraph.get(
                "text",
                "",
            )
        )

        if not text:
            continue

        if _BACK_MATTER_RE.fullmatch(text):
            _flush_chunk(
                chunks,
                current_chunk_paragraphs,
                book_id,
                current_chapter,
                current_chapter_title,
                current_section,
                book_name,
            )
            current_chunk_paragraphs = []
            break

        # ----------------------------------------------------
        # Chapter heading.
        # ----------------------------------------------------

        chapter_parsed = parse_chapter_line(
            text
        )

        if chapter_parsed:

            chapter_number, chapter_title = (
                chapter_parsed
            )

            if chapter_number > 0 and chapter_number != current_chapter:
                _flush_chunk(
                    chunks,
                    current_chunk_paragraphs,
                    book_id,
                    current_chapter,
                    current_chapter_title,
                    current_section,
                    book_name,
                )

                current_chunk_paragraphs = []

                current_chapter = chapter_number
                # If parse_chapter_line found no inline title, look ahead
                # to the next paragraph for the chapter title.
                if chapter_title is None and para_index + 1 < len(paragraphs):
                    _next_raw = clean_text(
                        paragraphs[para_index + 1].get("text", "")
                    )
                    if (
                        _next_raw
                        and len(_next_raw) <= 120
                        and not parse_chapter_line(_next_raw)
                        and not parse_bare_chapter_line(_next_raw)
                        and not _is_toc_line(_next_raw)
                    ):
                        chapter_title = _next_raw
                        _skip_para_index = para_index + 1  # consumed
                current_chapter_title = chapter_title
                current_section = None

                continue

        # ----------------------------------------------------
        # Bare chapter heading.
        # ----------------------------------------------------

        bare_chapter = parse_bare_chapter_line(
            text
        )

        if bare_chapter is not None and bare_chapter > 0:

            if bare_chapter != current_chapter:
                _flush_chunk(
                    chunks,
                    current_chunk_paragraphs,
                    book_id,
                    current_chapter,
                    current_chapter_title,
                    current_section,
                    book_name,
                )

                current_chunk_paragraphs = []

                current_chapter = bare_chapter
                # Look ahead one paragraph for the chapter title
                # (books that put "CHAPTER N" on one line and the
                # title on the next).
                _next_title: str | None = None
                if para_index + 1 < len(paragraphs):
                    _next_raw = clean_text(
                        paragraphs[para_index + 1].get("text", "")
                    )
                    if (
                        _next_raw
                        and len(_next_raw) <= 120
                        and not parse_chapter_line(_next_raw)
                        and not parse_bare_chapter_line(_next_raw)
                        and not _is_toc_line(_next_raw)
                    ):
                        _next_title = _next_raw
                        _skip_para_index = para_index + 1  # consumed
                current_chapter_title = _next_title
                current_section = None

                continue

        # ----------------------------------------------------
        # Ignore anything before Chapter 1, unless the PDF has no
        # recognizable chapter structure at all. In that case, start
        # chunking from the first substantial page rather than dropping
        # the entire document.
        # ----------------------------------------------------

        if current_chapter is None:
            if allow_unstructured_content:
                current_chapter = 1
                current_chapter_title = None
                allow_unstructured_content = False
            else:
                continue

        # ----------------------------------------------------
        # Section heading.
        # ----------------------------------------------------

        if paragraph.get(
            "is_heading_candidate",
            False,
        ) and _is_probable_section_heading(text):
            current_section = text

            continue

        # ----------------------------------------------------
        # Normal paragraph.
        # ----------------------------------------------------

        normal_paragraph = dict(
            paragraph
        )

        normal_paragraph[
            "text"
        ] = text
        normal_paragraph["section_title"] = current_section

        # If paragraph is oversized, split it before packing.
        split_parts = (
            split_oversized_paragraph(
                normal_paragraph
            )
        )

        for part in split_parts:

            part_tokens = count_tokens(
                part["text"]
            )

            candidate_text = clean_text(
                "\n\n".join(
                    [p["text"] for p in current_chunk_paragraphs]
                    + [part["text"]]
                )
            )
            candidate_tokens = count_tokens(candidate_text)

            if (
                current_chunk_paragraphs
                and candidate_tokens > MAX_TOKENS
            ):
                _flush_chunk(
                    chunks,
                    current_chunk_paragraphs,
                    book_id,
                    current_chapter,
                    current_chapter_title,
                    current_section,
                    book_name,
                )

                current_chunk_paragraphs = []
                candidate_tokens = part_tokens

            current_chunk_paragraphs.append(
                part
            )

            # Stop close to target size.
            if candidate_tokens >= TARGET_TOKENS:
                _flush_chunk(
                    chunks,
                    current_chunk_paragraphs,
                    book_id,
                    current_chapter,
                    current_chapter_title,
                    current_section,
                    book_name,
                )

                current_chunk_paragraphs = []

    # Final chunk.
    _flush_chunk(
        chunks,
        current_chunk_paragraphs,
        book_id,
        current_chapter,
        current_chapter_title,
        current_section,
        book_name,
    )

    # Merge accidental tiny chunks.
    chunks = _merge_small_chunks(
        chunks
    )

    # Prepend front-matter chunks (TOC, preface, dedication, etc.)
    # so page-number queries on those pages return content.
    if front_matter_chunks:
        chunks = front_matter_chunks + chunks

    # --------------------------------------------------------
    # Add stable IDs and neighbor links.
    # --------------------------------------------------------

    for index, chunk in enumerate(chunks):

        chunk["chunk_index"] = index

        chunk["chunk_id"] = _make_chunk_id(
            chunk
        )

        chunk["token_count"] = count_tokens(
            chunk["text"]
        )

    for index, chunk in enumerate(chunks):
        chunk["prev_chunk_id"] = (
            chunks[index - 1]["chunk_id"]
            if index > 0
            else None
        )

        chunk["next_chunk_id"] = (
            chunks[index + 1]["chunk_id"]
            if index + 1 < len(chunks)
            else None
        )

    return chunks


# ============================================================
# PUBLIC DOCUMENT LOADER
# ============================================================

def load_documents(
    knowledge_base_dir: Path | None = None,
) -> list[dict[str, Any]]:

    kb_dir = (
        Path(knowledge_base_dir)
        if knowledge_base_dir is not None
        else KNOWLEDGE_BASE_DIR
    )

    active_files = get_active_files(
        kb_dir
    )

    print(
        f"[KB] Found {len(active_files)} active knowledge file(s)"
    )

    if not active_files:
        active_raw = _get_active_kb_files_raw()

        if active_raw:
            print(
                f"[KB] ACTIVE_KB_FILES requested: {active_raw}"
            )

        return []

    all_documents: list[dict[str, Any]] = []

    for file_path in active_files:

        print(
            f"[KB] Loading: {file_path.name}"
        )

        suffix = file_path.suffix.lower()

        if suffix == ".pdf":

            try:
                document = fitz.open(
                    str(file_path)
                )

            except Exception as exc:
                print(
                    f"[KB] Failed to open "
                    f"{file_path.name}: {exc}"
                )
                continue

            try:
                page_count = len(document)

            finally:
                document.close()

            native_pages: list[
                dict[str, Any]
            ] = []

            for page_index in range(
                page_count
            ):
                try:
                    page_data = _extract_page_native(
                        str(file_path),
                        page_index,
                    )

                    native_pages.append(
                        page_data
                    )

                except Exception as exc:
                    print(
                        f"[PDF] Page {page_index + 1} "
                        f"failed: {exc}"
                    )

                    native_pages.append(
                        {
                            "page": page_index + 1,
                            "text": "",
                            "blocks": [],
                            "has_images": False,
                            "native_text_chars": 0,
                        }
                    )

            empty_pages = [
                index
                for index, page in enumerate(
                    native_pages
                )
                if len(
                    page.get(
                        "text",
                        "",
                    ).strip()
                ) < OCR_MIN_TEXT_CHARS
            ]

            if empty_pages:

                if OCR_MODE in {
                    "auto",
                    "on",
                }:
                    ocr_results = _ocr_pages(
                        str(file_path),
                        empty_pages,
                    )

                    for page_number, ocr_text in (
                        ocr_results.items()
                    ):
                        if not ocr_text:
                            continue

                        page = native_pages[
                            page_number - 1
                        ]

                        # Only replace native extraction if OCR
                        # produced meaningful text.
                        if len(
                            ocr_text.strip()
                        ) > len(
                            page.get(
                                "text",
                                "",
                            ).strip()
                        ):
                            page["text"] = ocr_text
                            page["ocr_used"] = True

                else:
                    print(
                        f"[KB] WARNING: "
                        f"{len(empty_pages)}/{page_count} "
                        f"pages have no text "
                        f"(OCR_MODE={OCR_MODE})"
                    )

            remaining_empty = sum(
                1
                for page in native_pages
                if len(
                    page.get(
                        "text",
                        "",
                    ).strip()
                ) < OCR_MIN_TEXT_CHARS
            )

            if remaining_empty:
                print(
                    f"[KB] WARNING: "
                    f"{remaining_empty}/{page_count} "
                    f"pages still have little/no text"
                )

            # Remove repeated headers/footers.
            native_pages = remove_running_headers_footers(
                native_pages
            )

            for page in native_pages:
                page["source"] = file_path.name
                page["path"] = str(file_path)

            all_documents.extend(
                native_pages
            )

            print(
                f"[KB] Loaded {page_count} page/document units "
                f"from {file_path.name}"
            )

        else:
            doc_pages = load_document(file_path)
            if doc_pages:
                all_documents.extend(doc_pages)
                print(
                    f"[KB] Loaded {len(doc_pages)} unit(s) from "
                    f"{file_path.name}"
                )

    return all_documents


# ============================================================
# INTERNAL BOOK GROUPING
# ============================================================

def _documents_by_source(
    documents: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:

    grouped: dict[
        str,
        list[dict[str, Any]]
    ] = {}

    for document in documents:

        source = document.get(
            "source"
        ) or document.get(
            "filename"
        ) or "document"

        grouped.setdefault(
            source,
            [],
        ).append(document)

    for source in grouped:
        grouped[source].sort(
            key=lambda item: item.get(
                "page",
                0,
            )
        )

    return grouped


# ============================================================
# PUBLIC CHUNKING API
# ============================================================

def chunk_documents(
    documents: list[dict[str, Any]],
    **kwargs: Any,
) -> list[dict[str, Any]]:

    if not documents:
        print(
            "[CHUNKING] No documents supplied"
        )
        return []

    grouped = _documents_by_source(
        documents
    )

    all_chunks: list[
        dict[str, Any]
    ] = []

    for source, pages in grouped.items():

        book_path = Path(
            pages[0].get(
                "path",
                source,
            )
        )

        book_chunks = _chunk_book(
            pages,
            book_path,
        )

        all_chunks.extend(
            book_chunks
        )

    # Re-number globally.
    for index, chunk in enumerate(
        all_chunks
    ):

        chunk["chunk_index"] = index

        chunk["chunk_id"] = _make_chunk_id(
            chunk
        )

    # Rebuild neighbor links after global numbering.
    for index, chunk in enumerate(
        all_chunks
    ):

        chunk["prev_chunk_id"] = (
            all_chunks[index - 1]["chunk_id"]
            if index > 0
            else None
        )

        chunk["next_chunk_id"] = (
            all_chunks[index + 1]["chunk_id"]
            if index + 1 < len(all_chunks)
            else None
        )

    print(
        f"[CHUNKING] Created "
        f"{len(all_chunks)} canonical chunks"
    )

    return all_chunks


# ============================================================
# CHUNK STATISTICS
# ============================================================

def get_chunk_statistics(
    chunks: list[dict[str, Any]],
) -> dict[str, Any]:

    if not chunks:
        return {
            "chunks": 0,
            "min_tokens": 0,
            "max_tokens": 0,
            "avg_tokens": 0.0,
            "chapters": [],
            "sections": [],
            "books": [],
            "over_max": 0,
            "under_min": 0,
            "no_chapter": 0,
            "no_section": 0,
        }

    token_counts = [
        int(
            chunk.get(
                "token_count",
                count_tokens(
                    chunk.get(
                        "text",
                        "",
                    )
                ),
            )
        )
        for chunk in chunks
    ]

    chapters = sorted(
        {
            chunk.get("chapter_num")
            for chunk in chunks
            if chunk.get("chapter_num")
            is not None
        }
    )

    sections = sorted(
        {
            chunk.get("section_title")
            for chunk in chunks
            if chunk.get("section_title")
        }
    )

    books = sorted(
        {
            chunk.get("book_id")
            for chunk in chunks
            if chunk.get("book_id")
        }
    )

    over_max = sum(
        1
        for count in token_counts
        if count > MAX_TOKENS
    )

    under_min = sum(
        1
        for count in token_counts
        if count < MIN_TOKENS
    )

    no_chapter = sum(
        1
        for chunk in chunks
        if chunk.get("chapter_num") is None
    )

    no_section = sum(
        1
        for chunk in chunks
        if not chunk.get("section_title")
    )

    return {
        "chunks": len(chunks),
        "min_tokens": min(token_counts),
        "max_tokens": max(token_counts),
        "avg_tokens": round(
            sum(token_counts)
            / len(token_counts),
            2,
        ),
        "chapters": chapters,
        "sections": sections,
        "books": books,
        "over_max": over_max,
        "under_min": under_min,
        "no_chapter": no_chapter,
        "no_section": no_section,
    }


def print_chunk_statistics(
    chunks: list[dict[str, Any]],
) -> dict[str, Any]:

    stats = get_chunk_statistics(
        chunks
    )

    print("\n=== CHUNK STATISTICS ===")

    print(
        f"Chunks: {stats['chunks']}"
    )

    print(
        f"Token range: "
        f"{stats['min_tokens']} - "
        f"{stats['max_tokens']}"
    )

    print(
        f"Average tokens: "
        f"{stats['avg_tokens']}"
    )

    print(
        f"Chapters: "
        f"{stats['chapters']}"
    )

    print(
        f"Sections detected: "
        f"{len(stats['sections'])}"
    )

    print(
        f"Books: "
        f"{stats['books']}"
    )

    print(
        f"Over MAX_TOKENS ({MAX_TOKENS}): "
        f"{stats['over_max']}"
    )

    print(
        f"Under MIN_TOKENS ({MIN_TOKENS}): "
        f"{stats['under_min']}"
    )

    print(
        f"Without chapter metadata: "
        f"{stats['no_chapter']}"
    )

    print(
        f"Without section metadata: "
        f"{stats['no_section']}"
    )

    return stats


# ============================================================
# CHROMA RECORD CONVERSION
# ============================================================

def chunk_to_chroma_record(
    chunk: dict[str, Any],
) -> dict[str, Any]:

    metadata: dict[str, Any] = {
        "book_id": chunk.get(
            "book_id",
            "",
        ),
        "source": chunk.get(
            "source",
            chunk.get(
                "source_name",
                chunk.get(
                    "book_name",
                    chunk.get(
                        "book_id",
                        "",
                    ),
                ),
            ),
        ),
        "source_name": chunk.get(
            "source_name",
            chunk.get(
                "book_name",
                "",
            ),
        ),
        "book_title": chunk.get(
            "book_title",
            chunk.get(
                "book_name",
                "",
            ),
        ),
        "source_label": chunk.get(
            "source_label",
            _make_source_label(
                chunk.get("book_title") or chunk.get("book_name"),
                chunk.get("chapter_num"),
                chunk.get("chapter_title"),
                chunk.get("section_title"),
                chunk.get("page_start"),
                chunk.get("page_end"),
            ),
        ),
        "chunk_index": int(
            chunk.get(
                "chunk_index",
                0,
            )
        ),
        "page_start": int(
            chunk.get(
                "page_start",
                0,
            )
        ),
        "page_end": int(
            chunk.get(
                "page_end",
                0,
            )
        ),
        "token_count": int(
            chunk.get(
                "token_count",
                0,
            )
        ),
    }

    if chunk.get("chapter_num") is not None:
        metadata["chapter_num"] = int(
            chunk["chapter_num"]
        )

    if chunk.get("chapter_title"):
        metadata[
            "chapter_title"
        ] = str(
            chunk["chapter_title"]
        )

    if chunk.get("section_title"):
        metadata[
            "section_title"
        ] = str(
            chunk["section_title"]
        )

    # Prev/next chunk IDs removed — not useful in UI and clutter metadata
    # if chunk.get("prev_chunk_id"):
    #     metadata["prev_chunk_id"] = str(chunk["prev_chunk_id"])
    # if chunk.get("next_chunk_id"):
    #     metadata["next_chunk_id"] = str(chunk["next_chunk_id"])

    return {
        "id": chunk.get(
            "chunk_id"
        ),
        "document": chunk.get(
            "text",
            "",
        ),
        "metadata": metadata,
    }


# ============================================================
# NEIGHBOR EXPANSION
# ============================================================

def expand_neighbor_chunks(
    chunks: list[dict[str, Any]],
    selected_indexes: Iterable[int],
    radius: int = 1,
) -> list[dict[str, Any]]:

    if not chunks:
        return []

    indexes = set()

    for index in selected_indexes:

        if not isinstance(
            index,
            int,
        ):
            continue

        start = max(
            0,
            index - radius,
        )

        end = min(
            len(chunks),
            index + radius + 1,
        )

        indexes.update(
            range(
                start,
                end,
            )
        )

    return [
        chunks[index]
        for index in sorted(indexes)
    ]


# ============================================================
# DEBUG PREVIEW
# ============================================================

def preview_chunks(
    chunks: list[dict[str, Any]],
    limit: int = 10,
) -> None:

    print(
        f"\n=== FIRST {min(limit, len(chunks))} CHUNKS ==="
    )

    for index, chunk in enumerate(
        chunks[:limit]
    ):

        print(
            "\n"
            + "=" * 80
        )

        print(
            f"Chunk: {index}"
            f" | ID: {chunk.get('chunk_id')}"
            f" | Pages: "
            f"{chunk.get('page_start')}-"
            f"{chunk.get('page_end')}"
            f" | Tokens: "
            f"{chunk.get('token_count')}"
            f" | Chapter: "
            f"{chunk.get('chapter_num')}"
            f" | Section: "
            f"{chunk.get('section_title')}"
        )

        print("-" * 80)

        print(
            chunk.get(
                "text",
                "",
            )[:1000]
        )


# ============================================================
# OPTIONAL COMPATIBILITY HELPERS
# ============================================================

def get_chunk_text(
    chunk: dict[str, Any],
) -> str:

    return str(
        chunk.get(
            "text",
            "",
        )
    )


def get_chunk_metadata(
    chunk: dict[str, Any],
) -> dict[str, Any]:

    return chunk_to_chroma_record(
        chunk
    )["metadata"]



# ============================================================
# PIPELINE COMPATIBILITY SHIMS
# These expose the names the RAG pipeline imports.
# Do NOT remove.  Do NOT modify chunking logic above.
# ============================================================

def list_knowledge_files(
    knowledge_base_dir: Path | None = None,
) -> list[Path]:
    """Return all active KB files; respects ACTIVE_KB_FILES."""
    return get_active_files(knowledge_base_dir)


def list_txt_files(
    knowledge_base_dir: Path | None = None,
) -> list[Path]:
    """Return active .txt files only (used by pipeline status())."""
    return [
        p for p in get_active_files(knowledge_base_dir)
        if p.suffix.lower() == ".txt"
    ]


class DocumentPageList(list):
    """
    List of document page dicts that additionally exposes an ingestion .stats dict:
    - pages_processed: total pages or logical page units extracted
    - ocr_pages: count of pages processed with OCR
    - image_captions: count of figures/images captioned or detected
    - chunks_created: count of chunks created (populated during chunking)
    - skipped_sections: count of blank pages or empty sections skipped
    """
    def __init__(self, items=(), stats: dict[str, Any] | None = None):
        super().__init__(items)
        self.stats: dict[str, Any] = stats or {
            "pages_processed": len(self),
            "ocr_pages": 0,
            "image_captions": 0,
            "chunks_created": 0,
            "skipped_sections": 0,
        }


def _detect_captions_in_text(text: str) -> int:
    """Count explicit figure / image / table captions in extracted text."""
    if not text:
        return 0
    matches = re.findall(r"\b(?:Figure|Fig\.|Illustration|Image|Chart|Table)\s*[\d\.:A-Za-z]+", text, re.IGNORECASE)
    return len(matches)


def load_document(
    file_path: "Path | str",
    content: "bytes | None" = None,
) -> DocumentPageList:
    """
    Universal document and ebook loader with OCR fallback and full telemetry tracking.
    Supports PDF, DOCX, EPUB, MOBI, XPS, TXT, MD, RST, RTF, HTML, CSV, JSON, and DOC.
    Returns a DocumentPageList exposing .stats for the Ingestion Health Panel.
    """
    file_path = Path(file_path)
    suffix = file_path.suffix.lower()
    filename = file_path.name

    # -------------------------------------------------------------------------
    # 1. PDF DOCUMENTS
    # -------------------------------------------------------------------------
    if suffix == ".pdf":
        try:
            doc = fitz.open(str(file_path)) if file_path.exists() else fitz.open(stream=content, filetype="pdf")
            total_doc_pages = len(doc)
        except Exception as exc:
            print(f"[load_document] Failed to open PDF {filename}: {exc}")
            return DocumentPageList([], {"pages_processed": 0, "ocr_pages": 0, "image_captions": 0, "chunks_created": 0, "skipped_sections": 0})

        pages: list[dict[str, Any]] = []
        total_images = 0
        total_captions = 0

        for idx in range(total_doc_pages):
            try:
                page_data = _extract_page_native(str(file_path), idx)
                img_list = doc[idx].get_images(full=True) if idx < len(doc) else []
                page_data["image_count"] = len(img_list)
                total_images += len(img_list)
                captions_found = _detect_captions_in_text(page_data.get("text", ""))
                total_captions += captions_found
                pages.append(page_data)
            except Exception as exc:
                print(f"[load_document] Page {idx + 1} extraction failed: {exc}")
                pages.append({
                    "page": idx + 1, "text": "", "blocks": [],
                    "has_images": False, "image_count": 0, "native_text_chars": 0,
                })

        try:
            doc.close()
        except Exception:
            pass

        # OCR fallback for pages with little or no native text
        empty_page_indexes = [
            idx for idx, page in enumerate(pages)
            if len((page.get("text") or "").strip()) < OCR_MIN_TEXT_CHARS
        ]
        ocr_performed_count = 0

        if empty_page_indexes and file_path.exists():
            print(f"[load_document] {filename}: {len(empty_page_indexes)} page(s) sent to OCR")
            try:
                ocr_results = _ocr_pages(str(file_path), empty_page_indexes)
                for idx, ocr_text in ocr_results.items():
                    if ocr_text.strip():
                        pages[idx]["text"] = ocr_text
                        pages[idx]["native_text_chars"] = len(ocr_text)
                        pages[idx]["ocr_performed"] = True
                        ocr_performed_count += 1
                        total_captions += _detect_captions_in_text(ocr_text)
            except Exception as exc:
                print(f"[load_document] OCR failed for {filename}: {exc}")

        # Clean running headers & footers
        pages = remove_running_headers_footers(pages)
        raw_page_count = len(pages)
        valid_pages = [p for p in pages if (p.get("text") or "").strip()]
        skipped_sections = total_doc_pages - len(valid_pages)

        for p in valid_pages:
            p["source"] = filename
            p["path"] = str(file_path)

        stats = {
            "pages_processed": total_doc_pages,
            "ocr_pages": ocr_performed_count,
            "image_captions": max(total_captions, total_images),
            "chunks_created": 0,
            "skipped_sections": max(0, skipped_sections),
        }
        return DocumentPageList(valid_pages, stats)

    # -------------------------------------------------------------------------
    # 2. EBOOKS (EPUB, MOBI, XPS, FB2, CBZ)
    # -------------------------------------------------------------------------
    if suffix in {".epub", ".mobi", ".xps", ".oxps", ".fb2", ".cbz"}:
        try:
            doc = fitz.open(str(file_path)) if file_path.exists() else fitz.open(stream=content, filetype=suffix.lstrip("."))
            total_doc_pages = len(doc)
        except Exception as exc:
            print(f"[load_document] Failed to open ebook {filename}: {exc}")
            return DocumentPageList([], {"pages_processed": 0, "ocr_pages": 0, "image_captions": 0, "chunks_created": 0, "skipped_sections": 0})

        pages = []
        total_images = 0
        total_captions = 0

        for idx in range(total_doc_pages):
            try:
                page = doc[idx]
                text = page.get_text()
                text = clean_text(text)
                img_list = page.get_images(full=True)
                img_count = len(img_list)
                total_images += img_count
                captions_found = _detect_captions_in_text(text)
                total_captions += captions_found
                
                if text.strip():
                    pages.append({
                        "page": idx + 1,
                        "text": text,
                        "source": filename,
                        "path": str(file_path),
                        "has_images": img_count > 0,
                        "image_count": img_count,
                        "native_text_chars": len(text),
                    })
            except Exception as exc:
                print(f"[load_document] Ebook page {idx + 1} read failed: {exc}")

        try:
            doc.close()
        except Exception:
            pass

        skipped_sections = total_doc_pages - len(pages)
        stats = {
            "pages_processed": total_doc_pages,
            "ocr_pages": 0,
            "image_captions": max(total_captions, total_images),
            "chunks_created": 0,
            "skipped_sections": max(0, skipped_sections),
        }
        return DocumentPageList(pages, stats)

    # -------------------------------------------------------------------------
    # 3. WORD DOCUMENTS (DOCX)
    # -------------------------------------------------------------------------
    if suffix in {".docx", ".doc"}:
        pages = []
        total_images = 0
        total_captions = 0
        doc_stream = io.BytesIO(content) if content is not None else (open(str(file_path), "rb").read() if file_path.exists() else None)
        
        extracted_sections: list[str] = []
        try:
            import docx
            stream_obj = io.BytesIO(content) if content is not None else str(file_path)
            doc_obj = docx.Document(stream_obj)
            current_section_lines: list[str] = []
            
            for p in doc_obj.paragraphs:
                p_text = clean_text(p.text)
                if not p_text:
                    continue
                # Split logical sections at major headings
                style_name = (p.style.name or "").lower() if p.style else ""
                if ("heading 1" in style_name or "title" in style_name) and current_section_lines:
                    extracted_sections.append("\n\n".join(current_section_lines))
                    current_section_lines = [p_text]
                else:
                    current_section_lines.append(p_text)
                    
            # Extract tables as structured text
            for table in doc_obj.tables:
                table_rows = []
                for row in table.rows:
                    cells = [clean_text(cell.text) for cell in row.cells if clean_text(cell.text)]
                    if cells:
                        table_rows.append(" | ".join(cells))
                if table_rows:
                    current_section_lines.append("\n".join(table_rows))
                    
            if current_section_lines:
                extracted_sections.append("\n\n".join(current_section_lines))
                
            # Count inline shapes / drawings as images
            try:
                for rel in doc_obj.part.rels.values():
                    if "image" in rel.target_ref.lower():
                        total_images += 1
            except Exception:
                pass

        except Exception as docx_err:
            print(f"[load_document] python-docx parsing failed, using XML/binary fallback: {docx_err}")
            # Robust fallback: extract text from word/document.xml in docx zip
            try:
                import zipfile
                import xml.etree.ElementTree as ET
                stream_for_zip = io.BytesIO(content) if content is not None else open(str(file_path), "rb")
                with zipfile.ZipFile(stream_for_zip) as z:
                    xml_data = z.read("word/document.xml")
                    root = ET.fromstring(xml_data)
                    texts = [elem.text for elem in root.iter() if elem.text and elem.text.strip()]
                    if texts:
                        extracted_sections = [" ".join(texts)]
            except Exception as zip_err:
                # Binary fallback for legacy .doc
                if doc_stream:
                    raw_str = doc_stream.decode("utf-8", errors="ignore") if isinstance(doc_stream, bytes) else ""
                    clean_words = re.findall(r"[A-Za-z0-9\.,;:'\"\(\)\-\?\!\s]{4,}", raw_str)
                    if clean_words:
                        extracted_sections = ["\n".join(clean_words)]

        # Group extracted text into logical pages (~350 words per page)
        words_per_page = 350
        page_idx = 1
        for sec_text in extracted_sections:
            sec_text = clean_text(sec_text)
            if not sec_text:
                continue
            total_captions += _detect_captions_in_text(sec_text)
            sec_words = sec_text.split()
            if len(sec_words) <= words_per_page:
                pages.append({
                    "page": page_idx, "text": sec_text, "source": filename,
                    "path": str(file_path), "has_images": total_images > 0, "image_count": total_images,
                })
                page_idx += 1
            else:
                # Chunk into logical pages
                for i in range(0, len(sec_words), words_per_page):
                    chunk_text = " ".join(sec_words[i:i + words_per_page])
                    pages.append({
                        "page": page_idx, "text": chunk_text, "source": filename,
                        "path": str(file_path), "has_images": total_images > 0, "image_count": total_images,
                    })
                    page_idx += 1

        stats = {
            "pages_processed": max(1, len(pages)),
            "ocr_pages": 0,
            "image_captions": max(total_captions, total_images),
            "chunks_created": 0,
            "skipped_sections": 0,
        }
        return DocumentPageList(pages, stats)

    # -------------------------------------------------------------------------
    # 4. PLAIN TEXT & MARKDOWN (TXT, MD, RST, RTF)
    # -------------------------------------------------------------------------
    if suffix in {".txt", ".md", ".markdown", ".rst", ".rtf"}:
        try:
            if content is not None:
                text = content.decode("utf-8", errors="replace")
            else:
                text = file_path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            try:
                raw_b = content if content is not None else file_path.read_bytes()
                text = raw_b.decode("latin-1", errors="ignore")
            except Exception as exc:
                print(f"[load_document] Failed to read {filename}: {exc}")
                return DocumentPageList([], {"pages_processed": 0, "ocr_pages": 0, "image_captions": 0, "chunks_created": 0, "skipped_sections": 0})

        text = clean_text(text)
        captions_found = _detect_captions_in_text(text)
        
        # Split into logical sections by form feed, markdown headers, or size
        sections = re.split(r"(?:\x0c|\n(?=#{1,3}\s+))", text)
        sections = [s.strip() for s in sections if s.strip()]
        if not sections:
            sections = [text] if text else []

        pages = []
        for idx, sec in enumerate(sections, start=1):
            pages.append({
                "page": idx, "text": sec, "source": filename,
                "path": str(file_path), "has_images": False, "image_count": 0,
            })

        stats = {
            "pages_processed": max(1, len(pages)),
            "ocr_pages": 0,
            "image_captions": captions_found,
            "chunks_created": 0,
            "skipped_sections": 0,
        }
        return DocumentPageList(pages, stats)

    # -------------------------------------------------------------------------
    # 5. WEB & DATA DOCUMENTS (HTML, HTM, CSV, TSV, JSON)
    # -------------------------------------------------------------------------
    if suffix in {".html", ".htm"}:
        raw_str = content.decode("utf-8", errors="ignore") if content is not None else file_path.read_text(encoding="utf-8", errors="ignore")
        # Strip HTML tags
        clean_html = re.sub(r"<script[^>]*>.*?</script>", "", raw_str, flags=re.DOTALL | re.IGNORECASE)
        clean_html = re.sub(r"<style[^>]*>.*?</style>", "", clean_html, flags=re.DOTALL | re.IGNORECASE)
        clean_html = re.sub(r"<[^>]+>", " ", clean_html)
        clean_html = clean_text(clean_html)
        captions_found = _detect_captions_in_text(clean_html)
        pages = [{"page": 1, "text": clean_html, "source": filename, "path": str(file_path), "has_images": False, "image_count": 0}] if clean_html else []
        return DocumentPageList(pages, {"pages_processed": 1, "ocr_pages": 0, "image_captions": captions_found, "chunks_created": 0, "skipped_sections": 0})

    if suffix in {".csv", ".tsv"}:
        raw_str = content.decode("utf-8", errors="ignore") if content is not None else file_path.read_text(encoding="utf-8", errors="ignore")
        lines = [line.strip() for line in raw_str.splitlines() if line.strip()]
        chunk_lines_count = 35
        pages = []
        for idx in range(0, len(lines), chunk_lines_count):
            p_text = "\n".join(lines[idx:idx + chunk_lines_count])
            pages.append({
                "page": (idx // chunk_lines_count) + 1, "text": p_text,
                "source": filename, "path": str(file_path), "has_images": False, "image_count": 0,
            })
        return DocumentPageList(pages, {"pages_processed": max(1, len(pages)), "ocr_pages": 0, "image_captions": 0, "chunks_created": 0, "skipped_sections": 0})

    if suffix == ".json":
        raw_str = content.decode("utf-8", errors="ignore") if content is not None else file_path.read_text(encoding="utf-8", errors="ignore")
        try:
            parsed_json = json.loads(raw_str)
            clean_str = json.dumps(parsed_json, indent=2)
        except Exception:
            clean_str = raw_str
        pages = [{"page": 1, "text": clean_str, "source": filename, "path": str(file_path), "has_images": False, "image_count": 0}] if clean_str else []
        return DocumentPageList(pages, {"pages_processed": 1, "ocr_pages": 0, "image_captions": 0, "chunks_created": 0, "skipped_sections": 0})

    # Fallback generic text loader
    try:
        raw_b = content if content is not None else file_path.read_bytes()
        text = raw_b.decode("utf-8", errors="ignore")
        if text.strip():
            pages = [{"page": 1, "text": clean_text(text), "source": filename, "path": str(file_path), "has_images": False, "image_count": 0}]
            return DocumentPageList(pages, {"pages_processed": 1, "ocr_pages": 0, "image_captions": 0, "chunks_created": 0, "skipped_sections": 0})
    except Exception as exc:
        print(f"[load_document] Unsupported format {suffix}: {exc}")

    return DocumentPageList([], {"pages_processed": 0, "ocr_pages": 0, "image_captions": 0, "chunks_created": 0, "skipped_sections": 0})



# ============================================================
# MAIN DEBUG ENTRY
# ============================================================

if __name__ == "__main__":

    print(
        "document_loader.py loaded successfully."
    )

    print(
        f"Knowledge base: "
        f"{KNOWLEDGE_BASE_DIR}"
    )

    print(
        f"Target tokens: "
        f"{TARGET_TOKENS}"
    )

    print(
        f"Maximum tokens: "
        f"{MAX_TOKENS}"
    )

    print(
        f"Minimum tokens: "
        f"{MIN_TOKENS}"
    )

    print(
        f"OCR mode: "
        f"{OCR_MODE}"
    )

    print(
        f"OCR workers: "
        f"{OCR_WORKERS}"
    )