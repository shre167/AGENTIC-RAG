"""
Image analysis using Gemini Vision via the OpenAI-compatible multimodal API.

Extracts images from PDF pages (using PyMuPDF) and sends them to the
configured vision model to generate descriptive text summaries.

This module is deliberately isolated from the core retrieval pipeline:
- It only runs during ingestion when ENABLE_IMAGE_ANALYSIS=true
- Results are stored as image_description in chunk metadata
- If the vision API call fails, ingestion continues without image descriptions
- The text RAG pipeline is never blocked on image analysis
"""
from __future__ import annotations

import base64
import logging
from typing import Any

logger = logging.getLogger(__name__)


def extract_page_images(pdf_path: str, page_index: int) -> list[bytes]:
    """
    Extract all embedded images from a PDF page as PNG bytes.

    Returns a list of image bytes. Empty list if no images or extraction fails.
    """
    try:
        import pymupdf as fitz
        doc = fitz.open(pdf_path)
        try:
            page = doc[page_index]
            image_list = page.get_images(full=True)
            images = []
            for img_info in image_list:
                xref = img_info[0]
                try:
                    base_image = doc.extract_image(xref)
                    img_bytes = base_image.get("image")
                    if img_bytes and len(img_bytes) > 1000:  # skip tiny icons
                        images.append(img_bytes)
                except Exception as e:
                    logger.debug("Failed to extract image xref %s: %s", xref, e)
            return images
        finally:
            doc.close()
    except Exception as e:
        logger.warning("extract_page_images failed for page %d: %s", page_index, e)
        return []


def describe_image(
    image_bytes: bytes,
    *,
    api_key: str,
    base_url: str,
    model: str,
    context_hint: str = "",
) -> str:
    """
    Send an image to Gemini Vision and return a text description.

    Uses the OpenAI-compatible multimodal message format:
    content: [{"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}]

    Returns empty string on failure (ingestion should continue).
    """
    try:
        import httpx
        from openai import OpenAI

        b64 = base64.b64encode(image_bytes).decode("utf-8")
        data_url = f"data:image/png;base64,{b64}"

        prompt = (
            "This image is from a book page. "
            "Describe what you see in 2-4 sentences: "
            "what type of visual is it (diagram, table, chart, figure, photo), "
            "what it shows, and any key labels or values visible. "
            "Be factual and concise."
        )
        if context_hint:
            prompt += f" Context: {context_hint}"

        normalized_url = base_url.strip()
        if not normalized_url.endswith("/"):
            normalized_url += "/"

        http_client = httpx.Client(verify=False, timeout=httpx.Timeout(30.0, connect=10.0))
        client = OpenAI(api_key=api_key, base_url=normalized_url, http_client=http_client)

        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                }
            ],
            max_tokens=300,
            temperature=0.1,
        )

        if response.choices:
            return (response.choices[0].message.content or "").strip()
        return ""

    except Exception as e:
        logger.warning("describe_image failed: %s", e)
        return ""


def analyze_pdf_page_images(
    pdf_path: str,
    page_index: int,
    *,
    api_key: str,
    base_url: str,
    model: str,
    context_hint: str = "",
) -> str:
    """
    Extract and describe all images on a PDF page.

    Returns a combined description string, or empty string if no images
    or analysis fails. This is the main entry point for the ingestion pipeline.
    """
    images = extract_page_images(pdf_path, page_index)
    if not images:
        return ""

    descriptions = []
    for i, img_bytes in enumerate(images[:3], 1):  # cap at 3 images per page
        desc = describe_image(
            img_bytes,
            api_key=api_key,
            base_url=base_url,
            model=model,
            context_hint=context_hint,
        )
        if desc:
            prefix = f"Figure {i}: " if len(images) > 1 else ""
            descriptions.append(f"{prefix}{desc}")

    return " | ".join(descriptions) if descriptions else ""
