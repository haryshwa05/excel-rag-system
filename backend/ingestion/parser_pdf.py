"""
Vision-capable PDF parser.

Strategy:
  1. Extract all text per page (prose, field descriptions, table of contents)
  2. Extract tables as structured DataFrame chunks
  3. For large images (>300x300px): describe with a vision LLM
     - UI screenshots, diagrams, form layouts get a text description
     - The description captures field names, tabs, buttons — all answerable content
  4. Deduplicate repeating header/logo images by MD5 hash + size filter
  5. Merge everything into one chunk stream with consistent metadata

This means "what tabs are on the Customer Accounts screen?" gets answered
from the image description, not just surrounding prose.

Requires:
    pip install pdfplumber pymupdf pillow
    (PyMuPDF provides 'fitz' for reliable image byte extraction)
"""

import hashlib
import asyncio
from pathlib import Path

import pdfplumber

try:
    import fitz          # PyMuPDF
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False

from core.config import settings

# ── Thresholds ────────────────────────────────────────────────────────────────

IMAGE_MIN_WIDTH  = 300    # narrower images are decorative
IMAGE_MIN_HEIGHT = 200

# Documents often have a repeating header banner — filter it by approximate size
HEADER_BANNER_WIDTH  = 436
HEADER_BANNER_HEIGHT = 62

MAX_IMAGES_PER_PAGE  = 3   # cap LLM calls per page
TEXT_LIGHT_THRESHOLD = 200  # chars; below this we consider page "mostly image"


# ── Image deduplication cache ─────────────────────────────────────────────────

_described_hashes: set[str] = set()
_vision_unavailable = False


def _img_hash(img_bytes: bytes) -> str:
    return hashlib.md5(img_bytes).hexdigest()


def _is_header(width: int, height: int) -> bool:
    return (
        abs(width  - HEADER_BANNER_WIDTH)  < 20 and
        abs(height - HEADER_BANNER_HEIGHT) < 20
    )


def _is_meaningful(width: int, height: int) -> bool:
    return (
        width  >= IMAGE_MIN_WIDTH  and
        height >= IMAGE_MIN_HEIGHT and
        not _is_header(width, height)
    )


# ── Vision LLM call ───────────────────────────────────────────────────────────

_VISION_PROMPT = (
    "This is a screenshot from a software user manual ({page_label}). "
    "Describe: screen or dialog name, all visible field labels, tab names, "
    "button labels, dropdown options, and any other text visible in the UI. "
    "Be specific — this description will answer user questions about the software."
)


async def _describe_image(img_bytes: bytes, page_label: str) -> str | None:
    """
    Ask a vision LLM to describe a UI screenshot. Returns a text description
    suitable for embedding, or None on failure/skip.
    """
    global _vision_unavailable

    h = _img_hash(img_bytes)
    if h in _described_hashes:
        return None
    _described_hashes.add(h)

    if _vision_unavailable:
        return None

    if not settings.enable_vision or settings.vision_provider == "none":
        return None

    try:
        if settings.vision_provider == "ollama":
            from ollama import AsyncClient
            client   = AsyncClient(host=settings.ollama_base_url)
            response = await asyncio.wait_for(
                client.chat(
                    model=settings.ollama_vision_model,
                    messages=[{
                        "role":    "user",
                        "content": _VISION_PROMPT.format(page_label=page_label),
                        "images":  [img_bytes],   # SDK accepts raw bytes directly
                    }],
                ),
                timeout=8,
            )
            return response.message.content

        elif settings.vision_provider == "openai":
            import base64 as _b64
            from openai import AsyncOpenAI
            b64        = _b64.b64encode(img_bytes).decode()
            media_type = "image/jpeg" if img_bytes[:3] == b'\xff\xd8\xff' else "image/png"
            client     = AsyncOpenAI(api_key=settings.openai_api_key)
            response   = await client.chat.completions.create(
                model="gpt-4o-mini",
                max_tokens=400,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{b64}", "detail": "low"}},
                        {"type": "text",      "text": _VISION_PROMPT.format(page_label=page_label)},
                    ],
                }],
            )
            return response.choices[0].message.content

        return None

    except Exception as e:
        if settings.vision_provider == "ollama":
            _vision_unavailable = True
            print("[parser_pdf] Vision disabled for this run: Ollama unavailable.")
        print(f"[parser_pdf] Vision failed on {page_label}: {e}")
        return None


# ── Image extraction ──────────────────────────────────────────────────────────

def _images_pymupdf(doc, page_num: int) -> list[tuple[bytes, int, int]]:
    """Extract meaningful images from a page using PyMuPDF."""
    page   = doc[page_num]
    result = []
    for img_info in page.get_images(full=True):
        xref = img_info[0]
        try:
            base_image = doc.extract_image(xref)
            w, h = base_image["width"], base_image["height"]
            if _is_meaningful(w, h):
                result.append((base_image["image"], w, h))
            if len(result) >= MAX_IMAGES_PER_PAGE:
                break
        except Exception:
            continue
    return result


def _images_pdfplumber(page) -> list[tuple[None, int, int]]:
    """Fallback: pdfplumber gives size metadata but not raw bytes."""
    result = []
    for img in (page.images or []):
        w = int(img.get("width",  0))
        h = int(img.get("height", 0))
        if _is_meaningful(w, h):
            result.append((None, w, h))
    return result


def _page_needs_vision(page, raw_text: str) -> bool:
    meaningful = [
        img for img in (page.images or [])
        if _is_meaningful(int(img.get("width", 0)), int(img.get("height", 0)))
    ]
    return len(meaningful) > 0


# ── Text chunking ─────────────────────────────────────────────────────────────

def _text_chunks(
    text: str,
    page_label: str,
    page_num: int,
    file_path: Path,
    file_id: str,
    chunk_size: int,
) -> list:
    """Split page text into chunks at paragraph boundaries."""
    from ingestion.parser import Chunk

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[Chunk] = []
    buffer: list[str]   = []
    buf_len = 0

    def flush():
        if not buffer:
            return
        content = "\n\n".join(buffer)
        chunks.append(Chunk(
            text=f"[PDF Text — {file_path.name} | {page_label}]\n\n{content}",
            metadata={
                "file_id":      file_id,
                "file_name":    file_path.name,
                "sheet_name":   f"{page_label} text",
                "row_start":    page_num + 1,
                "row_end":      page_num + 1,
                "column_names": "text",
                "num_rows":     1,
                "content_type": "text",
            },
        ))
        buffer.clear()

    for para in paragraphs:
        if buf_len + len(para) > chunk_size and buffer:
            flush()
            buf_len = 0
        buffer.append(para)
        buf_len += len(para)
    flush()
    return chunks


# ── Table extraction ──────────────────────────────────────────────────────────

def _table_chunks(page, page_label: str, file_path: Path, file_id: str) -> list:
    from ingestion.parser import Chunk, _dataframe_to_chunks
    import pandas as pd

    chunks: list[Chunk] = []
    for tbl_idx, table in enumerate(page.extract_tables() or []):
        if not table or len(table) < 2:
            continue

        raw_headers = [str(h).strip() if h else f"Col{i}" for i, h in enumerate(table[0])]
        seen: dict[str, int] = {}
        headers: list[str]   = []
        for h in raw_headers:
            if h in seen:
                seen[h] += 1
                headers.append(f"{h}_{seen[h]}")
            else:
                seen[h] = 0
                headers.append(h)

        rows = [[str(c).strip() if c else "" for c in row] for row in table[1:]]
        if not rows:
            continue

        df = pd.DataFrame(rows, columns=headers)
        chunks.extend(
            _dataframe_to_chunks(
                df, headers, file_id, file_path.name,
                f"{page_label} Table {tbl_idx + 1}", 0,
            )
        )
    return chunks


# ── Main entry point ──────────────────────────────────────────────────────────

async def parse_pdf_with_vision(
    file_path: Path,
    file_id: str,
    chunk_size_chars: int = 1200,
) -> list:
    """
    Parse a PDF into chunks: text + tables + optional vision image descriptions.
    Returns a list of Chunk objects compatible with the rest of the ingestion pipeline.
    """
    from ingestion.parser import Chunk

    all_chunks: list[Chunk] = []

    fitz_doc = None
    if HAS_PYMUPDF:
        fitz_doc = fitz.open(str(file_path))

    with pdfplumber.open(file_path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            page_label = f"Page {page_num + 1}"

            # Tables first (structured data)
            all_chunks.extend(_table_chunks(page, page_label, file_path, file_id))

            # Page text
            raw_text = (page.extract_text() or "").strip()
            if len(raw_text) > 30:
                all_chunks.extend(
                    _text_chunks(raw_text, page_label, page_num, file_path, file_id, chunk_size_chars)
                )

            # Vision: only if the page has meaningful images and vision is enabled
            if settings.enable_vision and _page_needs_vision(page, raw_text):
                images = (
                    _images_pymupdf(fitz_doc, page_num)
                    if (fitz_doc and HAS_PYMUPDF)
                    else _images_pdfplumber(page)
                )

                for img_bytes, width, height in images:
                    if img_bytes is None:
                        continue   # pdfplumber fallback has no raw bytes

                    description = await _describe_image(img_bytes, page_label)
                    if description:
                        all_chunks.append(Chunk(
                            text=(
                                f"[UI Screenshot — {file_path.name} | {page_label} | {width}×{height}px]\n\n"
                                f"{description}"
                            ),
                            metadata={
                                "file_id":      file_id,
                                "file_name":    file_path.name,
                                "sheet_name":   f"{page_label} screenshot",
                                "row_start":    page_num + 1,
                                "row_end":      page_num + 1,
                                "column_names": "screenshot_description",
                                "num_rows":     1,
                                "content_type": "image_description",
                                "image_size":   f"{width}x{height}",
                            },
                        ))

    if fitz_doc:
        fitz_doc.close()

    return all_chunks
