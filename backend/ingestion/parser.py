"""
Parser — reads Excel, CSV, and PDF files into chunks for embedding.
"""

from pathlib import Path
from dataclasses import dataclass, field

import pandas as pd

from core.config import settings


@dataclass
class Chunk:
    text: str
    metadata: dict = field(default_factory=dict)


def parse_file(file_path: Path, file_id: str) -> list[Chunk]:
    suffix = file_path.suffix.lower()

    if suffix == ".csv":
        chunks = _parse_csv(file_path, file_id)
    elif suffix in (".xlsx", ".xls"):
        chunks = _parse_excel(file_path, file_id)
    elif suffix == ".pdf":
        chunks = _parse_pdf(file_path, file_id)
    else:
        raise ValueError(f"Unsupported file type: '{suffix}'. Supported: .csv, .xlsx, .xls, .pdf")

    summary = _build_summary_chunk(file_path, file_id, chunks)
    if summary:
        chunks.insert(0, summary)

    return chunks


# ── CSV ───────────────────────────────────────────────────────────

def _parse_csv(file_path: Path, file_id: str) -> list[Chunk]:
    all_chunks: list[Chunk] = []
    row_offset = 0
    headers = None

    for batch_df in pd.read_csv(
        file_path, chunksize=settings.pandas_chunk_size, dtype=str, keep_default_na=False
    ):
        if headers is None:
            headers = list(batch_df.columns)
        all_chunks.extend(
            _dataframe_to_chunks(batch_df, headers, file_id, file_path.name, "default", row_offset)
        )
        row_offset += len(batch_df)

    return all_chunks


# ── Excel ─────────────────────────────────────────────────────────

def _parse_excel(file_path: Path, file_id: str) -> list[Chunk]:
    all_chunks: list[Chunk] = []

    with pd.ExcelFile(file_path) as xls:
        for sheet_name in xls.sheet_names:
            df = pd.read_excel(xls, sheet_name=sheet_name, dtype=str, keep_default_na=False)
            if df.empty:
                continue
            headers = list(df.columns)
            for start in range(0, len(df), settings.pandas_chunk_size):
                end = min(start + settings.pandas_chunk_size, len(df))
                all_chunks.extend(
                    _dataframe_to_chunks(df.iloc[start:end], headers, file_id, file_path.name, sheet_name, start)
                )

    return all_chunks


# ── PDF ───────────────────────────────────────────────────────────

def _parse_pdf(file_path: Path, file_id: str) -> list[Chunk]:
    try:
        import pdfplumber
    except ImportError:
        raise ImportError("pdfplumber required for PDF parsing. Run: pip install pdfplumber")

    all_chunks: list[Chunk] = []

    with pdfplumber.open(file_path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            page_label = f"Page {page_num + 1}"

            # Tables → structured DataFrame chunks
            for table_idx, table in enumerate(page.extract_tables() or []):
                if not table or len(table) < 2:
                    continue

                raw_headers = [str(h).strip() if h else f"Col{i}" for i, h in enumerate(table[0])]
                seen: dict[str, int] = {}
                headers: list[str] = []
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
                all_chunks.extend(
                    _dataframe_to_chunks(df, headers, file_id, file_path.name, f"{page_label} Table {table_idx + 1}", 0)
                )

            # Free text → paragraph chunks
            text = (page.extract_text() or "").strip()
            if len(text) > 80:
                paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
                buf: list[str] = []
                buf_len = 0
                target = 800

                def _flush():
                    nonlocal buf, buf_len
                    if not buf:
                        return
                    block = "\n\n".join(buf)
                    all_chunks.append(Chunk(
                        text=f"[PDF Text — {file_path.name} | {page_label}]\n\n{block}",
                        metadata={
                            "file_id": file_id,
                            "file_name": file_path.name,
                            "sheet_name": f"{page_label} text",
                            "row_start": page_num + 1,
                            "row_end": page_num + 1,
                            "column_names": "text",
                            "num_rows": 1,
                        },
                    ))
                    buf = []
                    buf_len = 0

                for para in paragraphs:
                    if buf_len + len(para) > target and buf:
                        _flush()
                    buf.append(para)
                    buf_len += len(para)

                _flush()

    return all_chunks


# ── Shared chunking ───────────────────────────────────────────────

def _dataframe_to_chunks(
    df: pd.DataFrame, headers: list[str],
    file_id: str, file_name: str, sheet_name: str, row_offset: int,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    header_line = "Columns: " + " | ".join(headers)
    chunk_size = settings.chunk_size
    overlap    = settings.chunk_overlap
    step       = max(chunk_size - overlap, 1)
    total      = len(df)

    for start in range(0, total, step):
        end = min(start + chunk_size, total)
        chunk_df = df.iloc[start:end]
        if len(chunk_df) < overlap and start > 0:
            break

        rows_text = _format_rows(chunk_df, headers, row_offset + start)
        abs_start = row_offset + start + 2
        abs_end   = row_offset + end + 1

        chunks.append(Chunk(
            text=f"{header_line}\n{rows_text}",
            metadata={
                "file_id": file_id,
                "file_name": file_name,
                "sheet_name": sheet_name,
                "row_start": abs_start,
                "row_end": abs_end,
                "column_names": " | ".join(headers),
                "num_rows": len(chunk_df),
            },
        ))
        if end >= total:
            break

    return chunks


def _format_rows(df: pd.DataFrame, headers: list[str], start_row: int) -> str:
    lines = []
    for idx, (_, row) in enumerate(df.iterrows()):
        values = " | ".join(str(v).strip() for v in row)
        lines.append(f"Row {start_row + idx + 2}: {values}")
    return "\n".join(lines)


# ── Summary chunk ─────────────────────────────────────────────────

CATEGORICAL_THRESHOLD = 50


def _build_summary_chunk(file_path: Path, file_id: str, chunks: list[Chunk]) -> Chunk | None:
    if not chunks:
        return None

    suffix = file_path.suffix.lower()
    if suffix == ".pdf":
        return _build_pdf_summary(file_path, file_id, chunks)

    try:
        if suffix == ".csv":
            df = pd.read_csv(file_path, dtype=str, keep_default_na=False)
        else:
            df = pd.read_excel(file_path, dtype=str, keep_default_na=False)
    except Exception:
        return None

    total_rows = len(df)
    columns    = list(df.columns)
    sheets     = list(set(c.metadata.get("sheet_name", "default") for c in chunks))

    lines = [
        f"=== DATASET SUMMARY for {file_path.name} ===",
        f"Total rows: {total_rows}",
        f"Total columns: {len(columns)}",
        f"Column names: {' | '.join(columns)}",
        f"Sheets: {', '.join(sheets)}",
        "",
        "=== COLUMN STATISTICS (computed over ALL rows) ===",
    ]

    for col in columns:
        non_empty = df[col][df[col].str.strip() != ""].shape[0]
        unique    = df[col].nunique()
        lines.append(f"\n  Column: {col}")
        lines.append(f"    Non-empty: {non_empty}/{total_rows}  |  Unique: {unique}")

        if unique == 0:
            continue

        vc = df[col].value_counts()
        if unique <= CATEGORICAL_THRESHOLD:
            lines.append(f"    Value counts (ALL): {', '.join(f'{v}={c}' for v, c in vc.items())}")
        else:
            lines.append(f"    Top 10: {', '.join(f'{v}={c}' for v, c in vc.head(10).items())}")

        try:
            num = pd.to_numeric(df[col], errors="coerce").dropna()
            if len(num) > total_rows * 0.5:
                lines.append(
                    f"    Numeric: min={num.min():.2f}, max={num.max():.2f}, "
                    f"mean={num.mean():.2f}, median={num.median():.2f}"
                )
        except Exception:
            pass

    lines.append("\n=== SAMPLE ROWS (first 3) ===")
    lines.append("Columns: " + " | ".join(columns))
    for idx, (_, row) in enumerate(df.head(3).iterrows()):
        lines.append(f"Row {idx + 2}: {' | '.join(str(v).strip() for v in row)}")

    return Chunk(
        text="\n".join(lines),
        metadata={
            "file_id": file_id,
            "file_name": file_path.name,
            "sheet_name": "summary",
            "row_start": 0,
            "row_end": total_rows,
            "column_names": " | ".join(columns),
            "num_rows": total_rows,
        },
    )


def _build_pdf_summary(file_path: Path, file_id: str, chunks: list[Chunk]) -> Chunk:
    data_chunks  = [c for c in chunks if c.metadata.get("sheet_name") != "summary"]
    sheets       = list(set(c.metadata.get("sheet_name", "") for c in data_chunks))
    table_sheets = [s for s in sheets if "Table" in s]
    text_sheets  = [s for s in sheets if "text" in s]

    lines = [
        f"=== PDF SUMMARY for {file_path.name} ===",
        f"Total pages with content: {len(set(c.metadata.get('row_start', 0) for c in data_chunks))}",
        f"Table sections: {len(table_sheets)}",
        f"Text sections: {len(text_sheets)}",
        f"Total chunks: {len(data_chunks)}",
        "",
        "Sections:",
    ]
    for s in sheets[:20]:
        lines.append(f"  - {s}")

    return Chunk(
        text="\n".join(lines),
        metadata={
            "file_id": file_id,
            "file_name": file_path.name,
            "sheet_name": "summary",
            "row_start": 0,
            "row_end": len(data_chunks),
            "column_names": "pdf_content",
            "num_rows": len(data_chunks),
        },
    )
