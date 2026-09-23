import re

from src.models import Document, SourceChunk

# Only explicit dotted numbering at the start of a line; no inferred sections.
SECTION_PATTERN = re.compile(
    r"^[ \t]*(\d+(?:\.\d+)+)\.?(?=[ \t]|\r?$)", re.MULTILINE
)


def detect_section(text: str) -> str | None:
    match = SECTION_PATTERN.search(text)
    return match.group(1) if match else None


def make_chunk(
    document: Document,
    source_key: str,
    locator: str,
    text: str,
    page: int | None = None,
) -> SourceChunk:
    """Source positions, not random IDs, identify chunks within an analysis."""
    return SourceChunk(
        chunk_id=f"{document.side.lower()}_{source_key}",
        document_id=document.id,
        filename=document.filename,
        page=page,
        section=detect_section(text),
        locator=locator,
        text=text,
    )


def split_pdf_page(document: Document, page: int, text: str) -> list[SourceChunk]:
    """Split at explicit line-start markers; concatenation preserves every character."""
    if not text.strip():
        return []
    starts = [match.start() for match in SECTION_PATTERN.finditer(text)]
    boundaries = sorted({0, *starts, len(text)})
    # Keep whitespace before the first heading attached to it, not as an empty chunk.
    if len(boundaries) > 2 and not text[:boundaries[1]].strip():
        boundaries.pop(1)
    chunks = []
    for index, (start, end) in enumerate(zip(boundaries, boundaries[1:]), start=1):
        chunk = make_chunk(
            document, f"p{page:03d}_{index:03d}",
            f"Page {page}, characters {start + 1}-{end}", text[start:end], page,
        )
        chunks.append(chunk)
    return chunks
