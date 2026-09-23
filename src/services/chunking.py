import re

from src.models import Document, SourceChunk

# Only explicit dotted numbering at the start of a line; no inferred sections.
SECTION_PATTERN = re.compile(r"^\s*(\d+(?:\.\d+)+)\.?\s+\S", re.MULTILINE)


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
