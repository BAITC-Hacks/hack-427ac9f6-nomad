from io import BytesIO

from docx import Document as WordDocument

from src.models import Document, SourceChunk
from src.services.chunking import make_chunk


def parse_docx(data: bytes, document: Document) -> list[SourceChunk]:
    word = WordDocument(BytesIO(data))
    chunks = []
    # Physical body paragraph indices include empty paragraphs.
    for index, paragraph in enumerate(word.paragraphs, start=1):
        text = paragraph.text
        if text.strip():
            chunks.append(make_chunk(
                document, f"para{index:04d}",
                f"Body paragraph {index}", text,
            ))
    return chunks
