import pymupdf

from src.models import Document, SourceChunk
from src.services.chunking import make_chunk


def parse_pdf(data: bytes, document: Document) -> list[SourceChunk]:
    chunks = []
    with pymupdf.open(stream=data, filetype="pdf") as pdf:
        if pdf.needs_pass:
            raise ValueError("Password-protected PDFs are not supported.")
        for page_number, page in enumerate(pdf, start=1):
            text = page.get_text("text")
            if text.strip():
                chunks.append(make_chunk(
                    document, f"p{page_number:03d}_001",
                    f"Page {page_number}", text, page=page_number,
                ))
    return chunks
