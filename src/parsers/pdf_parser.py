import pymupdf

from src.models import Document, SourceChunk
from src.services.chunking import split_pdf_page


def parse_pdf(data: bytes, document: Document) -> list[SourceChunk]:
    chunks = []
    with pymupdf.open(stream=data, filetype="pdf") as pdf:
        if pdf.needs_pass:
            raise ValueError("Password-protected PDFs are not supported.")
        for page_number, page in enumerate(pdf, start=1):
            text = page.get_text("text")
            chunks.extend(split_pdf_page(document, page_number, text))
    return chunks
