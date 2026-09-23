import hashlib
from pathlib import Path

from src.models import Document, Side
from src.parsers.docx_parser import parse_docx
from src.parsers.pdf_parser import parse_pdf
from src.parsers.xlsx_parser import parse_xlsx

PARSERS = {"pdf": parse_pdf, "docx": parse_docx, "xlsx": parse_xlsx}
MAX_FILE_BYTES = 20 * 1024 * 1024


class DocumentParseError(ValueError):
    """An upload error safe to display to the user."""


def parse_document(data: bytes, filename: str, side: Side) -> Document:
    if side not in ("BEFORE", "AFTER"):
        raise DocumentParseError("Document side must be BEFORE or AFTER.")
    extension = Path(filename).suffix.lower().lstrip(".")
    if extension not in PARSERS:
        raise DocumentParseError("Unsupported file. Please upload a PDF, DOCX, or XLSX file.")
    if not data:
        raise DocumentParseError("The file is empty. Please upload a non-empty document.")
    if len(data) > MAX_FILE_BYTES:
        raise DocumentParseError("The file is too large. The H1 limit is 20 MB per document.")

    digest = hashlib.sha256(filename.encode("utf-8") + b"\0" + data).hexdigest()
    document = Document(
        id=f"{side.lower()}_{digest}", side=side,
        filename=filename, format=extension,
    )
    try:
        document.chunks = PARSERS[extension](data, document)
    except Exception as exc:
        raise DocumentParseError(
            "Could not parse this file. Check that it is a valid, unencrypted "
            f"{extension.upper()} document and try saving it again."
        ) from exc
    if not document.chunks:
        raise DocumentParseError(
            "No readable text was found. Scanned PDFs require OCR, which is not "
            "included in H1. For DOCX, text must be in body paragraphs; "
            "for XLSX, at least one row must contain a value."
        )
    return document

