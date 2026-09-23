# AI Org Structure Analyzer

A HackAlem hackathon project for preparing organizational documents from before
and after reorganization for review.

## Current H1 functionality

Upload a BEFORE document and an AFTER document, then click **Analyze documents**
to extract text. Each side shows its filename, total chunk count, and first five
chunks with source IDs, location, detected section, and original extracted text.
Documents are held only in Python objects in the current Streamlit session.
Changing an upload clears the previous result.

Source chunks retain document ID and filename. IDs use the side and source
position (for example, `before_p006_001`) and are deterministic within an
analysis. IDs are scoped to the current pair of documents, not globally unique
across analyses. The stored chunks are the source of truth for original text.
Section detection uses explicit dotted numbers at the start of a line; a page
chunk records its first detected section, or null if none is found.

## Supported formats

- **PDF:** PyMuPDF extracts one chunk per non-empty page, with a 1-based page
  number. Scanned images have no OCR support. Reading order follows PDF text
  extraction and may be imperfect for complex layouts.
- **DOCX:** python-docx extracts non-empty body paragraphs, keeping their
  1-based physical paragraph index, including gaps for empty paragraphs.
  Tables, headers, footers, text boxes, and Word-generated list numbering
  are not extracted. DOCX page numbers are unavailable and remain null.
- **XLSX:** openpyxl extracts non-empty rows, keeping worksheet name, 1-based
  row number, and column letters. Formulas are preserved as formula text,
  not recalculated. Page and section may be null.

Uploads are limited to 20 MB each. Empty, unreadable, unsupported, encrypted,
or invalid documents show an error. An image-only PDF with some text pages
includes only its readable text pages. Large or heavily compressed Office
documents may require substantial memory; this MVP is for local use with
trusted hackathon documents.

## Installation

Requires Python 3.10 or newer.

```bash
python -m venv .venv
```

Activate the virtual environment:

- Windows PowerShell: `.venv\Scripts\Activate.ps1`
- macOS/Linux: `source .venv/bin/activate`

```bash
python -m pip install -r requirements.txt
```

No API keys or environment configuration are required.

## How to run

From the project directory:

```bash
python -m streamlit run app.py
```

Open the local URL shown by Streamlit, normally http://localhost:8501.
