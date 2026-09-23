# AI Org Structure Analyzer

HackAlem MVP for extracting organizational units and their functions from
BEFORE and AFTER organizational documents, with traceable original evidence.

## Current functionality

- Upload one BEFORE and one AFTER PDF, DOCX, or XLSX (20 MB maximum each).
- Click **Analyze documents** to parse both documents and run AI extraction.
- View units, parent names when explicit, function counts, expandable functions,
  and source filenames, pages/locators, sections, and original extracted text.
- Inspect the original H1 preview (total chunks and first five per document).
- Results live only in Streamlit session state. Changing an upload clears them.
  Ordinary UI reruns do not call the API; each Analyze click starts a fresh run.
- A failed AI run leaves the source previews available, without presenting a
  partially processed document pair as a complete extraction.

## Parsing and source evidence

PDF pages split at line-start dotted section markers such as 3.4., 5.3.2.,
and 9.21. Preamble text is retained. Concatenating a page's chunks reproduces
all of its extracted text, including whitespace. A page without detectable
markers remains one chunk. IDs are deterministic source positions such as
before_p006_003 and are scoped to the current analysis.

DOCX retains non-empty body paragraphs with physical paragraph indices.
XLSX retains non-empty rows with sheet names, row numbers and column letters;
formulas remain source text.

The AI returns structured unit/function interpretations and chunk IDs.
Python discards references not supplied in the current API batch, then resolves
them against the original document's SourceChunk objects. Units/functions with
no valid references are discarded. Source quotations, pages, and locators shown
in the UI always come from those original chunks, never from model output.
Reference validation establishes that evidence exists, not that the AI's
interpretation is correct. Review the cited text.

## AI extraction (H2.3)

Uses the official OpenAI Python SDK, Responses API responses.parse, and
Pydantic Structured Outputs. Default model: gpt-4.1-mini, configurable with
OPENAI_MODEL. Each document is processed independently in two stages.

**Stage A — structure only.** All source batches are processed before Stage B.
The schema contains units and structural/alias evidence, with no functions.
Existing structural validation and alias normalization produce a complete
registry with ID, canonical_name, aliases, parent and source_refs.
Roles, directors and working groups are not units merely because they have
subordinates. Title-like names require an explicit structural naming definition.
Aliases require an explicit equivalence in the source.

**Stage B — functions only.** Every source batch is read again with the complete
validated registry (IDs, canonical names, aliases and parents). This schema
cannot create units. Each function chooses an existing owner_unit_id and cites
source_chunk_ids. Ownership may be a direct assignment, an abbreviation marker
on a duty, a section dedicated to a unit, or a responsibility explicitly assigned
to that unit through its director/manager. Roles remain source context, not units.
The model marks ambiguous ownership or unsupported interpretations for rejection.

Python rejects unknown owners, empty evidence, ambiguous/unsupported candidates,
and functions containing any unknown or unsent evidence ID. It resolves source
references against original chunks. Semantic support and ownership interpretation
are assessed by the model, not proven by deterministic ID validation.
There is no exact-substring or narrow role/heading regex filter on Stage B output:
a specific responsibility or faithful concise formulation may come from a larger
chunk. The UI always displays the untouched source chunk as evidence.

The model is instructed to consolidate equivalent duties conservatively. Python
also merges matching normalized wording within one owner (list labels, case,
whitespace and punctuation), preserving all references. Different owners, scopes,
word order and negations are not merged automatically. Semantic paraphrases
across batches can still remain separate.

Source batches are at most 24,000 UTF-8 bytes with one excerpt of overlap.
Stage B adds a registry limited to 24,000 bytes plus small JSON overhead; an
oversized registry causes an explicit error rather than silently dropping units.
Oversized source chunks are sliced, retaining their original IDs for retrieval.
Each stage uses store=False, an 8,000-token output cap, a 90-second request timeout
and at most one SDK retry. No validated units means Stage B is skipped. Refusals,
incomplete output or API errors reject the run instead of showing partial success.
Two passes generally mean twice as many extraction requests as H2.2.
Old session extraction results are cleared when the policy changes to H2.3.

## Installation and key setup

Python 3.10 or newer:

~~~powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env
~~~

Create .env only if you do not already have one. Edit it locally:

~~~dotenv
OPENAI_API_KEY=your-key-here
OPENAI_MODEL=gpt-4.1-mini
~~~

The app reads the repository .env on each explicit run. Environment variables
take precedence. A missing/blank key shows a clear UI error without calling
OpenAI. Never commit .env or paste a real key into code. Analyze sends extracted
document text to OpenAI and may incur API charges.

## How to run

From this repository:

~~~powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
~~~

On macOS/Linux, use .venv/bin/python instead. Open the URL printed by Streamlit.

For the isolated Python 3.12 environment used to verify H2 on this machine:

~~~powershell
.\.venv-h2\Scripts\python.exe -m streamlit run app.py
~~~

The existing .venv was preserved; .venv-h2 is local and Git-ignored.

## Tests

~~~powershell
.\.venv-h2\Scripts\python.exe -m unittest discover -s tests -v
~~~

Tests preserve H1 coverage and use generated fixtures and mocked OpenAI calls.
They do not spend API credits. Real revision 8/9 extraction is not certified by
these tests.

## Limitations

No OCR; scanned pages without embedded text are skipped. PDF reading order and
section recognition depend on text layout; section continuations across pages
remain separate. DOCX tables, headers, footers, text boxes and automatic list
numbering are not extracted. XLSX formulas are not calculated.
Long sections may be fragmented across batches; limited overlap can miss
distant unit/function relationships. Deduplication does not resolve semantic
paraphrases or unrecognized name/parent variations. Alias definitions in other
formats may be missed. Model-based ownership may still omit duties or attach an
existing but irrelevant source reference. Review the original evidence. No live
revision 8/9 quality claim is established by mocked tests.
No BEFORE/AFTER matching, findings, recommendations, embeddings, reports or
persistent database are implemented. Use trusted local hackathon documents.

Official references:
- https://developers.openai.com/api/docs/guides/structured-outputs
- https://developers.openai.com/api/docs/models/gpt-4.1-mini

