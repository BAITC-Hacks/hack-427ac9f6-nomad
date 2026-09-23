import streamlit as st

from src.config import ConfigurationError, load_settings
from src.models import Document, ExtractionResult
from src.parsers.parser_factory import DocumentParseError, parse_document
from src.services.extraction import ExtractionError, extract_document, resolve_refs


def clear_results() -> None:
    st.session_state.pop("documents", None)
    st.session_state.pop("extractions", None)


def show_preview(document: Document) -> None:
    st.subheader(document.side)
    st.write(f"Filename: {document.filename}")
    st.write(f"Total extracted chunks: {len(document.chunks)}")
    for chunk in document.chunks[:5]:
        with st.container(border=True):
            st.code(chunk.chunk_id, language=None)
            st.write(f"Page / locator: {chunk.locator}")
            st.write(f"Section: {chunk.section or 'Not detected'}")
            st.text(chunk.text)


def show_sources(document: Document, refs: list[str]) -> None:
    for chunk in resolve_refs(document, refs):
        st.caption(
            f"{chunk.chunk_id} | {chunk.filename} | {chunk.locator} | "
            f"Section: {chunk.section or 'Not detected'}"
        )
        st.text(chunk.text)


def show_extraction(document: Document, result: ExtractionResult) -> None:
    st.subheader(f"{document.side} — Organizational Units")
    unit_column, function_column = st.columns(2)
    unit_column.metric(f"{document.side} units", len(result.units))
    function_column.metric(f"{document.side} functions", len(result.functions))
    st.caption("References checked against source chunks. AI interpretations still require review.")
    if result.discarded_items or result.discarded_references:
        st.warning(
            f"Discarded references: {result.discarded_references}; "
            f"items rejected by structural/ownership/evidence checks: {result.discarded_items}."
        )
    if not result.units:
        st.info("No organizational units with valid source references were extracted.")
    for unit in result.units:
        functions = [item for item in result.functions if item.unit_id == unit.id]
        with st.container(border=True):
            st.write(unit.name)
            if unit.aliases:
                st.caption("Aliases: " + ", ".join(unit.aliases))
            if unit.parent:
                st.write(f"Parent: {unit.parent}")
            st.write(f"Functions: {len(functions)}")
            st.caption("Source references: " + ", ".join(unit.source_refs))
            with st.expander("Unit source text"):
                show_sources(document, unit.source_refs)
            with st.expander(f"Functions ({len(functions)})"):
                for function in functions:
                    st.write(function.text)
                    show_sources(document, function.source_refs)
                    st.divider()


def main() -> None:
    st.set_page_config(page_title="AI Org Structure Analyzer", layout="wide")
    if st.session_state.get("extraction_policy") != "h2.3":
        st.session_state.pop("extractions", None)
        st.session_state["extraction_policy"] = "h2.3"
    st.title("AI Org Structure Analyzer")
    st.caption("H2.3: validate organizational structure, then extract functions for those units.")
    st.caption("Analyze documents sends extracted text to OpenAI. Each click starts a new API run.")
    before_column, after_column = st.columns(2)
    uploads = {}
    for side, column in (("BEFORE", before_column), ("AFTER", after_column)):
        with column:
            st.subheader(side)
            uploads[side] = st.file_uploader(
                "Upload document", type=["pdf", "docx", "xlsx"],
                key=f"upload_{side.lower()}", on_change=clear_results,
                help="PDF, DOCX, or XLSX. Maximum 20 MB per file.",
            )
            if uploads[side] is not None:
                st.write(f"Uploaded: {uploads[side].name}")

    if st.button("Analyze documents", type="primary"):
        clear_results()
        if any(upload is None for upload in uploads.values()):
            st.error("Please upload both BEFORE and AFTER documents.")
        else:
            documents = {}
            with st.spinner("Extracting document text..."):
                for side, upload in uploads.items():
                    try:
                        documents[side] = parse_document(
                            upload.getvalue(), upload.name, side,
                        )
                    except DocumentParseError as exc:
                        st.error(f"{side} — {upload.name}: {exc}")
            if len(documents) == 2:
                st.session_state["documents"] = documents
                try:
                    settings = load_settings()
                    results = {}
                    with st.spinner("Extracting organizational units and functions..."):
                        progress = st.empty()
                        for side, document in documents.items():
                            results[side] = extract_document(
                                document, settings,
                                progress=lambda done, total, side=side: progress.info(
                                    f"{side}: batch {done}/{total} completed"
                                ),
                            )
                        progress.empty()
                    st.session_state["extractions"] = results
                    st.success("Extraction complete. Review the original source references below.")
                except (ConfigurationError, ExtractionError) as exc:
                    st.error(str(exc))

    documents = st.session_state.get("documents")
    if documents:
        st.divider()
        left, right = st.columns(2)
        with left:
            if results := st.session_state.get("extractions"):
                show_extraction(documents["BEFORE"], results["BEFORE"])
            show_preview(documents["BEFORE"])
        with right:
            if results := st.session_state.get("extractions"):
                show_extraction(documents["AFTER"], results["AFTER"])
            show_preview(documents["AFTER"])


if __name__ == "__main__":
    main()
