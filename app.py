import streamlit as st

from src.models import Document
from src.parsers.parser_factory import DocumentParseError, parse_document


def clear_results() -> None:
    st.session_state.pop("documents", None)


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


def main() -> None:
    st.set_page_config(page_title="AI Org Structure Analyzer", layout="wide")
    st.title("AI Org Structure Analyzer")
    st.caption("H1: document text extraction and source preview.")
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
                st.success("Text extracted. Source previews are ready.")

    documents = st.session_state.get("documents")
    if documents:
        st.divider()
        left, right = st.columns(2)
        with left:
            show_preview(documents["BEFORE"])
        with right:
            show_preview(documents["AFTER"])


if __name__ == "__main__":
    main()
