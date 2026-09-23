import streamlit as st

from src.models import ComparisonResult, Document, ExtractionResult
from src.services.extraction import resolve_refs


def show_evidence(document: Document, ids: list[str]) -> None:
    st.write(document.side)
    for chunk in resolve_refs(document, ids):
        st.caption(f"{chunk.filename} | Page: {chunk.page or '—'} | "
                   f"Section: {chunk.section or '—'} | {chunk.locator} | {chunk.chunk_id}")
        st.text(chunk.text)
    if not ids:
        st.caption("Нет ссылок для этой стороны; отсутствие ссылки не доказывает отсутствие функции.")


def show_comparison(result: ComparisonResult, before: Document, after: Document,
                    before_result: ExtractionResult, after_result: ExtractionResult) -> None:
    names = {u.id: u.name for u in before_result.units + after_result.units}
    def name(unit_id):
        return names.get(unit_id, "—")
    for warning in result.warnings:
        st.warning(warning)
    st.caption(f"Source chunks used: BEFORE {result.before_chunks_used}/{len(before.chunks)}, "
               f"AFTER {result.after_chunks_used}/{len(after.chunks)}. "
               f"Rejected items: {result.discarded_items}. "
               "AI interpretations require source review; counts refer only to returned results.")
    overview, changes, findings = st.tabs(["OVERVIEW", "FUNCTION CHANGES", "FINDINGS"])
    with overview:
        columns = st.columns(4)
        for column, status in zip(columns, ("CREATED", "DELETED", "PRESERVED", "TRANSFORMED")):
            column.metric(status.title(), sum(c.status == status for c in result.unit_changes))
        columns = st.columns(3)
        for column, kind, label in zip(columns,
                ("FUNCTION_LOSS", "DUPLICATION", "CONFLICT_OF_INTEREST"),
                ("Potential losses", "Potential duplications", "Potential conflicts")):
            column.metric(label, sum(f.type == kind for f in result.findings))
        for change in result.unit_changes:
            with st.expander(f"{change.status}: {name(change.before_unit)} → {name(change.after_unit)}"):
                st.write(change.reason)
                st.caption(f"Confidence: {change.confidence:.0%}")
                show_evidence(before, change.before_evidence_ids)
                show_evidence(after, change.after_evidence_ids)
        if not result.unit_changes:
            st.info("Нет подтверждённых результатов сравнения структуры.")
    with changes:
        for change in result.responsibility_changes:
            label = change.status.replace("_", " ")
            with st.expander(f"{label} | {name(change.before_unit)} → {name(change.after_unit)} "
                             f"| {change.confidence:.0%}"):
                left, right = st.columns(2)
                with left:
                    st.write("BEFORE")
                    st.write(change.before_text)
                with right:
                    st.write("AFTER")
                    st.write(change.after_text or "Эквивалент не найден в проверенном контексте; требуется проверка.")
                st.write(change.reason)
                show_evidence(before, change.before_evidence_ids)
                show_evidence(after, change.after_evidence_ids)
        if not result.responsibility_changes:
            st.info("Нет подтверждённых результатов сравнения обязанностей; это не означает отсутствие изменений.")
    with findings:
        labels = {"FUNCTION_LOSS": "Потенциальная потеря функции",
                  "DUPLICATION": "Потенциальное дублирование",
                  "CONFLICT_OF_INTEREST": "Потенциальный конфликт интересов"}
        for finding in result.findings:
            with st.expander(f"{labels[finding.type]}: {finding.title}"):
                st.caption("Гипотеза для проверки, не окончательный или юридический вывод.")
                st.write(finding.description)
                st.write("Подразделения: " + ", ".join(name(u) for u in finding.affected_unit_ids))
                st.write(finding.reason)
                st.caption(f"Confidence: {finding.confidence:.0%}")
                if finding.recommendation:
                    st.write("Рекомендация: " + finding.recommendation)
                show_evidence(before, finding.before_evidence_ids)
                show_evidence(after, finding.after_evidence_ids)
        if not result.findings:
            st.info("Подтверждённых гипотез не получено; отсутствие гипотез не гарантирует отсутствие рисков.")

