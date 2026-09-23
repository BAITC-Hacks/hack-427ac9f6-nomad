import streamlit as st

from src.models import ComparisonResult, Document, ExtractionResult
from src.ui_common import (
    UNIT_STATUS, FUNCTION_STATUS, RISK_LABEL, badge, evidence_pair, friendly_warning,
)


def show_comparison(result: ComparisonResult, before: Document, after: Document,
                    before_result: ExtractionResult, after_result: ExtractionResult) -> None:
    names = {u.id: u.name for u in before_result.units + after_result.units}
    def name(unit_id):
        return names.get(unit_id, "Не указано")

    st.header("Результаты сравнения")
    st.caption("Изменения организационной структуры, распределения функций и возможные риски.")
    for warning in result.warnings:
        st.warning(friendly_warning(warning))
    columns = st.columns(4)
    for column, status in zip(columns, ("CREATED", "PRESERVED", "TRANSFORMED", "DELETED")):
        column.metric(UNIT_STATUS[status], sum(c.status == status for c in result.unit_changes))
    columns = st.columns(3)
    for column, kind, label in zip(columns,
            ("FUNCTION_LOSS", "DUPLICATION", "CONFLICT_OF_INTEREST"),
            ("Возможные потери", "Дублирования", "Конфликты")):
        column.metric(label, sum(f.type == kind for f in result.findings))

    overview, changes, findings = st.tabs(
        ["ОБЗОР ИЗМЕНЕНИЙ", "ИЗМЕНЕНИЯ ФУНКЦИЙ", "РИСКИ И РЕКОМЕНДАЦИИ"])
    with overview:
        for change in result.unit_changes:
            with st.container(border=True):
                badge(UNIT_STATUS[change.status])
                left, arrow, right = st.columns([5, 1, 5])
                with left:
                    st.caption("Подразделение до")
                    st.write(name(change.before_unit))
                with arrow:
                    st.write("→")
                with right:
                    st.caption("Подразделение после")
                    st.write(name(change.after_unit))
                st.write(change.reason)
                st.caption(f"Уверенность анализа: {change.confidence:.0%}")
                evidence_pair(before, after, change.before_evidence_ids, change.after_evidence_ids)
        if not result.unit_changes:
            st.info("Подтверждённых изменений структуры не получено.")
    with changes:
        for change in result.responsibility_changes:
            with st.container(border=True):
                badge(FUNCTION_STATUS[change.status], change.status == "POTENTIALLY_LOST")
                left, right = st.columns(2)
                with left:
                    st.caption("До реорганизации · " + name(change.before_unit))
                    st.write(change.before_text)
                with right:
                    st.caption("После реорганизации · " + name(change.after_unit))
                    st.write(change.after_text or
                             "Эквивалентная обязанность не найдена в проверенных материалах. Требуется уточнение.")
                st.write(change.reason)
                st.caption(f"Уверенность анализа: {change.confidence:.0%}")
                evidence_pair(before, after, change.before_evidence_ids, change.after_evidence_ids)
        if not result.responsibility_changes:
            st.info("Подтверждённых изменений функций не получено. Это не означает отсутствия изменений.")
    with findings:
        for finding in result.findings:
            with st.container(border=True):
                badge("Гипотеза для проверки", risk=True)
                st.subheader(RISK_LABEL[finding.type])
                st.write(finding.title)
                st.write(finding.description)
                st.write("Подразделения: " + " · ".join(name(u) for u in finding.affected_unit_ids))
                st.caption(f"Уверенность анализа: {finding.confidence:.0%}")
                st.markdown("**Почему система обратила внимание**")
                st.write(finding.reason)
                if finding.recommendation:
                    st.markdown("**Рекомендация**")
                    st.write(finding.recommendation)
                evidence_pair(before, after, finding.before_evidence_ids, finding.after_evidence_ids)
        if not result.findings:
            st.info("Подтверждённых гипотез не получено. Это не гарантирует отсутствия рисков.")
