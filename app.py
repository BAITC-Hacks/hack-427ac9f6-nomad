import streamlit as st

from src.config import ConfigurationError, load_settings
from src.models import Document, ExtractionResult
from src.parsers.parser_factory import DocumentParseError, parse_document
from src.services.extraction import ExtractionError
from src.services.pipeline import run_analysis
from src.comparison_ui import show_comparison
from src.ui_common import SIDE, style_and_header, show_evidence, friendly_error, show_filename


PARTIAL_COMPARISON_MESSAGE = (
    "Сравнение выполнено частично. Часть изменений организационной структуры "
    "не удалось уверенно сопоставить. Ниже представлены подтверждённые результаты. "
    "Рекомендуется проверить неподтверждённые изменения по исходным документам."
)
LEGACY_PARTIAL_ERROR = (
    "Не удалось полностью завершить сравнение. Анализ документов сохранён. "
    "Нажмите «Провести анализ», чтобы повторить сравнение."
)


def clear_results() -> None:
    for key in ("documents", "extractions", "comparison", "run_error", "workflow_message"):
        st.session_state.pop(key, None)


def show_extraction(document: Document, result: ExtractionResult) -> None:
    if not result.units:
        st.info("Структурные подразделения с подтверждением в документе не найдены.")
    for unit in result.units:
        functions = [item for item in result.functions if item.unit_id == unit.id]
        with st.expander(f"{unit.name} · Функций: {len(functions)}", expanded=False):
            if unit.parent:
                st.write(f"Вышестоящее подразделение: {unit.parent}")
            if unit.aliases:
                st.caption("Другие наименования: " + ", ".join(unit.aliases))
            with st.expander("Показать подтверждение", expanded=False):
                show_evidence(document, unit.source_refs, show_side=False)
            with st.expander(f"Функции подразделения · {len(functions)}", expanded=False):
                if not functions:
                    st.caption("Отдельные функции не выделены. Сравнение также учитывает исходный текст документа.")
                for function in functions:
                    st.write(function.text)
                    with st.expander("Показать подтверждение", expanded=False):
                        show_evidence(document, function.source_refs, show_side=False)
                    st.divider()


def show_structures(documents, results) -> None:
    before, after = st.columns(2)
    for side, column, label in (("BEFORE", before, "Структура до"),
                                ("AFTER", after, "Структура после")):
        with column:
            st.subheader(label)
            show_extraction(documents[side], results[side])


def request_run() -> None:
    if not st.session_state.get("analysis_running"):
        st.session_state["analysis_running"] = True
        st.session_state["run_requested"] = True


def analyze(uploads) -> None:
    st.session_state.pop("run_error", None)
    st.session_state.pop("workflow_message", None)
    stages = ["Читаем документы", "Определяем структуру и функции",
              "Сравниваем изменения", "Формируем риски и рекомендации"]
    try:
        if any(upload is None for upload in uploads.values()):
            st.session_state["run_error"] = "Загрузите оба документа: до и после реорганизации."
            return
        inputs = {side: (upload.name, upload.getvalue()) for side, upload in uploads.items()}
        with st.status("1. Читаем документы", expanded=True) as status:
            def progress(message):
                status.update(label=f"{stages.index(message) + 1}. {message}")
            try:
                settings = load_settings()
            except ConfigurationError:
                # Retain readable sources even when service access is not configured.
                st.session_state["documents"] = {
                    side: parse_document(data, name, side) for side, (name, data) in inputs.items()
                }
                raise
            result = run_analysis(inputs, st.session_state, settings, progress)
            if result.completed:
                status.update(label="Анализ завершён", state="complete", expanded=False)
                st.session_state["workflow_message"] = "Анализ завершён. Сравнение завершено — результаты готовы."
            else:
                status.update(label="Сравнение выполнено частично", state="complete", expanded=True)
                st.warning(PARTIAL_COMPARISON_MESSAGE)
    except (ConfigurationError, DocumentParseError, ExtractionError) as exc:
        st.session_state["run_error"] = friendly_error(exc)
    except Exception:
        st.session_state["run_error"] = (
            "Не удалось завершить обработку. Уже выполненный анализ документов сохранён. "
            "Повторите попытку кнопкой «Провести анализ».")
    finally:
        st.session_state["analysis_running"] = False


def main() -> None:
    st.set_page_config(page_title="ОргАналитик AI", layout="wide", initial_sidebar_state="collapsed")
    if st.session_state.get("extraction_policy") != "h2.3":
        st.session_state.pop("extractions", None)
        st.session_state.pop("comparison", None)
        st.session_state["extraction_policy"] = "h2.3"
    # Re-evaluate displayed session results too, including sessions opened before this fix.
    if st.session_state.get("comparison") and st.session_state.get("extractions"):
        from src.services.comparison import update_coverage
        saved = st.session_state["comparison"]
        extracted = st.session_state["extractions"]
        update_coverage(saved, extracted["BEFORE"].units, extracted["AFTER"].units)
        if not saved.completed:
            st.session_state.pop("workflow_message", None)
    stage = 3 if st.session_state.get("comparison") else (2 if st.session_state.get("extractions") else 1)
    style_and_header(stage)

    with st.expander("Документы для анализа", expanded=not bool(st.session_state.get("comparison"))):
        columns = st.columns(2, gap="large")
        uploads = {}
        for side, column in zip(("BEFORE", "AFTER"), columns):
            with column:
                with st.container(border=True):
                    st.markdown(f"**{SIDE[side].upper()}**")
                    uploads[side] = st.file_uploader(
                        "Загрузить документ", type=["pdf", "docx", "xlsx"],
                        key=f"upload_{side.lower()}", on_change=clear_results,
                        help="PDF, DOCX или XLSX · до 20 МБ",
                        disabled=bool(st.session_state.get("analysis_running")),
                    )
                    st.caption("PDF, DOCX или XLSX · до 20 МБ")
                    upload = uploads[side]
                    if upload is not None:
                        size = len(upload.getvalue()) / (1024 * 1024)
                        show_filename(upload.name, "✓ ")
                        st.caption(f"✓ Размер: {size:.2f} МБ".replace(".", ","))
        st.button("Провести анализ", type="primary", on_click=request_run,
                  disabled=bool(st.session_state.get("analysis_running")))
        if st.session_state.pop("run_requested", False):
            analyze(uploads)
            st.rerun()

    # Replace the old partial-result error in already-open sessions as well.
    comparison = st.session_state.get("comparison")
    if comparison and not comparison.completed:
        if st.session_state.get("run_error") == LEGACY_PARTIAL_ERROR:
            st.session_state.pop("run_error", None)
        if not st.session_state.get("run_error"):
            st.warning(PARTIAL_COMPARISON_MESSAGE)
    if st.session_state.get("run_error"):
        st.error(st.session_state["run_error"])
    if st.session_state.get("workflow_message"):
        st.success(st.session_state["workflow_message"])

    documents = st.session_state.get("documents")
    results = st.session_state.get("extractions")
    if documents and results:
        comparison = st.session_state.get("comparison")
        if comparison:
            show_comparison(comparison, documents["BEFORE"], documents["AFTER"],
                            results["BEFORE"], results["AFTER"])
            with st.expander("Структура и функции подразделений", expanded=False):
                show_structures(documents, results)
        else:
            st.subheader("Документы готовы к сравнению")
            columns = st.columns(4)
            values = (
                ("Подразделений до", len(results["BEFORE"].units)),
                ("Подразделений после", len(results["AFTER"].units)),
                ("Функций проанализировано", sum(len(r.functions) for r in results.values())),
                ("Выявлено изменений", "—"),
            )
            for column, (label, value) in zip(columns, values):
                column.metric(label, value)
            st.caption("Нажмите «Провести анализ», чтобы завершить сравнение. Сохранённое извлечение будет использовано повторно.")
            show_structures(documents, results)
    elif documents:
        st.info("Документы прочитаны. Завершите анализ, чтобы увидеть структуру и функции.")


if __name__ == "__main__":
    main()
