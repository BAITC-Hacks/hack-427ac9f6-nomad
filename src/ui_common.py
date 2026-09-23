"""Presentation-only labels, styling and source evidence."""
from html import escape

import streamlit as st

from src.models import Document
from src.services.extraction import resolve_refs

SIDE = {"BEFORE": "До реорганизации", "AFTER": "После реорганизации"}
UNIT_STATUS = {"PRESERVED": "Сохранено", "CREATED": "Создано",
               "DELETED": "Удалено", "TRANSFORMED": "Преобразовано"}
FUNCTION_STATUS = {"SAME": "Без изменений", "MODIFIED": "Изменена",
                   "MOVED": "Перераспределена", "POTENTIALLY_LOST": "Возможная потеря"}
RISK_LABEL = {"FUNCTION_LOSS": "Возможная потеря функции",
             "DUPLICATION": "Возможное дублирование функций",
             "CONFLICT_OF_INTEREST": "Потенциальный конфликт интересов"}

CSS = """
<style>
.stApp { background: #f5f8fc; color: #172b46; }
.block-container { max-width: 1280px; padding-top: 2.6rem; padding-bottom: 3rem; }
h1 { letter-spacing: -.045em; color: #12345b; font-size: 2.7rem !important; }
h2,h3 { color: #17365b; letter-spacing: -.02em; }
[data-testid="stHeader"] { background: #f5f8fc; }
[data-testid="stToolbar"], #MainMenu, footer { display: none; }
[data-testid="stVerticalBlockBorderWrapper"] > div { border-radius: 16px; }
[data-testid="stMetric"] { background: #fff; border: 1px solid #e2eaf3;
 border-radius: 14px; padding: 16px 18px; box-shadow: 0 3px 12px #16385805; }
[data-testid="stMetricValue"] { color: #175f9c; font-weight: 650; }
[data-testid="stMetricLabel"] { color: #52677e; }
[data-testid="stExpander"] { background: #fff; border-color: #dfe8f2; border-radius: 12px; }
[data-testid="stExpander"] details summary { padding: 12px 16px; }
[data-testid="stButton"] button { border-radius: 10px; padding: .6rem 1.2rem; font-weight: 600; }
[data-testid="stFileUploaderDropzone"] { background: #f8fbff; border: 1px dashed #bdd2e6; border-radius: 12px; }
[data-testid="stFileUploaderDropzoneInstructions"] > div,
[data-testid="stFileUploaderDropzoneInstructions"] span,
[data-testid="stFileUploaderDropzoneInstructions"] small { display: none; }
[data-testid="stFileUploaderDropzoneInstructions"]::after {
 content: "Перетащите документ сюда"; font-size: .9rem; color: #546b82; }
[data-testid="stFileUploaderDropzone"] button { font-size: 0 !important; }
[data-testid="stFileUploaderDropzone"] button * { display: none; }
[data-testid="stFileUploaderDropzone"] button::after {
 content: "Выбрать файл"; font-size: .875rem; }
[data-testid="stFileUploaderFileName"] { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
[data-testid="stFileUploaderFileData"] small { display: none; }
[data-testid="stFileUploader"] [data-testid="stAlertContentError"] > * { display: none; }
[data-testid="stFileUploader"] [data-testid="stAlertContentError"]::after {
 content: "Не удалось загрузить файл. Проверьте формат и размер: до 20 МБ."; }
[data-testid="stText"] { white-space: pre-wrap; overflow-wrap: anywhere; font-family: inherit;
 color: #334a62; font-size: .92rem; line-height: 1.65; }
[data-baseweb="tab-list"] { gap: 20px; border-bottom: 1px solid #dfe8f2; }
[data-baseweb="tab"] { font-size: .82rem; font-weight: 600; }
.oa-subtitle { font-size: 1.16rem; color: #345776; margin: -.25rem 0 .7rem; }
.oa-description { color: #5e7389; max-width: 810px; line-height: 1.65; }
.oa-steps { display: flex; gap: 28px; margin: 1.25rem 0 1.6rem; flex-wrap: wrap; }
.oa-step { color: #5d7087; display: flex; gap: 9px; align-items: center; font-size: .88rem; }
.oa-step b { display: inline-grid; place-items: center; width: 27px; height: 27px;
 border-radius: 50%; background: #e4edf7; color: #426080; font-size: .8rem; }
.oa-step.active { color: #145f9c; font-weight: 600; }
.oa-step.active b { background: #176fae; color: white; }
.oa-badge { display: inline-block; padding: 4px 10px; border-radius: 20px;
 background: #eaf3fb; color: #225e8f; font-size: .8rem; font-weight: 600; }
.oa-risk { background: #fff3de; color: #7a591e; }
@media(max-width: 640px) { .block-container { padding-top: 1.4rem; }
 h1 { font-size: 2rem !important; } .oa-steps { gap: 12px; } }
</style>
"""


def display_filename(name: str, limit: int = 64) -> str:
    if len(name) <= limit:
        return name
    tail = name[-min(22, limit // 3):]
    return name[:limit - len(tail) - 1].rstrip() + "…" + tail


def show_filename(name: str, prefix: str = "") -> None:
    st.markdown(f'<span title="{escape(name, quote=True)}">'
                f'{escape(prefix + display_filename(name))}</span>', unsafe_allow_html=True)


def style_and_header(stage: int) -> None:
    st.markdown(CSS, unsafe_allow_html=True)
    st.title("ОргАналитик AI")
    st.markdown('<p class="oa-subtitle">Интеллектуальный анализ организационной структуры и функций</p>'
                '<p class="oa-description">Сравните документы до и после реорганизации, '
                'чтобы выявить изменения структуры, перераспределение функций '
                'и потенциальные организационные риски.</p>', unsafe_allow_html=True)
    labels = ["Загрузите документы", "Запустите анализ", "Изучите изменения и риски"]
    steps = "".join(f'<span class="oa-step {"active" if i == stage else ""}">'
                    f'<b>{i}</b>{label}</span>' for i, label in enumerate(labels, 1))
    st.markdown(f'<div class="oa-steps">{steps}</div>', unsafe_allow_html=True)


def badge(label: str, risk: bool = False) -> None:
    st.markdown(f'<span class="oa-badge{" oa-risk" if risk else ""}">{escape(label)}</span>',
                unsafe_allow_html=True)


def friendly_error(error: Exception) -> str:
    text = str(error)
    if "OPENAI_API_KEY" in text:
        return "Доступ к сервису анализа не настроен. Обратитесь к администратору и повторите попытку."
    if "Unsupported file" in text:
        return "Формат не поддерживается. Загрузите документ PDF, DOCX или XLSX."
    if "file is empty" in text:
        return "Файл пуст. Выберите документ с содержимым."
    if "too large" in text:
        return "Документ слишком большой. Допустимый размер — до 20 МБ."
    if "No readable text" in text:
        return ("Не удалось найти доступный текст. Для PDF выберите документ с текстовым слоем, "
                "для DOCX — с текстом в абзацах, для XLSX — с заполненными строками.")
    if "Could not parse" in text:
        return "Не удалось прочитать документ. Проверьте, что файл открывается и не защищён паролем."
    return "Не удалось завершить анализ. Проверьте доступность сервиса и повторите попытку."


def friendly_warning(message: str) -> str:
    # Localize service wording at the UI boundary, without changing analysis behavior.
    return (message.replace("AFTER", "документа после реорганизации")
            .replace("BEFORE", "документа до реорганизации")
            .replace("доступ к модели, квоту и сеть", "доступность сервиса")
            .replace("Контекст сравнения слишком большой", "Объём материалов для сравнения слишком большой"))


def source_location(document: Document, locator: str) -> str:
    if document.format == "docx":
        return locator.replace("Body paragraph", "Абзац")
    if document.format == "xlsx":
        return locator.replace("Sheet ", "Лист ").replace(", row ", ", строка ")
    return ""


def show_evidence(document: Document, ids: list[str], show_side: bool = True) -> None:
    if show_side:
        st.markdown(f"**{SIDE[document.side]}**")
    chunks = resolve_refs(document, ids)
    if not chunks:
        st.caption("Подтверждающий фрагмент для этой стороны не найден.")
    for chunk in chunks:
        show_filename(chunk.filename, "Документ: ")
        st.caption(f"Страница: {chunk.page or '—'} · Пункт: {chunk.section or '—'}")
        location = source_location(document, chunk.locator)
        if location:
            st.caption(location)
        st.markdown("**Исходный фрагмент**")
        st.text(chunk.text)


def evidence_pair(before: Document, after: Document, before_ids: list[str], after_ids: list[str]) -> None:
    with st.expander("Показать подтверждение", expanded=False):
        left, right = st.columns(2)
        with left:
            show_evidence(before, before_ids)
        with right:
            show_evidence(after, after_ids)

