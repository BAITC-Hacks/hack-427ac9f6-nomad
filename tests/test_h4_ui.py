"""Minimal presentation regressions; no API requests."""
import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest
from src.models import ComparisonResult, UnitChange
from src.ui_common import friendly_error
from test_h3 import setup_documents, responsibility, finding


class RussianInterfaceTests(unittest.TestCase):
    def test_comparison_labels_and_collapsed_original_evidence(self):
        before, after, be, ae = setup_documents()
        result = ComparisonResult(
            unit_changes=[UnitChange(before_unit="bu", after_unit="au", status="PRESERVED",
                reason="Подразделение сохранено", confidence=.85,
                before_evidence_ids=["b1"], after_evidence_ids=["a1"])],
            responsibility_changes=[responsibility("MOVED", "other")],
            findings=[finding()],
        )
        app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=30)
        app.session_state["extraction_policy"] = "h2.3"
        app.session_state["documents"] = {"BEFORE": before, "AFTER": after}
        app.session_state["extractions"] = {"BEFORE": be, "AFTER": ae}
        app.session_state["comparison"] = result
        app.run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(app.header[0].value, "Результаты сравнения")
        self.assertEqual([m.label for m in app.metric],
                         ["Создано", "Сохранено", "Преобразовано", "Удалено",
                          "Возможные потери", "Дублирования", "Конфликты"])
        evidence = [e for e in app.expander if e.label == "Показать подтверждение"]
        self.assertTrue(evidence)
        self.assertTrue(all(not e.proto.expanded for e in evidence))
        self.assertTrue(any(t.value == before.chunks[0].text for t in app.text))
        visible_strings = "\n".join(str(x.value) for kind in
            (app.markdown, app.caption, app.title, app.header, app.subheader, app.info, app.warning)
            for x in kind)
        for forbidden in ("H2.3", "H3", "SourceChunk", "Discarded references",
                          "BEFORE", "AFTER", "PRESERVED", "MOVED", "DUPLICATION"):
            self.assertNotIn(forbidden, visible_strings)
        self.assertEqual(len(app.code), 0)

    def test_errors_are_localized_without_development_details(self):
        for message in ("The file is too large. The H1 limit is 20 MB per document.",
                        "Unsupported file", "No readable text", "OPENAI_API_KEY missing",
                        "Could not parse", "OpenAI request failure"):
            translated = friendly_error(ValueError(message))
            self.assertRegex(translated, "[А-Яа-я]")
            for forbidden in ("H1", "OpenAI", "OPENAI_API_KEY"):
                self.assertNotIn(forbidden, translated)


if __name__ == "__main__":
    unittest.main()

