import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from streamlit.testing.v1 import AppTest
from src.config import Settings
from src.models import (
    Document, SourceChunk, OrgUnit, ExtractionResult, ComparisonResult,
    UnitChange, ResponsibilityChange, Finding, UnitEvidence,
)
from src.services.comparison import (
    CoreOutput, RiskOutput, LossOutput, LossDecision, SourceSelection, ComparisonError,
    validate_core, validate_risks, apply_loss_review, candidate_id, compare_documents, select_chunks,
)


def setup_documents():
    before = Document("b", "BEFORE", "before.pdf", "pdf")
    after = Document("a", "AFTER", "after.pdf", "pdf")
    before.chunks = [SourceChunk("b1", "b", before.filename, 1, "3.4", "Page 1",
                                "Департамент анализа (ДА) проверяет качество.")]
    after.chunks = [
        SourceChunk("a1", "a", after.filename, 2, "4.1", "Page 2",
                    "Департамент анализа (ДА) планирует работу."),
        SourceChunk("a2", "a", after.filename, 3, "4.2", "Page 3",
                    "Департамент контроля (ДК) проверяет качество и выполняет операции."),
    ]
    bu = [OrgUnit("bu", "Департамент анализа", None, ["b1"], ["ДА"])]
    au = [OrgUnit("au", "Департамент анализа", None, ["a1"], ["ДА"]),
          OrgUnit("other", "Департамент контроля", None, ["a2"], ["ДК"])]
    return before, after, ExtractionResult("b", bu), ExtractionResult("a", au)


def responsibility(status="POTENTIALLY_LOST", owner=None):
    return ResponsibilityChange(
        before_unit="bu", after_unit=owner, before_text="Проверяет качество",
        after_text="Проверяет качество" if owner else None, status=status,
        reason="Возможное изменение распределения обязанности", confidence=.8,
        before_evidence_ids=["b1"], after_evidence_ids=["a2"] if owner else [],
    )


def finding(kind="DUPLICATION"):
    return Finding(type=kind, title="Возможное пересечение", description="Проверить распределение",
                   affected_unit_ids=["au", "other"], reason="Обязанности могут пересекаться",
                   confidence=.7, before_evidence_ids=[], after_evidence_ids=["a1", "a2"],
                   unit_evidence=[UnitEvidence(unit_id="au", evidence_ids=["a1"]),
                                  UnitEvidence(unit_id="other", evidence_ids=["a2"])],
                   recommendation="Уточнить полномочия")


def response(value):
    return SimpleNamespace(status="completed", output_parsed=value)


class GateTests(unittest.TestCase):
    def setUp(self):
        self.b, self.a, self.be, self.ae = setup_documents()

    def core(self, units=None, duties=None):
        return validate_core(CoreOutput(unit_changes=units or [], responsibility_changes=duties or []),
                             self.be.units, self.ae.units, self.b.chunks, self.a.chunks)

    def test_all_unit_statuses_validate(self):
        for status in ("CREATED", "DELETED", "PRESERVED", "TRANSFORMED"):
            with self.subTest(status=status):
                item = UnitChange(
                    before_unit=None if status == "CREATED" else "bu",
                    after_unit=None if status == "DELETED" else "au", status=status,
                    reason="Изменение структуры", confidence=.8,
                    before_evidence_ids=[] if status == "CREATED" else ["b1"],
                    after_evidence_ids=[] if status == "DELETED" else ["a1"],
                )
                self.assertEqual(len(self.core(units=[item]).unit_changes), 1)

    def test_unknown_unit_and_invalid_side_evidence_rejected(self):
        item = responsibility("MOVED", "unknown")
        self.assertEqual(self.core(duties=[item]).responsibility_changes, [])
        item = responsibility("MOVED", "other")
        item.before_evidence_ids = ["a1"]
        self.assertEqual(self.core(duties=[item]).responsibility_changes, [])

    def test_invalid_reference_discarded_when_valid_evidence_remains(self):
        item = responsibility("MOVED", "other")
        item.after_evidence_ids += ["invented"]
        result = self.core(duties=[item])
        self.assertEqual(result.responsibility_changes[0].after_evidence_ids, ["a2"])

    def test_moved_wins_over_same_loss_candidate_in_any_order(self):
        for duties in ([responsibility(), responsibility("MOVED", "other")],
                       [responsibility("MOVED", "other"), responsibility()]):
            result = self.core(duties=duties)
            self.assertEqual([c.status for c in result.responsibility_changes], ["MOVED"])

    def test_duplicate_and_conflict_validate_with_per_unit_evidence(self):
        for kind in ("DUPLICATION", "CONFLICT_OF_INTEREST"):
            accepted, dropped = validate_risks(RiskOutput(findings=[finding(kind)]),
                                               self.ae.units, self.a.chunks)
            self.assertEqual(len(accepted), 1)
            self.assertEqual(dropped, 0)

    def test_finding_without_evidence_unknown_owner_or_single_unit_duplicate_rejected(self):
        for modification in (
            {"after_evidence_ids": ["fake"]},
            {"affected_unit_ids": ["unknown"]},
            {"affected_unit_ids": ["au"], "unit_evidence": [UnitEvidence(unit_id="au", evidence_ids=["a1"])]},
            {"unit_evidence": [UnitEvidence(unit_id="au", evidence_ids=["a1"])]},
        ):
            item = finding().model_copy(update=modification)
            accepted, _ = validate_risks(RiskOutput(findings=[item]), self.ae.units, self.a.chunks)
            self.assertEqual(accepted, [])

    def test_loss_requires_review_of_all_after_units_and_complete_scope(self):
        candidate = responsibility()
        for checked, truncated, expected in [
            (["au"], False, 0), (["au", "other"], True, 0), (["au", "other"], False, 1),
        ]:
            result = ComparisonResult()
            review = LossOutput(decisions=[LossDecision(
                candidate_id=candidate_id(candidate), status="POTENTIALLY_LOST",
                after_unit=None, after_text=None, reason="Эквивалент не найден; требуется проверка",
                confidence=.5, after_evidence_ids=[], checked_after_unit_ids=checked,
            )])
            apply_loss_review(result, review, [candidate], self.ae.units,
                              SourceSelection(self.a.chunks, truncated))
            self.assertEqual(len(result.findings), expected)

    def test_source_selection_is_bounded(self):
        selection = select_chunks(self.a, self.ae.units, max_bytes=350)
        self.assertLessEqual(sum(len(c.text.encode("utf-8")) for c in selection.chunks), 350)
        self.assertTrue(selection.truncated)

    def test_loss_review_prefers_move_over_contradictory_loss(self):
        candidate = responsibility()
        common = dict(candidate_id=candidate_id(candidate), reason="Проверка", confidence=.7,
                      checked_after_unit_ids=["au", "other"])
        review = LossOutput(decisions=[
            LossDecision(**common, status="POTENTIALLY_LOST", after_unit=None,
                         after_text=None, after_evidence_ids=[]),
            LossDecision(**common, status="MOVED", after_unit="other",
                         after_text="Проверяет качество", after_evidence_ids=["a2"]),
        ])
        result = ComparisonResult()
        apply_loss_review(result, review, [candidate], self.ae.units, SourceSelection(self.a.chunks, False))
        self.assertEqual([c.status for c in result.responsibility_changes], ["MOVED"])
        self.assertEqual(result.findings, [])


class PipelineTests(unittest.TestCase):
    @patch("src.services.comparison.OpenAI")
    def test_loss_recheck_finds_move_with_no_h2_functions(self, factory):
        b, a, be, ae = setup_documents()
        candidate = responsibility()
        review = LossOutput(decisions=[LossDecision(
            candidate_id=candidate_id(candidate), status="MOVED", after_unit="other",
            after_text="Проверяет качество", reason="Обязанность указана у другого подразделения",
            confidence=.9, after_evidence_ids=["a2"], checked_after_unit_ids=["au", "other"],
        )])
        client = factory.return_value.__enter__.return_value
        client.responses.parse.side_effect = [
            response(CoreOutput(unit_changes=[], responsibility_changes=[candidate])),
            response(review), response(RiskOutput(findings=[])),
        ]
        result = compare_documents(b, a, be, ae, Settings("test-only"))
        self.assertEqual(be.functions, [])
        self.assertEqual([x.status for x in result.responsibility_changes], ["MOVED"])
        self.assertEqual(result.findings, [])
        payload = json.loads(client.responses.parse.call_args_list[0].kwargs["input"])
        self.assertIn("проверяет качество", payload["before"]["chunks"][0]["text"])
        self.assertEqual(client.responses.parse.call_count, 3)
        self.assertTrue(result.completed)

    @patch("src.services.comparison.OpenAI")
    def test_optional_failure_keeps_core_results(self, factory):
        b, a, be, ae = setup_documents()
        factory.return_value.__enter__.return_value.responses.parse.side_effect = [
            response(CoreOutput(unit_changes=[], responsibility_changes=[responsibility("MOVED", "other")])),
            ComparisonError("private error"),
        ]
        result = compare_documents(b, a, be, ae, Settings("test-only"))
        self.assertEqual(len(result.responsibility_changes), 1)
        self.assertTrue(result.warnings)
        self.assertFalse(result.completed)
        self.assertNotIn("private error", " ".join(result.warnings))

    @patch("src.services.comparison.OpenAI")
    def test_core_failure_still_allows_optional_risks(self, factory):
        b, a, be, ae = setup_documents()
        factory.return_value.__enter__.return_value.responses.parse.side_effect = [
            ComparisonError("private error"), response(RiskOutput(findings=[finding()])),
        ]
        result = compare_documents(b, a, be, ae, Settings("test-only"))
        self.assertEqual(len(result.findings), 1)
        self.assertTrue(result.warnings)
        self.assertFalse(result.completed)

    @patch("src.services.comparison.OpenAI")
    def test_failed_loss_recheck_never_shows_loss(self, factory):
        b, a, be, ae = setup_documents()
        factory.return_value.__enter__.return_value.responses.parse.side_effect = [
            response(CoreOutput(unit_changes=[], responsibility_changes=[responsibility()])),
            ComparisonError("private error"), response(RiskOutput(findings=[])),
        ]
        result = compare_documents(b, a, be, ae, Settings("test-only"))
        self.assertEqual(result.responsibility_changes, [])
        self.assertEqual(result.findings, [])
        self.assertFalse(result.completed)

    @patch("src.services.comparison.OpenAI")
    def test_mismatched_documents_never_call_api(self, factory):
        b, a, be, ae = setup_documents()
        be.document_id = "wrong"
        result = compare_documents(b, a, be, ae, Settings("test-only"))
        factory.assert_not_called()
        self.assertTrue(result.warnings)
        self.assertFalse(result.completed)

    @patch("src.services.comparison.compare_documents")
    @patch("src.config.dotenv_values", return_value={"OPENAI_API_KEY": "test-only"})
    def test_ui_failure_preserves_extraction_and_displays_tabs(self, _, compare):
        b, a, be, ae = setup_documents()
        compare.return_value = ComparisonResult(warnings=["Сравнение недоступно"])
        app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=30)
        app.session_state["extraction_policy"] = "h2.3"
        app.session_state["documents"] = {"BEFORE": b, "AFTER": a}
        app.session_state["extractions"] = {"BEFORE": be, "AFTER": ae}
        from io import BytesIO
        from test_h1 import word_bytes
        upload = BytesIO(word_bytes(["3.4. Department duties"]))
        upload.name = "test.docx"
        with patch("streamlit.file_uploader", return_value=upload), patch(
                "src.services.extraction.extract_document", side_effect=[be, ae]):
            app.run()
            app.button[0].click().run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual([tab.label for tab in app.tabs],
                         ["ОБЗОР ИЗМЕНЕНИЙ", "ИЗМЕНЕНИЯ ФУНКЦИЙ", "РИСКИ И РЕКОМЕНДАЦИИ"])
        self.assertEqual(app.session_state["extractions"]["AFTER"].units, ae.units)
        self.assertTrue(app.warning)
        app.run()
        self.assertEqual(compare.call_count, 1)


if __name__ == "__main__":
    unittest.main()
