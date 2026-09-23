import unittest
from pathlib import Path
from unittest.mock import patch
from streamlit.testing.v1 import AppTest
from src.models import ComparisonResult, OrgUnit, UnitChange
from src.config import Settings
from src.services.comparison import (CoreOutput, RiskOutput, validate_core, compare_documents,
                                     update_coverage, COVERAGE_WARNING)
from src.services.conclusion import build_conclusion, DISCLAIMER
from test_h3 import setup_documents, finding, response


def complete_changes(be, ae, b, a):
    changes = []
    for before, after in zip(be.units, ae.units):
        changes.append(UnitChange(before_unit=before.id, after_unit=after.id, status="PRESERVED",
            reason="Сохранено", confidence=.8, before_evidence_ids=[b.chunks[0].chunk_id],
            after_evidence_ids=[a.chunks[0].chunk_id]))
    for after in ae.units[len(be.units):]:
        changes.append(UnitChange(before_unit=None, after_unit=after.id, status="CREATED",
            reason="Создано", confidence=.8, before_evidence_ids=[], after_evidence_ids=[a.chunks[0].chunk_id]))
    return changes


def successful_comparison(b, a, be, ae, settings=None, **kwargs):
    result = ComparisonResult(completed=True, unit_changes=complete_changes(be, ae, b, a))
    update_coverage(result, be.units, ae.units)
    return result


class P0Tests(unittest.TestCase):
    def setUp(self):
        self.b, self.a, self.be, self.ae = setup_documents()
        self.be.units += [OrgUnit('b2', 'Отдел 2', None, ['b1']), OrgUnit('b3', 'Отдел 3', None, ['b1'])]
        self.ae.units += [OrgUnit('a3', 'Отдел 3', None, ['a1']), OrgUnit('a4', 'Отдел 4', None, ['a1']),
                          OrgUnit('a5', 'Отдел 5', None, ['a1'])]

    def validate(self, changes):
        return validate_core(CoreOutput(unit_changes=changes, responsibility_changes=[]),
                             self.be.units, self.ae.units, self.b.chunks, self.a.chunks)

    def summary(self, result):
        return "\n".join(text for _, text in build_conclusion(result))

    @patch('src.services.comparison.OpenAI')
    def test_empty_3_to_5_not_completed_despite_successful_requests(self, api):
        api.return_value.__enter__.return_value.responses.parse.side_effect = [
            response(CoreOutput(unit_changes=[], responsibility_changes=[])),
            response(RiskOutput(findings=[finding()]))]
        r = compare_documents(self.b, self.a, self.be, self.ae, Settings('mock'))
        self.assertFalse(r.completed)
        self.assertEqual((r.before_units_total, r.after_units_total), (3, 5))
        self.assertEqual(len(r.findings), 1)
        self.assertIn(COVERAGE_WARNING, r.warnings)

    def test_partial_results_preserved_and_uncovered_detected(self):
        changes = complete_changes(self.be, self.ae, self.b, self.a)[:1]
        r = self.validate(changes)
        self.assertEqual(r.unit_changes, changes)
        self.assertFalse(r.structure_complete)
        self.assertEqual((r.before_units_covered, r.after_units_covered), (1, 1))
        self.assertEqual(sum(d['code']=='uncovered_unit' for d in r.diagnostics), 6)

    def test_complete_coverage_accepted_including_deleted(self):
        changes = complete_changes(self.be, self.ae, self.b, self.a)
        changes[2:3] = [UnitChange(before_unit='b3', after_unit=None, status='DELETED', reason='Удалено',
            confidence=.8, before_evidence_ids=['b1'], after_evidence_ids=[]),
            UnitChange(before_unit=None, after_unit='a3', status='CREATED', reason='Создано',
            confidence=.8, before_evidence_ids=[], after_evidence_ids=['a1'])]
        changes[0].status = 'TRANSFORMED'
        r = self.validate(changes)
        self.assertTrue(r.structure_complete)
        self.assertEqual((r.before_units_covered, r.after_units_covered), (3, 5))
        self.assertNotIn(COVERAGE_WARNING, r.warnings)

    def test_diagnostics_and_evidence_gate_unchanged(self):
        base = complete_changes(self.be, self.ae, self.b, self.a)[0]
        cases = [(base.model_copy(update={'before_unit':'unknown'}), 'invalid_unit_id'),
                 (base.model_copy(update={'after_evidence_ids':['invented']}), 'invalid_evidence_id'),
                 (base.model_copy(update={'status':'CREATED','before_unit':None}), 'invalid_status_evidence_combination')]
        for item, code in cases:
            r = self.validate([item])
            self.assertFalse(r.unit_changes)
            self.assertIn(code, [d['code'] for d in r.diagnostics])
        mixed = base.model_copy(update={'after_evidence_ids':['a1','invented']})
        self.assertEqual(self.validate([mixed]).unit_changes[0].after_evidence_ids, ['a1'])

    def test_conclusion_counts_only_validated_results_and_recommendations(self):
        r = self.validate(complete_changes(self.be, self.ae, self.b, self.a))
        r.findings = [finding()]
        text = self.summary(r)
        self.assertIn('создано: 2; сохранено: 3', text)
        self.assertIn(r.findings[0].recommendation, text)
        self.assertIn('конфликты интересов — 0', text)
        self.assertIn(DISCLAIMER, text)
        self.assertNotIn('unit_', text)

    def test_empty_summary_cautious_and_no_invented_recommendations(self):
        r = self.validate([])
        text = self.summary(r)
        self.assertIn('В рамках выполненного анализа потенциальные потери не выявлены', text)
        self.assertIn('рекомендации отсутствуют', text)
        self.assertIn('выполнено частично', text)
        self.assertNotIn('изменений не обнаружено', text.lower())
        self.assertNotIn('Создано:', text)
        self.assertFalse(r.findings)

    def test_ui_empty_metrics_are_not_zero_and_summary_follows_results(self):
        app = AppTest.from_file(str(Path(__file__).resolve().parents[1]/'app.py'), default_timeout=30)
        app.session_state['extraction_policy']='h2.3'
        app.session_state['documents']={'BEFORE':self.b,'AFTER':self.a}
        app.session_state['extractions']={'BEFORE':self.be,'AFTER':self.ae}
        app.session_state['comparison']=ComparisonResult(completed=True)
        app.session_state['workflow_message']='Анализ завершён'
        app.run()
        self.assertFalse(app.exception)
        self.assertEqual([m.value for m in app.metric[:4]], ['—']*4)
        self.assertTrue(app.warning)
        self.assertFalse(app.success)
        self.assertTrue(any(s.value=='Итоговое аналитическое заключение' for s in app.subheader))

    def test_incomplete_not_cached_extraction_reused_and_complete_cached(self):
        from src.services.pipeline import run_analysis
        state = {}
        uploads = {'BEFORE':('b.pdf',b'b'),'AFTER':('a.pdf',b'a')}
        complete = successful_comparison(self.b,self.a,self.be,self.ae)
        with patch('src.services.pipeline.parse_document',side_effect=[self.b,self.a]), patch(
                'src.services.extraction.extract_document',side_effect=[self.be,self.ae]) as extract, patch(
                'src.services.comparison.compare_documents',side_effect=[ComparisonResult(completed=True),complete]) as compare:
            run_analysis(uploads,state,Settings('mock'))
            self.assertFalse(state['_comparison_cache'])
            self.assertTrue(state['_analysis_cache'])
            run_analysis(uploads,state,Settings('mock'))
            run_analysis(uploads,state,Settings('mock'))
            self.assertEqual(extract.call_count,2)
            self.assertEqual(compare.call_count,2)
            self.assertTrue(state['comparison'].completed)

    @patch('src.services.comparison.OpenAI')
    def test_complete_api_path_marks_completed(self, api):
        changes = complete_changes(self.be, self.ae, self.b, self.a)
        api.return_value.__enter__.return_value.responses.parse.side_effect = [
            response(CoreOutput(unit_changes=changes, responsibility_changes=[])),
            response(RiskOutput(findings=[]))]
        r = compare_documents(self.b,self.a,self.be,self.ae,Settings('mock'))
        self.assertTrue(r.completed)
        self.assertTrue(r.structure_complete)
        self.assertEqual(len(r.unit_changes), 5)

    def test_all_rejected_structure_incomplete(self):
        changes = complete_changes(self.be,self.ae,self.b,self.a)
        for change in changes:
            change.after_evidence_ids = ['invalid']
        r = self.validate(changes)
        self.assertFalse(r.unit_changes)
        self.assertFalse(r.structure_complete)
        self.assertEqual(r.discarded_items, len(changes))

    def test_summary_function_counts_and_recommendations_are_exact(self):
        from test_h3 import responsibility
        r = self.validate(complete_changes(self.be,self.ae,self.b,self.a))
        r.responsibility_changes = [responsibility(status, 'au')
                                   for status in ('SAME','MODIFIED','MOVED')]
        r.findings = [finding(),finding()]
        sections = dict(build_conclusion(r))
        self.assertIn('без изменений: 1; изменены: 1; перераспределены: 1',sections['Изменения функций'])
        self.assertEqual(sections['Рекомендации'],r.findings[0].recommendation)
        self.assertEqual(r.findings,[finding(),finding()])
