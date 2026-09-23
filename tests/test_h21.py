"""H2.1 quality regressions: synthetic evidence, never paid API calls."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from stage_mocks import staged_response

from src.config import Settings
from src.models import Document
from src.services.chunking import split_pdf_page
from src.services.extraction import BatchExtraction, extract_document, make_batches, merge_extractions


def fixture(text):
    doc = Document("quality", "BEFORE", "synthetic.pdf", "pdf")
    doc.chunks = split_pdf_page(doc, 1, text)
    return doc


def unit(name, ref, aliases=None, functions=None, kind="structural_unit"):
    return dict(name=name, parent=None, source_chunk_ids=[ref],
                entity_kind=kind, structural_basis="explicit_structure",
                structural_source_chunk_ids=[ref], aliases=aliases or [],
                functions=functions or [])


def duty(owner, ref, ownership="explicit", owner_ref=None):
    return dict(text="Планирует работу.", source_chunk_ids=[ref],
                owner_name=owner, ownership=ownership,
                role_name=None,
                ownership_source_chunk_ids=[owner_ref or ref])


def merge(doc, *units):
    return merge_extractions(doc, [BatchExtraction.model_validate({"units": list(units)})])


class QualityTests(unittest.TestCase):
    def test_roles_rejected_even_when_model_misclassifies_them(self):
        names = ["Главный аудитор", "Куратор проверки", "Руководитель объекта аудита",
                 "Главный риск-менеджер", "Рабочая группа"]
        doc = fixture("3.4. " + "; ".join(names))
        ref = doc.chunks[0].chunk_id
        result = merge(doc, *(unit(name, ref) for name in names))
        self.assertEqual(result.units, [])
        self.assertIn("Главный аудитор", doc.chunks[0].text)

    def test_temporary_group_and_generic_concept_excluded(self):
        doc = fixture("3.4. Команда проверки. Внутренний аудит организации.")
        ref = doc.chunks[0].chunk_id
        self.assertEqual(merge(doc, unit("Команда проверки", ref, kind="temporary_group"),
                               unit("Внутренний аудит организации", ref, kind="other")).units, [])

    def test_explicit_structural_group_kept(self):
        doc = fixture("3.4. Группа аналитики является структурным подразделением.")
        ref = doc.chunks[0].chunk_id
        self.assertEqual(len(merge(doc, unit("Группа аналитики", ref)).units), 1)

    def test_alias_merge_across_batches_and_order(self):
        full = "Департамент аналитики (ДА)"
        doc = fixture("3.4. В состав входят: Департамент аналитики (ДА).\n"
                      "3.5. ДА планирует работу.\n")
        first, second = [c.chunk_id for c in doc.chunks]
        a = unit(full, first, aliases=[dict(name="ДА", source_chunk_ids=[first])])
        b = unit("ДА", second, functions=[duty("ДА", second)])
        batches = [BatchExtraction.model_validate({"units": [x]}) for x in [b, a]]
        result = merge_extractions(doc, batches)
        self.assertEqual(len(result.units), 1)
        self.assertEqual(result.units[0].name, full)
        self.assertEqual(result.units[0].aliases, ["ДА"])
        self.assertEqual(len(result.functions), 1)
        self.assertEqual(result.functions[0].unit_id, result.units[0].id)
        self.assertEqual(result, merge_extractions(doc, reversed(batches)))

    def test_cooccurrence_is_not_alias_evidence(self):
        doc = fixture("3.4. Департамент аналитики сотрудничает с ДА.")
        ref = doc.chunks[0].chunk_id
        result = merge(doc, unit("Департамент аналитики", ref,
                                aliases=[dict(name="ДА", source_chunk_ids=[ref])]),
                       unit("ДА", ref))
        self.assertEqual(len(result.units), 2)
        self.assertEqual(result.units[0].aliases, [])

    def test_defined_as_alias_supported(self):
        doc = fixture("3.4. Департамент аналитики, далее — ДА, является подразделением.")
        ref = doc.chunks[0].chunk_id
        result = merge(doc, unit("Департамент аналитики", ref,
                                aliases=[dict(name="ДА", source_chunk_ids=[ref])]),
                       unit("ДА", ref))
        self.assertEqual(len(result.units), 1)

    def test_parent_does_not_accumulate_child_or_inferred_duties(self):
        doc = fixture("3.4. Блок контроля включает Департамент аналитики.\n"
                      "3.5. Департамент аналитики планирует работу.\n")
        first, second = [c.chunk_id for c in doc.chunks]
        result = merge(doc,
            unit("Блок контроля", first, functions=[
                duty("Департамент аналитики", second),
                duty("Блок контроля", second, ownership="inferred"),
                duty("Блок контроля", second, owner_ref=first),
            ]),
            unit("Департамент аналитики", first,
                 functions=[duty("Департамент аналитики", second)]))
        parent = next(u for u in result.units if u.name == "Блок контроля")
        self.assertEqual(len(result.functions), 1)
        self.assertNotEqual(result.functions[0].unit_id, parent.id)

    def test_function_heading_bounds_ownership(self):
        doc = fixture("5.3. Функции Департамент аналитики\n"
                      "5.3.1. Планирует работу.\n"
                      "5.4. Иная деятельность.\n")
        header, inside, outside = [c.chunk_id for c in doc.chunks]
        result = merge(doc, unit("Департамент аналитики", header, functions=[
            duty("Департамент аналитики", inside, "unit_heading", header),
            duty("Департамент аналитики", outside, "unit_heading", header),
        ]))
        self.assertEqual(len(result.functions), 1)
        self.assertEqual(result.functions[0].source_refs, [header, inside])

    def test_general_parent_heading_not_function_heading(self):
        doc = fixture("5.3. Блок контроля\n5.3.1. Планирует работу.")
        header, inside = [c.chunk_id for c in doc.chunks]
        result = merge(doc, unit("Блок контроля", header, functions=[
            duty("Блок контроля", inside, "unit_heading", header)]))
        self.assertEqual(result.functions, [])

    def test_unknown_structural_alias_and_ownership_refs(self):
        doc = fixture("3.4. Департамент аналитики (ДА) планирует работу.")
        ref = doc.chunks[0].chunk_id
        candidate = unit("Департамент аналитики", ref,
                         aliases=[dict(name="ДА", source_chunk_ids=["unknown"])],
                         functions=[duty("Департамент аналитики", ref, owner_ref="unknown")])
        result = merge(doc, candidate)
        self.assertEqual(result.units[0].aliases, [])
        self.assertEqual(result.functions, [])
        candidate["structural_source_chunk_ids"] = ["unknown"]
        self.assertEqual(merge(doc, candidate).units, [])

    @patch("src.services.extraction.OpenAI")
    def test_new_evidence_fields_restricted_to_current_batch(self, factory):
        doc = fixture("3.4. Департамент аналитики.\n3.5. Департамент аналитики (ДА).")
        first, second = [c.chunk_id for c in doc.chunks]
        candidate = unit("Департамент аналитики", first,
                         aliases=[dict(name="ДА", source_chunk_ids=[second])],
                         functions=[duty("Департамент аналитики", first, owner_ref=second)])
        factory.return_value.__enter__.return_value.responses.parse.side_effect = (
            staged_response(BatchExtraction.model_validate({"units": [candidate]})))
        with patch("src.services.extraction.make_batches", return_value=[[make_batches(doc)[0][0]]]):
            result = extract_document(doc, Settings("test-only"))
        self.assertEqual(result.units[0].aliases, [])
        self.assertEqual(result.functions, [])
        self.assertEqual(result.discarded_references, 2)


if __name__ == "__main__":
    unittest.main()
