"""H2.2 regression tests with synthetic units; no OpenAI requests."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from stage_mocks import staged_response

from src.config import Settings
from src.services.extraction import BatchExtraction, extract_document, make_batches, resolve_refs
from test_h21 import fixture, unit, duty, merge


class OwnershipTests(unittest.TestCase):
    def test_director_not_unit_even_with_nearby_structure_words(self):
        doc = fixture("3.4. Директор направления внутреннего аудита руководит структурным подразделением.")
        ref = doc.chunks[0].chunk_id
        self.assertEqual(merge(doc, unit("Директор направления внутреннего аудита", ref)).units, [])

    def test_title_named_structural_unit_exception(self):
        name = "Директор проектов"
        doc = fixture('3.4. Создано структурное подразделение под названием «Директор проектов».')
        ref = doc.chunks[0].chunk_id
        self.assertEqual(len(merge(doc, unit(name, ref)).units), 1)

    def test_explicit_role_responsibility_maps_to_validated_unit(self):
        doc = fixture("3.4. Департамент аналитики является структурным подразделением.\n"
                      "5.3. Директор департамента аналитики выполняет следующие обязанности:\n"
                      "5.3.1. Планирует работу.\n")
        structure, heading, responsibility = [c.chunk_id for c in doc.chunks]
        item = duty("Департамент аналитики", responsibility, "role_to_unit", heading)
        item["role_name"] = "Директор департамента аналитики"
        result = merge(doc, unit("Департамент аналитики", structure, functions=[item]),
                       unit(item["role_name"], heading))
        self.assertEqual(len(result.units), 1)
        self.assertEqual(len(result.functions), 1)
        self.assertEqual(result.functions[0].source_refs, [heading, responsibility])
        self.assertEqual([c.chunk_id for c in resolve_refs(doc, result.functions[0].source_refs)],
                         [heading, responsibility])

    def test_marker_routes_separate_items_not_entire_director_list(self):
        doc = fixture("3.4. Департамент аналитики (ДА) является подразделением.\n"
                      "3.5. Департамент контроля (ДК) является подразделением.\n"
                      "5.3. Директоры департаментов ДА и ДК:\n"
                      "а. Анализирует данные (ДА);\nб. Проверяет качество (ДК).\n")
        a, b, duties = [c.chunk_id for c in doc.chunks]
        a_fn = duty("ДА", duties, "unit_marker")
        a_fn["text"] = "Анализирует данные"
        b_fn = duty("ДК", duties, "unit_marker")
        b_fn["text"] = "Проверяет качество"
        wrong = dict(a_fn, owner_name="ДК")
        result = merge(doc,
            unit("Департамент аналитики", a,
                 aliases=[dict(name="ДА", source_chunk_ids=[a])], functions=[a_fn]),
            unit("Департамент контроля", b,
                 aliases=[dict(name="ДК", source_chunk_ids=[b])], functions=[b_fn, wrong]))
        self.assertEqual(len(result.units), 2)
        self.assertEqual(len(result.functions), 2)
        for function in result.functions:
            owner = next(u for u in result.units if u.id == function.unit_id)
            self.assertEqual("аналитики" in owner.name, function.text == "Анализирует данные")

    def test_ambiguous_multi_unit_role_not_mapped(self):
        doc = fixture("3.4. ДА и ДК являются подразделениями.\n"
                      "5.3. Директор ДА и ДК выполняет обязанности:\n"
                      "5.3.1. Планирует работу.")
        structure, heading, task = [c.chunk_id for c in doc.chunks]
        item = duty("ДА", task, "role_to_unit", heading)
        item["role_name"] = "Директор ДА"
        self.assertEqual(merge(doc, unit("ДА", structure, functions=[item]),
                               unit("ДК", structure)).functions, [])

    def test_role_without_named_unit_or_capacity_not_mapped(self):
        for header, role in [("Директор выполняет обязанности:", "Директор"),
                             ("Директор ДА присутствовал на встрече.", "Директор ДА")]:
            doc = fixture("3.4. ДА является подразделением.\n5.3. " + header +
                          "\n5.3.1. Планирует работу.")
            structure, heading, task = [c.chunk_id for c in doc.chunks]
            item = duty("ДА", task, "role_to_unit", heading)
            item["role_name"] = role
            self.assertEqual(merge(doc, unit("ДА", structure, functions=[item])).functions, [])

    def test_unknown_relationship_reference_rejected(self):
        doc = fixture("3.4. ДА является подразделением.\n5.3. Директор ДА выполняет обязанности:\n"
                      "Планирует работу.")
        structure, task = [c.chunk_id for c in doc.chunks]
        item = duty("ДА", task, "role_to_unit", "unknown")
        item["role_name"] = "Директор ДА"
        result = merge(doc, unit("ДА", structure, functions=[item]))
        self.assertEqual(result.functions, [])
        self.assertEqual(result.discarded_references, 1)

    def test_fabricated_text_not_accepted_with_real_marker(self):
        doc = fixture("3.4. ДА является подразделением.\n5.3. Проверяет качество (ДА).")
        structure, task = [c.chunk_id for c in doc.chunks]
        item = duty("ДА", task, "unit_marker")
        self.assertEqual(merge(doc, unit("ДА", structure, functions=[item])).functions, [])

    def test_fabricated_direct_function_rejected(self):
        doc = fixture("3.4. ДА является подразделением и проверяет качество.")
        ref = doc.chunks[0].chunk_id
        self.assertEqual(merge(doc, unit("ДА", ref, functions=[duty("ДА", ref)])).functions, [])

    def test_dedicated_structural_section_accepts_duty_not_sibling(self):
        doc = fixture("3.4. Департамент аналитики является подразделением.\n"
                      "5.3. Департамент аналитики\n5.3.1. Планирует работу.\n"
                      "5.4. Планирует работу.")
        structure, heading, task, sibling = [c.chunk_id for c in doc.chunks]
        result = merge(doc, unit("Департамент аналитики", structure, functions=[
            duty("Департамент аналитики", task, "structural_section", heading),
            duty("Департамент аналитики", sibling, "structural_section", heading)]))
        self.assertEqual(len(result.functions), 1)
        self.assertEqual(result.functions[0].source_refs, [heading, task])

    @patch("src.services.extraction.OpenAI")
    def test_role_relationship_cannot_reference_unsent_chunk(self, factory):
        doc = fixture("3.4. ДА является подразделением. Планирует работу.\n"
                      "5.3. Директор ДА выполняет обязанности:")
        structure, heading = [c.chunk_id for c in doc.chunks]
        item = duty("ДА", structure, "role_to_unit", heading)
        item["role_name"] = "Директор ДА"
        parsed = BatchExtraction.model_validate({"units": [unit("ДА", structure, functions=[item])]})
        factory.return_value.__enter__.return_value.responses.parse.side_effect = staged_response(parsed)
        with patch("src.services.extraction.make_batches", return_value=[[make_batches(doc)[0][0]]]):
            result = extract_document(doc, Settings("test-only"))
        self.assertEqual(result.functions, [])
        self.assertEqual(result.discarded_references, 1)


if __name__ == "__main__":
    unittest.main()
