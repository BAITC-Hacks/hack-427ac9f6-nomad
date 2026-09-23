"""H2.3 two-stage pipeline tests. All model responses are mocked."""
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from pydantic import ValidationError
from src.config import Settings
from src.services.extraction import (
    StructureExtraction, FunctionExtraction, validate_structure, validate_functions,
    extract_document, make_batches, ExtractionError, resolve_refs,
)
from test_h21 import fixture, unit


def structural(candidate):
    return {key: value for key, value in candidate.items() if key != "functions"}


def prepared():
    doc = fixture("3.4. Департамент аналитики (ДА) является структурным подразделением.\n"
                  "3.5. Департамент контроля (ДК) является структурным подразделением.\n"
                  "5.3. Директоры ДА и ДК отвечают за следующие направления:\n"
                  "а. анализ данных, построение прогнозов (ДА);\n"
                  "б. проверка качества процессов (ДК).\n")
    a, b, _ = [c.chunk_id for c in doc.chunks]
    output = StructureExtraction(units=[
        structural(unit("Департамент аналитики", a,
                        aliases=[dict(name="ДА", source_chunk_ids=[a])])),
        structural(unit("Департамент контроля", b,
                        aliases=[dict(name="ДК", source_chunk_ids=[b])])),
    ])
    return doc, output, validate_structure(doc, [output]).units


def fn(owner, ref, text="Анализ данных", ownership="unit_marker", supported=True):
    return dict(text=text, owner_unit_id=owner, source_chunk_ids=[ref],
                ownership=ownership, supported_by_source=supported)


def response(parsed):
    return SimpleNamespace(status="completed", output_parsed=parsed)


class TwoStageTests(unittest.TestCase):
    @patch("src.services.extraction.OpenAI")
    def test_all_structure_batches_finish_before_functions_and_registry_is_complete(self, factory):
        doc, structure, units = prepared()
        client = factory.return_value.__enter__.return_value
        parts = make_batches(doc)[0]
        batches = [parts[:2], parts[2:]]
        registry_seen = []

        def respond(**kwargs):
            if kwargs["text_format"] is StructureExtraction:
                chunks = json.loads(kwargs["input"])
                return response(structure.model_copy(deep=True) if len(chunks) == 2
                                else StructureExtraction(units=[]))
            registry_seen.append(json.loads(kwargs["input"])["validated_units"])
            return response(FunctionExtraction(functions=[]))

        client.responses.parse.side_effect = respond
        with patch("src.services.extraction.make_batches", return_value=batches):
            result = extract_document(doc, Settings("test-only"))
        schemas = [call.kwargs["text_format"] for call in client.responses.parse.call_args_list]
        self.assertEqual(schemas, [StructureExtraction, StructureExtraction,
                                   FunctionExtraction, FunctionExtraction])
        self.assertEqual(len(result.units), 2)
        for registry in registry_seen:
            self.assertEqual({u["id"] for u in registry}, {u.id for u in units})
            self.assertTrue(all("canonical_name" in u and u["aliases"] for u in registry))
        self.assertNotIn("functions", StructureExtraction.model_json_schema()["$defs"]["StructuralUnit"]["properties"])

    def test_stage_b_cannot_create_roles_or_units(self):
        with self.assertRaises(ValidationError):
            FunctionExtraction.model_validate({"functions": [], "units": [{"name": "Директор"}]})
        doc, _, units = prepared()
        out = FunctionExtraction(functions=[fn("Директор ДА", doc.chunks[-1].chunk_id)])
        result = validate_functions(doc, units, [out])
        self.assertEqual(result.units, units)
        self.assertEqual(result.functions, [])

    @patch("src.services.extraction.OpenAI")
    def test_marked_directors_list_assigns_to_registry_ids(self, factory):
        doc, structure, units = prepared()
        refs = doc.chunks[-1].chunk_id
        a = next(u for u in units if "ДА" in u.aliases)
        b = next(u for u in units if "ДК" in u.aliases)
        factory.return_value.__enter__.return_value.responses.parse.side_effect = [
            response(structure), response(FunctionExtraction(functions=[
                fn(a.id, refs, "Анализ данных и построение прогнозов"),
                fn(b.id, refs, "Проверка качества процессов"),
            ])),
        ]
        result = extract_document(doc, Settings("test-only"))
        self.assertEqual({f.unit_id for f in result.functions}, {a.id, b.id})
        self.assertEqual(len(result.functions), 2)
        self.assertIs(resolve_refs(doc, result.functions[0].source_refs)[0], doc.chunks[-1])

    def test_role_responsibility_can_map_to_validated_unit(self):
        doc, _, units = prepared()
        target = units[0]
        item = fn(target.id, doc.chunks[-1].chunk_id,
                  "Построение прогнозов", "role_to_unit")
        result = validate_functions(doc, units, [FunctionExtraction(functions=[item])])
        self.assertEqual(result.functions[0].unit_id, target.id)
        self.assertEqual(len(result.units), 2)

    def test_unknown_owner_rejected(self):
        doc, _, units = prepared()
        result = validate_functions(doc, units, [FunctionExtraction(
            functions=[fn("invented-id", doc.chunks[-1].chunk_id)])])
        self.assertEqual(result.functions, [])

    def test_unknown_evidence_rejected_even_with_one_valid_reference(self):
        doc, _, units = prepared()
        item = fn(units[0].id, "invented")
        item["source_chunk_ids"].append(doc.chunks[-1].chunk_id)
        result = validate_functions(doc, units, [FunctionExtraction(functions=[item])])
        self.assertEqual(result.functions, [])
        self.assertEqual(result.discarded_references, 1)

    def test_specific_responsibility_not_exact_substring_is_accepted(self):
        doc, _, units = prepared()
        text = "Анализ данных и построение прогнозов"
        self.assertNotIn(text.lower(), doc.chunks[-1].text.lower())
        result = validate_functions(doc, units, [FunctionExtraction(
            functions=[fn(units[0].id, doc.chunks[-1].chunk_id, text)])])
        self.assertEqual(result.functions[0].text, text)
        self.assertEqual(resolve_refs(doc, result.functions[0].source_refs)[0].text,
                         doc.chunks[-1].text)

    def test_ambiguous_or_unsupported_function_rejected(self):
        doc, _, units = prepared()
        ref = doc.chunks[-1].chunk_id
        result = validate_functions(doc, units, [FunctionExtraction(functions=[
            fn(units[0].id, ref, ownership="ambiguous"),
            fn(units[0].id, ref, supported=False),
        ])])
        self.assertEqual(result.functions, [])

    def test_conservative_dedup_preserves_distinct_units_and_responsibilities(self):
        doc, _, units = prepared()
        ref = doc.chunks[-1].chunk_id
        result = validate_functions(doc, units, [FunctionExtraction(functions=[
            fn(units[0].id, ref, "а. Анализ данных."),
            fn(units[0].id, ref, "Анализ   данных"),
            fn(units[0].id, ref, "Анализ финансовых данных"),
            fn(units[1].id, ref, "Анализ данных"),
        ])])
        self.assertEqual(len(result.functions), 3)

    @patch("src.services.extraction.OpenAI")
    def test_empty_structure_skips_stage_b(self, factory):
        doc, _, _ = prepared()
        factory.return_value.__enter__.return_value.responses.parse.return_value = (
            response(StructureExtraction(units=[])))
        result = extract_document(doc, Settings("test-only"))
        self.assertEqual(result.functions, [])
        self.assertEqual(factory.return_value.__enter__.return_value.responses.parse.call_count, 1)

    @patch("src.services.extraction.OpenAI")
    def test_stage_b_failure_does_not_return_partial_result(self, factory):
        doc, structure, _ = prepared()
        factory.return_value.__enter__.return_value.responses.parse.side_effect = [
            response(structure), SimpleNamespace(status="incomplete", output_parsed=None),
        ]
        with self.assertRaises(ExtractionError):
            extract_document(doc, Settings("test-only"))

    def test_director_with_subordinates_still_not_structure(self):
        doc = fixture("3.4. Директор направления внутреннего аудита имеет подчиненных.")
        candidate = structural(unit("Директор направления внутреннего аудита", doc.chunks[0].chunk_id))
        result = validate_structure(doc, [StructureExtraction(units=[candidate])])
        self.assertEqual(result.units, [])


if __name__ == "__main__":
    unittest.main()

