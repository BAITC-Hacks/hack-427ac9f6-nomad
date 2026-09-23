import json
import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from openai import APIConnectionError
import httpx
from stage_mocks import staged_response
from src.services.extraction import StructureExtraction

from src.config import ConfigurationError, Settings, load_settings
from src.models import Document
from src.services.chunking import split_pdf_page
from src.services.extraction import (
    BatchExtraction, ExtractionError, batch_payload, extract_document,
    make_batches, merge_extractions, resolve_refs,
)
from test_h1 import word_bytes
from streamlit.testing.v1 import AppTest
from pathlib import Path
from io import BytesIO


def document():
    doc = Document("document-1", "BEFORE", "source.pdf", "pdf")
    doc.chunks = split_pdf_page(doc, 1, "3.4. Alpha\nPlans work.\n3.5. Alpha\nPlans work.\n")
    return doc


def output(refs, name="Alpha", text="Plans work.", function_refs=None, parent=None):
    return BatchExtraction.model_validate({"units": [{
        "name": name, "parent": parent, "source_chunk_ids": refs,
        "entity_kind": "structural_unit", "structural_basis": "unit_definition",
        "structural_source_chunk_ids": refs[:1], "aliases": [],
        "functions": [{"text": text, "source_chunk_ids":
                       refs if function_refs is None else function_refs,
                       "owner_name": name, "ownership": "explicit",
                       "role_name": None,
                       "ownership_source_chunk_ids": refs[:1]}],
    }]})


class ChunkingTests(unittest.TestCase):
    def test_all_markers_preserve_text_and_positions(self):
        doc = document()
        text = "Preface\n3.4. One\n3.5. Two\n5.3.2.\nThree\n5.3.3. Four\n9.21. Five\n"
        chunks = split_pdf_page(doc, 6, text)
        self.assertEqual("".join(c.text for c in chunks), text)
        self.assertEqual([c.section for c in chunks],
                         [None, "3.4", "3.5", "5.3.2", "5.3.3", "9.21"])
        self.assertEqual(chunks, split_pdf_page(doc, 6, text))
        self.assertEqual(chunks[-1].chunk_id, "before_p006_006")
        self.assertTrue(all(c.page == 6 and c.document_id == doc.id for c in chunks))

    def test_fallback_and_whitespace(self):
        for text in ["No numbered section.\nOriginal text\n", "\n  3.4. Title\nBody\n"]:
            chunks = split_pdf_page(document(), 1, text)
            self.assertEqual(len(chunks), 1)
            self.assertEqual(chunks[0].text, text)

    def test_large_unicode_chunk_batch_budget(self):
        doc = document()
        text = ('Ж"\\\n' * 12000)
        doc.chunks = split_pdf_page(doc, 1, text)
        batches = make_batches(doc, max_bytes=4096)
        self.assertGreater(len(batches), 1)
        self.assertTrue(all(len(batch_payload(b).encode("utf-8")) <= 4096 for b in batches))
        # Remove the deliberate previous-excerpt overlap, not original text.
        parts = batches[0] + [part for batch in batches[1:] for part in batch[1:]]
        self.assertEqual("".join(p.text for p in parts), text)
        self.assertEqual({p.chunk_id for p in parts}, {doc.chunks[0].chunk_id})


class EvidenceTests(unittest.TestCase):
    def test_unknown_refs_discarded_and_resolved_to_original(self):
        doc = document()
        ref = doc.chunks[0].chunk_id
        result = merge_extractions(doc, [output([ref, "invented", ref])])
        self.assertEqual(result.units[0].source_refs, [ref])
        self.assertEqual(result.functions[0].source_refs, [ref])
        self.assertEqual(result.discarded_references, 2)
        self.assertIs(resolve_refs(doc, [ref, "invented"])[0], doc.chunks[0])

    def test_no_valid_unit_evidence_drops_unit_and_functions(self):
        result = merge_extractions(document(), [output(["invented"])])
        self.assertEqual(result.units, [])
        self.assertEqual(result.functions, [])
        self.assertEqual(result.discarded_items, 2)

    def test_function_cannot_borrow_unit_evidence(self):
        doc = document()
        result = merge_extractions(doc, [output([doc.chunks[0].chunk_id],
                                                function_refs=["invented"])])
        self.assertEqual(len(result.units), 1)
        self.assertEqual(result.functions, [])

    def test_merge_duplicates_and_union_evidence(self):
        doc = document()
        first, second = [c.chunk_id for c in doc.chunks]
        outputs = [output([first]), output([second], name=" ALPHA ", text="Plans   work.")]
        result = merge_extractions(doc, outputs)
        self.assertEqual(len(result.units), 1)
        self.assertEqual(len(result.functions), 1)
        self.assertEqual(result.functions[0].source_refs, [first, second])
        self.assertEqual(result.units[0].id, merge_extractions(doc, outputs).units[0].id)

    def test_different_parents_not_merged(self):
        doc = document()
        ref = doc.chunks[0].chunk_id
        result = merge_extractions(doc, [output([ref], parent="A"), output([ref], parent="B")])
        self.assertEqual(len(result.units), 2)

    def test_foreign_document_chunks_do_not_validate(self):
        doc = document()
        doc.id = "different-document"
        self.assertEqual(resolve_refs(doc, [doc.chunks[0].chunk_id]), [])


class ConfigurationTests(unittest.TestCase):
    @patch.dict(os.environ, {}, clear=True)
    @patch("src.config.dotenv_values", return_value={})
    def test_missing_key(self, _):
        with self.assertRaisesRegex(ConfigurationError, "OPENAI_API_KEY"):
            load_settings()

    @patch.dict(os.environ, {}, clear=True)
    @patch("src.config.dotenv_values",
           return_value={"OPENAI_API_KEY": "test-only", "OPENAI_MODEL": "configured-model"})
    def test_dotenv_and_secret_repr(self, _):
        settings = load_settings()
        self.assertEqual(settings.model, "configured-model")
        self.assertEqual(settings.api_key, "test-only")
        self.assertNotIn("test-only", repr(settings))


class APITests(unittest.TestCase):
    @patch("src.services.extraction.OpenAI")
    def test_structured_call_and_batch_id_scope(self, factory):
        doc = document()
        ref = doc.chunks[0].chunk_id
        client = factory.return_value.__enter__.return_value
        # A real document ID not supplied in this particular request is also rejected.
        client.responses.parse.side_effect = staged_response(output([ref, doc.chunks[1].chunk_id]))
        first_batch = make_batches(doc)[:1]
        first_batch[0] = first_batch[0][:1]
        with patch("src.services.extraction.make_batches", return_value=first_batch):
            result = extract_document(doc, Settings("test-only"))
        args = client.responses.parse.call_args_list[0].kwargs
        self.assertEqual(args["model"], "gpt-4.1-mini")
        self.assertIs(args["text_format"], StructureExtraction)
        self.assertFalse(args["store"])
        self.assertEqual(json.loads(args["input"])[0]["chunk_id"], ref)
        self.assertEqual(result.units[0].source_refs, [ref])
        self.assertEqual(result.discarded_references, 2)

    @patch("src.services.extraction.OpenAI")
    def test_refusal_and_incomplete_fail_closed(self, factory):
        client = factory.return_value.__enter__.return_value
        for status, parsed in [("completed", None), ("incomplete", output(["fake"]))]:
            client.responses.parse.return_value = SimpleNamespace(
                status=status, output_parsed=parsed)
            with self.assertRaises(ExtractionError):
                extract_document(document(), Settings("test-only"))

    @patch("src.services.extraction.OpenAI")
    def test_api_failure_is_sanitized(self, factory):
        factory.return_value.__enter__.return_value.responses.parse.side_effect = (
            APIConnectionError(message="sensitive-error", request=httpx.Request("POST", "https://example.test"))
        )
        with self.assertRaises(ExtractionError) as error:
            extract_document(document(), Settings("test-only"))
        self.assertNotIn("sensitive-error", str(error.exception))


class H2AppTests(unittest.TestCase):
    def setUp(self):
        self.app_path = str(Path(__file__).resolve().parents[1] / "app.py")

    @patch("src.services.extraction.OpenAI")
    @patch("src.config.dotenv_values", return_value={})
    @patch.dict(os.environ, {}, clear=True)
    def test_missing_key_ui_and_no_api_call(self, _, factory):
        data = word_bytes(["3.4. Alpha"])
        upload = BytesIO(data)
        upload.name = "test.docx"
        with patch("streamlit.file_uploader", return_value=upload):
            app = AppTest.from_file(self.app_path, default_timeout=30).run()
            app.button[0].click().run()
        self.assertEqual(len(app.exception), 0)
        self.assertIn("Доступ к сервису анализа не настроен", app.error[0].value)
        factory.assert_not_called()
        self.assertEqual(len(app.code), 0)
        self.assertEqual(len(app.session_state["documents"]), 2)

    @patch("src.services.extraction.OpenAI")
    @patch("src.config.dotenv_values", return_value={"OPENAI_API_KEY": "test-only"})
    @patch.dict(os.environ, {}, clear=True)
    def test_mocked_end_to_end_metrics_and_original_evidence(self, _, factory):
        data = word_bytes(["3.4. Alpha plans work."])
        upload = BytesIO(data)
        upload.name = "test.docx"
        client = factory.return_value.__enter__.return_value
        def respond(**kwargs):
            payload = json.loads(kwargs["input"])
            chunks = payload if isinstance(payload, list) else payload["chunks"]
            return staged_response(output([chunks[0]["chunk_id"]]))(**kwargs)
        client.responses.parse.side_effect = respond
        from test_p0 import successful_comparison
        with patch("src.services.comparison.compare_documents", side_effect=successful_comparison) as compare, patch("streamlit.file_uploader", return_value=upload):
            app = AppTest.from_file(self.app_path, default_timeout=30).run()
            app.button[0].click().run()
            self.assertEqual(len(app.exception), 0)
            self.assertEqual(len(app.metric), 7)
            self.assertEqual(len(app.session_state["extractions"]["BEFORE"].units), 1)
            self.assertTrue(any(item.value == "3.4. Alpha plans work." for item in app.text))
            app.run()  # Ordinary reruns must not incur new API calls.
            app.button[0].click().run()  # Explicit repeat also uses the success cache.
            self.assertFalse(app.exception)
            self.assertTrue(app.session_state["comparison"].completed)
            self.assertEqual(compare.call_count, 1)
            self.assertEqual(len(app.button), 1)
        self.assertEqual(client.responses.parse.call_count, 4)


if __name__ == "__main__":
    unittest.main()
