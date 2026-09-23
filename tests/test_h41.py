"""Session success-cache regressions; all analysis services are mocked."""
import unittest
from dataclasses import replace
from unittest.mock import patch

from src.config import Settings
from src.models import ComparisonResult, ExtractionResult, OrgUnit
from src.services import pipeline
from src.ui_common import display_filename
from test_h1 import word_bytes
from test_p0 import successful_comparison


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.uploads = {side: (f"{side}.docx", word_bytes(["3.4. Department duties"]))
                        for side in ("BEFORE", "AFTER")}
        self.state = {}
        self.settings = Settings("test-only")
        self.extract_patch = patch("src.services.extraction.extract_document",
                                  side_effect=lambda doc, settings: ExtractionResult(document_id=doc.id, units=[OrgUnit(doc.id, "Отдел", None, [doc.chunks[0].chunk_id])]))
        self.compare_patch = patch("src.services.comparison.compare_documents",
                                  side_effect=successful_comparison)
        self.extract = self.extract_patch.start()
        self.compare = self.compare_patch.start()
        self.addCleanup(self.extract_patch.stop)
        self.addCleanup(self.compare_patch.stop)

    def run_pipeline(self, **kwargs):
        return pipeline.run_analysis(self.uploads, self.state, self.settings, **kwargs)

    def test_repeat_reuses_both_stages_and_keeps_original_sources(self):
        stages = []
        self.run_pipeline(progress=stages.append)
        original = self.state["documents"]["BEFORE"].chunks[0].text
        self.state["documents"]["BEFORE"].chunks[0] = replace(
            self.state["documents"]["BEFORE"].chunks[0], text="mutated display state")
        self.run_pipeline()
        self.assertEqual(self.extract.call_count, 2)
        self.assertEqual(self.compare.call_count, 1)
        self.assertEqual(self.state["documents"]["BEFORE"].chunks[0].text, original)
        self.assertIn("Определяем структуру и функции", stages)
        self.assertIn("Сравниваем изменения", stages)

    def test_content_model_filename_and_versions_invalidate(self):
        initial = pipeline.cache_keys(self.uploads, self.settings.model)
        for side in self.uploads:
            changed = dict(self.uploads)
            name, data = changed[side]
            changed[side] = (name, data + b"changed")
            self.assertNotEqual(initial, pipeline.cache_keys(changed, self.settings.model))
            changed[side] = ("renamed.docx", data)
            self.assertNotEqual(initial, pipeline.cache_keys(changed, self.settings.model))
        self.assertNotEqual(initial, pipeline.cache_keys(self.uploads, "another-model"))
        with patch.object(pipeline, "ANALYSIS_VERSION", "next"):
            self.assertNotEqual(initial[0], pipeline.cache_keys(self.uploads, self.settings.model)[0])
        with patch.object(pipeline, "COMPARISON_VERSION", "next"):
            changed = pipeline.cache_keys(self.uploads, self.settings.model)
            self.assertEqual(initial[0], changed[0])
            self.assertNotEqual(initial[1], changed[1])

    def test_failed_extraction_is_not_cached(self):
        self.extract.side_effect = RuntimeError("failure")
        with self.assertRaises(RuntimeError):
            self.run_pipeline()
        self.assertFalse(self.state["_analysis_cache"])
        self.compare.assert_not_called()

    def test_partial_comparison_retries_without_extraction(self):
        def compare_once_failed(*args, **kwargs):
            return ComparisonResult() if self.compare.call_count == 1 else successful_comparison(*args, **kwargs)
        self.compare.side_effect = compare_once_failed
        self.run_pipeline()
        self.assertFalse(self.state["_comparison_cache"])
        self.assertEqual(len(self.state["extractions"]), 2)
        self.run_pipeline()
        self.run_pipeline()
        self.assertEqual(self.extract.call_count, 2)
        self.assertEqual(self.compare.call_count, 2)

    def test_comparison_exception_preserves_extraction(self):
        self.compare.side_effect = RuntimeError("failure")
        with self.assertRaises(RuntimeError):
            self.run_pipeline()
        self.assertEqual(len(self.state["extractions"]), 2)
        self.assertFalse(self.state["_comparison_cache"])
        self.compare.side_effect = successful_comparison
        self.run_pipeline()
        self.assertEqual(self.extract.call_count, 2)

    def test_sessions_are_isolated_and_cache_is_bounded(self):
        self.run_pipeline()
        pipeline.run_analysis(self.uploads, {}, self.settings)
        self.assertEqual(self.extract.call_count, 4)
        for i in range(5):
            self.settings = Settings("test-only", model=f"model-{i}")
            self.run_pipeline()
        self.assertEqual(len(self.state["_analysis_cache"]), pipeline.MAX_PAIRS)
        self.assertEqual(len(self.state["_comparison_cache"]), pipeline.MAX_PAIRS)

    def test_filename_truncation_preserves_revision_tail_and_extension(self):
        name = "Положение о внутреннем аудите " * 5 + "редакция №8.pdf"
        label = display_filename(name)
        self.assertLessEqual(len(label), 64)
        self.assertIn("…", label)
        self.assertTrue(label.endswith("редакция №8.pdf"))
        self.assertEqual(display_filename("short.docx"), "short.docx")


if __name__ == "__main__":
    unittest.main()
