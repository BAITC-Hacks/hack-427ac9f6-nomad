"""Session-local success cache; no secrets, disk cache or cross-user state."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from src.parsers.parser_factory import parse_document
from src.services import extraction, comparison

ANALYSIS_VERSION = "extraction-2.3"
COMPARISON_VERSION = "comparison-3-coverage-1"
ROOT = Path(__file__).resolve().parents[2]
MAX_PAIRS = 3


def source_version(paths) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.encode())
        digest.update((ROOT / path).read_bytes())
    return digest.hexdigest()


def cache_keys(uploads, model: str):
    # Original filenames matter for document IDs and citations: a rename is a cache miss.
    identity = [(side, name, hashlib.sha256(data).hexdigest())
                for side, (name, data) in sorted(uploads.items())]
    analysis_files = ["src/models.py", "src/services/extraction.py", "src/services/chunking.py",
                      "requirements.txt"] + [
        str(path.relative_to(ROOT)).replace("\\", "/")
        for path in (ROOT / "src/parsers").glob("*.py")]
    analysis = hashlib.sha256(json.dumps(
        [identity, model, ANALYSIS_VERSION, source_version(analysis_files)],
        ensure_ascii=False).encode()).hexdigest()
    comparison_key = hashlib.sha256(json.dumps(
        [analysis, COMPARISON_VERSION, source_version(["src/services/comparison.py"])],
    ).encode()).hexdigest()
    return analysis, comparison_key


def remember(cache, key, value):
    cache[key] = deepcopy(value)
    while len(cache) > MAX_PAIRS:
        del cache[next(iter(cache))]


def run_analysis(uploads, state, settings, progress=lambda message: None):
    """Run only on explicit submission. Successful stage outputs survive later failure."""
    analysis_key, comparison_key = cache_keys(uploads, settings.model)
    analysis_cache = state.setdefault("_analysis_cache", {})
    comparison_cache = state.setdefault("_comparison_cache", {})
    progress("Читаем документы")
    cached = analysis_cache.get(analysis_key)
    if cached is not None:
        documents, results = deepcopy(cached)
        progress("Определяем структуру и функции")
    else:
        documents = {side: parse_document(data, name, side)
                     for side, (name, data) in uploads.items()}
        state["documents"] = documents
        progress("Определяем структуру и функции")
        results = {}
        for side, document in documents.items():
            results[side] = extraction.extract_document(document, settings)
        # Exceptions/refusals cannot reach this cache write.
        remember(analysis_cache, analysis_key, (documents, results))
    state["documents"] = documents
    state["extractions"] = results
    state.pop("comparison", None)
    progress("Сравниваем изменения")
    if comparison_key in comparison_cache:
        cached_comparison = comparison_cache[comparison_key]
        comparison.update_coverage(cached_comparison, results["BEFORE"].units, results["AFTER"].units)
        if not cached_comparison.completed:
            del comparison_cache[comparison_key]
    if comparison_key in comparison_cache:
        result = deepcopy(comparison_cache[comparison_key])
        progress("Формируем риски и рекомендации")
    else:
        result = comparison.compare_documents(
            documents["BEFORE"], documents["AFTER"],
            results["BEFORE"], results["AFTER"], settings, progress=progress,
        )
        comparison.update_coverage(result, results["BEFORE"].units, results["AFTER"].units)
        if result.completed:
            remember(comparison_cache, comparison_key, result)
    state["comparison"] = result
    return result

