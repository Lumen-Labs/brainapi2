import ast
import os
import time
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent

ENV_DEFAULTS = {
    "BRAINPAT_TOKEN": "test-token",
    "MODELS_MODE": "local",
    "EMBEDDINGS_LOCAL_MODEL": "local-model",
    "EMBEDDINGS_SMALL_MODEL": "small-model",
    "EMBEDDING_NODES_DIMENSION": "3",
    "EMBEDDING_TRIPLETS_DIMENSION": "3",
    "EMBEDDING_OBSERVATIONS_DIMENSION": "3",
    "EMBEDDING_DATA_DIMENSION": "3",
    "EMBEDDING_RELATIONSHIPS_DIMENSION": "3",
    "REDIS_HOST": "localhost",
    "REDIS_PORT": "6379",
    "NEO4J_HOST": "localhost",
    "NEO4J_PORT": "7687",
    "NEO4J_USERNAME": "neo4j",
    "NEO4J_PASSWORD": "password",
    "MILVUS_HOST": "localhost",
    "MILVUS_PORT": "19530",
    "MONGO_CONNECTION_STRING": "mongodb://localhost",
    "CELERY_WORKER_CONCURRENCY": "1",
    "OLLAMA_HOST": "localhost",
    "OLLAMA_PORT": "11434",
    "OLLAMA_LLM_SMALL_MODEL": "small",
    "OLLAMA_LLM_LARGE_MODEL": "large",
}
for _k, _v in ENV_DEFAULTS.items():
    os.environ.setdefault(_k, _v)


class TestDockerfileSpacyModels(unittest.TestCase):
    def test_builder_installs_models_via_preload_script(self):
        dockerfile = (ROOT / "Dockerfile").read_text()
        preload = (ROOT / "scripts" / "preload_docker_models.py").read_text()
        self.assertIn("preload_docker_models.py --spacy-only", dockerfile)
        self.assertIn("--local-ml-only", dockerfile)
        self.assertIn("SPACY_MODEL_NAMES", preload)
        self.assertIn("spacy download", preload)
        self.assertIn("src.constants.spacy_models", preload)
        self.assertIn("--spacy-only", preload)
        spacy_run, _, rest = dockerfile.partition(
            "python scripts/preload_docker_models.py --spacy-only"
        )
        self.assertTrue(rest)
        self.assertNotIn("INSTALL_LOCAL_ML", spacy_run.splitlines()[-1])

    def test_preload_detects_wrapped_huggingface_rate_limits(self):
        import scripts.preload_docker_models as preload

        class FakeResponse:
            status_code = 429
            headers = {"Retry-After": "7"}

        class HubError(Exception):
            def __init__(self):
                super().__init__("429 Client Error: Too Many Requests")
                self.response = FakeResponse()

        wrapped = OSError(
            "We couldn't connect to 'https://huggingface.co' to load the files"
        )
        wrapped.__cause__ = HubError()

        self.assertTrue(preload._is_rate_limited(wrapped))
        self.assertTrue(preload._is_transient_hub_error(wrapped))
        self.assertEqual(preload._retry_after_seconds(wrapped), 7.0)

    def test_spacy_only_skips_sentence_transformer_preload(self):
        import scripts.preload_docker_models as preload

        with (
            patch.object(preload, "_preload_spacy") as spacy_fn,
            patch.object(preload, "_preload_sentence_transformers") as st_fn,
            patch.object(preload, "_configure_hf_token") as hf_fn,
        ):
            preload.main(["--spacy-only"])
        spacy_fn.assert_called_once()
        st_fn.assert_not_called()
        hf_fn.assert_not_called()

    def test_chinese_uses_web_pipeline_name(self):
        from src.constants.spacy_models import SPACY_MODEL_NAMES

        self.assertEqual(SPACY_MODEL_NAMES["zh"], "zh_core_web_sm")

    def test_no_languages_without_spacy_3_8_pipelines(self):
        from src.constants.spacy_models import SPACY_MODEL_NAMES

        for code in (
            "ar",
            "bn",
            "fil",
            "hi",
            "id",
            "km",
            "lo",
            "ml",
            "ms",
            "my",
            "pa",
            "ta",
            "te",
            "th",
            "tr",
            "vi",
        ):
            self.assertNotIn(code, SPACY_MODEL_NAMES)


class TestFastAPIRedirectSlashes(unittest.TestCase):
    def test_app_disables_trailing_slash_redirect(self):
        src = (ROOT / "src" / "services" / "api" / "app.py").read_text()
        tree = ast.parse(src)
        found = False
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if not isinstance(func, ast.Name) or func.id != "FastAPI":
                continue
            for kw in node.keywords:
                if kw.arg == "redirect_slashes" and isinstance(
                    kw.value, ast.Constant
                ):
                    if kw.value.value is False:
                        found = True
        self.assertTrue(
            found, "FastAPI(...) should pass redirect_slashes=False"
        )


class TestSpacyManagerDownloadBehavior(unittest.TestCase):
    def test_request_path_module_does_not_import_subprocess(self):
        source = (ROOT / "src" / "utils" / "nlp" / "spacy.py").read_text()
        self.assertNotIn("subprocess", source)
        self.assertNotIn("spacy download", source)

    @patch("src.utils.nlp.spacy.spacy.load")
    def test_failed_load_raises_fast_and_is_cached(self, mock_load):
        mock_load.side_effect = OSError("missing")

        from src.utils.nlp.spacy import SpacyManager

        mgr = SpacyManager()
        started = time.perf_counter()
        with patch("src.utils.nlp.spacy.logger") as mock_logger:
            with self.assertRaises(OSError) as ctx:
                mgr.get_model("it")
            with self.assertRaises(OSError):
                mgr.get_model("it")
        elapsed = time.perf_counter() - started
        self.assertIn("it_core_news_sm", str(ctx.exception))
        self.assertEqual(mock_load.call_count, 1)
        self.assertEqual(mock_logger.error.call_count, 1)
        self.assertLess(elapsed, 0.5)

    @patch("src.utils.nlp.spacy.spacy.load")
    def test_extract_elements_returns_empty_on_failed_load(self, mock_load):
        mock_load.side_effect = OSError("missing")
        from src.utils.nlp.ner import MultiLangEntityExtractor
        from src.utils.nlp.spacy import SpacyManager

        extractor = MultiLangEntityExtractor()
        extractor.spacy_manager = SpacyManager()
        started = time.perf_counter()
        first = extractor.extract_elements("hello world", lang="en")
        second = extractor.extract_elements("hello world", lang="en")
        elapsed = time.perf_counter() - started
        self.assertEqual(first.tokens, [])
        self.assertEqual(first.noun_chunks, [])
        self.assertEqual(second.tokens, [])
        self.assertEqual(mock_load.call_count, 1)
        self.assertLess(elapsed, 0.5)

    @patch("src.utils.nlp.spacy.spacy.load")
    def test_successful_load_is_cached(self, mock_load):
        nlp = object()
        mock_load.return_value = nlp

        from src.utils.nlp.spacy import SpacyManager

        mgr = SpacyManager()
        mgr.keep_models_in_memory = True
        self.assertIs(mgr.get_model("en"), nlp)
        self.assertIs(mgr.get_model("en"), nlp)
        self.assertEqual(mock_load.call_count, 1)

    def test_keep_models_in_memory_defaults_true(self):
        config_src = (ROOT / "src" / "config.py").read_text()
        self.assertIn(
            'os.getenv("SPACY_KEEP_MODELS_IN_MEMORY", "true")',
            config_src,
        )
        if os.getenv("SPACY_KEEP_MODELS_IN_MEMORY") in (None, "true"):
            from src.utils.nlp.spacy import SpacyManager

            self.assertTrue(SpacyManager().keep_models_in_memory)

    @patch("src.utils.nlp.spacy.spacy.load")
    def test_warmup_loads_default_language_once(self, mock_load):
        nlp = object()
        mock_load.return_value = nlp

        from src.utils.nlp.spacy import SpacyManager

        class ImmediateThread:
            def __init__(self, target=None, name=None, daemon=None):
                self._target = target
                self.daemon = daemon

            def start(self):
                self._target()

        mgr = SpacyManager()
        mgr.keep_models_in_memory = True
        with patch("src.utils.nlp.spacy.threading.Thread", ImmediateThread):
            mgr.warmup("en")
        self.assertIs(mgr.get_model("en"), nlp)
        self.assertEqual(mock_load.call_count, 1)


if __name__ == "__main__":
    unittest.main()
