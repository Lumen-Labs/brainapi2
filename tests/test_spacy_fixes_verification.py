import ast
import os
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

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
        from src.constants.spacy_models import (
            SPACY_BLANK_FALLBACK_LANGS,
            SPACY_MODEL_NAMES,
        )

        for code in SPACY_BLANK_FALLBACK_LANGS:
            self.assertNotIn(code, SPACY_MODEL_NAMES)
        for code in (
            "ar",
            "he",
            "hi",
            "id",
            "th",
            "tr",
            "vi",
        ):
            self.assertIn(code, SPACY_BLANK_FALLBACK_LANGS)

    def test_additional_small_pipelines_are_registered(self):
        from src.constants.spacy_models import SPACY_MODEL_NAMES

        expected = {
            "ca": "ca_core_news_sm",
            "da": "da_core_news_sm",
            "fi": "fi_core_news_sm",
            "el": "el_core_news_sm",
            "hr": "hr_core_news_sm",
            "lt": "lt_core_news_sm",
            "mk": "mk_core_news_sm",
            "nb": "nb_core_news_sm",
            "pl": "pl_core_news_sm",
            "ro": "ro_core_news_sm",
            "sl": "sl_core_news_sm",
            "sv": "sv_core_news_sm",
            "uk": "uk_core_news_sm",
        }
        for code, name in expected.items():
            self.assertEqual(SPACY_MODEL_NAMES[code], name)

    def test_norwegian_detector_codes_map_to_bokmaal(self):
        from src.constants.spacy_models import resolve_spacy_lang

        self.assertEqual(resolve_spacy_lang("no"), "nb")
        self.assertEqual(resolve_spacy_lang("nn"), "nb")
        self.assertEqual(resolve_spacy_lang("nb"), "nb")
        self.assertEqual(resolve_spacy_lang("NO"), "nb")


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
            with self.assertRaises(OSError) as first:
                mgr.get_model("it")
            with self.assertRaises(OSError) as second:
                mgr.get_model("it")
        elapsed = time.perf_counter() - started
        self.assertIn("it_core_news_sm", str(first.exception))
        self.assertEqual(str(first.exception), str(second.exception))
        self.assertIsNot(first.exception, second.exception)
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

        mgr = SpacyManager()
        mgr.keep_models_in_memory = True
        with patch("src.utils.nlp.spacy.threading.Thread", ImmediateThread):
            mgr.warmup("en")
        self.assertIs(mgr.get_model("en"), nlp)
        self.assertEqual(mock_load.call_count, 1)

    @patch("src.utils.nlp.spacy.spacy.load")
    def test_norwegian_alias_loads_bokmaal_pipeline(self, mock_load):
        nlp = object()
        mock_load.return_value = nlp

        from src.utils.nlp.spacy import SpacyManager

        mgr = SpacyManager()
        mgr.keep_models_in_memory = True
        self.assertIs(mgr.get_model("no"), nlp)
        self.assertIs(mgr.get_model("nb"), nlp)
        mock_load.assert_called_once_with("nb_core_news_sm")

    @patch("src.utils.nlp.spacy.spacy.load")
    @patch("src.utils.nlp.spacy.spacy.blank")
    def test_unsupported_language_uses_blank_without_error(
        self, mock_blank, mock_load
    ):
        blank_nlp = MagicMock(name="blank")
        mock_blank.return_value = blank_nlp

        from src.utils.nlp.spacy import SpacyManager

        mgr = SpacyManager()
        mgr.keep_models_in_memory = True
        with patch("src.utils.nlp.spacy.logger") as mock_logger:
            out = mgr.get_model("ar")
        self.assertIs(out, blank_nlp)
        mock_blank.assert_called_once_with("ar")
        mock_load.assert_not_called()
        mock_logger.error.assert_not_called()

    @patch("src.utils.nlp.spacy.spacy.load")
    @patch("src.utils.nlp.spacy.spacy.blank")
    def test_unknown_language_falls_back_to_xx_blank(self, mock_blank, mock_load):
        fallback = object()
        mock_blank.side_effect = [ImportError("missing lang"), fallback]

        from src.utils.nlp.spacy import SpacyManager

        mgr = SpacyManager()
        with patch("src.utils.nlp.spacy.logger") as mock_logger:
            out = mgr.get_model("zz")
        self.assertIs(out, fallback)
        self.assertEqual(mock_blank.call_args_list[0].args, ("zz",))
        self.assertEqual(mock_blank.call_args_list[1].args, ("xx",))
        mock_load.assert_not_called()
        mock_logger.error.assert_not_called()

    def test_extract_elements_fallback_returns_tokens_without_entities(self):
        from src.utils.nlp.ner import MultiLangEntityExtractor
        from src.utils.nlp.spacy import SpacyManager

        extractor = MultiLangEntityExtractor()
        extractor.spacy_manager = SpacyManager()
        extractor.spacy_manager.keep_models_in_memory = True
        result = extractor.extract_elements(
            "hello world from retrieval", lang="hi"
        )
        self.assertTrue(result.tokens)
        self.assertEqual(result.noun_chunks, [])
        texts = [token["text"].lower() for token in result.tokens]
        self.assertIn("hello", texts)
        self.assertFalse(any(token.get("entity") for token in result.tokens))

    @patch("src.utils.nlp.spacy.spacy.load")
    @patch("src.utils.nlp.spacy.time.monotonic")
    def test_failed_load_retries_after_ttl(self, mock_mono, mock_load):
        clock = {"now": 0.0}
        mock_mono.side_effect = lambda: clock["now"]
        nlp = object()
        mock_load.side_effect = [OSError("missing"), nlp]

        from src.utils.nlp.spacy import SpacyManager

        mgr = SpacyManager()
        mgr.load_retry_ttl_seconds = 300.0
        with patch("src.utils.nlp.spacy.logger"):
            with self.assertRaises(OSError):
                mgr.get_model("pl")
            clock["now"] = 10.0
            with self.assertRaises(OSError):
                mgr.get_model("pl")
            self.assertEqual(mock_load.call_count, 1)
            clock["now"] = 300.0
            self.assertIs(mgr.get_model("pl"), nlp)
        self.assertEqual(mock_load.call_count, 2)

    @patch("src.utils.nlp.spacy.spacy.load")
    def test_stale_failed_sentinel_without_failure_record_retries(self, mock_load):
        nlp = object()
        mock_load.return_value = nlp
        from src.utils.nlp.spacy import SpacyManager, _FAILED

        mgr = SpacyManager()
        mgr.keep_models_in_memory = True
        mgr._cache["pl"] = _FAILED
        self.assertIs(mgr.get_model("pl"), nlp)
        mock_load.assert_called_once_with("pl_core_news_sm")

    @patch("src.utils.nlp.spacy.spacy.load")
    def test_unload_alias_clears_canonical_cache(self, mock_load):
        mock_load.side_effect = OSError("missing")
        from src.utils.nlp.spacy import SpacyManager

        mgr = SpacyManager()
        mgr.keep_models_in_memory = True
        with patch("src.utils.nlp.spacy.logger"):
            with self.assertRaises(OSError):
                mgr.get_model("no")
        mgr.unload("no")
        nlp = object()
        mock_load.side_effect = None
        mock_load.return_value = nlp
        self.assertIs(mgr.get_model("nb"), nlp)
        self.assertEqual(mock_load.call_count, 2)

    def test_load_retry_ttl_defaults_to_five_minutes(self):
        from src.config import _spacy_load_retry_ttl_seconds
        from src.constants.spacy_models import DEFAULT_SPACY_LOAD_RETRY_TTL_SECONDS

        self.assertEqual(DEFAULT_SPACY_LOAD_RETRY_TTL_SECONDS, 300.0)
        config_src = (ROOT / "src" / "config.py").read_text()
        self.assertIn(
            'os.getenv("SPACY_LOAD_RETRY_TTL_SECONDS", "300")',
            config_src,
        )
        with patch.dict(os.environ, {"SPACY_LOAD_RETRY_TTL_SECONDS": "12.5"}):
            self.assertEqual(_spacy_load_retry_ttl_seconds(), 12.5)

    def test_api_startup_calls_warmup_default_model(self):
        src = (ROOT / "src" / "services" / "api" / "app.py").read_text()
        self.assertIn("warmup_default_model()", src)
        warmup_src = (ROOT / "src" / "utils" / "nlp" / "spacy.py").read_text()
        self.assertIn("warmup_language_detector", warmup_src)
        self.assertIn("lang-detect-warmup", warmup_src)

    @patch("src.utils.nlp.spacy.spacy.load", return_value=object())
    def test_warmup_default_model_warms_detector(self, _mock_load):
        from src.utils.nlp.spacy import warmup_default_model

        with (
            patch("src.utils.nlp.spacy.threading.Thread", ImmediateThread),
            patch(
                "src.utils.nlp.lang_detect.warmup_language_detector"
            ) as mock_detector,
        ):
            warmup_default_model("en")
        mock_detector.assert_called_once()

    def test_warmup_language_detector_invokes_langid(self):
        from src.utils.nlp import lang_detect as lang_detect_mod

        lang_detect_mod._langid_cached.cache_clear()
        with (
            patch.object(
                lang_detect_mod, "classify", return_value=("en", 1.0)
            ) as mock_classify,
            patch.object(
                lang_detect_mod, "detect_language", return_value="en"
            ) as mock_detect,
        ):
            lang_detect_mod.warmup_language_detector()
        mock_classify.assert_called()
        mock_detect.assert_called()


class ImmediateThread:
    def __init__(self, target=None, name=None, daemon=None):
        self._target = target
        self.name = name
        self.daemon = daemon

    def start(self):
        self._target()


if __name__ == "__main__":
    unittest.main()
