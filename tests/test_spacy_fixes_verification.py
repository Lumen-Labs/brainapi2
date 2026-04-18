import ast
import os
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
    def test_builder_installs_models_via_spacy_model_names(self):
        dockerfile = (ROOT / "Dockerfile").read_text()
        self.assertIn("SPACY_MODEL_NAMES", dockerfile)
        self.assertIn("spacy download", dockerfile)
        self.assertIn("src.constants.spacy_models", dockerfile)


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
    @patch("src.utils.nlp.spacy.subprocess.run")
    @patch("src.utils.nlp.spacy.spacy.load")
    def test_failed_download_raises_oserror_not_sysexit(
        self, mock_load, mock_run
    ):
        mock_load.side_effect = OSError("missing")
        mock_run.return_value = MagicMock(returncode=1, stderr="nope", stdout="")

        from src.utils.nlp.spacy import SpacyManager

        mgr = SpacyManager()
        with self.assertRaises(OSError) as ctx:
            mgr.get_model("it")
        self.assertIn("it_core_news_sm", str(ctx.exception))
        mock_run.assert_called_once()

    @patch("src.utils.nlp.spacy.subprocess.run")
    @patch("src.utils.nlp.spacy.spacy.load")
    def test_successful_download_then_load(
        self, mock_load, mock_run
    ):
        nlp = object()
        mock_load.side_effect = [OSError("missing"), nlp]
        mock_run.return_value = MagicMock(returncode=0, stderr="", stdout="")

        from src.utils.nlp.spacy import SpacyManager

        mgr = SpacyManager()
        out = mgr.get_model("en")
        self.assertIs(out, nlp)
        self.assertEqual(mock_load.call_count, 2)


if __name__ == "__main__":
    unittest.main()
