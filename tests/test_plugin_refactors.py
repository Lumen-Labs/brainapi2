import tempfile
import unittest
from pathlib import Path

from src.core.plugins.catalog import PluginManifestCatalog
from src.core.plugins.entrypoints import PluginEntryPointResolver
from src.core.plugins.loader import PluginLoader
from src.core.plugins.manifest import MANIFEST_FILENAME


def write_manifest(
    plugin_dir: Path,
    *,
    name: str,
    version: str = "1.0.0",
    entry_point: str = "main.py",
    priority: int = 100,
) -> None:
    plugin_dir.mkdir(parents=True, exist_ok=True)
    content = (
        f"name: {name}\n"
        f"version: {version}\n"
        f"entry_point: \"{entry_point}\"\n"
        f"priority: {priority}\n"
    )
    (plugin_dir / MANIFEST_FILENAME).write_text(content, encoding="utf-8")


class PluginEntryPointResolverTests(unittest.TestCase):
    def test_resolver_handles_python_suffix_safely(self):
        resolver = PluginEntryPointResolver()
        resolved = resolver.resolve("mycopy.py", Path("/tmp/demo-plugin"))
        self.assertEqual(resolved, "mycopy")

    def test_resolver_maps_init_to_package(self):
        resolver = PluginEntryPointResolver()
        self.assertEqual(
            resolver.resolve("__init__.py", Path("/tmp/demo-plugin")),
            "demo-plugin",
        )
        self.assertEqual(
            resolver.resolve("pkg/__init__.py", Path("/tmp/demo-plugin")),
            "pkg",
        )

    def test_resolver_normalizes_relative_paths(self):
        resolver = PluginEntryPointResolver()
        resolved = resolver.resolve("./nested/module.py", Path("/tmp/demo-plugin"))
        self.assertEqual(resolved, "nested.module")


class PluginManifestCatalogTests(unittest.TestCase):
    def test_catalog_can_toggle_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_manifest(root / "valid", name="valid", entry_point="main.py")
            write_manifest(root / "invalid", name="invalid", entry_point="")

            catalog = PluginManifestCatalog(root)

            non_validated = catalog.discover(validate=False)
            validated = catalog.discover(validate=True)

            self.assertEqual({m.name for m in non_validated}, {"valid", "invalid"})
            self.assertEqual({m.name for m in validated}, {"valid"})


class PluginLoaderRefactorTests(unittest.TestCase):
    class DummyContext:
        def __init__(self):
            self.loaded = []

    def test_loader_discovers_valid_plugins_sorted_by_priority(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_manifest(root / "slow", name="slow", priority=20, entry_point="main.py")
            write_manifest(root / "fast", name="fast", priority=5, entry_point="main.py")
            write_manifest(root / "broken", name="broken", priority=1, entry_point="")

            loader = PluginLoader(root, self.DummyContext())
            discovered = loader.discover()

            self.assertEqual([m.name for m in discovered], ["fast", "slow"])

    def test_loader_uses_entry_point_resolver_for_python_module(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plugin_dir = root / "mycopy"
            write_manifest(plugin_dir, name="mycopy", entry_point="mycopy.py")
            (plugin_dir / "mycopy.py").write_text(
                "def register(context):\n    context.loaded.append('mycopy')\n",
                encoding="utf-8",
            )
            manifest = loader_manifest = PluginManifestCatalog(root).discover(validate=True)[0]
            loader = PluginLoader(root, self.DummyContext())

            loaded = loader.load(loader_manifest)

            self.assertTrue(loaded)
            self.assertEqual(loader.context.loaded, ["mycopy"])
            self.assertIn(manifest.name, loader.loaded_plugins)


if __name__ == "__main__":
    unittest.main()
