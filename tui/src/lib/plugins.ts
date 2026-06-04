import { access, cp } from "node:fs/promises";
import path from "node:path";
import { runInherit, runQuiet } from "./exec.js";
import { sourcePath, venvBin } from "./paths.js";
import type { PluginSource } from "../types.js";

export interface PluginSearchResult {
  name: string;
  version: string;
  description: string;
  source: PluginSource;
  path?: string;
}

interface LocalPluginManifest {
  name: string;
  version: string;
  path: string;
  description: string;
}

function pluginScript(script: string): string[] {
  return ["-c", script];
}

async function runPluginJson<T>(script: string): Promise<T> {
  const result = await runQuiet(venvBin("python"), pluginScript(script), {
    cwd: sourcePath(),
  });
  if (!result.ok) {
    const details = result.stderr.trim() || result.stdout.trim() || "unknown error";
    throw new Error(details);
  }
  const text = result.stdout.trim();
  if (!text) {
    throw new Error("Plugin command returned empty output");
  }
  return JSON.parse(text) as T;
}

export async function listInstalledPlugins(): Promise<PluginSearchResult[]> {
  const script = [
    "import json",
    "from src.core.plugins.manager import PluginManager",
    "manager = PluginManager(plugins_dir=__import__('pathlib').Path('plugins'), registry_url=None)",
    "plugins = manager.list_installed()",
    "print(json.dumps([{'name': p.name, 'version': p.version, 'description': p.description or ''} for p in plugins]))",
  ].join(";");
  const installed = await runPluginJson<Array<{ name: string; version: string; description: string }>>(script);
  return installed.map((plugin) => ({ ...plugin, source: "local" }));
}

async function listRegistryPlugins(): Promise<PluginSearchResult[]> {
  const script = [
    "import json",
    "from src.core.plugins.manager import PluginManager",
    "registry_url = __import__('os').getenv('PLUGIN_REGISTRY_URL', 'https://registry.brain-api.dev')",
    "manager = PluginManager(plugins_dir=__import__('pathlib').Path('plugins'), registry_url=registry_url)",
    "plugins = manager.list_available()",
    "print(json.dumps([{'name': p.name, 'version': p.version, 'description': p.description or ''} for p in plugins]))",
  ].join(";");
  const available = await runPluginJson<Array<{ name: string; version: string; description: string }>>(script);
  return available.map((plugin) => ({ ...plugin, source: "registry" }));
}

async function listWorkspacePlugins(): Promise<LocalPluginManifest[]> {
  const script = [
    "import json",
    "from pathlib import Path",
    "from src.core.plugins.manifest import MANIFEST_FILENAME, parse_manifest",
    "base = Path.cwd() / 'plugins'",
    "results = []",
    "if base.exists():",
    "  for child in sorted(base.iterdir()):",
    "    manifest_path = child / MANIFEST_FILENAME",
    "    if not manifest_path.exists():",
    "      continue",
    "    try:",
    "      m = parse_manifest(manifest_path)",
    "    except Exception:",
    "      continue",
    "    results.append({'name': m.name, 'version': m.version, 'description': m.description or '', 'path': str(child.resolve())})",
    "print(json.dumps(results))",
  ].join("\n");
  return runPluginJson<LocalPluginManifest[]>(script);
}

export async function searchPlugins(query: string): Promise<PluginSearchResult[]> {
  const normalized = query.trim().toLowerCase();
  if (!normalized) return [];
  const [localInstalled, localWorkspace, registry] = await Promise.all([
    listInstalledPlugins().catch(() => []),
    listWorkspacePlugins().catch(() => []),
    listRegistryPlugins().catch(() => []),
  ]);
  const workspaceCandidates: PluginSearchResult[] = localWorkspace.map((plugin) => ({
    name: plugin.name,
    version: plugin.version,
    description: plugin.description,
    source: "local",
    path: plugin.path,
  }));
  const merged = [...localInstalled, ...workspaceCandidates, ...registry];
  const dedup = new Map<string, PluginSearchResult>();
  for (const candidate of merged) {
    const key = `${candidate.source}:${candidate.name}:${candidate.path ?? ""}`;
    if (!dedup.has(key)) dedup.set(key, candidate);
  }
  return [...dedup.values()].filter((plugin) =>
    plugin.name.toLowerCase().includes(normalized),
  );
}

export async function installRegistryPlugin(name: string, version?: string): Promise<void> {
  const args = ["-m", "src.core.plugins.cli", "install", name];
  if (version && version.trim()) {
    args.push("--version", version.trim());
  }
  const result = await runInherit(venvBin("python"), args, { cwd: sourcePath() });
  if (!result.ok) {
    throw new Error(`Failed to install registry plugin '${name}'`);
  }
}

export async function uninstallPlugin(name: string): Promise<void> {
  const result = await runInherit(
    venvBin("python"),
    ["-m", "src.core.plugins.cli", "uninstall", name],
    { cwd: sourcePath() },
  );
  if (!result.ok) {
    throw new Error(`Failed to uninstall plugin '${name}'`);
  }
}

async function resolveLocalPluginName(pluginPath: string): Promise<string> {
  const quotedPath = JSON.stringify(pluginPath);
  const script = [
    "from pathlib import Path",
    "from src.core.plugins.manifest import MANIFEST_FILENAME, parse_manifest",
    `p = Path(${quotedPath})`,
    "manifest = p / MANIFEST_FILENAME",
    "if not manifest.exists():",
    "  raise SystemExit('plugin.yaml not found')",
    "parsed = parse_manifest(manifest)",
    "print(parsed.name)",
  ].join("\n");
  const result = await runQuiet(venvBin("python"), pluginScript(script), { cwd: sourcePath() });
  if (!result.ok) {
    throw new Error(result.stderr.trim() || result.stdout.trim() || "Invalid local plugin");
  }
  const name = result.stdout.trim();
  if (!name) throw new Error("Invalid local plugin manifest");
  return name;
}

export async function installLocalPlugin(pluginPath: string): Promise<string> {
  await access(pluginPath);
  const name = await resolveLocalPluginName(pluginPath);
  const targetPath = path.join(sourcePath(), "plugins", name);
  await cp(pluginPath, targetPath, { recursive: true, force: true });
  return name;
}
