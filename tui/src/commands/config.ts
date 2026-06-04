import * as p from "@clack/prompts";
import pc from "picocolors";
import { readState, updateState } from "../lib/state.js";
import { askUseDefaults, DEFAULT_DBS } from "../flows/defaults.js";
import { askDatabases } from "../flows/services.js";
import { askModels } from "../flows/models.js";
import {
  askPipeline,
  ocrToExtras,
  postgresBackendExtras,
} from "../flows/pipeline.js";
import { askConnections } from "../flows/connections.js";
import { askAuth } from "../flows/auth.js";
import { askServicesRuntime } from "../flows/services-runtime.js";
import {
  runSetupWizard,
  toInitChoices,
  type SetupDraft,
} from "../flows/setup-wizard.js";
import { askPluginUninstall, askPlugins } from "../flows/plugins.js";
import { writeEnvFromChoices } from "../lib/write-env.js";
import { installExtras, missingExtras } from "../lib/python.js";
import { askText, isPromptBack, pickOne } from "../lib/prompts.js";
import {
  installLocalPlugin,
  installRegistryPlugin,
  listInstalledPlugins,
  uninstallPlugin,
} from "../lib/plugins.js";
import type { InitChoices, InstallState } from "../types.js";

const MENU_BACK_HINT = "Return to configuration menu";

type ConfigMenuSection =
  | "defaults"
  | "databases"
  | "runtime"
  | "models"
  | "pipeline"
  | "connections"
  | "auth"
  | "plugins"
  | "wizard"
  | "save"
  | "exit";

interface ConfigDraft extends SetupDraft {}

function draftSummary(draft: ConfigDraft): string {
  const lines: string[] = [];
  if (draft.dbs) {
    lines.push(
      `Databases: ${draft.dbs.vectorDb} / ${draft.dbs.dataDb} / ${draft.dbs.graphDb}`,
    );
  } else {
    lines.push("Databases: not configured");
  }
  lines.push(
    draft.servicesRuntime
      ? `Services runtime: ${draft.servicesRuntime}`
      : "Services runtime: not configured",
  );
  lines.push(draft.models ? "Models: configured" : "Models: not configured");
  lines.push(
    draft.pipeline
      ? `Pipeline OCR: ${draft.pipeline.ocrMode}`
      : "Pipeline OCR: not configured",
  );
  lines.push(
    draft.connections ? "Connections: configured" : "Connections: not configured",
  );
  lines.push(
    draft.auth?.brainpatToken ? "Auth token: set" : "Auth token: not configured",
  );
  lines.push(`Plugins selected: ${draft.plugins?.length ?? 0}`);
  return lines.join("\n");
}

function hasCompleteDraft(draft: ConfigDraft): boolean {
  return Boolean(
    draft.dbs &&
      draft.servicesRuntime &&
      draft.models &&
      draft.pipeline &&
      draft.connections &&
      draft.auth?.brainpatToken,
  );
}

async function configMainMenu(): Promise<ConfigMenuSection> {
  return pickOne<ConfigMenuSection>({
    message: "What would you like to configure?",
    options: [
      {
        value: "defaults",
        label: "Apply default stack",
        hint: "NetworkX + Postgres + remote LLM",
      },
      { value: "databases", label: "Databases" },
      { value: "runtime", label: "Services runtime" },
      { value: "models", label: "Models (LLM & embeddings)" },
      { value: "pipeline", label: "Pipeline (OCR)" },
      { value: "connections", label: "Connection details" },
      { value: "auth", label: "Authentication token" },
      { value: "plugins", label: "Plugins (install/uninstall)" },
      {
        value: "wizard",
        label: "Configure all (step-by-step)",
        hint: "Full flow with back between steps",
      },
      { value: "save", label: "Save & apply changes" },
      { value: "exit", label: "Exit without saving" },
    ],
  });
}

async function applyDefaultsSection(draft: ConfigDraft): Promise<void> {
  const useDefaults = await askUseDefaults({
    allowBack: true,
    backHint: MENU_BACK_HINT,
  });
  if (isPromptBack(useDefaults) || !useDefaults) {
    return;
  }
  draft.usedDefaults = true;
  draft.dbs = DEFAULT_DBS;
  draft.pipeline = { ocrMode: "docparser" };
  const models = await askModels({ prechosenMode: "remote" });
  if (!isPromptBack(models)) {
    draft.models = models;
  }
  p.log.success("Default stack applied to draft.");
}

async function configureDatabases(draft: ConfigDraft): Promise<void> {
  const dbs = await askDatabases({
    allowBack: true,
    backHint: MENU_BACK_HINT,
    initial: draft.dbs,
  });
  if (isPromptBack(dbs)) {
    return;
  }
  draft.dbs = dbs;
  draft.usedDefaults = false;
  p.log.success("Databases updated.");
}

async function configureRuntime(draft: ConfigDraft): Promise<void> {
  if (!draft.dbs) {
    p.log.warn("Configure databases first.");
    return;
  }
  const runtime = await askServicesRuntime(draft.dbs, {
    allowBack: true,
    backHint: MENU_BACK_HINT,
    initialValue: draft.servicesRuntime,
  });
  if (isPromptBack(runtime)) {
    return;
  }
  draft.servicesRuntime = runtime;
  p.log.success("Services runtime updated.");
}

async function configureModels(draft: ConfigDraft): Promise<void> {
  const models = await askModels({
    allowBack: true,
    backHint: MENU_BACK_HINT,
    initialMode: draft.models?.mode,
  });
  if (isPromptBack(models)) {
    return;
  }
  draft.models = models;
  p.log.success("Models updated.");
}

async function configurePipeline(draft: ConfigDraft): Promise<void> {
  const pipeline = await askPipeline({
    allowBack: true,
    backHint: MENU_BACK_HINT,
    initialOcrMode: draft.pipeline?.ocrMode,
  });
  if (isPromptBack(pipeline)) {
    return;
  }
  draft.pipeline = pipeline;
  draft.usedDefaults = false;
  p.log.success("Pipeline updated.");
}

async function configureConnections(draft: ConfigDraft): Promise<void> {
  if (!draft.dbs) {
    p.log.warn("Configure databases first.");
    return;
  }
  const connections = await askConnections(draft.dbs);
  draft.connections = connections;
  p.log.success("Connections updated.");
}

async function configureAuth(draft: ConfigDraft): Promise<void> {
  const auth = await askAuth({
    allowBack: true,
    backHint: MENU_BACK_HINT,
  });
  if (isPromptBack(auth)) {
    return;
  }
  draft.auth = auth;
  p.log.success("Authentication token updated.");
}

async function configurePlugins(draft: ConfigDraft): Promise<void> {
  const action = await pickOne<"add_search" | "install_path" | "uninstall" | "list" | "back">({
    message: "Plugin management",
    options: [
      { value: "add_search", label: "Search and add plugin (local/registry)" },
      { value: "install_path", label: "Install local plugin by path" },
      { value: "uninstall", label: "Uninstall installed plugin" },
      { value: "list", label: "List installed plugins" },
      { value: "back", label: "Back" },
    ],
    initialValue: "add_search",
  });
  if (action === "back") return;
  if (action === "list") {
    const installed = await listInstalledPlugins();
    if (installed.length === 0) {
      p.log.info("No installed plugins.");
      return;
    }
    p.note(
      installed.map((plugin) => `${plugin.name} (${plugin.version})`).join("\n"),
      "Installed plugins",
    );
    return;
  }
  if (action === "install_path") {
    const pluginPath = await askText({
      message: "Enter local plugin directory path",
      placeholder: "/path/to/plugin",
    });
    if (!pluginPath.trim()) {
      p.log.warn("No path provided.");
      return;
    }
    const installedName = await installLocalPlugin(pluginPath.trim());
    const existing = draft.plugins ?? [];
    draft.plugins = [
      ...existing.filter((plugin) => plugin.name !== installedName),
      { name: installedName, source: "local", path: pluginPath.trim() },
    ];
    p.log.success(`Installed local plugin ${installedName}.`);
    return;
  }
  if (action === "uninstall") {
    const selected = await askPluginUninstall({ allowBack: true, backHint: MENU_BACK_HINT });
    if (selected === null || isPromptBack(selected)) {
      return;
    }
    await uninstallPlugin(selected);
    draft.plugins = (draft.plugins ?? []).filter((plugin) => plugin.name !== selected);
    p.log.success(`Uninstalled plugin ${selected}.`);
    return;
  }
  const selectedPlugins = await askPlugins({
    allowBack: true,
    backHint: MENU_BACK_HINT,
    initial: draft.plugins ?? [],
  });
  if (isPromptBack(selectedPlugins)) {
    return;
  }
  for (const plugin of selectedPlugins) {
    if (plugin.source === "local") {
      if (!plugin.path) continue;
      await installLocalPlugin(plugin.path);
      p.log.success(`Installed local plugin ${plugin.name}.`);
      continue;
    }
    await installRegistryPlugin(plugin.name, plugin.version);
    p.log.success(`Installed registry plugin ${plugin.name}.`);
  }
  draft.plugins = selectedPlugins;
}

async function runConfigWizard(draft: ConfigDraft): Promise<void> {
  const completed = await runSetupWizard(draft, {
    firstStepBack: "return",
    firstStepBackHint: MENU_BACK_HINT,
    stepBackHint: "Previous step",
  });
  if (completed) {
    p.log.success("Full configuration complete.");
  }
}

async function saveDraft(draft: ConfigDraft): Promise<boolean> {
  const statePatch: Partial<InstallState> = {};
  if (draft.servicesRuntime) {
    statePatch.servicesRuntime = draft.servicesRuntime;
  }
  if (draft.plugins) {
    statePatch.installedPlugins = draft.plugins.map((plugin) => plugin.name);
  }
  if (Object.keys(statePatch).length > 0) {
    await updateState(statePatch);
  }

  if (!hasCompleteDraft(draft)) {
    p.log.success("Changes saved.");
    p.log.warn(
      "Draft is partial, so .env was not rewritten. Use Configure all to fully regenerate it.",
    );
    p.outro(pc.green("Partial configuration applied."));
    return true;
  }

  const choices = toInitChoices(draft);
  await writeEnvFromChoices(choices);
  await updateState({
    envWritten: true,
    servicesRuntime: choices.servicesRuntime,
    installedPlugins: choices.plugins.map((plugin) => plugin.name),
  });

  const extras = [
    ...ocrToExtras(choices.pipeline.ocrMode),
    ...postgresBackendExtras(choices.dbs),
  ];
  const missingExtrasList = await missingExtras(extras);
  if (missingExtrasList.length > 0) {
    p.log.info(`Installing missing extras: ${missingExtrasList.join(", ")}`);
    await installExtras(missingExtrasList);
    p.log.success("Extras installed.");
  }

  p.outro(pc.green(".env rewritten."));
  return true;
}

export async function runConfig(): Promise<void> {
  p.intro(pc.bgCyan(pc.black(" brainapi ")) + " " + pc.dim("config"));

  const state = await readState();
  if (!state || !state.cloned) {
    p.cancel(
      "No brainapi install detected. Run " + pc.cyan("brainapi init") + " first.",
    );
    process.exit(1);
  }

  const draft: ConfigDraft = {
    servicesRuntime: state.servicesRuntime ?? undefined,
    plugins: (state.installedPlugins ?? []).map((name) => ({
      name,
      source: "local",
    })),
  };

  p.note(
    "Pick a section to update, or use step-by-step mode. Select Back on any prompt to go back.",
    "Configuration",
  );

  while (true) {
    p.note(draftSummary(draft), "Current draft");
    const section = await configMainMenu();

    switch (section) {
      case "exit":
        p.cancel("No changes saved.");
        return;
      case "save":
        if (await saveDraft(draft)) {
          return;
        }
        break;
      case "defaults":
        await applyDefaultsSection(draft);
        break;
      case "databases":
        await configureDatabases(draft);
        break;
      case "runtime":
        await configureRuntime(draft);
        break;
      case "models":
        await configureModels(draft);
        break;
      case "pipeline":
        await configurePipeline(draft);
        break;
      case "connections":
        await configureConnections(draft);
        break;
      case "auth":
        await configureAuth(draft);
        break;
      case "plugins":
        await configurePlugins(draft);
        break;
      case "wizard":
        await runConfigWizard(draft);
        break;
    }
  }
}
