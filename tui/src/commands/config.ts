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
import { writeEnvFromChoices } from "../lib/write-env.js";
import { installExtras, missingExtras } from "../lib/python.js";
import { isPromptBack, pickOne } from "../lib/prompts.js";
import type { InitChoices } from "../types.js";

const MENU_BACK_HINT = "Return to configuration menu";

type ConfigMenuSection =
  | "defaults"
  | "databases"
  | "runtime"
  | "models"
  | "pipeline"
  | "connections"
  | "auth"
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
  return lines.join("\n");
}

function missingForSave(draft: ConfigDraft): string[] {
  const missing: string[] = [];
  if (!draft.dbs) missing.push("databases");
  if (!draft.servicesRuntime) missing.push("services runtime");
  if (!draft.models) missing.push("models");
  if (!draft.pipeline) missing.push("pipeline");
  if (!draft.connections) missing.push("connections");
  if (!draft.auth?.brainpatToken) missing.push("authentication");
  return missing;
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
  const missing = missingForSave(draft);
  if (missing.length > 0) {
    p.log.error(
      "Cannot save yet. Still need: " + missing.join(", ") + ".",
    );
    p.note(draftSummary(draft), "Current draft");
    return false;
  }

  const choices = toInitChoices(draft);
  await writeEnvFromChoices(choices);
  await updateState({ envWritten: true, servicesRuntime: choices.servicesRuntime });

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
      case "wizard":
        await runConfigWizard(draft);
        break;
    }
  }
}
