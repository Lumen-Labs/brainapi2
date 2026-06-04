import * as p from "@clack/prompts";
import { askUseDefaults, DEFAULT_DBS } from "./defaults.js";
import { askDatabases } from "./services.js";
import { askModels } from "./models.js";
import { askPipeline } from "./pipeline.js";
import { askConnections } from "./connections.js";
import { askAuth } from "./auth.js";
import { askServicesRuntime } from "./services-runtime.js";
import { askPlugins } from "./plugins.js";
import { isPromptBack } from "../lib/prompts.js";
import type {
  AuthChoices,
  Connections,
  DbChoices,
  InitChoices,
  ModelsChoices,
  PluginChoice,
  PipelineChoices,
  ServicesRuntime,
} from "../types.js";

export interface SetupDraft {
  dbs?: DbChoices;
  servicesRuntime?: ServicesRuntime;
  models?: ModelsChoices;
  pipeline?: PipelineChoices;
  connections?: Connections;
  auth?: AuthChoices;
  plugins?: PluginChoice[];
  usedDefaults?: boolean;
}

export interface RunSetupWizardOptions {
  firstStepBack?: "return" | "cancel";
  stepBackHint?: string;
  firstStepBackHint?: string;
}

export function toInitChoices(draft: SetupDraft): InitChoices {
  return {
    dbs: draft.dbs!,
    servicesRuntime: draft.servicesRuntime!,
    models: draft.models!,
    pipeline: draft.pipeline!,
    connections: draft.connections!,
    auth: draft.auth!,
    plugins: draft.plugins ?? [],
    usedDefaults: draft.usedDefaults ?? false,
  };
}

export async function runSetupWizard(
  draft: SetupDraft,
  opts: RunSetupWizardOptions = {},
): Promise<boolean> {
  const stepBackHint = opts.stepBackHint ?? "Previous step";
  const firstStepBackHint = opts.firstStepBackHint ?? stepBackHint;
  let step = 0;

  while (step >= 0) {
    if (step === 0) {
      const useDefaults = await askUseDefaults({
        allowBack: true,
        backHint: firstStepBackHint,
      });
      if (isPromptBack(useDefaults)) {
        if (opts.firstStepBack === "cancel") {
          p.cancel("Setup cancelled.");
          process.exit(0);
        }
        return false;
      }
      draft.usedDefaults = useDefaults;
      if (useDefaults) {
        draft.dbs = DEFAULT_DBS;
        draft.pipeline = { ocrMode: "docparser" };
        step = 2;
        continue;
      }
      step = 1;
      continue;
    }

    if (step === 1) {
      const dbs = await askDatabases({
        allowBack: true,
        backHint: stepBackHint,
        initial: draft.dbs,
      });
      if (isPromptBack(dbs)) {
        step = 0;
        continue;
      }
      draft.dbs = dbs;
      step = 2;
      continue;
    }

    if (step === 2) {
      if (!draft.dbs) {
        step = 1;
        continue;
      }
      const runtime = await askServicesRuntime(draft.dbs, {
        allowBack: true,
        backHint: stepBackHint,
        initialValue: draft.servicesRuntime,
      });
      if (isPromptBack(runtime)) {
        step = draft.usedDefaults ? 0 : 1;
        continue;
      }
      draft.servicesRuntime = runtime;
      step = 3;
      continue;
    }

    if (step === 3) {
      const models = draft.usedDefaults
        ? await askModels({
            prechosenMode: "remote",
            initialMode: draft.models?.mode,
          })
        : await askModels({
            allowBack: true,
            backHint: stepBackHint,
            initialMode: draft.models?.mode,
          });
      if (isPromptBack(models)) {
        step = 2;
        continue;
      }
      draft.models = models;
      step = draft.usedDefaults ? 5 : 4;
      continue;
    }

    if (step === 4) {
      const pipeline = await askPipeline({
        allowBack: true,
        backHint: stepBackHint,
        initialOcrMode: draft.pipeline?.ocrMode,
      });
      if (isPromptBack(pipeline)) {
        step = 3;
        continue;
      }
      draft.pipeline = pipeline;
      step = 5;
      continue;
    }

    if (step === 5) {
      if (!draft.dbs) {
        step = draft.usedDefaults ? 0 : 1;
        continue;
      }
      draft.connections = await askConnections(draft.dbs);
      step = 6;
      continue;
    }

    if (step === 6) {
      const auth = await askAuth({
        allowBack: true,
        backHint: stepBackHint,
      });
      if (isPromptBack(auth)) {
        step = 5;
        continue;
      }
      draft.auth = auth;
      step = 7;
      continue;
    }

    if (step === 7) {
      const plugins = await askPlugins({
        allowBack: true,
        backHint: stepBackHint,
        initial: draft.plugins,
      });
      if (isPromptBack(plugins)) {
        step = 6;
        continue;
      }
      draft.plugins = plugins;
      return true;
    }
  }

  return false;
}
