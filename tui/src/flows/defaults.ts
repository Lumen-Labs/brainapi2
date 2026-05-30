import * as p from "@clack/prompts";
import pc from "picocolors";
import {
  askConfirm,
  askConfirmOrBack,
  isPromptBack,
  pickOne,
  type PromptBack,
} from "../lib/prompts.js";
import type { DbChoices } from "../types.js";

export async function askUseDefaults(opts?: {
  allowBack?: false;
}): Promise<boolean>;
export async function askUseDefaults(opts: {
  allowBack: true;
  backHint?: string;
}): Promise<boolean | PromptBack>;
export async function askUseDefaults(opts?: {
  allowBack?: boolean;
  backHint?: string;
}): Promise<boolean | PromptBack> {
  const choice = opts?.allowBack
    ? await pickOne<"defaults" | "custom">({
        message: "How would you like to configure BrainAPI?",
        options: [
          { value: "defaults", label: "Use default settings", hint: "Postgres + pgvector + NetworkX + remote LLM" },
          { value: "custom", label: "Configure each option manually" },
        ],
        initialValue: "defaults",
        allowBack: true,
        backHint: opts.backHint,
      })
    : await pickOne<"defaults" | "custom">({
        message: "How would you like to configure BrainAPI?",
        options: [
          { value: "defaults", label: "Use default settings", hint: "Postgres + pgvector + NetworkX + remote LLM" },
          { value: "custom", label: "Configure each option manually" },
        ],
        initialValue: "defaults",
      });

  if (isPromptBack(choice)) {
    return choice;
  }

  if (choice === "custom") return false;

  p.note(
    [
      "Stack: NetworkX (on Postgres) + pgvector + Postgres data layer + remote LLM.",
      "Backing services: " + pc.cyan("Redis") + " + " + pc.cyan("Postgres") + ".",
      "",
      "You'll still be asked how to run those services (Docker or manual),",
      "for the remote LLM provider, GCP credentials, and a BRAINPAT_TOKEN.",
    ].join("\n"),
    "Defaults",
  );

  if (opts?.allowBack) {
    return askConfirmOrBack({
      message: "Continue with defaults?",
      initialValue: true,
      backHint: opts.backHint,
    });
  }

  return askConfirm({
    message: "Continue with defaults?",
    initialValue: true,
  });
}

export const DEFAULT_DBS: DbChoices = {
  vectorDb: "postgresql",
  dataDb: "postgresql",
  graphDb: "networkx",
};
