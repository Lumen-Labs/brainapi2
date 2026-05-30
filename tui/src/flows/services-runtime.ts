import * as p from "@clack/prompts";
import pc from "picocolors";
import { isPromptBack, pickOne, type PromptBack } from "../lib/prompts.js";
import type { DbChoices, ServicesRuntime } from "../types.js";

function nonRedisPicked(dbs: DbChoices): string[] {
  const picked: string[] = [];
  if (
    dbs.vectorDb === "postgresql" ||
    dbs.dataDb === "postgresql" ||
    dbs.graphDb === "networkx"
  ) {
    picked.push("PostgreSQL");
  }
  if (dbs.graphDb === "neo4j") picked.push("Neo4j");
  if (dbs.vectorDb === "milvus") picked.push("Milvus");
  if (dbs.dataDb === "mongo") picked.push("MongoDB");
  return picked;
}

export async function askServicesRuntime(
  dbs: DbChoices,
  opts?: { allowBack?: false; initialValue?: ServicesRuntime },
): Promise<ServicesRuntime>;
export async function askServicesRuntime(
  dbs: DbChoices,
  opts: {
    allowBack: true;
    backHint?: string;
    initialValue?: ServicesRuntime;
  },
): Promise<ServicesRuntime | PromptBack>;
export async function askServicesRuntime(
  dbs: DbChoices,
  opts?: {
    allowBack?: boolean;
    backHint?: string;
    initialValue?: ServicesRuntime;
  },
): Promise<ServicesRuntime | PromptBack> {
  const sidecars = ["Redis", ...nonRedisPicked(dbs)];
  p.note(
    [
      `BrainAPI needs these backing services for your setup:`,
      `  ${pc.cyan(sidecars.join(", "))}`,
      ``,
      `You can let BrainAPI manage them via Docker, or run them yourself.`,
    ].join("\n"),
    "Backing services",
  );

  const picked = opts?.allowBack
    ? await pickOne<ServicesRuntime>({
        message: "How do you want to run them?",
        options: [
          {
            value: "docker",
            label: "Docker Compose (recommended)",
            hint: "isolated, reproducible, one extra dependency",
          },
          {
            value: "manual",
            label: "Manual — I'll install/run them myself",
            hint: "no Docker dependency; the TUI just probes for reachability",
          },
        ],
        initialValue: opts.initialValue ?? "docker",
        allowBack: true,
        backHint: opts.backHint,
      })
    : await pickOne<ServicesRuntime>({
        message: "How do you want to run them?",
        options: [
          {
            value: "docker",
            label: "Docker Compose (recommended)",
            hint: "isolated, reproducible, one extra dependency",
          },
          {
            value: "manual",
            label: "Manual — I'll install/run them myself",
            hint: "no Docker dependency; the TUI just probes for reachability",
          },
        ],
        initialValue: opts?.initialValue ?? "docker",
      });
  if (isPromptBack(picked)) {
    return picked;
  }
  return picked;
}
