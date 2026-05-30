import * as p from "@clack/prompts";
import pc from "picocolors";
import { envFilePath, sourcePath } from "../lib/paths.js";
import { readState } from "../lib/state.js";
import { pullRepo } from "../lib/git.js";
import { installDeps } from "../lib/python.js";
import { readEnvFile } from "../lib/env.js";
import { syncDevSourceIfAvailable } from "../lib/sync-source.js";

function getEnv(
  env: Awaited<ReturnType<typeof readEnvFile>>,
  key: string,
): string | undefined {
  const idx = env.keyIndex.get(key);
  if (idx === undefined) return undefined;
  const raw = (env.lines[idx] ?? "").split("=").slice(1).join("=");
  return raw.trim().replace(/^"|"$/g, "");
}

async function extrasFromEnv(): Promise<string[]> {
  try {
    const env = await readEnvFile(envFilePath());
    const ocr = getEnv(env, "OCR_MODE");
    const extras = ocr === "docling" ? ["docling-ocr"] : [];
    const vector = getEnv(env, "VECTOR_DB");
    const data = getEnv(env, "DATA_DB");
    const graph = getEnv(env, "GRAPH_DB");
    if (
      vector === "postgresql" ||
      data === "postgresql" ||
      graph === "networkx"
    ) {
      extras.push("postgresql-backend");
    }
    return extras;
  } catch {
    return [];
  }
}

export async function runUpdate(): Promise<void> {
  p.intro(pc.bgCyan(pc.black(" brainapi ")) + " " + pc.dim("update"));

  const state = await readState();
  if (!state || !state.cloned) {
    p.cancel(
      "No brainapi install detected. Run " + pc.cyan("brainapi init") + " first.",
    );
    process.exit(1);
  }

  const branch = state.branch || "main";
  const spinner = p.spinner();
  spinner.start(`git pull origin ${branch} in ${sourcePath()}`);
  try {
    await pullRepo(branch);
    spinner.stop("Repository updated");
  } catch (err) {
    spinner.stop(pc.red("git pull failed"));
    throw err;
  }

  await syncDevSourceIfAvailable();

  const extras = await extrasFromEnv();
  if (extras.length === 0) {
    p.log.step("Reinstalling Python base dependencies");
  } else {
    p.log.step(`Reinstalling Python base + extras (${extras.join(", ")})`);
  }
  await installDeps({ extras });

  p.outro(pc.green("brainapi is up to date."));
}
