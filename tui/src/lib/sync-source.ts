import { access, cp } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import * as p from "@clack/prompts";
import pc from "picocolors";
import { sourcePath } from "./paths.js";

const SYNC_PATHS = [
  "src/config.py",
  "src/core/instances.py",
  "src/lib/llm",
  "src/lib/embeddings",
  "src/lib/postgresql",
  "src/services/api/app.py",
  "src/services/api/console_static.py",
  "src/services/api/middlewares/auth.py",
  "src/services/api/middlewares/brains.py",
  "src/services/api/controllers/meta.py",
  "src/services/api/routes/meta.py",
  "src/adapters/interfaces/data.py",
  "src/adapters/data.py",
  "src/services/api/controllers/vectors.py",
  "src/services/api/routes/retrieve.py",
  "src/adapters/interfaces/embeddings.py",
  "src/adapters/embeddings.py",
  "src/lib/milvus/client.py",
  "console",
] as const;

async function pathExists(target: string): Promise<boolean> {
  try {
    await access(target);
    return true;
  } catch {
    return false;
  }
}

export async function devWorkspaceRoot(): Promise<string | null> {
  const fromEnv = process.env.BRAINAPI_DEV_SOURCE?.trim();
  if (fromEnv && (await pathExists(path.join(fromEnv, "src", "config.py")))) {
    return fromEnv;
  }

  const distDir = path.dirname(fileURLToPath(import.meta.url));
  const candidate = path.resolve(distDir, "..", "..", "..");
  if (await pathExists(path.join(candidate, "src", "config.py"))) {
    return candidate;
  }
  return null;
}

export async function syncDevSourceIfAvailable(): Promise<boolean> {
  const root = await devWorkspaceRoot();
  if (!root) return false;

  const destRoot = sourcePath();
  let synced = false;
  for (const rel of SYNC_PATHS) {
    const src = path.join(root, rel);
    const dest = path.join(destRoot, rel);
    if (!(await pathExists(src))) continue;
    await cp(src, dest, { recursive: true, force: true });
    synced = true;
  }
  if (synced) {
    p.log.info(
      `Synced Python sources from ${pc.cyan(root)} → ${pc.cyan(destRoot)}`,
    );
  }
  return synced;
}
