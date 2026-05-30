import { access } from "node:fs/promises";
import path from "node:path";
import * as p from "@clack/prompts";
import pc from "picocolors";
import { sourcePath } from "./paths.js";
import { runInherit } from "./exec.js";

const CONSOLE_DIST = path.join(sourcePath(), "console", "dist", "index.html");

export async function consoleDistExists(): Promise<boolean> {
  try {
    await access(CONSOLE_DIST);
    return true;
  } catch {
    return false;
  }
}

export async function ensureConsoleBuilt(): Promise<void> {
  if (await consoleDistExists()) {
    return;
  }
  const consoleDir = path.join(sourcePath(), "console");
  const spinner = p.spinner();
  spinner.start("Building local console (first run)");
  try {
    const install = await runInherit("npm", ["install"], { cwd: consoleDir });
    if (!install.ok) {
      throw new Error("npm install failed in console/");
    }
    const build = await runInherit("npm", ["run", "build"], { cwd: consoleDir });
    if (!build.ok) {
      throw new Error("npm run build failed in console/");
    }
    spinner.stop("Local console built");
  } catch (err) {
    spinner.stop(pc.red("Console build failed"));
    p.log.warn(
      `Could not build console UI: ${
        err instanceof Error ? err.message : String(err)
      }. Run ${pc.cyan("make build-console")} manually.`,
    );
  }
}
