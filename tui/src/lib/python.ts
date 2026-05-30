import { access, mkdir, readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import {
  MIN_PYTHON_VERSION,
  PYTHON_CANDIDATES,
} from "../constants.js";
import { sourcePath, venvBin, venvPath } from "./paths.js";
import { runQuiet, runInherit } from "./exec.js";
import { updateState } from "./state.js";
import { INSTALL_EXTRAS_SCRIPT as EMBEDDED_INSTALL_EXTRAS_SCRIPT } from "../embedded/install-extras-script.js";

const INSTALL_EXTRAS_SCRIPT_REL = "scripts/install_extras.py";

async function ensureInstallExtrasScript(): Promise<void> {
  const target = path.join(sourcePath(), INSTALL_EXTRAS_SCRIPT_REL);
  let existing: string | null = null;
  try {
    existing = await readFile(target, "utf8");
  } catch {
    existing = null;
  }
  if (existing === EMBEDDED_INSTALL_EXTRAS_SCRIPT) return;
  await mkdir(path.dirname(target), { recursive: true });
  await writeFile(target, EMBEDDED_INSTALL_EXTRAS_SCRIPT, "utf8");
}

export interface DetectedPython {
  bin: string;
  version: [number, number, number];
}

function meetsMinimum(version: [number, number, number]): boolean {
  const [major, minor] = version;
  const [minMajor, minMinor] = MIN_PYTHON_VERSION;
  if (major > minMajor) return true;
  if (major < minMajor) return false;
  return minor >= minMinor;
}

export async function detectPython(): Promise<DetectedPython | null> {
  for (const candidate of PYTHON_CANDIDATES) {
    const result = await runQuiet(candidate, ["--version"]);
    if (!result.ok) continue;
    const output = result.stdout || result.stderr;
    const match = output.match(/Python (\d+)\.(\d+)\.(\d+)/);
    if (!match) continue;
    const major = Number(match[1]);
    const minor = Number(match[2]);
    const patch = Number(match[3]);
    const version: [number, number, number] = [major, minor, patch];
    if (meetsMinimum(version)) {
      return { bin: candidate, version };
    }
  }
  return null;
}

export async function venvExists(): Promise<boolean> {
  try {
    await access(path.join(venvPath(), "pyvenv.cfg"));
    return true;
  } catch {
    return false;
  }
}

export async function createVenv(pythonBin: string): Promise<void> {
  const result = await runInherit(pythonBin, ["-m", "venv", venvPath()]);
  if (!result.ok) {
    throw new Error(`python -m venv failed with exit code ${result.exitCode}`);
  }
  await updateState({ venvCreated: true });
}

export interface InstallDepsOptions {
  extras?: string[];
  skipBase?: boolean;
}

function scriptArgs(opts: InstallDepsOptions, withUpgrade: boolean): string[] {
  const args: string[] = [INSTALL_EXTRAS_SCRIPT_REL];
  if (!opts.skipBase) args.push("--include-base");
  if (withUpgrade) args.push("--upgrade-pip");
  for (const extra of opts.extras ?? []) {
    args.push(extra);
  }
  return args;
}

export async function installDeps(options: InstallDepsOptions = {}): Promise<void> {
  await ensureInstallExtrasScript();
  const pythonBin = venvBin("python");
  const args = scriptArgs(options, true);
  const result = await runInherit(pythonBin, args, { cwd: sourcePath() });
  if (!result.ok) {
    throw new Error(`scripts/install_extras.py failed with exit code ${result.exitCode}`);
  }
  if (!options.skipBase) {
    await updateState({ depsInstalled: true });
  }
}

export async function installExtras(extras: string[]): Promise<void> {
  if (extras.length === 0) return;
  await installDeps({ extras, skipBase: true });
}

export async function missingExtras(extras: string[]): Promise<string[]> {
  if (extras.length === 0) return [];
  await ensureInstallExtrasScript();
  const pythonBin = venvBin("python");
  const result = await runQuiet(
    pythonBin,
    [INSTALL_EXTRAS_SCRIPT_REL, "--check", ...extras],
    { cwd: sourcePath() },
  );
  if (result.ok) return [];
  return [...extras];
}

export async function missingRequiredPythonDeps(): Promise<boolean> {
  await ensureInstallExtrasScript();
  const pythonBin = venvBin("python");
  const result = await runQuiet(
    pythonBin,
    [INSTALL_EXTRAS_SCRIPT_REL, "--check"],
    { cwd: sourcePath() },
  );
  return !result.ok;
}
