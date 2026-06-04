import { mkdir, readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import { brainapiHome, sourcePath, stateFilePath } from "./paths.js";
import { DEFAULT_BRANCH, DEFAULT_REPO_URL } from "../constants.js";
import type { InstallState } from "../types.js";

export function defaultState(): InstallState {
  return {
    cloned: false,
    venvCreated: false,
    depsInstalled: false,
    envWritten: false,
    containersStarted: false,
    selectedServices: null,
    installedPlugins: null,
    servicesRuntime: null,
    sourcePath: sourcePath(),
    repoUrl: process.env.BRAINAPI_REPO_URL ?? DEFAULT_REPO_URL,
    branch: process.env.BRAINAPI_BRANCH ?? DEFAULT_BRANCH,
  };
}

export async function readState(): Promise<InstallState | null> {
  try {
    const raw = await readFile(stateFilePath(), "utf8");
    const parsed = JSON.parse(raw) as Partial<InstallState>;
    return { ...defaultState(), ...parsed };
  } catch (err) {
    if ((err as NodeJS.ErrnoException).code === "ENOENT") return null;
    throw err;
  }
}

export async function writeState(state: InstallState): Promise<void> {
  await mkdir(path.dirname(stateFilePath()), { recursive: true });
  await writeFile(stateFilePath(), JSON.stringify(state, null, 2), "utf8");
}

export async function updateState(
  patch: Partial<InstallState>,
): Promise<InstallState> {
  const current = (await readState()) ?? defaultState();
  const next = { ...current, ...patch };
  await writeState(next);
  return next;
}

export async function ensureStateDir(): Promise<void> {
  await mkdir(brainapiHome(), { recursive: true });
}
