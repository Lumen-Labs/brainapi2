import { readFile, writeFile, access } from "node:fs/promises";
import path from "node:path";
import { ENV_KEYS } from "../constants.js";
import type { PipelineMode } from "../types.js";
import { envFilePath, sourcePath } from "./paths.js";

const QUOTE_NEEDED = /[\s#"'=]/;

function escapeValue(value: string): string {
  if (value === "") return '""';
  if (!QUOTE_NEEDED.test(value)) return value;
  return `"${value.replace(/\\/g, "\\\\").replace(/"/g, '\\"')}"`;
}

function parseLine(
  line: string,
): { key: string; rawValue: string } | null {
  const trimmed = line.trim();
  if (!trimmed || trimmed.startsWith("#")) return null;
  const idx = line.indexOf("=");
  if (idx < 0) return null;
  return {
    key: line.slice(0, idx).trim(),
    rawValue: line.slice(idx + 1),
  };
}

export interface EnvFile {
  /** Original line content in file order. */
  lines: string[];
  /** Map of key -> index into `lines` for quick lookup/update. */
  keyIndex: Map<string, number>;
}

export async function readEnvFile(filePath: string): Promise<EnvFile> {
  let text = "";
  try {
    text = await readFile(filePath, "utf8");
  } catch (err) {
    if ((err as NodeJS.ErrnoException).code !== "ENOENT") throw err;
  }
  const lines = text.length === 0 ? [] : text.split(/\r?\n/);
  if (lines.length > 0 && lines[lines.length - 1] === "") lines.pop();
  const keyIndex = new Map<string, number>();
  lines.forEach((line, i) => {
    const parsed = parseLine(line);
    if (parsed) keyIndex.set(parsed.key, i);
  });
  return { lines, keyIndex };
}

export function setEnvValue(env: EnvFile, key: string, value: string): void {
  const newLine = `${key}=${escapeValue(value)}`;
  const existing = env.keyIndex.get(key);
  if (existing !== undefined) {
    env.lines[existing] = newLine;
  } else {
    env.lines.push(newLine);
    env.keyIndex.set(key, env.lines.length - 1);
  }
}

export function removeEnvValue(env: EnvFile, key: string): void {
  const existing = env.keyIndex.get(key);
  if (existing === undefined) return;
  env.lines.splice(existing, 1);
  env.keyIndex.clear();
  env.lines.forEach((line, i) => {
    const parsed = parseLine(line);
    if (parsed) env.keyIndex.set(parsed.key, i);
  });
}

export function applyEnvValues(
  env: EnvFile,
  values: Record<string, string | number | undefined>,
): void {
  for (const [key, value] of Object.entries(values)) {
    if (value === undefined) continue;
    setEnvValue(env, key, String(value));
  }
}

export async function writeEnvFile(
  filePath: string,
  env: EnvFile,
): Promise<void> {
  const content = env.lines.join("\n") + "\n";
  await writeFile(filePath, content, "utf8");
}

export async function loadOrSeedEnv(): Promise<EnvFile> {
  const targetPath = envFilePath();
  try {
    await access(targetPath);
    return await readEnvFile(targetPath);
  } catch {
    const examplePath = path.join(sourcePath(), ".env.example");
    return await readEnvFile(examplePath);
  }
}

export async function saveEnv(env: EnvFile): Promise<void> {
  await writeEnvFile(envFilePath(), env);
}

export async function setPipelineMode(mode: PipelineMode): Promise<void> {
  const targetPath = envFilePath();
  const env = await readEnvFile(targetPath);
  setEnvValue(env, ENV_KEYS.pipelineMode, mode);
  await writeEnvFile(targetPath, env);
}

export function getEnvValue(env: EnvFile, key: string): string | undefined {
  const idx = env.keyIndex.get(key);
  if (idx === undefined) return undefined;
  const raw = (env.lines[idx] ?? "").split("=").slice(1).join("=");
  return raw.trim().replace(/^"|"$/g, "");
}

export function resolvedModelsMode(env: EnvFile): "local" | "remote" {
  const small = getEnvValue(env, "LLM_SMALL_PROVIDER");
  const large = getEnvValue(env, "LLM_LARGE_PROVIDER");
  const embeddings = getEnvValue(env, "EMBEDDINGS_PROVIDER");
  if (small === "ollama" && large === "ollama" && embeddings === "ollama") {
    return "local";
  }
  const mode = getEnvValue(env, "MODELS_MODE");
  if (mode === "local" || mode === "remote") return mode;
  return "remote";
}

export function envFileToProcessEnv(env: EnvFile): NodeJS.ProcessEnv {
  const out: Record<string, string> = {};
  for (const [key, idx] of env.keyIndex.entries()) {
    const value = getEnvValue(env, key);
    if (value !== undefined) out[key] = value;
  }
  out.MODELS_MODE = resolvedModelsMode(env);
  return { ...process.env, ...out };
}
