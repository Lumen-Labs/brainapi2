import { access } from "node:fs/promises";
import path from "node:path";
import { sourcePath } from "./paths.js";
import { runInherit, runQuiet } from "./exec.js";
import {
  SERVICE_COMPOSE_FILES,
  type ServiceName,
} from "../constants.js";

export type DockerState = "ok" | "missing" | "daemon_down";

export async function detectDocker(): Promise<DockerState> {
  const cli = await runQuiet("docker", ["--version"]);
  if (!cli.ok) return "missing";
  const compose = await runQuiet("docker", ["compose", "version"]);
  if (!compose.ok) {
    const legacy = await runQuiet("docker-compose", ["--version"]);
    if (!legacy.ok) return "missing";
  }
  const info = await runQuiet("docker", ["info"]);
  if (!info.ok) return "daemon_down";
  return "ok";
}

export function composeFileFor(service: ServiceName): string {
  return path.join(sourcePath(), SERVICE_COMPOSE_FILES[service]);
}

export async function composeFileExists(service: ServiceName): Promise<boolean> {
  try {
    await access(composeFileFor(service));
    return true;
  } catch {
    return false;
  }
}

export async function composeUp(service: ServiceName): Promise<void> {
  const file = composeFileFor(service);
  const result = await runInherit("docker", [
    "compose",
    "-f",
    file,
    "up",
    "-d",
  ]);
  if (!result.ok) {
    throw new Error(`docker compose up failed for ${service}`);
  }
}

export async function composeDown(service: ServiceName): Promise<void> {
  const file = composeFileFor(service);
  await runInherit("docker", ["compose", "-f", file, "down"]);
}

export async function waitForDaemon(timeoutMs = 60_000): Promise<boolean> {
  const start = Date.now();
  while (Date.now() - start < timeoutMs) {
    const info = await runQuiet("docker", ["info"]);
    if (info.ok) return true;
    await new Promise((resolve) => setTimeout(resolve, 2000));
  }
  return false;
}
