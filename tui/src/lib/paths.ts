import { homedir, platform as osPlatform } from "node:os";
import path from "node:path";
import type { Platform } from "../types.js";

export function brainapiHome(): string {
  return process.env.BRAINAPI_HOME ?? path.join(homedir(), ".brainapi");
}

export function sourcePath(): string {
  return path.join(brainapiHome(), "source");
}

export function venvPath(): string {
  return path.join(sourcePath(), ".venv");
}

export function envFilePath(): string {
  return path.join(sourcePath(), ".env");
}

export function stateFilePath(): string {
  return path.join(brainapiHome(), "state.json");
}

export function venvBin(name: string): string {
  if (currentPlatform() === "win32") {
    return path.join(venvPath(), "Scripts", `${name}.exe`);
  }
  return path.join(venvPath(), "bin", name);
}

export function currentPlatform(): Platform {
  const p = osPlatform();
  if (p === "darwin" || p === "linux" || p === "win32") {
    return p;
  }
  return "linux";
}
