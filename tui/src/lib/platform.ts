import { currentPlatform } from "./paths.js";
import { which } from "./exec.js";
import type { Platform } from "../types.js";

export type LinuxPackageManager = "apt-get" | "dnf" | "pacman" | "apk" | "zypper";

export interface PlatformCapabilities {
  platform: Platform;
  hasBrew: boolean;
  hasPyenv: boolean;
  hasWinget: boolean;
  hasChoco: boolean;
  linuxPackageManagers: LinuxPackageManager[];
}

const LINUX_PMS: LinuxPackageManager[] = [
  "apt-get",
  "dnf",
  "pacman",
  "apk",
  "zypper",
];

export async function detectCapabilities(): Promise<PlatformCapabilities> {
  const platform = currentPlatform();
  const [hasBrew, hasPyenv, hasWinget, hasChoco] = await Promise.all([
    platform === "darwin" || platform === "linux" ? which("brew") : Promise.resolve(false),
    which("pyenv"),
    platform === "win32" ? which("winget") : Promise.resolve(false),
    platform === "win32" ? which("choco") : Promise.resolve(false),
  ]);
  const linuxPackageManagers: LinuxPackageManager[] = [];
  if (platform === "linux") {
    for (const pm of LINUX_PMS) {
      if (await which(pm)) linuxPackageManagers.push(pm);
    }
  }
  return {
    platform,
    hasBrew,
    hasPyenv,
    hasWinget,
    hasChoco,
    linuxPackageManagers,
  };
}
