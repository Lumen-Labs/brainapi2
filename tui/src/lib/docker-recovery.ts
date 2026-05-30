import { access } from "node:fs/promises";
import * as p from "@clack/prompts";
import pc from "picocolors";
import { runInherit } from "./exec.js";
import { detectDocker, waitForDaemon, type DockerState } from "./docker.js";
import { detectCapabilities } from "./platform.js";
import { askConfirm, pickOne } from "./prompts.js";

interface DockerOption {
  value: string;
  label: string;
  command?: { bin: string; args: string[] };
  manualInstructions?: string;
  openUrl?: string;
}

const MAX_RETRIES = 3;

async function openUrl(url: string): Promise<void> {
  const cmd = process.platform === "darwin"
    ? { bin: "open", args: [url] }
    : process.platform === "win32"
      ? { bin: "cmd", args: ["/c", "start", "", url] }
      : { bin: "xdg-open", args: [url] };
  await runInherit(cmd.bin, cmd.args);
}

async function dockerDesktopExistsOnMac(): Promise<boolean> {
  try {
    await access("/Applications/Docker.app");
    return true;
  } catch {
    return false;
  }
}

async function buildOptions(): Promise<DockerOption[]> {
  const caps = await detectCapabilities();
  const options: DockerOption[] = [];

  if (caps.platform === "darwin") {
    if (caps.hasBrew) {
      options.push({
        value: "brew-cask",
        label: "Install Docker Desktop via Homebrew Cask (brew install --cask docker)",
        command: { bin: "brew", args: ["install", "--cask", "docker"] },
      });
    }
    options.push({
      value: "docker-download-mac",
      label: "Open Docker Desktop download page",
      openUrl: "https://www.docker.com/products/docker-desktop/",
    });
  } else if (caps.platform === "linux") {
    for (const pm of caps.linuxPackageManagers) {
      const args = pm === "pacman"
        ? ["sudo", pm, "-S", "--noconfirm", "docker", "docker-compose"]
        : pm === "apk"
          ? ["sudo", pm, "add", "docker", "docker-compose"]
          : pm === "zypper"
            ? ["sudo", pm, "install", "-y", "docker", "docker-compose"]
            : ["sudo", pm, "install", "-y", "docker.io", "docker-compose-plugin"];
      options.push({
        value: `pm-${pm}`,
        label: `Install via ${pm} (${args.slice(1).join(" ")})`,
        command: { bin: args[0]!, args: args.slice(1) },
      });
    }
    options.push({
      value: "docker-script",
      label: "Install via Docker's convenience script (curl -fsSL https://get.docker.com | sh)",
      manualInstructions:
        "curl -fsSL https://get.docker.com | sh\nsudo usermod -aG docker $USER\nnewgrp docker",
    });
  } else if (caps.platform === "win32") {
    if (caps.hasWinget) {
      options.push({
        value: "winget",
        label: "Install via winget (winget install Docker.DockerDesktop)",
        command: { bin: "winget", args: ["install", "Docker.DockerDesktop"] },
      });
    }
    if (caps.hasChoco) {
      options.push({
        value: "choco",
        label: "Install via Chocolatey (choco install docker-desktop)",
        command: { bin: "choco", args: ["install", "docker-desktop", "-y"] },
      });
    }
    options.push({
      value: "docker-download-win",
      label: "Open Docker Desktop download page",
      openUrl: "https://www.docker.com/products/docker-desktop/",
    });
  }

  options.push({
    value: "manual",
    label: "I'll install it manually — wait and retry",
  });
  options.push({
    value: "cancel",
    label: "Cancel — skip container start",
  });

  return options;
}

function manualInstructionsFor(): string {
  if (process.platform === "darwin") {
    return [
      "# Option A: Homebrew Cask",
      "brew install --cask docker",
      "",
      "# Option B: Download installer",
      "open https://www.docker.com/products/docker-desktop/",
    ].join("\n");
  }
  if (process.platform === "linux") {
    return [
      "# Debian/Ubuntu",
      "sudo apt-get install -y docker.io docker-compose-plugin",
      "",
      "# Fedora/RHEL",
      "sudo dnf install -y docker docker-compose",
      "",
      "# Add yourself to the docker group:",
      "sudo usermod -aG docker $USER",
      "newgrp docker",
      "",
      "# Or convenience script:",
      "curl -fsSL https://get.docker.com | sh",
    ].join("\n");
  }
  return [
    "winget install Docker.DockerDesktop",
    "",
    "# Or download:",
    "https://www.docker.com/products/docker-desktop/",
  ].join("\n");
}

async function handleMissing(): Promise<"installed" | "cancelled"> {
  let attempt = 0;
  while (attempt < MAX_RETRIES) {
    attempt += 1;
    p.log.warn(
      attempt === 1
        ? "Docker (or the compose plugin) was not found on your PATH."
        : `Still no Docker detected (attempt ${attempt}/${MAX_RETRIES}).`,
    );
    const options = await buildOptions();
    const choice = await pickOne<string>({
      message: "How would you like to install Docker?",
      options: options.map((o) => ({ value: o.value, label: o.label })),
    });
    if (choice === "cancel") return "cancelled";

    const selected = options.find((o) => o.value === choice);
    if (!selected) continue;

    if (selected.openUrl) {
      await openUrl(selected.openUrl);
    } else if (selected.command) {
      const cmdStr = `${selected.command.bin} ${selected.command.args.join(" ")}`;
      p.note(cmdStr, "Command to run");
      const confirmRun = await askConfirm({
        message: "Run this for me now?",
        initialValue: true,
      });
      if (confirmRun) {
        await runInherit(selected.command.bin, selected.command.args);
      }
    } else if (selected.value === "manual") {
      p.note(manualInstructionsFor(), "Manual install — copy and run");
    } else if (selected.manualInstructions) {
      p.note(selected.manualInstructions, "Copy and run");
    }

    await askConfirm({
      message: "Press enter when Docker is installed",
      initialValue: true,
    });

    const state = await detectDocker();
    if (state === "ok") return "installed";
    if (state === "daemon_down") {
      const recovered = await handleDaemonDown();
      if (recovered === "ok") return "installed";
      if (recovered === "cancelled") return "cancelled";
    }
  }
  return "cancelled";
}

async function handleDaemonDown(): Promise<"ok" | "cancelled"> {
  p.log.warn("Docker is installed but the daemon is not responding.");

  if (process.platform === "darwin" && (await dockerDesktopExistsOnMac())) {
    const launch = await askConfirm({
      message: "Launch Docker Desktop now?",
      initialValue: true,
    });
    if (launch) {
      await runInherit("open", ["-a", "Docker"]);
    }
  } else if (process.platform === "linux") {
    p.note(
      "sudo systemctl start docker",
      "Try starting the daemon",
    );
  }

  const spinner = p.spinner();
  spinner.start("Waiting for Docker daemon (up to 60s)");
  const ok = await waitForDaemon(60_000);
  if (ok) {
    spinner.stop(pc.green("Docker daemon is up."));
    return "ok";
  }
  spinner.stop(pc.yellow("Docker daemon did not start in time."));
  const retry = await askConfirm({
    message: "Start Docker manually and press enter to retry — or cancel.",
    initialValue: true,
  });
  if (!retry) return "cancelled";
  const state = await detectDocker();
  return state === "ok" ? "ok" : "cancelled";
}

export async function ensureDocker(): Promise<"ready" | "cancelled"> {
  const state = await detectDocker();
  if (state === "ok") return "ready";
  if (state === "missing") {
    const result = await handleMissing();
    return result === "installed" ? "ready" : "cancelled";
  }
  const result = await handleDaemonDown();
  return result === "ok" ? "ready" : "cancelled";
}

export function describeDockerState(state: DockerState): string {
  switch (state) {
    case "ok":
      return "Docker is installed and the daemon is running";
    case "missing":
      return "Docker is not installed (or compose plugin missing)";
    case "daemon_down":
      return "Docker is installed but the daemon is not responding";
  }
}
