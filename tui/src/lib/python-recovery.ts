import * as p from "@clack/prompts";
import pc from "picocolors";
import { runInherit } from "./exec.js";
import { detectCapabilities } from "./platform.js";
import { detectPython, type DetectedPython } from "./python.js";
import { askConfirm, pickOne } from "./prompts.js";

interface PythonOption {
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

async function buildOptions(): Promise<PythonOption[]> {
  const caps = await detectCapabilities();
  const options: PythonOption[] = [];

  if (caps.platform === "darwin") {
    if (caps.hasBrew) {
      options.push({
        value: "brew",
        label: "Install via Homebrew (brew install python@3.12)",
        command: { bin: "brew", args: ["install", "python@3.12"] },
      });
    }
    options.push({
      value: "pyenv-mac",
      label: caps.hasPyenv
        ? "Install via pyenv (pyenv install 3.12)"
        : "Install pyenv + Python 3.12 (recommended for multiple versions)",
      manualInstructions: caps.hasPyenv
        ? "pyenv install 3.12.7\npyenv global 3.12.7"
        : "brew install pyenv\npyenv install 3.12.7\npyenv global 3.12.7",
    });
    options.push({
      value: "python-org-mac",
      label: "Open python.org downloads in browser",
      openUrl: "https://www.python.org/downloads/macos/",
    });
  } else if (caps.platform === "linux") {
    for (const pm of caps.linuxPackageManagers) {
      const installArgs = pm === "pacman"
        ? ["sudo", pm, "-S", "--noconfirm", "python"]
        : pm === "apk"
          ? ["sudo", pm, "add", "python3", "py3-pip"]
          : ["sudo", pm, "install", "-y", "python3.12", "python3.12-venv", "python3-pip"];
      options.push({
        value: `pm-${pm}`,
        label: `Install via ${pm} (${installArgs.slice(1).join(" ")})`,
        command: { bin: installArgs[0]!, args: installArgs.slice(1) },
      });
    }
    if (caps.hasPyenv) {
      options.push({
        value: "pyenv-linux",
        label: "Install via pyenv (pyenv install 3.12)",
        manualInstructions: "pyenv install 3.12.7\npyenv global 3.12.7",
      });
    }
    options.push({
      value: "python-org-linux",
      label: "Open python.org downloads in browser",
      openUrl: "https://www.python.org/downloads/source/",
    });
  } else if (caps.platform === "win32") {
    if (caps.hasWinget) {
      options.push({
        value: "winget",
        label: "Install via winget (winget install Python.Python.3.12)",
        command: { bin: "winget", args: ["install", "Python.Python.3.12"] },
      });
    }
    if (caps.hasChoco) {
      options.push({
        value: "choco",
        label: "Install via Chocolatey (choco install python --version=3.12)",
        command: { bin: "choco", args: ["install", "python", "--version=3.12"] },
      });
    }
    options.push({
      value: "python-org-win",
      label: "Open python.org downloads in browser",
      openUrl: "https://www.python.org/downloads/windows/",
    });
  }

  options.push({
    value: "manual",
    label: "I'll install it manually — wait and retry",
  });
  options.push({
    value: "cancel",
    label: "Cancel — I'll re-run `brainapi init` later",
  });

  return options;
}

function manualInstructionsFor(): string {
  if (process.platform === "darwin") {
    return [
      "# Option A: Homebrew",
      "brew install python@3.12",
      "",
      "# Option B: pyenv (recommended if you juggle Python versions)",
      "brew install pyenv",
      "pyenv install 3.12.7",
      "pyenv global 3.12.7",
      "",
      "# Option C: Download installer",
      "open https://www.python.org/downloads/macos/",
    ].join("\n");
  }
  if (process.platform === "linux") {
    return [
      "# Debian/Ubuntu",
      "sudo apt-get update && sudo apt-get install -y python3.12 python3.12-venv python3-pip",
      "",
      "# Fedora/RHEL",
      "sudo dnf install -y python3.12",
      "",
      "# Arch",
      "sudo pacman -S python",
      "",
      "# Or via pyenv",
      "curl https://pyenv.run | bash",
      "pyenv install 3.12.7 && pyenv global 3.12.7",
    ].join("\n");
  }
  return [
    "winget install Python.Python.3.12",
    "",
    "# Or download:",
    "https://www.python.org/downloads/windows/",
  ].join("\n");
}

export async function recoverPython(): Promise<DetectedPython> {
  let attempt = 0;
  while (attempt < MAX_RETRIES) {
    attempt += 1;
    p.log.warn(
      attempt === 1
        ? "Python 3.11+ was not found on your PATH."
        : `Still no Python 3.11+ detected (attempt ${attempt}/${MAX_RETRIES}).`,
    );
    const options = await buildOptions();
    const choice = await pickOne<string>({
      message: "How would you like to install Python?",
      options: options.map((o) => ({ value: o.value, label: o.label })),
    });
    const selected = options.find((o) => o.value === choice);
    if (!selected) continue;

    if (selected.value === "cancel") {
      p.cancel("Setup cancelled. Re-run `brainapi init` once Python is installed.");
      process.exit(1);
    }

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
      message: "Press enter when Python is installed and on your PATH",
      initialValue: true,
    });

    const detected = await detectPython();
    if (detected) {
      p.log.success(
        `Found ${pc.cyan(detected.bin)} (Python ${detected.version.join(".")})`,
      );
      return detected;
    }
  }

  p.cancel(
    "Could not detect Python 3.11+ after several attempts. Open a fresh shell so PATH refreshes, then re-run `brainapi init`. Any cloned source is preserved.",
  );
  process.exit(1);
}

export async function ensurePython(): Promise<DetectedPython> {
  const detected = await detectPython();
  if (detected) return detected;
  return recoverPython();
}
