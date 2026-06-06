import * as p from "@clack/prompts";
import pc from "picocolors";
import { runInit } from "./commands/init.js";
import { runStart } from "./commands/start.js";
import { runConfig } from "./commands/config.js";
import { runDoctor } from "./commands/doctor.js";
import { runUpdate } from "./commands/update.js";
import { runPlugins, type PluginsOptions, type PluginsSubcommand } from "./commands/plugins.js";
import { readState } from "./lib/state.js";
import { isPipelineMode, type PipelineMode } from "./types.js";

type Command = "init" | "start" | "config" | "doctor" | "update" | "plugins" | "help";

interface ParsedArgs {
  command: Command;
  repo?: string;
  branch?: string;
  force?: boolean;
  noApi?: boolean;
  noMcp?: boolean;
  noWorker?: boolean;
  noServices?: boolean;
  only?: Array<"api" | "mcp" | "worker" | "services">;
  pipeline?: PipelineMode;
  invalidPipeline?: string;
  plugins?: PluginsOptions;
  unknown: string[];
}

const PLUGIN_SUBCOMMANDS = new Set<PluginsSubcommand>([
  "install",
  "uninstall",
  "list",
  "info",
  "update",
]);

function isPluginsSubcommand(value: string): value is PluginsSubcommand {
  return PLUGIN_SUBCOMMANDS.has(value as PluginsSubcommand);
}

function parseArgs(argv: string[]): ParsedArgs {
  const args = [...argv];
  let command: Command = "help";
  const unknown: string[] = [];
  const out: ParsedArgs = { command, unknown };

  while (args.length > 0) {
    const arg = args.shift()!;
    if (arg === "--repo" || arg === "-r") {
      out.repo = args.shift();
    } else if (arg === "--branch" || arg === "-b") {
      out.branch = args.shift();
    } else if (arg === "--force" || arg === "-f") {
      out.force = true;
    } else if (arg === "--no-api") {
      out.noApi = true;
    } else if (arg === "--no-mcp") {
      out.noMcp = true;
    } else if (arg === "--no-worker") {
      out.noWorker = true;
    } else if (arg === "--no-services") {
      out.noServices = true;
    } else if (arg === "--pipeline") {
      const raw = (args.shift() ?? "").trim().toLowerCase();
      if (isPipelineMode(raw)) {
        out.pipeline = raw;
      } else {
        out.invalidPipeline = raw || "(missing)";
      }
    } else if (arg === "--only") {
      const value = args.shift() ?? "";
      const parts = value
        .split(",")
        .map((s) => s.trim().toLowerCase())
        .filter((s) =>
          s === "api" || s === "mcp" || s === "worker" || s === "services",
        ) as Array<"api" | "mcp" | "worker" | "services">;
      if (parts.length === 0) {
        unknown.push(`--only ${value}`);
      } else {
        out.only = parts;
      }
    } else if (arg === "--help" || arg === "-h") {
      out.command = "help";
    } else if (arg === "--version" || arg === "-v") {
      printVersion();
      process.exit(0);
    } else if (out.command === "help" && !arg.startsWith("-")) {
      switch (arg) {
        case "init":
        case "start":
        case "config":
        case "doctor":
        case "update":
        case "plugins":
        case "help":
          out.command = arg;
          break;
        default:
          unknown.push(arg);
      }
    } else if (out.command === "plugins") {
      if (!out.plugins) out.plugins = {};
      const plugins = out.plugins;
      if (!plugins.subcommand && !arg.startsWith("-")) {
        if (isPluginsSubcommand(arg)) {
          plugins.subcommand = arg;
        } else {
          unknown.push(arg);
        }
      } else if (arg === "--version") {
        plugins.version = args.shift();
      } else if (arg === "--remote" || arg === "-r") {
        plugins.remote = true;
      } else if (
        plugins.subcommand &&
        plugins.subcommand !== "list" &&
        !plugins.name &&
        !arg.startsWith("-")
      ) {
        plugins.name = arg;
      } else {
        unknown.push(arg);
      }
    } else {
      unknown.push(arg);
    }
  }
  return out;
}

function printHelp(): void {
  const lines = [
    pc.bold("brainapi") + " — interactive installer for BrainAPI",
    "",
    pc.dim("Usage:") + " brainapi <command> [options]",
    "",
    pc.bold("Commands:"),
    "  init               Clone repo, set up venv, run interactive config, optional containers",
    "  start [...]        Start backing services + API + MCP + celery worker (Ctrl-C to stop all)",
    "  config             Re-run the interactive flow and rewrite .env",
    "  doctor             Check Python, Docker, Ollama, GCP credentials, services",
    "  update             Fetch latest source + reinstall Python dependencies",
    "  plugins <cmd>      Manage plugins (install, uninstall, list, info, update)",
    "  help               Show this message",
    "",
    pc.bold("Options:"),
    "  -r, --repo <url>           Git repo to clone (default $BRAINAPI_REPO_URL or upstream)",
    "  -b, --branch <ref>         Branch to checkout (default $BRAINAPI_BRANCH or 'main')",
    "  -f, --force                Re-run init even if state exists",
    "      --no-services          For 'start': skip bringing backing services up",
    "      --no-api               For 'start': skip the API server",
    "      --no-mcp               For 'start': skip the MCP server",
    "      --no-worker            For 'start': skip the celery worker",
    "      --only api,mcp,worker  For 'start': run only the listed processes (shortcut)",
    "      --pipeline accurate|lightweight   Set PIPELINE_MODE in ~/.brainapi/source/.env before start",
    "  -v, --version              Print the CLI version",
    "  -h, --help                 Show this message",
    "",
    pc.dim("Environment overrides: BRAINAPI_HOME, BRAINAPI_REPO_URL, BRAINAPI_BRANCH"),
  ];
  console.log(lines.join("\n"));
}

function printVersion(): void {
  console.log("brainapi tui CLI");
}

async function autoInitIfNeeded(command: Command): Promise<Command> {
  if (command === "help" || command === "init") return command;
  const state = await readState();
  if (state) return command;
  p.log.info(
    "No brainapi install found yet — running " + pc.cyan("brainapi init") + " first.",
  );
  return "init";
}

async function main(): Promise<void> {
  const parsed = parseArgs(process.argv.slice(2));
  if (parsed.invalidPipeline !== undefined) {
    p.log.error(
      `Invalid --pipeline value "${parsed.invalidPipeline}". Use accurate or lightweight.`,
    );
    process.exit(1);
  }
  if (parsed.unknown.length > 0 && parsed.command !== "help") {
    p.log.warn("Unknown argument(s): " + parsed.unknown.join(", "));
  }

  let cmd = parsed.command;
  if (cmd === "help") {
    if (parsed.unknown.length === 0 && process.argv.slice(2).length === 0) {
      cmd = await autoInitIfNeeded("init");
    } else {
      printHelp();
      return;
    }
  } else {
    cmd = await autoInitIfNeeded(cmd);
  }

  try {
    switch (cmd) {
      case "init":
        await runInit({
          repoUrl: parsed.repo,
          branch: parsed.branch,
          force: parsed.force,
        });
        return;
      case "start": {
        const startOpts = parsed.only
          ? {
              api: parsed.only.includes("api"),
              mcp: parsed.only.includes("mcp"),
              worker: parsed.only.includes("worker"),
              services: parsed.only.includes("services") || true,
            }
          : {
              api: !parsed.noApi,
              mcp: !parsed.noMcp,
              worker: !parsed.noWorker,
              services: !parsed.noServices,
            };
        if (parsed.only) {
          startOpts.services = parsed.only.includes("services");
        }
        await runStart({
          ...startOpts,
          pipelineMode: parsed.pipeline,
        });
        return;
      }
      case "config":
        await runConfig();
        return;
      case "doctor":
        await runDoctor();
        return;
      case "update":
        await runUpdate();
        return;
      case "plugins":
        await runPlugins(parsed.plugins ?? {});
        return;
      case "help":
        printHelp();
        return;
    }
  } catch (err) {
    p.log.error(err instanceof Error ? err.message : String(err));
    process.exit(1);
  }
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
