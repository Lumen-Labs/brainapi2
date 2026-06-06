import * as p from "@clack/prompts";
import pc from "picocolors";
import { installRegistryPlugin, runPluginCli, uninstallPlugin } from "../lib/plugins.js";
import { readState } from "../lib/state.js";

export type PluginsSubcommand = "install" | "uninstall" | "list" | "info" | "update";

export interface PluginsOptions {
  subcommand?: PluginsSubcommand;
  name?: string;
  version?: string;
  remote?: boolean;
}

async function ensureInstalled(): Promise<void> {
  const state = await readState();
  if (!state || !state.envWritten) {
    p.cancel(
      "No brainapi install detected. Run " + pc.cyan("brainapi init") + " first.",
    );
    process.exit(1);
  }
}

export async function runPlugins(opts: PluginsOptions): Promise<void> {
  await ensureInstalled();

  const subcommand = opts.subcommand;
  if (!subcommand) {
    p.log.error(
      "Usage: brainapi plugins <install|uninstall|list|info|update> [name] [--version <ver>] [--remote]",
    );
    process.exit(1);
  }

  p.intro(pc.bgCyan(pc.black(" brainapi ")) + " " + pc.dim(`plugins ${subcommand}`));

  switch (subcommand) {
    case "install": {
      if (!opts.name) {
        p.log.error("Plugin name is required. Example: brainapi plugins install chatbot");
        process.exit(1);
      }
      await installRegistryPlugin(opts.name, opts.version);
      p.outro(`Installed ${opts.name}`);
      return;
    }
    case "uninstall": {
      if (!opts.name) {
        p.log.error("Plugin name is required. Example: brainapi plugins uninstall chatbot");
        process.exit(1);
      }
      await uninstallPlugin(opts.name);
      p.outro(`Uninstalled ${opts.name}`);
      return;
    }
    case "list": {
      const args = ["list"];
      if (opts.remote) args.push("--remote");
      await runPluginCli(args);
      return;
    }
    case "info": {
      if (!opts.name) {
        p.log.error("Plugin name is required. Example: brainapi plugins info chatbot");
        process.exit(1);
      }
      await runPluginCli(["info", opts.name]);
      return;
    }
    case "update": {
      if (!opts.name) {
        p.log.error("Plugin name is required. Example: brainapi plugins update chatbot");
        process.exit(1);
      }
      await runPluginCli(["update", opts.name]);
      return;
    }
  }
}
