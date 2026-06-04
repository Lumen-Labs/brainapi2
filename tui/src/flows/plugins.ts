import * as p from "@clack/prompts";
import { askConfirm, askText, isPromptBack, pickOne, PROMPT_BACK } from "../lib/prompts.js";
import type { PluginChoice, PluginSource } from "../types.js";
import { listInstalledPlugins, searchPlugins } from "../lib/plugins.js";

interface AskPluginsOptions {
  allowBack?: boolean;
  backHint?: string;
  initial?: PluginChoice[];
}

function pluginLabel(plugin: {
  name: string;
  version: string;
  source: PluginSource;
  description: string;
}): string {
  const sourceTag = plugin.source === "registry" ? "registry" : "local";
  return `${plugin.name} (${sourceTag})`;
}

export async function askPlugins(
  opts: AskPluginsOptions = {},
): Promise<PluginChoice[] | typeof PROMPT_BACK> {
  const selections = new Map<string, PluginChoice>();
  for (const plugin of opts.initial ?? []) {
    selections.set(`${plugin.source}:${plugin.name}:${plugin.path ?? ""}`, plugin);
  }

  while (true) {
    const query = await askText({
      message: "Search plugin by name (or leave empty to continue)",
      placeholder: "chatbot",
    });
    if (!query.trim()) {
      if (opts.allowBack && selections.size === 0) {
        const shouldGoBack = await pickOne<"continue" | "back">({
          message: "No plugins selected. Continue or go back?",
          options: [
            { value: "continue", label: "Continue" },
            { value: "back", label: "Back", hint: opts.backHint },
          ],
          initialValue: "continue",
        });
        if (shouldGoBack === "back") {
          return PROMPT_BACK;
        }
      }
      return [...selections.values()];
    }

    const results = await searchPlugins(query);
    if (results.length === 0) {
      p.log.warn(`No plugin found for "${query}".`);
    } else {
      const choice = await pickOne<string>({
        message: "Select a plugin to add",
        options: results.map((plugin, index) => ({
          value: String(index),
          label: pluginLabel(plugin),
          hint: plugin.description || `${plugin.version}`,
        })),
        allowBack: true,
        backHint: opts.backHint,
      });
      if (isPromptBack(choice)) {
        if (opts.allowBack) {
          return PROMPT_BACK;
        }
        continue;
      }
      const selected = results[Number(choice)];
      if (!selected) {
        p.log.warn("Invalid plugin selection.");
        continue;
      }
      const confirmed = await askConfirm({
        message: `Add ${selected.name} from ${selected.source}?`,
        initialValue: true,
      });
      if (confirmed) {
        const key = `${selected.source}:${selected.name}:${selected.path ?? ""}`;
        selections.set(key, {
          name: selected.name,
          source: selected.source,
          version: selected.version,
          path: selected.path,
        });
        p.log.success(`Added ${selected.name}.`);
      }
    }

    const addMore = await askConfirm({
      message: "Add another plugin?",
      initialValue: true,
    });
    if (!addMore) {
      return [...selections.values()];
    }
  }
}

export async function askPluginUninstall(
  opts: { allowBack?: boolean; backHint?: string } = {},
): Promise<string | typeof PROMPT_BACK | null> {
  const installed = await listInstalledPlugins();
  if (installed.length === 0) {
    p.log.info("No installed plugins found.");
    return null;
  }
  const options = installed.map((plugin) => ({
    value: plugin.name,
    label: plugin.name,
    hint: plugin.version,
  }));
  const choice = opts.allowBack
    ? await pickOne<string>({
        message: "Select plugin to uninstall",
        options,
        allowBack: true,
        backHint: opts.backHint,
      })
    : await pickOne<string>({
        message: "Select plugin to uninstall",
        options,
      });
  if (isPromptBack(choice)) {
    return PROMPT_BACK;
  }
  return choice;
}
