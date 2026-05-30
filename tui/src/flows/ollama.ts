import { access } from "node:fs/promises";
import * as p from "@clack/prompts";
import pc from "picocolors";
import {
  OLLAMA_DEFAULT_HOST,
  OLLAMA_DEFAULT_LARGE_MODEL,
  OLLAMA_DEFAULT_PORT,
  OLLAMA_DEFAULT_SMALL_MODEL,
  EMBEDDINGS_DEFAULTS,
} from "../constants.js";
import {
  hasOllamaModel,
  isOllamaRunning,
  pullOllamaModel,
  type OllamaTarget,
} from "../lib/ollama.js";
import { askConfirm, askText, pickOne } from "../lib/prompts.js";
import type { OllamaChoices } from "../types.js";

async function waitForRunning(target: OllamaTarget): Promise<void> {
  while (!(await isOllamaRunning(target))) {
    p.log.warn(
      `Ollama did not respond at ${pc.cyan(`http://${target.host}:${target.port}`)}.`,
    );
    p.note(
      [
        "Start Ollama in a separate terminal:",
        "  ollama serve",
        "Or launch the Ollama desktop app.",
      ].join("\n"),
      "Ollama not running",
    );
    await askConfirm({
      message: "Press enter when Ollama is running",
      initialValue: true,
    });
  }
  p.log.success("Ollama is up.");
}

type MissingModelAction = "pull" | "change";

async function waitForModel(
  target: OllamaTarget,
  initialModelName: string,
  role: "small" | "large",
): Promise<string> {
  let modelName = initialModelName;
  while (!(await hasOllamaModel(target, modelName))) {
    p.log.warn(`Model ${pc.cyan(modelName)} is not available locally.`);

    const action = await pickOne<MissingModelAction>({
      message: `How do you want to provide the ${role} model?`,
      options: [
        {
          value: "pull",
          label: "Pull this model now",
          hint: `runs: ollama pull ${modelName}`,
        },
        {
          value: "change",
          label: "Use a different model name",
        },
      ],
      initialValue: "pull",
    });

    if (action === "pull") {
      const spinner = p.spinner();
      spinner.start(`Pulling ${modelName}`);
      const result = await pullOllamaModel(modelName);
      if (result.ok) {
        spinner.stop(`Pulled ${pc.cyan(modelName)}`);
        if (await hasOllamaModel(target, modelName)) {
          continue;
        }
        p.log.warn(
          `Pull finished but ${pc.cyan(modelName)} is still not listed by Ollama. Check the tag name.`,
        );
        continue;
      }
      spinner.stop(pc.red(`Failed to pull ${modelName}`));
      p.log.error(result.error ?? `ollama pull exited with code ${result.exitCode}`);
      continue;
    }

    const nextModel = await askText({
      message: `${role[0]!.toUpperCase()}${role.slice(1)} model name (Ollama tag)`,
      placeholder: modelName,
      defaultValue: modelName,
      validate: (value) =>
        value.trim().length === 0 ? "Model name is required" : undefined,
    });
    modelName = nextModel.trim();
  }
  p.log.success(`Model ${pc.cyan(modelName)} is available.`);
  return modelName;
}

async function askEmbeddingsLocal(defaultModel: string): Promise<string> {
  const raw = await askText({
    message: "Local sentence-transformers embeddings model",
    placeholder: defaultModel,
    defaultValue: defaultModel,
  });
  return raw.trim() || defaultModel;
}

export async function askOllama(): Promise<OllamaChoices> {
  p.log.step("Configure Ollama");

  const hostRaw = await askText({
    message: "Ollama host",
    placeholder: OLLAMA_DEFAULT_HOST,
    defaultValue: OLLAMA_DEFAULT_HOST,
  });
  const portRaw = await askText({
    message: "Ollama port",
    placeholder: String(OLLAMA_DEFAULT_PORT),
    defaultValue: String(OLLAMA_DEFAULT_PORT),
    validate: (value) => {
      const n = Number(value);
      if (!Number.isInteger(n) || n <= 0 || n > 65535) return "Port must be 1-65535";
      return undefined;
    },
  });
  const target: OllamaTarget = {
    host: hostRaw.trim() || OLLAMA_DEFAULT_HOST,
    port: Number(portRaw) || OLLAMA_DEFAULT_PORT,
  };

  await waitForRunning(target);

  const smallModelRaw = await askText({
    message: "Small LLM model name (Ollama tag)",
    placeholder: OLLAMA_DEFAULT_SMALL_MODEL,
    defaultValue: OLLAMA_DEFAULT_SMALL_MODEL,
  });
  const smallModel = await waitForModel(
    target,
    smallModelRaw.trim() || OLLAMA_DEFAULT_SMALL_MODEL,
    "small",
  );

  const largeModelRaw = await askText({
    message: "Large LLM model name (Ollama tag)",
    placeholder: OLLAMA_DEFAULT_LARGE_MODEL,
    defaultValue: OLLAMA_DEFAULT_LARGE_MODEL,
  });
  const largeModel = await waitForModel(
    target,
    largeModelRaw.trim() || OLLAMA_DEFAULT_LARGE_MODEL,
    "large",
  );

  const embeddingsLocalModel = await askEmbeddingsLocal(EMBEDDINGS_DEFAULTS.localModel);

  return {
    host: target.host,
    port: target.port,
    smallModel,
    largeModel,
    embeddingsLocalModel,
  };
}

export async function isFileReadable(filePath: string): Promise<boolean> {
  try {
    await access(filePath);
    return true;
  } catch {
    return false;
  }
}
