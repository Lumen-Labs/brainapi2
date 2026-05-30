import { readFile, stat } from "node:fs/promises";
import path from "node:path";
import { homedir } from "node:os";
import * as p from "@clack/prompts";
import pc from "picocolors";
import {
  AZURE_DEFAULT_EMBEDDING_MODEL,
  AZURE_DEFAULT_SMALL_MODEL,
  AZURE_DEFAULT_LARGE_API_VERSION,
  AZURE_DEFAULT_LARGE_MODEL,
  BEDROCK_DEFAULT_EMBEDDING_MODEL,
  BEDROCK_DEFAULT_LARGE_MODEL,
  BEDROCK_DEFAULT_REGION,
  BEDROCK_DEFAULT_SMALL_MODEL,
  GCP_DEFAULT_EMBEDDING_MODEL,
  GCP_DEFAULT_LARGE_MODEL,
  GCP_DEFAULT_SMALL_MODEL,
  OPENAI_DEFAULT_EMBEDDING_MODEL,
  OPENAI_DEFAULT_LARGE_MODEL,
  OPENAI_DEFAULT_SMALL_MODEL,
} from "../constants.js";
import {
  askPassword,
  askText,
  isPromptBack,
  pickOne,
  type PromptBack,
} from "../lib/prompts.js";
import type {
  AzureChoices,
  BedrockChoices,
  GcpChoices,
  ModelProvider,
  ModelsChoices,
  ModelsMode,
  OpenAIChoices,
} from "../types.js";
import { askOllama } from "./ollama.js";
import { attachEmbeddingDimensions } from "./embedding-dimensions.js";

function expandHome(filePath: string): string {
  if (filePath.startsWith("~")) {
    return path.join(homedir(), filePath.slice(1));
  }
  return filePath;
}

interface GcpCredentialsCheck {
  error?: string;
  projectId?: string;
}

async function inspectGcpCredentials(filePath: string): Promise<GcpCredentialsCheck> {
  const expanded = expandHome(filePath.trim());
  if (!expanded) return { error: "Path is required" };
  try {
    const info = await stat(expanded);
    if (!info.isFile()) return { error: "Path is not a file" };
    const raw = await readFile(expanded, "utf8");
    const parsed = JSON.parse(raw) as { project_id?: unknown };
    const projectId =
      typeof parsed.project_id === "string" && parsed.project_id.trim().length > 0
        ? parsed.project_id.trim()
        : undefined;
    return { projectId };
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    return { error: `Could not read credentials JSON: ${message}` };
  }
}

async function askGcp(): Promise<GcpChoices> {
  p.log.step("Configure GCP Vertex");

  let credentialsPath = "";
  let projectIdFromFile: string | undefined;
  while (true) {
    const raw = await askText({
      message: "Path to GCP service-account credentials JSON",
      placeholder: "~/.config/gcloud/brainapi.json",
    });
    const result = await inspectGcpCredentials(raw);
    if (!result.error) {
      credentialsPath = expandHome(raw.trim());
      projectIdFromFile = result.projectId;
      break;
    }
    p.log.error(result.error);
  }

  let projectId: string;
  if (projectIdFromFile) {
    projectId = projectIdFromFile;
    p.log.info(`Using GCP project ${pc.cyan(projectId)} (from credentials file)`);
  } else {
    p.log.warn(
      "No project_id field found in the credentials JSON — please enter it manually.",
    );
    const entered = await askText({
      message: "GCP project id",
      placeholder: "my-gcp-project",
      validate: (value) => (value.trim().length === 0 ? "Project id is required" : undefined),
    });
    projectId = entered.trim();
  }

  const smallModel = await askText({
    message: "GCP small LLM model",
    placeholder: GCP_DEFAULT_SMALL_MODEL,
    defaultValue: GCP_DEFAULT_SMALL_MODEL,
  });
  const largeModel = await askText({
    message: "GCP large LLM model",
    placeholder: GCP_DEFAULT_LARGE_MODEL,
    defaultValue: GCP_DEFAULT_LARGE_MODEL,
  });
  const embeddingModel = await askText({
    message: "GCP embedding model",
    placeholder: GCP_DEFAULT_EMBEDDING_MODEL,
    defaultValue: GCP_DEFAULT_EMBEDDING_MODEL,
  });

  return {
    credentialsPath,
    projectId,
    smallLlmModel: smallModel.trim() || GCP_DEFAULT_SMALL_MODEL,
    largeLlmModel: largeModel.trim() || GCP_DEFAULT_LARGE_MODEL,
    embeddingModel: embeddingModel.trim() || GCP_DEFAULT_EMBEDDING_MODEL,
  };
}

async function askAzure(): Promise<AzureChoices> {
  p.log.step("Configure Azure OpenAI");
  const llmEndpoint = await askText({
    message: "Azure LLM endpoint",
    placeholder: "https://yourproject.openai.azure.com",
    validate: (value) => (value.trim().length === 0 ? "Endpoint is required" : undefined),
  });
  const llmApiVersion = await askText({
    message: "Azure LLM API version",
    placeholder: AZURE_DEFAULT_LARGE_API_VERSION,
    defaultValue: AZURE_DEFAULT_LARGE_API_VERSION,
  });
  const llmSubscriptionKey = await askPassword({
    message: "Azure LLM subscription key",
    validate: (value) => (value.trim().length === 0 ? "API key is required" : undefined),
  });
  const smallLlmModel = await askText({
    message: "Azure small LLM deployment/model",
    placeholder: AZURE_DEFAULT_SMALL_MODEL,
    defaultValue: AZURE_DEFAULT_SMALL_MODEL,
  });
  const largeLlmModel = await askText({
    message: "Azure large LLM deployment/model",
    placeholder: AZURE_DEFAULT_LARGE_MODEL,
    defaultValue: AZURE_DEFAULT_LARGE_MODEL,
  });
  const embeddingEndpoint = await askText({
    message: "Azure embeddings endpoint URL",
    placeholder:
      "https://yourproject.openai.azure.com/openai/deployments/text-embedding-3-large/embeddings?api-version=2023-05-15",
    validate: (value) => (value.trim().length === 0 ? "Endpoint URL is required" : undefined),
  });
  const embeddingKey = await askPassword({
    message: "Azure embeddings API key",
    validate: (value) => (value.trim().length === 0 ? "API key is required" : undefined),
  });
  const embeddingModel = await askText({
    message: "Azure embedding model/deployment",
    placeholder: AZURE_DEFAULT_EMBEDDING_MODEL,
    defaultValue: AZURE_DEFAULT_EMBEDDING_MODEL,
  });

  return {
    smallLlmModel: smallLlmModel.trim() || AZURE_DEFAULT_SMALL_MODEL,
    largeLlmModel: largeLlmModel.trim() || AZURE_DEFAULT_LARGE_MODEL,
    llmApiVersion: llmApiVersion.trim() || AZURE_DEFAULT_LARGE_API_VERSION,
    llmEndpoint: llmEndpoint.trim(),
    llmSubscriptionKey: llmSubscriptionKey.trim(),
    embeddingEndpoint: embeddingEndpoint.trim(),
    embeddingKey: embeddingKey.trim(),
    embeddingModel: embeddingModel.trim() || AZURE_DEFAULT_EMBEDDING_MODEL,
  };
}

async function askBedrock(): Promise<BedrockChoices> {
  p.log.step("Configure Amazon Bedrock");
  const region = await askText({
    message: "AWS region",
    placeholder: BEDROCK_DEFAULT_REGION,
    defaultValue: BEDROCK_DEFAULT_REGION,
  });
  const accessKeyId = await askText({
    message: "AWS access key id",
    validate: (value) =>
      value.trim().length === 0 ? "Access key id is required" : undefined,
  });
  const secretAccessKey = await askPassword({
    message: "AWS secret access key",
    validate: (value) =>
      value.trim().length === 0 ? "Secret access key is required" : undefined,
  });
  const sessionToken = await askPassword({
    message: "AWS session token (optional)",
    validate: () => undefined,
  });
  const smallLlmModel = await askText({
    message: "Bedrock small LLM model id",
    placeholder: BEDROCK_DEFAULT_SMALL_MODEL,
    defaultValue: BEDROCK_DEFAULT_SMALL_MODEL,
  });
  const largeLlmModel = await askText({
    message: "Bedrock large LLM model id",
    placeholder: BEDROCK_DEFAULT_LARGE_MODEL,
    defaultValue: BEDROCK_DEFAULT_LARGE_MODEL,
  });
  const embeddingModel = await askText({
    message: "Bedrock embedding model id",
    placeholder: BEDROCK_DEFAULT_EMBEDDING_MODEL,
    defaultValue: BEDROCK_DEFAULT_EMBEDDING_MODEL,
  });
  return {
    region: region.trim() || BEDROCK_DEFAULT_REGION,
    accessKeyId: accessKeyId.trim(),
    secretAccessKey: secretAccessKey.trim(),
    sessionToken: sessionToken.trim() || undefined,
    smallLlmModel: smallLlmModel.trim() || BEDROCK_DEFAULT_SMALL_MODEL,
    largeLlmModel: largeLlmModel.trim() || BEDROCK_DEFAULT_LARGE_MODEL,
    embeddingModel: embeddingModel.trim() || BEDROCK_DEFAULT_EMBEDDING_MODEL,
  };
}

async function askOpenAI(): Promise<OpenAIChoices> {
  p.log.step("Configure OpenAI");

  const apiKey = await askPassword({
    message: "OpenAI API key",
    validate: (value) =>
      value.trim().length === 0 ? "API key is required" : undefined,
  });
  const baseUrl = await askText({
    message: "OpenAI base URL (optional)",
    placeholder: "https://api.openai.com/v1",
    defaultValue: "",
  });
  const smallLlmModel = await askText({
    message: "OpenAI small LLM model",
    placeholder: OPENAI_DEFAULT_SMALL_MODEL,
    defaultValue: OPENAI_DEFAULT_SMALL_MODEL,
  });
  const largeLlmModel = await askText({
    message: "OpenAI large LLM model",
    placeholder: OPENAI_DEFAULT_LARGE_MODEL,
    defaultValue: OPENAI_DEFAULT_LARGE_MODEL,
  });
  const embeddingModel = await askText({
    message: "OpenAI embedding model",
    placeholder: OPENAI_DEFAULT_EMBEDDING_MODEL,
    defaultValue: OPENAI_DEFAULT_EMBEDDING_MODEL,
  });

  return {
    apiKey: apiKey.trim(),
    baseUrl: baseUrl.trim() || undefined,
    smallLlmModel: smallLlmModel.trim() || OPENAI_DEFAULT_SMALL_MODEL,
    largeLlmModel: largeLlmModel.trim() || OPENAI_DEFAULT_LARGE_MODEL,
    embeddingModel: embeddingModel.trim() || OPENAI_DEFAULT_EMBEDDING_MODEL,
  };
}

async function askProvider(message: string, initialValue: ModelProvider): Promise<ModelProvider> {
  return pickOne<ModelProvider>({
    message,
    options: [
      { value: "ollama", label: "Ollama" },
      { value: "openai", label: "OpenAI" },
      { value: "azure", label: "Azure OpenAI" },
      { value: "gcp_vertex", label: "Google Cloud — Vertex AI" },
      { value: "amazon_bedrock", label: "Amazon Bedrock" },
    ],
    initialValue,
  });
}

export async function askModels(options?: {
  prechosenMode?: ModelsMode;
  allowBack?: false;
  initialMode?: ModelsMode;
}): Promise<ModelsChoices>;
export async function askModels(options: {
  prechosenMode?: ModelsMode;
  allowBack: true;
  backHint?: string;
  initialMode?: ModelsMode;
}): Promise<ModelsChoices | PromptBack>;
export async function askModels(options?: {
  prechosenMode?: ModelsMode;
  allowBack?: boolean;
  backHint?: string;
  initialMode?: ModelsMode;
}): Promise<ModelsChoices | PromptBack> {
  let mode = options?.prechosenMode;
  if (!mode) {
    const picked = options?.allowBack
      ? await pickOne<ModelsMode>({
          message: "Models mode",
          options: [
            { value: "remote", label: "Remote (cloud provider)" },
            { value: "local", label: "Local (Ollama)" },
          ],
          initialValue: options.initialMode ?? "remote",
          allowBack: true,
          backHint: options.backHint,
        })
      : await pickOne<ModelsMode>({
          message: "Models mode",
          options: [
            { value: "remote", label: "Remote (cloud provider)" },
            { value: "local", label: "Local (Ollama)" },
          ],
          initialValue: options?.initialMode ?? "remote",
        });
    if (isPromptBack(picked)) {
      return picked;
    }
    mode = picked;
  }

  if (mode === "local") {
    const ollama = await askOllama();
    return attachEmbeddingDimensions(
      {
        mode,
        llmSmallProvider: "ollama",
        llmLargeProvider: "ollama",
        embeddingsProvider: "ollama",
        ollama,
      },
      {
        allowBack: options?.allowBack,
        backHint: options?.backHint,
      },
    );
  }

  const llmSmallProvider = await askProvider("Small LLM provider", "gcp_vertex");
  const llmLargeProvider = await askProvider("Large LLM provider", "azure");
  const embeddingsProvider = await askProvider("Embeddings provider", "azure");

  const providers = new Set([llmSmallProvider, llmLargeProvider, embeddingsProvider]);

  let ollama = undefined;
  let gcp = undefined;
  let azure = undefined;
  let openai = undefined;
  let bedrock = undefined;

  if (providers.has("ollama")) {
    ollama = await askOllama();
  }
  if (providers.has("gcp_vertex")) {
    gcp = await askGcp();
  }
  if (providers.has("azure")) {
    azure = await askAzure();
  }
  if (providers.has("openai")) {
    openai = await askOpenAI();
  }
  if (providers.has("amazon_bedrock")) {
    bedrock = await askBedrock();
  }

  return attachEmbeddingDimensions(
    {
      mode,
      llmSmallProvider,
      llmLargeProvider,
      embeddingsProvider,
      ollama,
      gcp,
      azure,
      openai,
      bedrock,
    },
    {
      allowBack: options?.allowBack,
      backHint: options?.backHint,
    },
  );
}
