import * as p from "@clack/prompts";
import {
  AZURE_DEFAULT_EMBEDDING_MODEL,
  BEDROCK_DEFAULT_EMBEDDING_MODEL,
  EMBEDDINGS_DEFAULTS,
  GCP_DEFAULT_EMBEDDING_MODEL,
  KNOWN_EMBEDDING_MODEL_DIMENSIONS,
  lookupEmbeddingDimensions,
  OPENAI_DEFAULT_EMBEDDING_MODEL,
} from "../constants.js";
import {
  askConfirm,
  askConfirmOrBack,
  askText,
  isPromptBack,
  type PromptBack,
} from "../lib/prompts.js";
import type { EmbeddingDimensions, ModelsChoices, ModelProvider } from "../types.js";

function dimensionsFromSize(size: number): EmbeddingDimensions {
  return {
    nodesDimension: size,
    tripletsDimension: size,
    observationsDimension: size,
    dataDimension: size,
    relationshipsDimension: size,
  };
}

async function askDimensionSize(initial?: number): Promise<number> {
  const raw = await askText({
    message: "Embedding vector dimensions",
    placeholder: initial ? String(initial) : "3072",
    defaultValue: initial ? String(initial) : "3072",
    validate: (value) => {
      const n = Number(value.trim());
      if (!Number.isInteger(n) || n <= 0) {
        return "Enter a positive whole number";
      }
      return undefined;
    },
  });
  const n = Number(raw.trim());
  if (!Number.isInteger(n) || n <= 0) {
    throw new Error("Invalid embedding dimensions");
  }
  return n;
}

export async function askEmbeddingDimensions(
  embeddingModel: string,
  opts?: {
    allowBack?: boolean;
    backHint?: string;
    initial?: EmbeddingDimensions;
  },
): Promise<EmbeddingDimensions | PromptBack> {
  p.log.step("Embedding dimensions");
  const known = lookupEmbeddingDimensions(embeddingModel);
  const initialSize =
    opts?.initial?.nodesDimension ?? known ?? EMBEDDINGS_DEFAULTS.nodesDimension;

  if (known !== undefined) {
    p.note(
      [
        `Model: ${embeddingModel}`,
        `Known output size: ${known} dimensions`,
        "",
        "All vector stores (nodes, triplets, observations, data, relationships)",
        "must use the same dimension as your embedding model.",
      ].join("\n"),
      "Embedding dimensions",
    );

    if (opts?.allowBack) {
      const picked = await askConfirmOrBack({
        message: `Use ${known} dimensions for all embedding vector stores?`,
        initialValue: true,
        backHint: opts.backHint,
      });
      if (isPromptBack(picked)) {
        return picked;
      }
      if (picked) {
        return dimensionsFromSize(known);
      }
    } else {
      const useKnown = await askConfirm({
        message: `Use ${known} dimensions for all embedding vector stores?`,
        initialValue: true,
      });
      if (useKnown) {
        return dimensionsFromSize(known);
      }
    }
  } else {
    p.note(
      [
        `Model: ${embeddingModel}`,
        "This model is not in the built-in dimension map.",
        "Enter the embedding output size from your provider docs.",
      ].join("\n"),
      "Embedding dimensions",
    );
  }

  const size = await askDimensionSize(initialSize);
  return dimensionsFromSize(size);
}

export function resolveActiveEmbeddingModel(models: ModelsChoices): string {
  const provider: ModelProvider = models.embeddingsProvider;
  switch (provider) {
    case "ollama":
      return models.ollama?.embeddingsLocalModel ?? EMBEDDINGS_DEFAULTS.localModel;
    case "openai":
      return models.openai?.embeddingModel ?? OPENAI_DEFAULT_EMBEDDING_MODEL;
    case "azure":
      return models.azure?.embeddingModel ?? AZURE_DEFAULT_EMBEDDING_MODEL;
    case "gcp_vertex":
      return models.gcp?.embeddingModel ?? GCP_DEFAULT_EMBEDDING_MODEL;
    case "amazon_bedrock":
      return models.bedrock?.embeddingModel ?? BEDROCK_DEFAULT_EMBEDDING_MODEL;
    default:
      return EMBEDDINGS_DEFAULTS.localModel;
  }
}

export async function attachEmbeddingDimensions(
  models: ModelsChoices,
  opts?: {
    allowBack?: boolean;
    backHint?: string;
    initial?: EmbeddingDimensions;
  },
): Promise<ModelsChoices | PromptBack> {
  const embeddingModel = resolveActiveEmbeddingModel(models);
  const dimensions = await askEmbeddingDimensions(embeddingModel, opts);
  if (isPromptBack(dimensions)) {
    return dimensions;
  }
  return { ...models, embeddingDimensions: dimensions };
}

export { KNOWN_EMBEDDING_MODEL_DIMENSIONS };
