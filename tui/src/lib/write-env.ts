import {
  AZURE_DEFAULT_EMBEDDING_MODEL,
  AZURE_DEFAULT_LARGE_API_VERSION,
  AZURE_DEFAULT_LARGE_MODEL,
  AZURE_DEFAULT_SMALL_MODEL,
  BEDROCK_DEFAULT_EMBEDDING_MODEL,
  BEDROCK_DEFAULT_LARGE_MODEL,
  BEDROCK_DEFAULT_REGION,
  BEDROCK_DEFAULT_SMALL_MODEL,
  EMBEDDINGS_DEFAULTS,
  OPENAI_DEFAULT_EMBEDDING_MODEL,
  OPENAI_DEFAULT_LARGE_MODEL,
  OPENAI_DEFAULT_SMALL_MODEL,
  OLLAMA_EMBEDDING_DIMENSIONS,
  ENV_KEYS,
  GCP_DEFAULT_EMBEDDING_MODEL,
  GCP_DEFAULT_LARGE_MODEL,
  GCP_DEFAULT_SMALL_MODEL,
  OLLAMA_DEFAULT_HOST,
  OLLAMA_DEFAULT_LARGE_MODEL,
  OLLAMA_DEFAULT_PORT,
  OLLAMA_DEFAULT_SMALL_MODEL,
  SERVICE_DEFAULTS,
} from "../constants.js";
import { applyEnvValues, loadOrSeedEnv, removeEnvValue, saveEnv } from "./env.js";
import type { InitChoices } from "../types.js";
import { resolveActiveEmbeddingModel } from "../flows/embedding-dimensions.js";
import { lookupEmbeddingDimensions } from "../constants.js";

const AZURE_ENDPOINT_PLACEHOLDER = "https://example.openai.azure.com";
const AZURE_EMBEDDING_PLACEHOLDER =
  "https://example.openai.azure.com/openai/deployments/text-embedding-3-large/embeddings?api-version=2023-05-15";
const AZURE_KEY_PLACEHOLDER = "azure-key-placeholder";
const OPENAI_KEY_PLACEHOLDER = "openai-api-key-placeholder";

function resolveModelsMode(choices: InitChoices): "local" | "remote" {
  const { llmSmallProvider, llmLargeProvider, embeddingsProvider } = choices.models;
  if (
    llmSmallProvider === "ollama" &&
    llmLargeProvider === "ollama" &&
    embeddingsProvider === "ollama"
  ) {
    return "local";
  }
  return choices.models.mode;
}

export async function writeEnvFromChoices(choices: InitChoices): Promise<void> {
  const env = await loadOrSeedEnv();
  const values: Record<string, string | number | undefined> = {};

  values[ENV_KEYS.graphDb] = choices.dbs.graphDb;
  values[ENV_KEYS.dataDb] = choices.dbs.dataDb;
  values[ENV_KEYS.vectorDb] = choices.dbs.vectorDb;
  values[ENV_KEYS.modelsMode] = resolveModelsMode(choices);
  values[ENV_KEYS.ocrMode] = choices.pipeline.ocrMode;
  values[ENV_KEYS.pipelineMode] = "accurate";
  values[ENV_KEYS.agenticArchitecture] = "custom";
  values[ENV_KEYS.runGraphConsolidator] = "true";
  values[ENV_KEYS.celeryWorkerConcurrency] = 4;

  values[ENV_KEYS.brainpatToken] = choices.auth.brainpatToken;

  values[ENV_KEYS.redisHost] = choices.connections.redis.host;
  values[ENV_KEYS.redisPort] = choices.connections.redis.port;

  applyPostgresValues(values, choices);
  applyNeo4jValues(values, choices);
  applyMilvusValues(values, choices, env);
  applyMongoValues(values, choices);
  applyOllamaValues(values, choices);
  applyRemoteModelValues(values, choices);
  applyEmbeddingDefaults(values, choices);

  applyEnvValues(env, values);
  await saveEnv(env);
}

function applyPostgresValues(
  values: Record<string, string | number | undefined>,
  choices: InitChoices,
): void {
  const pg = choices.connections.postgresql;
  const defaults = SERVICE_DEFAULTS.postgresql;
  values[ENV_KEYS.postgresHost] = pg?.host ?? defaults.host;
  values[ENV_KEYS.postgresPort] = pg?.port ?? defaults.port;
  values[ENV_KEYS.postgresUsername] = pg?.username ?? defaults.username;
  values[ENV_KEYS.postgresPassword] = pg?.password ?? defaults.password;
  values[ENV_KEYS.postgresSystemDatabase] =
    pg?.systemDatabase ?? defaults.systemDatabase;
  values[ENV_KEYS.postgresMaintenanceDatabase] =
    pg?.maintenanceDatabase ?? defaults.maintenanceDatabase;
}

function applyNeo4jValues(
  values: Record<string, string | number | undefined>,
  choices: InitChoices,
): void {
  const neo = choices.connections.neo4j;
  const defaults = SERVICE_DEFAULTS.neo4j;
  values[ENV_KEYS.neo4jHost] = neo?.host ?? defaults.host;
  values[ENV_KEYS.neo4jPort] = neo?.port ?? defaults.port;
  values[ENV_KEYS.neo4jUsername] = neo?.username ?? defaults.username;
  values[ENV_KEYS.neo4jPassword] = neo?.password ?? defaults.password;
}

function applyMilvusValues(
  values: Record<string, string | number | undefined>,
  choices: InitChoices,
  env: Awaited<ReturnType<typeof loadOrSeedEnv>>,
): void {
  const mv = choices.connections.milvus;
  const defaults = SERVICE_DEFAULTS.milvus;
  values[ENV_KEYS.milvusHost] = mv?.host ?? defaults.host;
  values[ENV_KEYS.milvusPort] = mv?.port ?? defaults.port;
  if (mv?.uri) {
    values[ENV_KEYS.milvusUri] = mv.uri;
  } else {
    removeEnvValue(env, ENV_KEYS.milvusUri);
  }
  if (mv?.token) {
    values[ENV_KEYS.milvusToken] = mv.token;
  } else {
    removeEnvValue(env, ENV_KEYS.milvusToken);
  }
}

function applyMongoValues(
  values: Record<string, string | number | undefined>,
  choices: InitChoices,
): void {
  const mg = choices.connections.mongo;
  const defaults = SERVICE_DEFAULTS.mongo;
  values[ENV_KEYS.mongoHost] = mg?.host ?? defaults.host;
  values[ENV_KEYS.mongoPort] = mg?.port ?? defaults.port;
  values[ENV_KEYS.mongoUsername] = mg?.username ?? defaults.username;
  values[ENV_KEYS.mongoPassword] = mg?.password ?? defaults.password;
  values[ENV_KEYS.mongoSystemDatabase] = "system";
}

function applyOllamaValues(
  values: Record<string, string | number | undefined>,
  choices: InitChoices,
): void {
  const usesOllama =
    choices.models.llmSmallProvider === "ollama" ||
    choices.models.llmLargeProvider === "ollama" ||
    choices.models.embeddingsProvider === "ollama";
  if (usesOllama && choices.models.ollama) {
    const ol = choices.models.ollama;
    values[ENV_KEYS.ollamaHost] = ol.host;
    values[ENV_KEYS.ollamaPort] = ol.port;
    values[ENV_KEYS.ollamaSmallModel] = ol.smallModel;
    values[ENV_KEYS.ollamaLargeModel] = ol.largeModel;
    values[ENV_KEYS.embeddingsLocalModel] = ol.embeddingsLocalModel;
    return;
  }
  values[ENV_KEYS.ollamaHost] = OLLAMA_DEFAULT_HOST;
  values[ENV_KEYS.ollamaPort] = OLLAMA_DEFAULT_PORT;
  values[ENV_KEYS.ollamaSmallModel] = OLLAMA_DEFAULT_SMALL_MODEL;
  values[ENV_KEYS.ollamaLargeModel] = OLLAMA_DEFAULT_LARGE_MODEL;
}

function applyRemoteModelValues(
  values: Record<string, string | number | undefined>,
  choices: InitChoices,
): void {
  const gcp = choices.models.gcp;
  const azure = choices.models.azure;
  const openai = choices.models.openai;
  const bedrock = choices.models.bedrock;
  values[ENV_KEYS.llmSmallProvider] = choices.models.llmSmallProvider;
  values[ENV_KEYS.llmLargeProvider] = choices.models.llmLargeProvider;
  values[ENV_KEYS.embeddingsProvider] = choices.models.embeddingsProvider;
  values[ENV_KEYS.gcpCredentialsPath] = gcp?.credentialsPath ?? "gcp_credentials.json";
  values[ENV_KEYS.gcpProjectId] = gcp?.projectId ?? "your-project-id";
  values[ENV_KEYS.gcpSmallLlmModel] = gcp?.smallLlmModel ?? GCP_DEFAULT_SMALL_MODEL;
  values[ENV_KEYS.gcpExtraSmallLlmModel] =
    gcp?.smallLlmModel ?? GCP_DEFAULT_SMALL_MODEL;
  values[ENV_KEYS.gcpLargeLlmModel] = gcp?.largeLlmModel ?? GCP_DEFAULT_LARGE_MODEL;
  values[ENV_KEYS.gcpEmbeddingModel] = gcp?.embeddingModel ?? GCP_DEFAULT_EMBEDDING_MODEL;

  values[ENV_KEYS.azureSmallLlmModel] =
    azure?.smallLlmModel ?? AZURE_DEFAULT_SMALL_MODEL;
  values[ENV_KEYS.azureLargeLlmModel] =
    azure?.largeLlmModel ?? AZURE_DEFAULT_LARGE_MODEL;
  values[ENV_KEYS.azureLlmApiVersion] =
    azure?.llmApiVersion ?? AZURE_DEFAULT_LARGE_API_VERSION;
  values[ENV_KEYS.azureLlmEndpoint] =
    azure?.llmEndpoint ?? AZURE_ENDPOINT_PLACEHOLDER;
  values[ENV_KEYS.azureLlmSubscriptionKey] =
    azure?.llmSubscriptionKey ?? AZURE_KEY_PLACEHOLDER;
  values[ENV_KEYS.azureLargeLlmApiVersion] =
    azure?.llmApiVersion ?? AZURE_DEFAULT_LARGE_API_VERSION;
  values[ENV_KEYS.azureLargeLlmEndpoint] =
    azure?.llmEndpoint ?? AZURE_ENDPOINT_PLACEHOLDER;
  values[ENV_KEYS.azureLargeLlmSubscriptionKey] =
    azure?.llmSubscriptionKey ?? AZURE_KEY_PLACEHOLDER;
  values[ENV_KEYS.azureEmbeddingModel] =
    azure?.embeddingModel ?? AZURE_DEFAULT_EMBEDDING_MODEL;
  values[ENV_KEYS.azureEmbeddingFullEndpoint] =
    azure?.embeddingEndpoint ?? AZURE_EMBEDDING_PLACEHOLDER;
  values[ENV_KEYS.azureEmbeddingKey] =
    azure?.embeddingKey ?? AZURE_KEY_PLACEHOLDER;

  values[ENV_KEYS.openaiApiKey] = openai?.apiKey ?? OPENAI_KEY_PLACEHOLDER;
  values[ENV_KEYS.openaiSmallLlmModel] =
    openai?.smallLlmModel ?? OPENAI_DEFAULT_SMALL_MODEL;
  values[ENV_KEYS.openaiLargeLlmModel] =
    openai?.largeLlmModel ?? OPENAI_DEFAULT_LARGE_MODEL;
  values[ENV_KEYS.openaiEmbeddingModel] =
    openai?.embeddingModel ?? OPENAI_DEFAULT_EMBEDDING_MODEL;
  if (openai?.baseUrl) {
    values[ENV_KEYS.openaiBaseUrl] = openai.baseUrl;
  } else {
    values[ENV_KEYS.openaiBaseUrl] = "";
  }
  values[ENV_KEYS.bedrockAccessKeyId] = bedrock?.accessKeyId ?? "your-access-key-id";
  values[ENV_KEYS.bedrockSecretAccessKey] =
    bedrock?.secretAccessKey ?? "your-secret-access-key";
  values[ENV_KEYS.bedrockSessionToken] = bedrock?.sessionToken ?? "";
  values[ENV_KEYS.bedrockSmallLlmModel] =
    bedrock?.smallLlmModel ?? BEDROCK_DEFAULT_SMALL_MODEL;
  values[ENV_KEYS.bedrockLargeLlmModel] =
    bedrock?.largeLlmModel ?? BEDROCK_DEFAULT_LARGE_MODEL;
  values[ENV_KEYS.bedrockEmbeddingModel] =
    bedrock?.embeddingModel ?? BEDROCK_DEFAULT_EMBEDDING_MODEL;
}

function fallbackEmbeddingDimensions(choices: InitChoices) {
  const embeddingModel = resolveActiveEmbeddingModel(choices.models);
  const known = lookupEmbeddingDimensions(embeddingModel);
  if (known !== undefined) {
    return {
      nodesDimension: known,
      tripletsDimension: known,
      observationsDimension: known,
      dataDimension: known,
      relationshipsDimension: known,
    };
  }
  if (choices.models.embeddingsProvider === "ollama") {
    return OLLAMA_EMBEDDING_DIMENSIONS;
  }
  return EMBEDDINGS_DEFAULTS;
}

function applyEmbeddingDefaults(
  values: Record<string, string | number | undefined>,
  choices: InitChoices,
): void {
  values[ENV_KEYS.embeddingsSmallModel] = EMBEDDINGS_DEFAULTS.smallModel;
  values[ENV_KEYS.embeddingsLocalModel] =
    values[ENV_KEYS.embeddingsLocalModel] ?? EMBEDDINGS_DEFAULTS.localModel;

  const embeddingDims = choices.models.embeddingDimensions ?? fallbackEmbeddingDimensions(choices);

  values[ENV_KEYS.embeddingNodesDimension] = embeddingDims.nodesDimension;
  values[ENV_KEYS.embeddingTripletsDimension] = embeddingDims.tripletsDimension;
  values[ENV_KEYS.embeddingObservationsDimension] =
    embeddingDims.observationsDimension;
  values[ENV_KEYS.embeddingDataDimension] = embeddingDims.dataDimension;
  values[ENV_KEYS.embeddingRelationshipsDimension] =
    embeddingDims.relationshipsDimension;

  if (choices.models.embeddingsProvider === "azure" && choices.models.azure) {
    values[ENV_KEYS.embeddingsFullEndpoint] =
      choices.models.azure.embeddingEndpoint;
    values[ENV_KEYS.embeddingsKey] = choices.models.azure.embeddingKey;
  } else {
    values[ENV_KEYS.embeddingsFullEndpoint] =
      values[ENV_KEYS.embeddingsFullEndpoint] ?? AZURE_EMBEDDING_PLACEHOLDER;
    values[ENV_KEYS.embeddingsKey] =
      values[ENV_KEYS.embeddingsKey] ?? AZURE_KEY_PLACEHOLDER;
  }
}
