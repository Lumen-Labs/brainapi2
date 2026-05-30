export const DEFAULT_REPO_URL = "https://github.com/Lumen-Labs/brainapi2.git";
export const DEFAULT_BRANCH = "main";
export const MIN_PYTHON_VERSION: [number, number] = [3, 11];
export const PYTHON_CANDIDATES = [
  "python3.13",
  "python3.12",
  "python3.11",
  "python3",
  "python",
];

export const OLLAMA_DEFAULT_HOST = "localhost";
export const OLLAMA_DEFAULT_PORT = 11434;
export const OLLAMA_DEFAULT_SMALL_MODEL = "hf.co/unsloth/Qwen3-14B-GGUF:Q4_K_M";
export const OLLAMA_DEFAULT_LARGE_MODEL = "hf.co/unsloth/Qwen3-14B-GGUF:Q4_K_M";

export const GCP_DEFAULT_SMALL_MODEL = "gemini-3-flash-preview";
export const GCP_DEFAULT_LARGE_MODEL = "gemini-3-flash-preview";
export const GCP_DEFAULT_EMBEDDING_MODEL = "text-embedding-005";

export const AZURE_DEFAULT_LARGE_MODEL = "gpt-4o";
export const AZURE_DEFAULT_SMALL_MODEL = "gpt-4o-mini";
export const AZURE_DEFAULT_LARGE_API_VERSION = "2024-12-01-preview";
export const AZURE_DEFAULT_EMBEDDING_MODEL = "text-embedding-3-large";

export const OPENAI_DEFAULT_SMALL_MODEL = "gpt-4o-mini";
export const OPENAI_DEFAULT_LARGE_MODEL = "gpt-4o";
export const OPENAI_DEFAULT_EMBEDDING_MODEL = "text-embedding-3-large";

export const KNOWN_EMBEDDING_MODEL_DIMENSIONS: Record<string, number> = {
  "nomic-embed-text": 768,
  "mxbai-embed-large": 1024,
  "all-minilm": 384,
  "text-embedding-3-large": 3072,
  "text-embedding-3-small": 1536,
  "text-embedding-ada-002": 1536,
  "text-embedding-005": 768,
  "text-embedding-004": 768,
  "amazon.titan-embed-text-v2:0": 1024,
  "amazon.titan-embed-text-v1": 1536,
  "intfloat/e5-small": 384,
  "paraphrase-multilingual-minilm-l12-v2": 384,
};

export function lookupEmbeddingDimensions(model: string): number | undefined {
  const normalized = model.trim().toLowerCase();
  if (normalized in KNOWN_EMBEDDING_MODEL_DIMENSIONS) {
    return KNOWN_EMBEDDING_MODEL_DIMENSIONS[normalized];
  }
  const entries = Object.entries(KNOWN_EMBEDDING_MODEL_DIMENSIONS).sort(
    (a, b) => b[0].length - a[0].length,
  );
  for (const [key, dimensions] of entries) {
    if (normalized.includes(key.toLowerCase())) {
      return dimensions;
    }
  }
  return undefined;
}

export const OPENAI_EMBEDDING_DIMENSIONS = {
  nodesDimension: 3072,
  tripletsDimension: 3072,
  observationsDimension: 3072,
  dataDimension: 3072,
  relationshipsDimension: 3072,
} as const;

export const OPENAI_EMBEDDING_SMALL_DIMENSIONS = {
  nodesDimension: 1536,
  tripletsDimension: 1536,
  observationsDimension: 1536,
  dataDimension: 1536,
  relationshipsDimension: 1536,
} as const;

export const BEDROCK_DEFAULT_REGION = "us-east-1";
export const BEDROCK_DEFAULT_SMALL_MODEL = "us.anthropic.claude-3-5-haiku-20241022-v1:0";
export const BEDROCK_DEFAULT_LARGE_MODEL = "us.anthropic.claude-3-7-sonnet-20250219-v1:0";
export const BEDROCK_DEFAULT_EMBEDDING_MODEL = "amazon.titan-embed-text-v2:0";

export const EMBEDDINGS_DEFAULTS = {
  smallModel: "paraphrase-multilingual-MiniLM-L12-v2",
  localModel: "nomic-embed-text:latest",
  nodesDimension: 3072,
  tripletsDimension: 3072,
  observationsDimension: 3072,
  dataDimension: 3072,
  relationshipsDimension: 3072,
} as const;

export const OLLAMA_EMBEDDING_DIMENSIONS = {
  nodesDimension: 768,
  tripletsDimension: 768,
  observationsDimension: 768,
  dataDimension: 768,
  relationshipsDimension: 768,
} as const;

export const SERVICE_DEFAULTS = {
  redis: { host: "localhost", port: 6379 },
  postgresql: {
    host: "localhost",
    port: 5432,
    username: "postgres",
    password: "password",
    systemDatabase: "brainapi",
    maintenanceDatabase: "postgres",
  },
  neo4j: {
    host: "localhost",
    port: 7687,
    username: "neo4j",
    password: "your_password",
  },
  milvus: {
    host: "localhost",
    port: 19530,
  },
  mongo: {
    host: "localhost",
    port: 27017,
    username: "root",
    password: "password",
  },
  rabbitmq: {
    host: "localhost",
    port: 5672,
    username: "kalo",
    password: "kalo",
  },
} as const;

export const ENV_KEYS = {
  graphDb: "GRAPH_DB",
  dataDb: "DATA_DB",
  vectorDb: "VECTOR_DB",
  modelsMode: "MODELS_MODE",
  brainpatToken: "BRAINPAT_TOKEN",
  pipelineMode: "PIPELINE_MODE",
  ocrMode: "OCR_MODE",
  agenticArchitecture: "AGENTIC_ARCHITECTURE",
  runGraphConsolidator: "RUN_GRAPH_CONSOLIDATOR",

  redisHost: "REDIS_HOST",
  redisPort: "REDIS_PORT",
  celeryBackend: "CELERY_BACKEND",
  celeryWorkerConcurrency: "CELERY_WORKER_CONCURRENCY",
  celeryUnblockedWindowSeconds: "CELERY_UNBLOCKED_WINDOW_SECONDS",
  celeryUnblockedMaxRetries: "CELERY_UNBLOCKED_MAX_RETRIES",
  celeryQueueHealthThreshold: "CELERY_QUEUE_HEALTH_THRESHOLD",

  postgresHost: "POSTGRES_HOST",
  postgresPort: "POSTGRES_PORT",
  postgresUsername: "POSTGRES_USERNAME",
  postgresPassword: "POSTGRES_PASSWORD",
  postgresSystemDatabase: "POSTGRES_SYSTEM_DATABASE",
  postgresMaintenanceDatabase: "POSTGRES_MAINTENANCE_DATABASE",

  neo4jHost: "NEO4J_HOST",
  neo4jPort: "NEO4J_PORT",
  neo4jUsername: "NEO4J_USERNAME",
  neo4jPassword: "NEO4J_PASSWORD",

  milvusHost: "MILVUS_HOST",
  milvusPort: "MILVUS_PORT",
  milvusUri: "MILVUS_URI",
  milvusToken: "MILVUS_TOKEN",

  mongoHost: "MONGO_HOST",
  mongoPort: "MONGO_PORT",
  mongoUsername: "MONGO_USERNAME",
  mongoPassword: "MONGO_PASSWORD",
  mongoConnectionString: "MONGO_CONNECTION_STRING",
  mongoSystemDatabase: "MONGO_SYSTEM_DATABASE",

  ollamaHost: "OLLAMA_HOST",
  ollamaPort: "OLLAMA_PORT",
  ollamaSmallModel: "OLLAMA_LLM_SMALL_MODEL",
  ollamaLargeModel: "OLLAMA_LLM_LARGE_MODEL",

  gcpProjectId: "GCP_PROJECT_ID",
  gcpCredentialsPath: "GCP_CREDENTIALS_PATH",
  llmSmallProvider: "LLM_SMALL_PROVIDER",
  llmLargeProvider: "LLM_LARGE_PROVIDER",
  embeddingsProvider: "EMBEDDINGS_PROVIDER",
  gcpExtraSmallLlmModel: "GCP_EXTRA_SMALL_LLM_MODEL",
  gcpSmallLlmModel: "GCP_SMALL_LLM_MODEL",
  gcpLargeLlmModel: "GCP_LARGE_LLM_MODEL",
  gcpEmbeddingModel: "GCP_EMBEDDING_MODEL",

  azureSmallLlmModel: "AZURE_SMALL_LLM_MODEL",
  azureLargeLlmModel: "AZURE_LARGE_LLM_MODEL",
  azureLlmApiVersion: "AZURE_LLM_API_VERSION",
  azureLlmEndpoint: "AZURE_LLM_ENDPOINT",
  azureLlmSubscriptionKey: "AZURE_LLM_SUBSCRIPTION_KEY",
  azureLargeLlmApiVersion: "AZURE_LARGE_LLM_API_VERSION",
  azureLargeLlmEndpoint: "AZURE_LARGE_LLM_ENDPOINT",
  azureLargeLlmSubscriptionKey: "AZURE_LARGE_LLM_SUBSCRIPTION_KEY",
  azureEmbeddingModel: "AZURE_EMBEDDING_MODEL",
  azureEmbeddingFullEndpoint: "AZURE_EMBEDDING_FULL_ENDPOINT",
  azureEmbeddingKey: "AZURE_EMBEDDING_KEY",

  openaiApiKey: "OPENAI_API_KEY",
  openaiBaseUrl: "OPENAI_BASE_URL",
  openaiSmallLlmModel: "OPENAI_SMALL_LLM_MODEL",
  openaiLargeLlmModel: "OPENAI_LARGE_LLM_MODEL",
  openaiEmbeddingModel: "OPENAI_EMBEDDING_MODEL",

  bedrockRegion: "BEDROCK_REGION",
  bedrockAccessKeyId: "BEDROCK_ACCESS_KEY_ID",
  bedrockSecretAccessKey: "BEDROCK_SECRET_ACCESS_KEY",
  bedrockSessionToken: "BEDROCK_SESSION_TOKEN",
  bedrockSmallLlmModel: "BEDROCK_SMALL_LLM_MODEL",
  bedrockLargeLlmModel: "BEDROCK_LARGE_LLM_MODEL",
  bedrockEmbeddingModel: "BEDROCK_EMBEDDING_MODEL",

  embeddingsFullEndpoint: "EMBEDDINGS_FULL_ENDPOINT",
  embeddingsKey: "EMBEDDINGS_KEY",

  embeddingsSmallModel: "EMBEDDINGS_SMALL_MODEL",
  embeddingsLocalModel: "EMBEDDINGS_LOCAL_MODEL",
  embeddingNodesDimension: "EMBEDDING_NODES_DIMENSION",
  embeddingTripletsDimension: "EMBEDDING_TRIPLETS_DIMENSION",
  embeddingObservationsDimension: "EMBEDDING_OBSERVATIONS_DIMENSION",
  embeddingDataDimension: "EMBEDDING_DATA_DIMENSION",
  embeddingRelationshipsDimension: "EMBEDDING_RELATIONSHIPS_DIMENSION",
} as const;

export const SERVICE_COMPOSE_FILES = {
  redis: "src/lib/redis/docker-compose.yaml",
  postgresql: "src/lib/postgresql/docker-compose.yaml",
  neo4j: "src/lib/neo4j/docker-compose.yaml",
  milvus: "src/lib/milvus/docker-compose.yaml",
  mongo: "src/lib/mongo/docker-compose.yaml",
  rabbitmq: "src/lib/rabbitmq/docker-compose.yaml",
} as const;

export type ServiceName = keyof typeof SERVICE_COMPOSE_FILES;

export const API_DEFAULT_PORT = 8000;
export const MCP_DEFAULT_PORT = 8001;
