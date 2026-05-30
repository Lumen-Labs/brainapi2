export type GraphDb = "networkx" | "neo4j";
export type VectorDb = "postgresql" | "milvus";
export type DataDb = "postgresql" | "mongo";
export type ModelsMode = "local" | "remote";
export type ModelProvider = "ollama" | "azure" | "openai" | "gcp_vertex" | "amazon_bedrock";
export type OcrMode = "docparser" | "docling";
export type PipelineMode = "accurate" | "lightweight";
export type ServicesRuntime = "docker" | "manual";

export function isPipelineMode(value: string): value is PipelineMode {
  return value === "accurate" || value === "lightweight";
}

export interface DbChoices {
  vectorDb: VectorDb;
  dataDb: DataDb;
  graphDb: GraphDb;
}

export interface OllamaChoices {
  host: string;
  port: number;
  smallModel: string;
  largeModel: string;
  embeddingsLocalModel: string;
}

export interface GcpChoices {
  credentialsPath: string;
  projectId: string;
  smallLlmModel: string;
  largeLlmModel: string;
  embeddingModel: string;
}

export interface AzureChoices {
  smallLlmModel: string;
  largeLlmModel: string;
  llmApiVersion: string;
  llmEndpoint: string;
  llmSubscriptionKey: string;
  embeddingEndpoint: string;
  embeddingKey: string;
  embeddingModel: string;
}

export interface OpenAIChoices {
  apiKey: string;
  baseUrl?: string;
  smallLlmModel: string;
  largeLlmModel: string;
  embeddingModel: string;
}

export interface BedrockChoices {
  region: string;
  accessKeyId: string;
  secretAccessKey: string;
  sessionToken?: string;
  smallLlmModel: string;
  largeLlmModel: string;
  embeddingModel: string;
}

export interface EmbeddingDimensions {
  nodesDimension: number;
  tripletsDimension: number;
  observationsDimension: number;
  dataDimension: number;
  relationshipsDimension: number;
}

export interface ModelsChoices {
  mode: ModelsMode;
  llmSmallProvider: ModelProvider;
  llmLargeProvider: ModelProvider;
  embeddingsProvider: ModelProvider;
  embeddingDimensions?: EmbeddingDimensions;
  ollama?: OllamaChoices;
  gcp?: GcpChoices;
  azure?: AzureChoices;
  openai?: OpenAIChoices;
  bedrock?: BedrockChoices;
}

export interface PostgresConnection {
  host: string;
  port: number;
  username: string;
  password: string;
  systemDatabase: string;
  maintenanceDatabase: string;
}

export interface Neo4jConnection {
  host: string;
  port: number;
  username: string;
  password: string;
}

export interface MilvusConnection {
  host: string;
  port: number;
  uri?: string;
  token?: string;
}

export interface MongoConnection {
  host: string;
  port: number;
  username: string;
  password: string;
}

export interface RedisConnection {
  host: string;
  port: number;
}

export interface Connections {
  redis: RedisConnection;
  postgresql?: PostgresConnection;
  neo4j?: Neo4jConnection;
  milvus?: MilvusConnection;
  mongo?: MongoConnection;
}

export interface AuthChoices {
  brainpatToken: string;
}

export interface PipelineChoices {
  ocrMode: OcrMode;
}

export interface InitChoices {
  dbs: DbChoices;
  models: ModelsChoices;
  pipeline: PipelineChoices;
  connections: Connections;
  auth: AuthChoices;
  servicesRuntime: ServicesRuntime;
  usedDefaults: boolean;
}

export interface InstallState {
  cloned: boolean;
  venvCreated: boolean;
  depsInstalled: boolean;
  envWritten: boolean;
  containersStarted: boolean;
  selectedServices: string[] | null;
  servicesRuntime: ServicesRuntime | null;
  sourcePath: string;
  repoUrl: string;
  branch: string;
}

export type Platform = "darwin" | "linux" | "win32";
