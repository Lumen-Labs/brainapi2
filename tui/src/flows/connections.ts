import * as p from "@clack/prompts";
import { SERVICE_DEFAULTS } from "../constants.js";
import { askPassword, askText, pickOne } from "../lib/prompts.js";
import type {
  Connections,
  DbChoices,
  MilvusConnection,
  MongoConnection,
  Neo4jConnection,
  PostgresConnection,
  RedisConnection,
} from "../types.js";

function portValidator(value: string): string | undefined {
  const n = Number(value);
  if (!Number.isInteger(n) || n <= 0 || n > 65535)
    return "Port must be 1-65535";
  return undefined;
}

async function askRedis(): Promise<RedisConnection> {
  p.log.step("Redis (required for cache + Celery)");
  const host = await askText({
    message: "Redis host",
    placeholder: SERVICE_DEFAULTS.redis.host,
    defaultValue: SERVICE_DEFAULTS.redis.host,
  });
  const port = await askText({
    message: "Redis port",
    placeholder: String(SERVICE_DEFAULTS.redis.port),
    defaultValue: String(SERVICE_DEFAULTS.redis.port),
    validate: portValidator,
  });
  return {
    host: host.trim() || SERVICE_DEFAULTS.redis.host,
    port: Number(port) || SERVICE_DEFAULTS.redis.port,
  };
}

async function askPostgres(): Promise<PostgresConnection> {
  p.log.step("PostgreSQL");
  p.note(
    [
      "BrainAPI uses a single Postgres database to host all brains.",
      "Each brain is isolated logically via a `brain_id` column, not via separate databases.",
      "Press Enter to accept the default name.",
    ].join("\n"),
    "Heads up"
  );
  const host = await askText({
    message: "Postgres host",
    placeholder: SERVICE_DEFAULTS.postgresql.host,
    defaultValue: SERVICE_DEFAULTS.postgresql.host,
  });
  const port = await askText({
    message: "Postgres port",
    placeholder: String(SERVICE_DEFAULTS.postgresql.port),
    defaultValue: String(SERVICE_DEFAULTS.postgresql.port),
    validate: portValidator,
  });
  const username = await askText({
    message: "Postgres username",
    placeholder: SERVICE_DEFAULTS.postgresql.username,
    defaultValue: SERVICE_DEFAULTS.postgresql.username,
  });
  const password = await askPassword({
    message: "Postgres password (default is 'password')",
    validate: (value) =>
      value.length === 0 ? "Password is required" : undefined,
  });
  const systemDatabase = await askText({
    message: `System database (brains registry, default ${SERVICE_DEFAULTS.postgresql.systemDatabase})`,
    placeholder: SERVICE_DEFAULTS.postgresql.systemDatabase,
    defaultValue: SERVICE_DEFAULTS.postgresql.systemDatabase,
  });
  const maintenanceDatabase = await askText({
    message: `Maintenance database (for CREATE DATABASE, default ${SERVICE_DEFAULTS.postgresql.maintenanceDatabase})`,
    placeholder: SERVICE_DEFAULTS.postgresql.maintenanceDatabase,
    defaultValue: SERVICE_DEFAULTS.postgresql.maintenanceDatabase,
  });
  return {
    host: host.trim() || SERVICE_DEFAULTS.postgresql.host,
    port: Number(port) || SERVICE_DEFAULTS.postgresql.port,
    username: username.trim() || SERVICE_DEFAULTS.postgresql.username,
    password,
    systemDatabase:
      systemDatabase.trim() || SERVICE_DEFAULTS.postgresql.systemDatabase,
    maintenanceDatabase:
      maintenanceDatabase.trim() ||
      SERVICE_DEFAULTS.postgresql.maintenanceDatabase,
  };
}

async function askNeo4j(): Promise<Neo4jConnection> {
  p.log.step("Neo4j");
  const host = await askText({
    message: "Neo4j host",
    placeholder: SERVICE_DEFAULTS.neo4j.host,
    defaultValue: SERVICE_DEFAULTS.neo4j.host,
  });
  const port = await askText({
    message: "Neo4j Bolt port",
    placeholder: String(SERVICE_DEFAULTS.neo4j.port),
    defaultValue: String(SERVICE_DEFAULTS.neo4j.port),
    validate: portValidator,
  });
  const username = await askText({
    message: "Neo4j username",
    placeholder: SERVICE_DEFAULTS.neo4j.username,
    defaultValue: SERVICE_DEFAULTS.neo4j.username,
  });
  const password = await askPassword({
    message: "Neo4j password (default is 'your_password')",
    validate: (value) =>
      value.length === 0 ? "Password is required" : undefined,
  });
  return {
    host: host.trim() || SERVICE_DEFAULTS.neo4j.host,
    port: Number(port) || SERVICE_DEFAULTS.neo4j.port,
    username: username.trim() || SERVICE_DEFAULTS.neo4j.username,
    password,
  };
}

async function askMilvus(): Promise<MilvusConnection> {
  p.log.step("Milvus");
  const target = await pickOne<"local" | "managed">({
    message: "Milvus deployment",
    options: [
      { value: "local", label: "Local docker compose" },
      { value: "managed", label: "Managed (Zilliz Cloud) — uri + token" },
    ],
    initialValue: "local",
  });

  if (target === "managed") {
    const uri = await askText({
      message: "Milvus URI",
      placeholder: "https://your-cluster.api.gcp-us-west1.zillizcloud.com",
      validate: (value) =>
        value.trim().length === 0 ? "URI is required" : undefined,
    });
    const token = await askPassword({
      message: "Milvus token",
      validate: (value) =>
        value.length === 0 ? "Token is required" : undefined,
    });
    return {
      host: SERVICE_DEFAULTS.milvus.host,
      port: SERVICE_DEFAULTS.milvus.port,
      uri: uri.trim(),
      token,
    };
  }

  const host = await askText({
    message: "Milvus host",
    placeholder: SERVICE_DEFAULTS.milvus.host,
    defaultValue: SERVICE_DEFAULTS.milvus.host,
  });
  const port = await askText({
    message: "Milvus port",
    placeholder: String(SERVICE_DEFAULTS.milvus.port),
    defaultValue: String(SERVICE_DEFAULTS.milvus.port),
    validate: portValidator,
  });
  return {
    host: host.trim() || SERVICE_DEFAULTS.milvus.host,
    port: Number(port) || SERVICE_DEFAULTS.milvus.port,
  };
}

async function askMongo(): Promise<MongoConnection> {
  p.log.step("MongoDB");
  const host = await askText({
    message: "Mongo host",
    placeholder: SERVICE_DEFAULTS.mongo.host,
    defaultValue: SERVICE_DEFAULTS.mongo.host,
  });
  const port = await askText({
    message: "Mongo port",
    placeholder: String(SERVICE_DEFAULTS.mongo.port),
    defaultValue: String(SERVICE_DEFAULTS.mongo.port),
    validate: portValidator,
  });
  const username = await askText({
    message: "Mongo username",
    placeholder: SERVICE_DEFAULTS.mongo.username,
    defaultValue: SERVICE_DEFAULTS.mongo.username,
  });
  const password = await askPassword({
    message: "Mongo password (default is 'password')",
    validate: (value) =>
      value.length === 0 ? "Password is required" : undefined,
  });
  return {
    host: host.trim() || SERVICE_DEFAULTS.mongo.host,
    port: Number(port) || SERVICE_DEFAULTS.mongo.port,
    username: username.trim() || SERVICE_DEFAULTS.mongo.username,
    password,
  };
}

export async function askConnections(dbs: DbChoices): Promise<Connections> {
  const redis = await askRedis();
  const result: Connections = { redis };

  const needsPostgres =
    dbs.vectorDb === "postgresql" ||
    dbs.dataDb === "postgresql" ||
    dbs.graphDb === "networkx";
  if (needsPostgres) {
    result.postgresql = await askPostgres();
  }
  if (dbs.graphDb === "neo4j") {
    result.neo4j = await askNeo4j();
  }
  if (dbs.vectorDb === "milvus") {
    result.milvus = await askMilvus();
  }
  if (dbs.dataDb === "mongo") {
    result.mongo = await askMongo();
  }

  return result;
}
