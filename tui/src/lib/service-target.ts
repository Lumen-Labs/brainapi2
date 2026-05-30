import type { Connections } from "../types.js";
import type { ServiceName } from "../constants.js";
import { SERVICE_DEFAULTS } from "../constants.js";

export interface ServiceTarget {
  host: string;
  port: number;
  managed?: boolean;
}

export function targetFromConnections(
  service: ServiceName,
  connections: Connections,
): ServiceTarget | null {
  switch (service) {
    case "redis":
      return {
        host: connections.redis.host,
        port: connections.redis.port,
      };
    case "postgresql":
      if (!connections.postgresql) return null;
      return {
        host: connections.postgresql.host,
        port: connections.postgresql.port,
      };
    case "neo4j":
      if (!connections.neo4j) return null;
      return {
        host: connections.neo4j.host,
        port: connections.neo4j.port,
      };
    case "milvus":
      if (!connections.milvus) return null;
      if (connections.milvus.uri) {
        return {
          host: connections.milvus.host,
          port: connections.milvus.port,
          managed: true,
        };
      }
      return {
        host: connections.milvus.host,
        port: connections.milvus.port,
      };
    case "mongo":
      if (!connections.mongo) return null;
      return {
        host: connections.mongo.host,
        port: connections.mongo.port,
      };
    case "rabbitmq":
      return {
        host: SERVICE_DEFAULTS.rabbitmq.host,
        port: SERVICE_DEFAULTS.rabbitmq.port,
      };
    default:
      return null;
  }
}

export function targetFromEnv(
  service: ServiceName,
  getter: (key: string) => string | undefined,
): ServiceTarget | null {
  const defaults = SERVICE_DEFAULTS;
  switch (service) {
    case "redis":
      return {
        host: getter("REDIS_HOST") ?? defaults.redis.host,
        port: Number(getter("REDIS_PORT") ?? defaults.redis.port),
      };
    case "postgresql":
      return {
        host: getter("POSTGRES_HOST") ?? defaults.postgresql.host,
        port: Number(getter("POSTGRES_PORT") ?? defaults.postgresql.port),
      };
    case "neo4j":
      return {
        host: getter("NEO4J_HOST") ?? defaults.neo4j.host,
        port: Number(getter("NEO4J_PORT") ?? defaults.neo4j.port),
      };
    case "milvus": {
      const uri = getter("MILVUS_URI");
      if (uri && uri.length > 0) {
        return {
          host: getter("MILVUS_HOST") ?? defaults.milvus.host,
          port: Number(getter("MILVUS_PORT") ?? defaults.milvus.port),
          managed: true,
        };
      }
      return {
        host: getter("MILVUS_HOST") ?? defaults.milvus.host,
        port: Number(getter("MILVUS_PORT") ?? defaults.milvus.port),
      };
    }
    case "mongo":
      return {
        host: getter("MONGO_HOST") ?? defaults.mongo.host,
        port: Number(getter("MONGO_PORT") ?? defaults.mongo.port),
      };
    case "rabbitmq":
      return {
        host: getter("RABBITMQ_HOST") ?? defaults.rabbitmq.host,
        port: Number(getter("RABBITMQ_PORT") ?? defaults.rabbitmq.port),
      };
    default:
      return null;
  }
}
