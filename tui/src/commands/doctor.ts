import { access } from "node:fs/promises";
import * as p from "@clack/prompts";
import pc from "picocolors";
import { detectPython } from "../lib/python.js";
import { detectDocker } from "../lib/docker.js";
import { tcpProbe } from "../lib/net.js";
import { isOllamaRunning } from "../lib/ollama.js";
import { readEnvFile } from "../lib/env.js";
import { envFilePath, venvPath } from "../lib/paths.js";
import { readState } from "../lib/state.js";

interface CheckResult {
  name: string;
  ok: boolean | "warn";
  detail?: string;
}

function getValue(
  env: Awaited<ReturnType<typeof readEnvFile>>,
  key: string,
): string | undefined {
  const idx = env.keyIndex.get(key);
  if (idx === undefined) return undefined;
  const raw = (env.lines[idx] ?? "").split("=").slice(1).join("=");
  return raw.trim().replace(/^"|"$/g, "");
}

async function fileExists(path: string): Promise<boolean> {
  try {
    await access(path);
    return true;
  } catch {
    return false;
  }
}

function statusGlyph(ok: boolean | "warn"): string {
  if (ok === true) return pc.green("OK");
  if (ok === "warn") return pc.yellow("WARN");
  return pc.red("FAIL");
}

export async function runDoctor(): Promise<void> {
  p.intro(pc.bgCyan(pc.black(" brainapi ")) + " " + pc.dim("doctor"));

  const results: CheckResult[] = [];

  const py = await detectPython();
  results.push({
    name: "Python >= 3.11",
    ok: py !== null,
    detail: py ? `${py.bin} (${py.version.join(".")})` : "Not found on PATH",
  });

  results.push({
    name: "Python venv",
    ok: await fileExists(venvPath()),
    detail: venvPath(),
  });

  const state = await readState();
  results.push({
    name: "Install state",
    ok: state !== null,
    detail: state ? "~/.brainapi/state.json present" : "missing",
  });

  let env: Awaited<ReturnType<typeof readEnvFile>> | null = null;
  if (await fileExists(envFilePath())) {
    env = await readEnvFile(envFilePath());
    results.push({ name: ".env", ok: true, detail: envFilePath() });
  } else {
    results.push({ name: ".env", ok: false, detail: "missing — run brainapi init" });
  }

  const runtime = state?.servicesRuntime ?? "docker";
  if (runtime === "docker") {
    const dockerState = await detectDocker();
    results.push({
      name: "Docker",
      ok: dockerState === "ok" ? true : dockerState === "daemon_down" ? "warn" : false,
      detail:
        dockerState === "ok"
          ? "installed and daemon up"
          : dockerState === "daemon_down"
            ? "installed but daemon not running"
            : "not installed",
    });
  } else {
    results.push({
      name: "Docker",
      ok: "warn",
      detail: "not required (services runtime = manual)",
    });
  }

  if (env) {
    const modelsMode = getValue(env, "MODELS_MODE");
    if (modelsMode === "local") {
      const host = getValue(env, "OLLAMA_HOST") ?? "localhost";
      const port = Number(getValue(env, "OLLAMA_PORT") ?? "11434");
      const ok = await isOllamaRunning({ host, port });
      results.push({
        name: "Ollama",
        ok,
        detail: `http://${host}:${port}`,
      });
    } else if (modelsMode === "remote") {
      const credPath = getValue(env, "GCP_CREDENTIALS_PATH");
      results.push({
        name: "GCP credentials",
        ok: credPath ? await fileExists(credPath) : false,
        detail: credPath ?? "GCP_CREDENTIALS_PATH not set",
      });
    }

    const services: Array<[string, string, string]> = [
      ["Redis", getValue(env, "REDIS_HOST") ?? "localhost", getValue(env, "REDIS_PORT") ?? "6379"],
    ];
    if (
      getValue(env, "VECTOR_DB") === "postgresql" ||
      getValue(env, "DATA_DB") === "postgresql" ||
      getValue(env, "GRAPH_DB") === "networkx"
    ) {
      services.push([
        "PostgreSQL",
        getValue(env, "POSTGRES_HOST") ?? "localhost",
        getValue(env, "POSTGRES_PORT") ?? "5432",
      ]);
    }
    if (getValue(env, "GRAPH_DB") === "neo4j") {
      services.push([
        "Neo4j",
        getValue(env, "NEO4J_HOST") ?? "localhost",
        getValue(env, "NEO4J_PORT") ?? "7687",
      ]);
    }
    if (getValue(env, "VECTOR_DB") === "milvus") {
      services.push([
        "Milvus",
        getValue(env, "MILVUS_HOST") ?? "localhost",
        getValue(env, "MILVUS_PORT") ?? "19530",
      ]);
    }
    if (getValue(env, "DATA_DB") === "mongo") {
      services.push([
        "Mongo",
        getValue(env, "MONGO_HOST") ?? "localhost",
        getValue(env, "MONGO_PORT") ?? "27017",
      ]);
    }
    if (getValue(env, "CELERY_BACKEND") === "rabbitmq") {
      services.push([
        "RabbitMQ",
        getValue(env, "RABBITMQ_HOST") ?? "localhost",
        getValue(env, "RABBITMQ_PORT") ?? "5672",
      ]);
    }

    for (const [name, host, portRaw] of services) {
      const port = Number(portRaw);
      const ok = await tcpProbe({ host, port });
      results.push({
        name,
        ok,
        detail: `${host}:${port}`,
      });
    }
  }

  const lines = results.map((r) => `${statusGlyph(r.ok)}  ${r.name}${r.detail ? pc.dim(` — ${r.detail}`) : ""}`);
  p.note(lines.join("\n"), "Diagnostics");

  const fails = results.filter((r) => r.ok === false).length;
  if (fails === 0) {
    p.outro(pc.green("All checks passed."));
  } else {
    p.outro(pc.red(`${fails} check(s) failed.`));
    process.exit(1);
  }
}
