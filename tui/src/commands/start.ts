import * as p from "@clack/prompts";
import pc from "picocolors";
import { readState } from "../lib/state.js";
import { ensureDocker } from "../lib/docker-recovery.js";
import { bringUpServices, bringUpServicesManually } from "../lib/bring-up.js";
import { targetFromEnv } from "../lib/service-target.js";
import { envFileToProcessEnv, readEnvFile, setPipelineMode } from "../lib/env.js";
import { syncDevSourceIfAvailable } from "../lib/sync-source.js";
import { envFilePath, sourcePath, venvBin } from "../lib/paths.js";
import {
  installDeps,
  missingRequiredPythonDeps,
} from "../lib/python.js";
import { runParallel } from "../lib/parallel.js";
import {
  API_DEFAULT_PORT,
  MCP_DEFAULT_PORT,
  SERVICE_COMPOSE_FILES,
  type ServiceName,
} from "../constants.js";
import type { PipelineMode, ServicesRuntime } from "../types.js";

function getEnv(
  env: Awaited<ReturnType<typeof readEnvFile>>,
  key: string,
): string | undefined {
  const idx = env.keyIndex.get(key);
  if (idx === undefined) return undefined;
  const raw = (env.lines[idx] ?? "").split("=").slice(1).join("=");
  return raw.trim().replace(/^"|"$/g, "");
}

function pickedFromEnv(env: Awaited<ReturnType<typeof readEnvFile>>): ServiceName[] {
  const services = new Set<ServiceName>(["redis"]);
  const vector = getEnv(env, "VECTOR_DB");
  const data = getEnv(env, "DATA_DB");
  const graph = getEnv(env, "GRAPH_DB");
  if (vector === "postgresql" || data === "postgresql" || graph === "networkx") {
    services.add("postgresql");
  }
  if (graph === "neo4j") services.add("neo4j");
  if (vector === "milvus") services.add("milvus");
  if (data === "mongo") services.add("mongo");
  if (getEnv(env, "CELERY_BACKEND") === "rabbitmq") services.add("rabbitmq");
  return [...services].filter((s) => s in SERVICE_COMPOSE_FILES);
}

export interface StartOptions {
  api?: boolean;
  mcp?: boolean;
  worker?: boolean;
  services?: boolean;
  pipelineMode?: PipelineMode;
}

export async function runStart(opts: StartOptions = {}): Promise<void> {
  const wantApi = opts.api !== false;
  const wantMcp = opts.mcp !== false;
  const wantWorker = opts.worker !== false;
  const wantServices = opts.services !== false;

  p.intro(pc.bgCyan(pc.black(" brainapi ")) + " " + pc.dim("start"));

  await syncDevSourceIfAvailable();

  const state = await readState();
  if (!state || !state.envWritten) {
    p.cancel(
      "No brainapi install detected. Run " + pc.cyan("brainapi init") + " first.",
    );
    process.exit(1);
  }

  if (opts.pipelineMode) {
    await setPipelineMode(opts.pipelineMode);
    p.log.info(`PIPELINE_MODE set to ${pc.cyan(opts.pipelineMode)}`);
  }

  const env = await readEnvFile(envFilePath());

  if (await missingRequiredPythonDeps()) {
    p.log.info("Installing missing Python dependencies for current .env...");
    try {
      await installDeps({ skipBase: true });
      p.log.success("Python dependencies installed.");
    } catch (err) {
      p.log.error(
        `Failed to install Python dependencies: ${
          err instanceof Error ? err.message : String(err)
        }`,
      );
    }
  }

  if (wantServices) {
    const services = pickedFromEnv(env);
    const runtime: ServicesRuntime = state.servicesRuntime ?? "docker";
    const getter = (key: string) => getEnv(env, key);
    const items = services.map((service) => ({
      service,
      target: targetFromEnv(service, getter),
    }));

    if (runtime === "manual") {
      p.log.step("Checking backing services (manual runtime)");
      await bringUpServicesManually(items);
    } else {
      const ready = await ensureDocker();
      if (ready === "cancelled") {
        p.cancel("Docker is not available. Aborting start.");
        process.exit(1);
      }
      await bringUpServices(items);
    }
  } else {
    p.log.info("Skipping backing services (--no-services).");
  }

  if (!wantApi && !wantMcp && !wantWorker) {
    p.outro(pc.green("Services up. Skipping API/MCP/worker."));
    return;
  }

  const processEnv = envFileToProcessEnv(env);
  const jobs = buildJobs({ wantApi, wantMcp, wantWorker, processEnv });

  p.note(
    jobs
      .map((j) => `${pc.bold(j.name.padEnd(7))} ${pc.dim(`${j.bin} ${j.args.join(" ")}`)}`)
      .join("\n"),
    "Starting (Ctrl-C to stop everything)",
  );

  const results = await runParallel(jobs);
  const failed = results.filter((r) => r.code !== 0 && r.signal === null);
  if (failed.length > 0) {
    p.outro(
      pc.red(
        `Exited: ${failed.map((f) => `${f.name}=${f.code}`).join(", ")}`,
      ),
    );
    process.exit(1);
  }
  p.outro(pc.green("All processes exited cleanly."));
}

function buildJobs(opts: {
  wantApi: boolean;
  wantMcp: boolean;
  wantWorker: boolean;
  processEnv: NodeJS.ProcessEnv;
}): Array<{
  name: string;
  bin: string;
  args: string[];
  cwd: string;
  env: NodeJS.ProcessEnv;
}> {
  const jobs: Array<{
    name: string;
    bin: string;
    args: string[];
    cwd: string;
    env: NodeJS.ProcessEnv;
  }> = [];
  const cwd = sourcePath();

  if (opts.wantApi) {
    jobs.push({
      name: "api",
      bin: venvBin("uvicorn"),
      args: [
        "src.services.api.app:app",
        "--host",
        "0.0.0.0",
        "--port",
        String(API_DEFAULT_PORT),
        "--reload",
      ],
      cwd,
      env: opts.processEnv,
    });
  }
  if (opts.wantMcp) {
    jobs.push({
      name: "mcp",
      bin: venvBin("uvicorn"),
      args: [
        "src.services.mcp.app:app",
        "--host",
        "0.0.0.0",
        "--port",
        String(MCP_DEFAULT_PORT),
        "--reload",
      ],
      cwd,
      env: opts.processEnv,
    });
  }
  if (opts.wantWorker) {
    jobs.push({
      name: "worker",
      bin: venvBin("celery"),
      args: [
        "-A",
        "src.workers.app",
        "worker",
        "--loglevel=info",
        "--pool=threads",
      ],
      cwd,
      env: opts.processEnv,
    });
  }
  return jobs;
}
