import * as p from "@clack/prompts";
import pc from "picocolors";
import { composeFileExists, composeFileFor, composeUp } from "./docker.js";
import { tcpProbe } from "./net.js";
import { nativeRecipeFor } from "./native-install.js";
import { detectCapabilities } from "./platform.js";
import type { ServiceName } from "../constants.js";
import type { ServiceTarget } from "./service-target.js";

export interface BringUpInput {
  service: ServiceName;
  target: ServiceTarget | null;
}

export async function bringUpServices(items: BringUpInput[]): Promise<void> {
  for (const { service, target } of items) {
    if (target?.managed) {
      p.log.info(
        `${service}: managed deployment configured — skipping local docker compose up`,
      );
      continue;
    }

    if (!(await composeFileExists(service))) {
      p.log.warn(
        `${service}: compose file not found at ${pc.cyan(
          composeFileFor(service),
        )}. Skipping container start.`,
      );
      p.log.info(
        `Fix: push ${pc.cyan(
          `src/lib/${service}/docker-compose.yaml`,
        )} to your remote, then re-run ${pc.cyan(
          "brainapi update",
        )}. Or copy it manually into ${pc.cyan("~/.brainapi/source/")}.`,
      );
      continue;
    }

    if (target && (await tcpProbe({ host: target.host, port: target.port, timeoutMs: 750 }))) {
      p.log.info(
        `${service} already reachable on ${target.host}:${target.port} — skipping ${pc.cyan(
          "docker compose up",
        )} (using existing service).`,
      );
      continue;
    }

    const spinner = p.spinner();
    spinner.start(`docker compose up -d ${service}`);
    try {
      await composeUp(service);
      spinner.stop(`${service} up`);
    } catch (err) {
      spinner.stop(pc.red(`${service} failed to start`));
      p.log.error(err instanceof Error ? err.message : String(err));
    }
  }
}

const DEFAULT_WAIT_SECONDS = 60;
const PROBE_INTERVAL_MS = 1500;

export async function bringUpServicesManually(
  items: BringUpInput[],
): Promise<void> {
  const caps = await detectCapabilities();

  for (const { service, target } of items) {
    if (target?.managed) {
      p.log.info(
        `${service}: managed deployment configured — using ${pc.cyan(
          `${target.host}:${target.port}`,
        )}`,
      );
      continue;
    }

    if (!target) {
      p.log.warn(`${service}: no connection details configured. Skipping.`);
      continue;
    }

    if (await tcpProbe({ host: target.host, port: target.port, timeoutMs: 750 })) {
      p.log.success(
        `${service} reachable on ${pc.cyan(`${target.host}:${target.port}`)}`,
      );
      continue;
    }

    p.log.warn(
      `${service} is not running on ${pc.cyan(`${target.host}:${target.port}`)}.`,
    );
    printNativeRecipe(service, caps);

    const ok = await waitForReachable(
      service,
      target.host,
      target.port,
      DEFAULT_WAIT_SECONDS,
    );
    if (!ok) {
      p.log.warn(
        `${service} still not reachable after ${DEFAULT_WAIT_SECONDS}s — continuing anyway. ` +
          `BrainAPI will fail at runtime until you start ${pc.cyan(service)} on ${pc.cyan(
            `${target.host}:${target.port}`,
          )}.`,
      );
    }
  }
}

function printNativeRecipe(
  service: ServiceName,
  caps: Awaited<ReturnType<typeof detectCapabilities>>,
): void {
  const recipe = nativeRecipeFor(service, caps);
  if (!recipe) {
    p.log.info(
      `No native install recipe known for ${pc.cyan(
        service,
      )} on your platform. Please install it manually.`,
    );
    return;
  }
  if (recipe.unsupported) {
    p.log.warn(`${service}: ${recipe.unsupported}`);
    if (recipe.docsUrl) p.log.info(`Docs: ${pc.cyan(recipe.docsUrl)}`);
    return;
  }
  const lines: string[] = [`Install ${pc.bold(service)}:`];
  for (const cmd of recipe.install) lines.push(`  ${pc.cyan(cmd)}`);
  if (recipe.start.length > 0) {
    lines.push("", "Start it:");
    for (const cmd of recipe.start) lines.push(`  ${pc.cyan(cmd)}`);
  }
  if (recipe.postInstall && recipe.postInstall.length > 0) {
    lines.push("", "Post-install:");
    for (const cmd of recipe.postInstall) lines.push(`  ${pc.dim(cmd)}`);
  }
  if (recipe.docsUrl) {
    lines.push("", `Docs: ${pc.cyan(recipe.docsUrl)}`);
  }
  p.note(lines.join("\n"), `${service} — manual install`);
}

async function waitForReachable(
  service: ServiceName,
  host: string,
  port: number,
  totalSeconds: number,
): Promise<boolean> {
  const spinner = p.spinner();
  spinner.start(`Waiting for ${service} to become reachable on ${host}:${port}`);
  const deadline = Date.now() + totalSeconds * 1000;
  while (Date.now() < deadline) {
    if (await tcpProbe({ host, port, timeoutMs: 750 })) {
      spinner.stop(`${service} reachable on ${host}:${port}`);
      return true;
    }
    await new Promise((resolve) => setTimeout(resolve, PROBE_INTERVAL_MS));
  }
  spinner.stop(pc.yellow(`${service} not reachable after ${totalSeconds}s`));
  return false;
}
