import * as p from "@clack/prompts";
import pc from "picocolors";
import { sourcePath } from "../lib/paths.js";
import {
  defaultState,
  ensureStateDir,
  readState,
  updateState,
  writeState,
} from "../lib/state.js";
import { ensureRepo, isClonedRepo } from "../lib/git.js";
import { ensurePython } from "../lib/python-recovery.js";
import {
  createVenv,
  installDeps,
  venvExists,
} from "../lib/python.js";
import { runSetupWizard, toInitChoices, type SetupDraft } from "../flows/setup-wizard.js";
import {
  ocrToExtras,
  postgresBackendExtras,
} from "../flows/pipeline.js";
import { writeEnvFromChoices } from "../lib/write-env.js";
import { ensureDocker } from "../lib/docker-recovery.js";
import { bringUpServices, bringUpServicesManually } from "../lib/bring-up.js";
import { targetFromConnections } from "../lib/service-target.js";
import { askConfirm } from "../lib/prompts.js";
import type {
  Connections,
  DbChoices,
  InitChoices,
  ServicesRuntime,
} from "../types.js";
import { type ServiceName } from "../constants.js";

export interface InitOptions {
  repoUrl?: string;
  branch?: string;
  force?: boolean;
}

function intro(): void {
  p.intro(pc.bgCyan(pc.black(" brainapi ")) + " " + pc.dim("init"));
}

function outro(): void {
  p.outro(
    `${pc.green("Setup complete.")} Try ${pc.cyan("brainapi start")} when you're ready.`,
  );
}

function pickedServices(dbs: DbChoices): ServiceName[] {
  const set = new Set<ServiceName>(["redis"]);
  if (
    dbs.vectorDb === "postgresql" ||
    dbs.dataDb === "postgresql" ||
    dbs.graphDb === "networkx"
  ) {
    set.add("postgresql");
  }
  if (dbs.graphDb === "neo4j") set.add("neo4j");
  if (dbs.vectorDb === "milvus") set.add("milvus");
  if (dbs.dataDb === "mongo") set.add("mongo");
  return [...set];
}


async function clonePhase(opts: InitOptions): Promise<void> {
  if (await isClonedRepo()) {
    p.log.info("Repository already cloned — pulling latest");
  }
  const spinner = p.spinner();
  spinner.start("Cloning brainapi source");
  try {
    const state = await readState();
    const repoUrl = opts.repoUrl ?? state?.repoUrl ?? defaultState().repoUrl;
    const branch = opts.branch ?? state?.branch ?? defaultState().branch;
    const result = await ensureRepo({ repoUrl, branch });
    spinner.stop(
      result === "cloned"
        ? `Cloned to ${pc.cyan(sourcePath())}`
        : `Updated ${pc.cyan(sourcePath())}`,
    );
  } catch (err) {
    spinner.stop(pc.red("Clone failed."));
    throw err;
  }
}

async function ensureVenv(pythonBin: string): Promise<void> {
  const spinner = p.spinner();
  if (await venvExists()) {
    p.log.info("Virtualenv already exists — skipping creation");
    return;
  }
  spinner.start("Creating Python venv (.venv)");
  try {
    await createVenv(pythonBin);
    spinner.stop("Virtualenv created");
  } catch (err) {
    spinner.stop(pc.red("Failed to create venv"));
    throw err;
  }
}

async function installPhase(extras: string[]): Promise<void> {
  if (extras.length === 0) {
    p.log.info("Installing Python base dependencies");
  } else {
    p.log.info(
      `Installing Python base dependencies + extras (${extras.join(", ")})`,
    );
  }
  await installDeps({ extras });
  p.log.success("Dependencies installed");
}

async function configurePhase(): Promise<InitChoices> {
  const draft: SetupDraft = {};
  await runSetupWizard(draft, {
    firstStepBack: "cancel",
    firstStepBackHint: "Cancel setup",
    stepBackHint: "Previous step",
  });
  return toInitChoices(draft);
}

async function servicesPhase(
  dbs: DbChoices,
  connections: Connections,
  runtime: ServicesRuntime,
): Promise<void> {
  const services = pickedServices(dbs);
  const items = services.map((service) => ({
    service,
    target: targetFromConnections(service, connections),
  }));

  if (runtime === "manual") {
    p.log.step("Checking backing services (manual runtime)");
    await bringUpServicesManually(items);
    await updateState({
      containersStarted: false,
      selectedServices: services,
      servicesRuntime: "manual",
    });
    return;
  }

  const startNow = await askConfirm({
    message: "Start docker compose containers now for the services you picked?",
    initialValue: true,
  });
  if (!startNow) {
    p.log.info(
      "Skipping. Start them later with " +
        pc.cyan("brainapi start") +
        " or " +
        pc.cyan("docker compose -f src/lib/<service>/docker-compose.yaml up -d"),
    );
    await updateState({
      selectedServices: services,
      servicesRuntime: "docker",
    });
    return;
  }

  const ready = await ensureDocker();
  if (ready === "cancelled") {
    p.log.warn(
      "Docker is not available — setup is otherwise complete. Re-run " +
        pc.cyan("brainapi start") +
        " once Docker is up.",
    );
    return;
  }

  await bringUpServices(items);
  await updateState({
    containersStarted: true,
    selectedServices: services,
    servicesRuntime: "docker",
  });
}

export async function runInit(opts: InitOptions = {}): Promise<void> {
  intro();
  await ensureStateDir();
  const existing = await readState();
  if (existing && !opts.force) {
    p.log.info(
      "Resuming from previous install state at " + pc.cyan("~/.brainapi/state.json"),
    );
  } else if (!existing) {
    await writeState(defaultState());
  }

  let python = await ensurePython();
  await clonePhase(opts);

  const recheck = await ensurePython();
  if (recheck.bin !== python.bin) {
    p.log.info(`Using ${recheck.bin} (${recheck.version.join(".")})`);
  }
  python = recheck;

  await ensureVenv(python.bin);

  const choices = await configurePhase();
  const extras = [
    ...ocrToExtras(choices.pipeline.ocrMode),
    ...postgresBackendExtras(choices.dbs),
  ];

  await installPhase(extras);

  await writeEnvFromChoices(choices);
  await updateState({ envWritten: true });
  p.log.success(".env written");

  await servicesPhase(
    choices.dbs,
    choices.connections,
    choices.servicesRuntime,
  );

  outro();
}
