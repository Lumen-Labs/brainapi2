import { access, mkdir } from "node:fs/promises";
import path from "node:path";
import { simpleGit } from "simple-git";
import { brainapiHome, sourcePath } from "./paths.js";
import { updateState } from "./state.js";

export interface CloneOptions {
  repoUrl: string;
  branch: string;
  depth?: number;
  onProgress?: (line: string) => void;
}

export async function isClonedRepo(): Promise<boolean> {
  try {
    await access(path.join(sourcePath(), ".git"));
    return true;
  } catch {
    return false;
  }
}

export async function cloneRepo(opts: CloneOptions): Promise<void> {
  await mkdir(brainapiHome(), { recursive: true });
  const target = sourcePath();
  const git = simpleGit({
    progress: ({ method, stage, progress }) => {
      opts.onProgress?.(`${method} ${stage} ${progress}%`);
    },
  });
  await git.clone(opts.repoUrl, target, [
    "--branch",
    opts.branch,
    ...(opts.depth ? ["--depth", String(opts.depth)] : []),
  ]);
  await updateState({ cloned: true, repoUrl: opts.repoUrl, branch: opts.branch });
}

export async function pullRepo(branch: string): Promise<void> {
  const git = simpleGit(sourcePath());
  const remoteRef = `origin/${branch}`;
  await git.fetch("origin", branch);
  await git.clean("f", ["-d", "-e", ".env", "-e", ".venv"]);
  await git.checkout(["-B", branch, "-f"]);
  await git.reset(["--hard", remoteRef]);
}

export async function ensureRepo(opts: CloneOptions): Promise<"cloned" | "pulled"> {
  if (await isClonedRepo()) {
    await pullRepo(opts.branch);
    return "pulled";
  }
  await cloneRepo(opts);
  return "cloned";
}
