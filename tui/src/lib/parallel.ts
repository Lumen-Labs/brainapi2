import { spawn, type ChildProcess } from "node:child_process";
import pc from "picocolors";

export interface ParallelJob {
  name: string;
  bin: string;
  args: string[];
  cwd?: string;
  env?: NodeJS.ProcessEnv;
  color?: (s: string) => string;
}

export interface ParallelResult {
  name: string;
  code: number | null;
  signal: NodeJS.Signals | null;
}

const COLORS: Array<(s: string) => string> = [
  pc.cyan,
  pc.magenta,
  pc.yellow,
  pc.green,
  pc.blue,
  pc.red,
];

function makePrefixer(prefix: string): (chunk: Buffer | string) => string {
  let pending = "";
  return (chunk: Buffer | string): string => {
    const text = pending + (typeof chunk === "string" ? chunk : chunk.toString("utf8"));
    const lines = text.split(/\r?\n/);
    pending = lines.pop() ?? "";
    if (lines.length === 0) return "";
    return lines.map((line) => `${prefix} ${line}`).join("\n") + "\n";
  };
}

export async function runParallel(jobs: ParallelJob[]): Promise<ParallelResult[]> {
  if (jobs.length === 0) return [];

  const labelWidth = Math.max(...jobs.map((j) => j.name.length));
  const children: Array<{ name: string; child: ChildProcess }> = [];

  const cleanup = (signal: NodeJS.Signals): void => {
    for (const { child } of children) {
      if (!child.killed) {
        try {
          child.kill(signal);
        } catch {
        }
      }
    }
  };

  const onSigint = (): void => cleanup("SIGINT");
  const onSigterm = (): void => cleanup("SIGTERM");
  process.on("SIGINT", onSigint);
  process.on("SIGTERM", onSigterm);

  try {
    const results = await Promise.all(
      jobs.map((job, idx) => {
        const color = job.color ?? COLORS[idx % COLORS.length]!;
        const label = job.name.padEnd(labelWidth);
        const prefix = color(`[${label}]`);

        const child = spawn(job.bin, job.args, {
          cwd: job.cwd,
          env: { ...process.env, ...job.env },
          stdio: ["ignore", "pipe", "pipe"],
        });
        children.push({ name: job.name, child });

        const outPrefixer = makePrefixer(prefix);
        const errPrefixer = makePrefixer(prefix);
        child.stdout?.on("data", (chunk) => {
          const out = outPrefixer(chunk);
          if (out) process.stdout.write(out);
        });
        child.stderr?.on("data", (chunk) => {
          const out = errPrefixer(chunk);
          if (out) process.stderr.write(out);
        });

        return new Promise<ParallelResult>((resolve) => {
          child.on("exit", (code, signal) => {
            resolve({ name: job.name, code, signal });
          });
          child.on("error", (err) => {
            process.stderr.write(`${prefix} failed to spawn: ${err.message}\n`);
            resolve({ name: job.name, code: 127, signal: null });
          });
        });
      }),
    );

    return results;
  } finally {
    process.off("SIGINT", onSigint);
    process.off("SIGTERM", onSigterm);
  }
}
