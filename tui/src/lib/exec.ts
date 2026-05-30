import { execa, type ExecaError, type Options } from "execa";

export type RunResult = {
  exitCode: number;
  stdout: string;
  stderr: string;
  ok: boolean;
};

export async function runQuiet(
  bin: string,
  args: string[] = [],
  options: Options = {},
): Promise<RunResult> {
  try {
    const result = await execa(bin, args, {
      reject: false,
      stdout: "pipe",
      stderr: "pipe",
      ...options,
    });
    return {
      exitCode: result.exitCode ?? 1,
      stdout: result.stdout?.toString() ?? "",
      stderr: result.stderr?.toString() ?? "",
      ok: (result.exitCode ?? 1) === 0,
    };
  } catch (err) {
    const ee = err as ExecaError;
    return {
      exitCode: typeof ee.exitCode === "number" ? ee.exitCode : 1,
      stdout: ee.stdout?.toString() ?? "",
      stderr: ee.stderr?.toString() ?? (ee.message ?? ""),
      ok: false,
    };
  }
}

export async function runInherit(
  bin: string,
  args: string[] = [],
  options: Options = {},
): Promise<RunResult> {
  try {
    const result = await execa(bin, args, {
      reject: false,
      stdout: "inherit",
      stderr: "inherit",
      ...options,
    });
    return {
      exitCode: result.exitCode ?? 1,
      stdout: "",
      stderr: "",
      ok: (result.exitCode ?? 1) === 0,
    };
  } catch (err) {
    const ee = err as ExecaError;
    return {
      exitCode: typeof ee.exitCode === "number" ? ee.exitCode : 1,
      stdout: "",
      stderr: ee.message ?? "",
      ok: false,
    };
  }
}

export async function which(bin: string): Promise<boolean> {
  const finder = process.platform === "win32" ? "where" : "which";
  const result = await runQuiet(finder, [bin]);
  return result.ok && result.stdout.trim().length > 0;
}
