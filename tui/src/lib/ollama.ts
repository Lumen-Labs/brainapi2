import { httpProbe } from "./net.js";
import { runInherit, which } from "./exec.js";

export interface OllamaTarget {
  host: string;
  port: number;
}

export interface OllamaPullResult {
  ok: boolean;
  exitCode: number;
  error?: string;
}

export async function pullOllamaModel(modelName: string): Promise<OllamaPullResult> {
  const bin = (await which("ollama")) ? "ollama" : null;
  if (!bin) {
    return {
      ok: false,
      exitCode: 127,
      error: "ollama CLI not found on PATH",
    };
  }
  const result = await runInherit(bin, ["pull", modelName]);
  if (result.ok) {
    return { ok: true, exitCode: result.exitCode };
  }
  return {
    ok: false,
    exitCode: result.exitCode,
    error: result.stderr || `ollama pull exited with code ${result.exitCode}`,
  };
}

export async function isOllamaRunning(target: OllamaTarget): Promise<boolean> {
  const result = await httpProbe({
    url: `http://${target.host}:${target.port}/api/tags`,
    expectStatus: 200,
  });
  return result.ok;
}

export async function listOllamaModels(target: OllamaTarget): Promise<string[]> {
  const result = await httpProbe({
    url: `http://${target.host}:${target.port}/api/tags`,
    expectStatus: 200,
  });
  if (!result.ok || !result.body) return [];
  try {
    const json = JSON.parse(result.body) as { models?: Array<{ name?: string }> };
    return (json.models ?? [])
      .map((m) => m.name?.toString() ?? "")
      .filter((name) => name.length > 0);
  } catch {
    return [];
  }
}

export async function hasOllamaModel(
  target: OllamaTarget,
  modelName: string,
): Promise<boolean> {
  const models = await listOllamaModels(target);
  if (models.includes(modelName)) return true;
  const base = modelName.split(":")[0];
  return base !== undefined && models.some((m) => m.split(":")[0] === base);
}
