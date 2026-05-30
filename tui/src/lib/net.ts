import net from "node:net";

export interface TcpProbeOptions {
  host: string;
  port: number;
  timeoutMs?: number;
}

export async function tcpProbe({
  host,
  port,
  timeoutMs = 1500,
}: TcpProbeOptions): Promise<boolean> {
  return new Promise<boolean>((resolve) => {
    const socket = new net.Socket();
    let settled = false;
    const finish = (ok: boolean) => {
      if (settled) return;
      settled = true;
      socket.destroy();
      resolve(ok);
    };
    socket.setTimeout(timeoutMs);
    socket.once("connect", () => finish(true));
    socket.once("timeout", () => finish(false));
    socket.once("error", () => finish(false));
    socket.connect(port, host);
  });
}

export interface HttpProbeOptions {
  url: string;
  timeoutMs?: number;
  expectStatus?: number;
}

export async function httpProbe({
  url,
  timeoutMs = 2000,
  expectStatus,
}: HttpProbeOptions): Promise<{ ok: boolean; status?: number; body?: string }> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const response = await fetch(url, { signal: controller.signal });
    const body = await response.text();
    const status = response.status;
    const matches = expectStatus !== undefined ? status === expectStatus : status < 500;
    return { ok: matches, status, body };
  } catch {
    return { ok: false };
  } finally {
    clearTimeout(timer);
  }
}
