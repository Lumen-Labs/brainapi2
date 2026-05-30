import { clearSession, loadSession, normalizeBaseUrl, type AuthSession } from "./auth";

export class ApiError extends Error {
  status: number;

  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

export interface BrainRecord {
  id?: string;
  name_key: string;
  pat?: string;
}

export interface ApiFetchOptions {
  logoutOn401?: boolean;
}

let session: AuthSession | null = loadSession();

export function getSession(): AuthSession | null {
  return session;
}

export function setSession(next: AuthSession | null): void {
  session = next;
}

function authHeaders(pat: string, brainId?: string, path?: string): Headers {
  const headers = new Headers();
  headers.set("BrainPAT", pat);
  headers.set("Authorization", `Bearer ${pat}`);
  if (brainId && path && !path.startsWith("/system")) {
    headers.set("X-Brain-ID", brainId);
  }
  return headers;
}

function normalizeBrains(data: unknown): BrainRecord[] {
  if (Array.isArray(data)) {
    return data.filter(
      (item): item is BrainRecord =>
        typeof item === "object" &&
        item !== null &&
        typeof (item as BrainRecord).name_key === "string",
    );
  }
  if (typeof data === "object" && data !== null && "brains" in data) {
    return normalizeBrains((data as { brains: unknown }).brains);
  }
  return [];
}

export async function probeSystemPat(
  pat: string,
  apiBaseUrl: string,
): Promise<boolean> {
  const token = pat.trim();
  const url = `${normalizeBaseUrl(apiBaseUrl)}/system/brains-list`;
  const response = await fetch(url, {
    headers: authHeaders(token),
  });
  return response.ok;
}

export async function fetchBrainsList(
  s: AuthSession = session!,
): Promise<BrainRecord[]> {
  const pat = s.pat.trim();
  const url = `${normalizeBaseUrl(s.apiBaseUrl)}/system/brains-list`;
  const response = await fetch(url, {
    headers: authHeaders(pat),
  });
  if (!response.ok) {
    throw new ApiError(response.status, `Failed to load brains (${response.status})`);
  }
  const data = await response.json();
  return normalizeBrains(data);
}

export async function apiFetch<T>(
  path: string,
  options: RequestInit = {},
  overrideBrainId?: string,
  fetchOptions: ApiFetchOptions = {},
): Promise<T> {
  if (!session) {
    throw new ApiError(401, "Not authenticated");
  }

  const headers = authHeaders(session.pat, overrideBrainId ?? session.brainId, path);
  for (const [key, value] of new Headers(options.headers).entries()) {
    headers.set(key, value);
  }
  if (options.body && !headers.has("Content-Type")) {
    headers.set("Content-Type", "application/json");
  }

  const url = `${normalizeBaseUrl(session.apiBaseUrl)}${path}`;
  const response = await fetch(url, { ...options, headers });

  if (response.status === 401 && fetchOptions.logoutOn401 !== false) {
    clearSession();
    setSession(null);
    if (!window.location.pathname.endsWith("/login")) {
      window.location.href = "/console/login";
    }
    throw new ApiError(401, "Unauthorized");
  }

  const text = await response.text();
  let data: unknown = null;
  if (text) {
    try {
      data = JSON.parse(text);
    } catch {
      data = text;
    }
  }

  if (!response.ok) {
    const detail =
      typeof data === "object" && data && "detail" in data
        ? String((data as { detail: unknown }).detail)
        : `Request failed (${response.status})`;
    throw new ApiError(response.status, detail);
  }

  return data as T;
}

export interface LoginInfo {
  is_system_pat: boolean;
  brain_id: string;
}

export async function resolveLoginInfo(
  pat: string,
  apiBaseUrl: string,
): Promise<LoginInfo> {
  const token = pat.trim();
  const url = `${normalizeBaseUrl(apiBaseUrl)}/meta/login-info`;
  const response = await fetch(url, {
    headers: authHeaders(token),
  });
  if (!response.ok) {
    throw new ApiError(response.status, "Invalid BrainPAT");
  }
  return (await response.json()) as LoginInfo;
}

export async function connectSession(
  apiBaseUrl: string,
  pat: string,
): Promise<AuthSession> {
  const loginInfo = await resolveLoginInfo(pat, apiBaseUrl);
  const next: AuthSession = {
    apiBaseUrl: normalizeBaseUrl(apiBaseUrl),
    pat: pat.trim(),
    brainId: loginInfo.brain_id,
    isSystemPat: loginInfo.is_system_pat,
  };
  session = next;
  await apiFetch<string[]>("/meta/entity-labels");
  return next;
}

export async function validateSession(s: AuthSession): Promise<boolean> {
  session = s;
  try {
    await apiFetch<string[]>("/meta/entity-labels");
    return s.isSystemPat ?? false;
  } finally {
    session = loadSession() ?? s;
  }
}

export function mergeBrainOptions(
  brains: BrainRecord[],
  activeBrainId: string,
): BrainRecord[] {
  const byKey = new Map<string, BrainRecord>();
  for (const brain of brains) {
    byKey.set(brain.name_key, brain);
  }
  if (activeBrainId && !byKey.has(activeBrainId)) {
    byKey.set(activeBrainId, { name_key: activeBrainId });
  }
  return [...byKey.values()].sort((a, b) =>
    a.name_key.localeCompare(b.name_key),
  );
}
