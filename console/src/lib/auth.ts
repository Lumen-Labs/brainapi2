export interface AuthSession {
  apiBaseUrl: string;
  pat: string;
  brainId: string;
  isSystemPat?: boolean;
}

const STORAGE_KEY = "brainapi-console-session";

export function loadSession(): AuthSession | null {
  const raw = sessionStorage.getItem(STORAGE_KEY);
  if (!raw) return null;
  try {
    return JSON.parse(raw) as AuthSession;
  } catch {
    return null;
  }
}

export function saveSession(session: AuthSession): void {
  sessionStorage.setItem(STORAGE_KEY, JSON.stringify(session));
}

export function clearSession(): void {
  sessionStorage.removeItem(STORAGE_KEY);
}

export function normalizeBaseUrl(url: string): string {
  return url.replace(/\/+$/, "");
}
