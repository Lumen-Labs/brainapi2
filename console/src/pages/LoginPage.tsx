import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { loadSession, saveSession } from "../lib/auth";
import { connectSession, setSession } from "../lib/api";

export default function LoginPage() {
  const navigate = useNavigate();
  const existing = loadSession();
  const [apiBaseUrl, setApiBaseUrl] = useState(
    existing?.apiBaseUrl ?? "http://localhost:8000",
  );
  const [pat, setPat] = useState(existing?.pat ?? "");
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    setLoading(true);
    try {
      const session = await connectSession(apiBaseUrl, pat);
      saveSession(session);
      setSession(session);
      navigate("/");
    } catch (err) {
      setSession(null);
      setError(err instanceof Error ? err.message : "Login failed");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="flex min-h-screen items-center justify-center p-6">
      <div className="w-full max-w-md rounded-xl border border-slate-800 bg-slate-950 p-8 shadow-xl">
        <h1 className="mb-1 text-2xl font-semibold text-cyan-400">BrainAPI Console</h1>
        <p className="mb-6 text-sm text-slate-500">
          Connect to your local BrainAPI instance with a BrainPAT token.
        </p>
        <form onSubmit={handleSubmit} className="flex flex-col gap-4">
          <label className="flex flex-col gap-1 text-sm">
            <span className="text-slate-400">API base URL</span>
            <input
              type="url"
              value={apiBaseUrl}
              onChange={(e) => setApiBaseUrl(e.target.value)}
              className="rounded-lg border border-slate-700 bg-slate-900 px-3 py-2 text-slate-100 outline-none focus:border-cyan-600"
              required
            />
          </label>
          <label className="flex flex-col gap-1 text-sm">
            <span className="text-slate-400">BrainPAT</span>
            <input
              type="password"
              value={pat}
              onChange={(e) => setPat(e.target.value)}
              placeholder="brainpat_… or system token"
              className="rounded-lg border border-slate-700 bg-slate-900 px-3 py-2 text-slate-100 outline-none focus:border-cyan-600"
              required
            />
          </label>
          <div className="rounded-lg border border-slate-800 bg-slate-900/50 p-3 text-xs text-slate-500">
            <p className="mb-1">
              <strong className="text-slate-400">System PAT</strong> — from{" "}
              <code className="text-cyan-600">BRAINPAT_TOKEN</code> in your{" "}
              <code className="text-cyan-600">.env</code>. Starts on the default
              brain; switch brains from the sidebar after login.
            </p>
            <p>
              <strong className="text-slate-400">Per-brain PAT</strong> — returned
              when creating a brain. Scoped to that brain only.
            </p>
          </div>
          {error && (
            <p className="rounded-lg border border-red-900 bg-red-950/50 px-3 py-2 text-sm text-red-300">
              {error}
            </p>
          )}
          <button
            type="submit"
            disabled={loading}
            className="rounded-lg bg-cyan-700 px-4 py-2 text-sm font-medium text-white hover:bg-cyan-600 disabled:opacity-50"
          >
            {loading ? "Connecting…" : "Connect"}
          </button>
        </form>
      </div>
    </div>
  );
}
