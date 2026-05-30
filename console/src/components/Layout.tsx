import { useCallback, useEffect, useState } from "react";
import { NavLink, Outlet, useNavigate } from "react-router-dom";
import {
  fetchBrainsList,
  getSession,
  mergeBrainOptions,
  setSession,
  type BrainRecord,
} from "../lib/api";
import { clearSession, loadSession, saveSession } from "../lib/auth";

const navItems = [
  { to: "/", label: "Overview", end: true },
  { to: "/graph", label: "Graph" },
  { to: "/data", label: "Data" },
  { to: "/observations", label: "Observations" },
  { to: "/vectors", label: "Vectors" },
  { to: "/tasks", label: "Tasks" },
  { to: "/ingest", label: "Ingest" },
];

export default function Layout() {
  const navigate = useNavigate();
  const session = getSession();
  const [brains, setBrains] = useState<BrainRecord[]>([]);
  const [brainId, setBrainId] = useState(session?.brainId ?? "default");
  const [canSwitchBrains, setCanSwitchBrains] = useState(
    session?.isSystemPat ?? false,
  );
  const [loadingBrains, setLoadingBrains] = useState(false);
  const [brainsError, setBrainsError] = useState<string | null>(null);

  const loadBrains = useCallback(async () => {
    const current = getSession();
    if (!current?.isSystemPat) {
      setCanSwitchBrains(false);
      setBrains([]);
      return;
    }

    setCanSwitchBrains(true);
    setLoadingBrains(true);
    setBrainsError(null);
    try {
      const list = await fetchBrainsList(current);
      setBrains(mergeBrainOptions(list, current.brainId));
    } catch (err) {
      setBrainsError(
        err instanceof Error ? err.message : "Failed to load brains",
      );
      setBrains(mergeBrainOptions([], current.brainId));
    } finally {
      setLoadingBrains(false);
    }
  }, []);

  useEffect(() => {
    setBrainId(session?.brainId ?? "default");
    setCanSwitchBrains(session?.isSystemPat ?? false);
    loadBrains();
  }, [loadBrains, session?.brainId, session?.isSystemPat]);

  function switchBrain(id: string) {
    setBrainId(id);
    const current = loadSession();
    if (current) {
      const next = { ...current, brainId: id };
      saveSession(next);
      setSession(next);
      window.location.reload();
    }
  }

  function logout() {
    clearSession();
    setSession(null);
    navigate("/login");
  }

  const brainOptions = mergeBrainOptions(brains, brainId);

  return (
    <div className="flex min-h-screen">
      <aside className="flex w-56 shrink-0 flex-col border-r border-slate-800 bg-slate-950 p-4">
        <div className="mb-6">
          <div className="text-lg font-semibold text-cyan-400">BrainAPI</div>
          <div className="text-xs text-slate-500">Local Console</div>
        </div>
        {canSwitchBrains ? (
          <label className="mb-4 flex flex-col gap-1 text-xs">
            <span className="text-slate-500">Brain</span>
            <select
              value={brainId}
              onChange={(e) => switchBrain(e.target.value)}
              disabled={loadingBrains}
              className="rounded-lg border border-slate-700 bg-slate-900 px-2 py-1.5 text-sm text-slate-200 disabled:opacity-50"
            >
              {brainOptions.map((b) => (
                <option key={b.name_key} value={b.name_key}>
                  {b.name_key}
                </option>
              ))}
            </select>
            {loadingBrains && (
              <span className="text-slate-600">Loading brains…</span>
            )}
            {brainsError && (
              <button
                type="button"
                onClick={loadBrains}
                className="text-left text-amber-500 hover:underline"
              >
                {brainsError} — retry
              </button>
            )}
          </label>
        ) : (
          <p className="mb-4 text-xs text-slate-600">
            Brain: <span className="text-cyan-600">{brainId}</span>
            <span className="mt-1 block text-slate-700">
              Per-brain PAT — scoped to this brain
            </span>
          </p>
        )}
        <nav className="flex flex-1 flex-col gap-1">
          {navItems.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              end={item.end}
              className={({ isActive }) =>
                `rounded-lg px-3 py-2 text-sm transition-colors ${
                  isActive
                    ? "bg-cyan-950 text-cyan-300"
                    : "text-slate-400 hover:bg-slate-900 hover:text-slate-200"
                }`
              }
            >
              {item.label}
            </NavLink>
          ))}
        </nav>
        <button
          type="button"
          onClick={logout}
          className="mt-4 rounded-lg border border-slate-700 px-3 py-2 text-sm text-slate-400 hover:border-slate-500 hover:text-slate-200"
        >
          Log out
        </button>
      </aside>
      <main className="flex-1 overflow-auto p-6">
        <Outlet />
      </main>
    </div>
  );
}
