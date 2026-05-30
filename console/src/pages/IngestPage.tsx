import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { apiFetch, fetchBrainsList, getSession, type BrainRecord } from "../lib/api";

export default function IngestPage() {
  const session = getSession();
  const [text, setText] = useState("");
  const [ingesting, setIngesting] = useState(false);
  const [ingestResult, setIngestResult] = useState<string | null>(null);
  const [ingestError, setIngestError] = useState<string | null>(null);
  const [taskId, setTaskId] = useState<string | null>(null);

  const [brains, setBrains] = useState<BrainRecord[]>([]);
  const [brainsError, setBrainsError] = useState<string | null>(null);
  const [newBrainId, setNewBrainId] = useState("");
  const [creating, setCreating] = useState(false);
  const [createdPat, setCreatedPat] = useState<string | null>(null);

  useEffect(() => {
    if (!getSession()?.isSystemPat) {
      setBrainsError("Brain list requires system BRAINPAT_TOKEN");
      return;
    }
    fetchBrainsList()
      .then(setBrains)
      .catch(() =>
        setBrainsError("Brain list requires system BRAINPAT_TOKEN"),
      );
  }, []);

  async function handleIngest(e: React.FormEvent) {
    e.preventDefault();
    setIngesting(true);
    setIngestError(null);
    setIngestResult(null);
    setTaskId(null);
    try {
      const res = await apiFetch<{ message: string; task_id: string }>("/ingest/", {
        method: "POST",
        body: JSON.stringify({
          data: { data_type: "text", text_data: text },
        }),
      });
      setIngestResult(res.message);
      setTaskId(res.task_id);
      setText("");
    } catch (err) {
      setIngestError(err instanceof Error ? err.message : "Ingest failed");
    } finally {
      setIngesting(false);
    }
  }

  async function handleCreateBrain(e: React.FormEvent) {
    e.preventDefault();
    setCreating(true);
    setBrainsError(null);
    setCreatedPat(null);
    try {
      const res = await apiFetch<BrainRecord>("/system/brains", {
        method: "POST",
        body: JSON.stringify({ brain_id: newBrainId.trim() }),
      });
      setCreatedPat(res.pat ?? null);
      setBrains((prev) => [...prev, res]);
      setNewBrainId("");
    } catch (err) {
      setBrainsError(err instanceof Error ? err.message : "Failed to create brain");
    } finally {
      setCreating(false);
    }
  }

  return (
    <div className="max-w-2xl space-y-8">
      <div>
        <h1 className="mb-1 text-2xl font-semibold">Ingest</h1>
        <p className="text-sm text-slate-500">
          Ingesting into brain{" "}
          <span className="text-cyan-400">{session?.brainId}</span>
        </p>
      </div>

      <form onSubmit={handleIngest} className="space-y-4 rounded-xl border border-slate-800 bg-slate-950 p-6">
        <label className="flex flex-col gap-2 text-sm">
          <span className="text-slate-400">Text to ingest</span>
          <textarea
            value={text}
            onChange={(e) => setText(e.target.value)}
            rows={6}
            required
            className="rounded-lg border border-slate-700 bg-slate-900 px-3 py-2 text-slate-100 outline-none focus:border-cyan-600"
            placeholder="Emily organized the AI Ethics Meetup in London on March 8, 2024."
          />
        </label>
        {ingestError && (
          <p className="text-sm text-red-400">{ingestError}</p>
        )}
        {ingestResult && (
          <p className="text-sm text-emerald-400">
            {ingestResult}
            {taskId && (
              <>
                {" "}
                —{" "}
                <Link to="/tasks" className="text-cyan-500 hover:underline">
                  View task {taskId}
                </Link>
              </>
            )}
          </p>
        )}
        <button
          type="submit"
          disabled={ingesting || !text.trim()}
          className="rounded-lg bg-cyan-700 px-4 py-2 text-sm font-medium hover:bg-cyan-600 disabled:opacity-50"
        >
          {ingesting ? "Submitting…" : "Ingest text"}
        </button>
      </form>

      <div className="rounded-xl border border-slate-800 bg-slate-950 p-6">
        <h2 className="mb-1 font-semibold">Brains</h2>
        <p className="mb-4 text-xs text-slate-500">
          Requires system BRAINPAT_TOKEN from your .env
        </p>

        {brainsError && (
          <p className="mb-3 text-sm text-amber-400">{brainsError}</p>
        )}

        {brains.length > 0 && (
          <ul className="mb-4 space-y-1 text-sm">
            {brains.map((b) => (
              <li key={b.id ?? b.name_key} className="text-slate-400">
                <span className="text-cyan-400">{b.name_key}</span>
                {b.pat && (
                  <span className="ml-2 font-mono text-xs text-slate-600">
                    pat: {b.pat.slice(0, 8)}…
                  </span>
                )}
              </li>
            ))}
          </ul>
        )}

        <form onSubmit={handleCreateBrain} className="flex gap-2">
          <input
            type="text"
            value={newBrainId}
            onChange={(e) => setNewBrainId(e.target.value)}
            pattern="[a-zA-Z][a-zA-Z0-9]*"
            placeholder="newBrainId"
            className="flex-1 rounded-lg border border-slate-700 bg-slate-900 px-3 py-2 text-sm"
          />
          <button
            type="submit"
            disabled={creating || !newBrainId.trim()}
            className="rounded-lg border border-slate-700 px-4 py-2 text-sm hover:border-cyan-700 disabled:opacity-50"
          >
            Create
          </button>
        </form>

        {createdPat && (
          <div className="mt-4 rounded-lg border border-emerald-900 bg-emerald-950/30 p-3">
            <p className="mb-1 text-xs text-emerald-400">New brain PAT (copy now):</p>
            <code className="break-all text-xs text-slate-300">{createdPat}</code>
          </div>
        )}
      </div>
    </div>
  );
}
