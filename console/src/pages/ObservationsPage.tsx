import { useEffect, useState } from "react";
import { apiFetch } from "../lib/api";

interface Observation {
  id: string;
  text: string;
  resource_id?: string;
  metadata?: Record<string, unknown>;
  inserted_at?: string;
}

export default function ObservationsPage() {
  const [observations, setObservations] = useState<Observation[]>([]);
  const [labels, setLabels] = useState<string[]>([]);
  const [query, setQuery] = useState("");
  const [resourceId, setResourceId] = useState("");
  const [selectedLabel, setSelectedLabel] = useState("");
  const [skip, setSkip] = useState(0);
  const [limit] = useState(20);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selected, setSelected] = useState<Observation | null>(null);

  useEffect(() => {
    apiFetch<{ labels?: string[] }>("/retrieve/observations/labels")
      .then((res) => setLabels(res.labels ?? []))
      .catch(() => setLabels([]));
  }, []);

  useEffect(() => {
    let cancelled = false;
    async function load() {
      setLoading(true);
      setError(null);
      try {
        const params = new URLSearchParams({
          limit: String(limit),
          skip: String(skip),
        });
        if (query) params.set("query_text", query);
        if (resourceId) params.set("resource_id", resourceId);
        if (selectedLabel) params.set("labels", selectedLabel);
        const res = await apiFetch<{ observations: Observation[] }>(
          `/retrieve/observations?${params}`,
        );
        if (!cancelled) setObservations(res.observations ?? []);
      } catch (err) {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : "Failed to load observations");
        }
      } finally {
        if (!cancelled) setLoading(false);
      }
    }
    load();
    return () => {
      cancelled = true;
    };
  }, [query, resourceId, selectedLabel, skip, limit]);

  return (
    <div>
      <h1 className="mb-4 text-2xl font-semibold">Observations</h1>
      <div className="mb-4 flex flex-wrap gap-2">
        <input
          type="search"
          placeholder="Search text…"
          value={query}
          onChange={(e) => {
            setQuery(e.target.value);
            setSkip(0);
          }}
          className="rounded-lg border border-slate-700 bg-slate-900 px-3 py-2 text-sm"
        />
        <input
          type="text"
          placeholder="Resource ID"
          value={resourceId}
          onChange={(e) => {
            setResourceId(e.target.value);
            setSkip(0);
          }}
          className="rounded-lg border border-slate-700 bg-slate-900 px-3 py-2 text-sm"
        />
        <select
          value={selectedLabel}
          onChange={(e) => {
            setSelectedLabel(e.target.value);
            setSkip(0);
          }}
          className="rounded-lg border border-slate-700 bg-slate-900 px-3 py-2 text-sm"
        >
          <option value="">All labels</option>
          {labels.map((l) => (
            <option key={l} value={l}>
              {l}
            </option>
          ))}
        </select>
      </div>

      {error && <p className="mb-2 text-sm text-red-400">{error}</p>}

      <div className="flex gap-4">
        <div className="flex-1 space-y-2">
          {loading ? (
            <p className="text-slate-500">Loading…</p>
          ) : observations.length === 0 ? (
            <p className="text-slate-500">No observations</p>
          ) : (
            observations.map((obs) => (
              <button
                key={obs.id}
                type="button"
                onClick={() => setSelected(obs)}
                className={`w-full rounded-xl border p-4 text-left transition-colors ${
                  selected?.id === obs.id
                    ? "border-cyan-700 bg-cyan-950/30"
                    : "border-slate-800 bg-slate-950 hover:border-slate-700"
                }`}
              >
                <p className="text-sm text-slate-200">{obs.text}</p>
                <p className="mt-2 font-mono text-xs text-slate-500">{obs.id}</p>
              </button>
            ))
          )}
          <div className="flex gap-2 pt-2">
            <button
              type="button"
              disabled={skip === 0}
              onClick={() => setSkip(Math.max(0, skip - limit))}
              className="rounded border border-slate-700 px-3 py-1 text-sm disabled:opacity-40"
            >
              Prev
            </button>
            <button
              type="button"
              disabled={observations.length < limit}
              onClick={() => setSkip(skip + limit)}
              className="rounded border border-slate-700 px-3 py-1 text-sm disabled:opacity-40"
            >
              Next
            </button>
          </div>
        </div>

        {selected && (
          <pre className="w-96 shrink-0 overflow-auto rounded-xl border border-slate-800 bg-slate-950 p-4 text-xs text-slate-400">
            {JSON.stringify(selected, null, 2)}
          </pre>
        )}
      </div>
    </div>
  );
}
