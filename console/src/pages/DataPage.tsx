import { useEffect, useState } from "react";
import { apiFetch } from "../lib/api";

type Tab = "chunks" | "structured";

export default function DataPage() {
  const [tab, setTab] = useState<Tab>("chunks");
  const [query, setQuery] = useState("");
  const [skip, setSkip] = useState(0);
  const [limit] = useState(20);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [items, setItems] = useState<Record<string, unknown>[]>([]);
  const [total, setTotal] = useState(0);
  const [selected, setSelected] = useState<Record<string, unknown> | null>(null);
  const [types, setTypes] = useState<string[]>([]);
  const [selectedType, setSelectedType] = useState("");

  useEffect(() => {
    if (tab === "structured") {
      apiFetch<{ types?: string[] }>("/retrieve/structured-data/types")
        .then((res) => setTypes(res.types ?? []))
        .catch(() => setTypes([]));
    }
  }, [tab]);

  useEffect(() => {
    let cancelled = false;
    async function load() {
      setLoading(true);
      setError(null);
      try {
        if (tab === "chunks") {
          const params = new URLSearchParams({
            limit: String(limit),
            skip: String(skip),
          });
          if (query) params.set("query_text", query);
          const res = await apiFetch<{
            data: Record<string, unknown>[];
            total: number;
          }>(`/retrieve/text-chunks?${params}`);
          if (!cancelled) {
            setItems(res.data ?? []);
            setTotal(res.total ?? 0);
          }
        } else {
          const params = new URLSearchParams({
            limit: String(limit),
            skip: String(skip),
          });
          if (query) params.set("query_text", query);
          if (selectedType) params.set("types", selectedType);
          const res = await apiFetch<{
            data: Record<string, unknown>[];
            total: number;
          }>(`/retrieve/structured-data?${params}`);
          if (!cancelled) {
            setItems(res.data ?? []);
            setTotal(res.total ?? 0);
          }
        }
      } catch (err) {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : "Failed to load data");
        }
      } finally {
        if (!cancelled) setLoading(false);
      }
    }
    load();
    return () => {
      cancelled = true;
    };
  }, [tab, query, skip, limit, selectedType]);

  return (
    <div>
      <h1 className="mb-4 text-2xl font-semibold">Data</h1>
      <div className="mb-4 flex gap-2">
        {(["chunks", "structured"] as Tab[]).map((t) => (
          <button
            key={t}
            type="button"
            onClick={() => {
              setTab(t);
              setSkip(0);
              setSelected(null);
            }}
            className={`rounded-lg px-4 py-2 text-sm ${
              tab === t
                ? "bg-cyan-900 text-cyan-300"
                : "bg-slate-900 text-slate-400 hover:text-slate-200"
            }`}
          >
            {t === "chunks" ? "Text chunks" : "Structured data"}
          </button>
        ))}
      </div>

      <div className="mb-4 flex flex-wrap gap-2">
        <input
          type="search"
          placeholder="Search…"
          value={query}
          onChange={(e) => {
            setQuery(e.target.value);
            setSkip(0);
          }}
          className="rounded-lg border border-slate-700 bg-slate-900 px-3 py-2 text-sm"
        />
        {tab === "structured" && (
          <select
            value={selectedType}
            onChange={(e) => {
              setSelectedType(e.target.value);
              setSkip(0);
            }}
            className="rounded-lg border border-slate-700 bg-slate-900 px-3 py-2 text-sm"
          >
            <option value="">All types</option>
            {types.map((t) => (
              <option key={t} value={t}>
                {t}
              </option>
            ))}
          </select>
        )}
      </div>

      {error && <p className="mb-2 text-sm text-red-400">{error}</p>}

      <div className="flex gap-4">
        <div className="flex-1 overflow-hidden rounded-xl border border-slate-800">
          <table className="w-full text-left text-sm">
            <thead className="border-b border-slate-800 bg-slate-950 text-slate-500">
              <tr>
                <th className="px-4 py-2">ID</th>
                <th className="px-4 py-2">Preview</th>
              </tr>
            </thead>
            <tbody>
              {loading ? (
                <tr>
                  <td colSpan={2} className="px-4 py-8 text-center text-slate-500">
                    Loading…
                  </td>
                </tr>
              ) : items.length === 0 ? (
                <tr>
                  <td colSpan={2} className="px-4 py-8 text-center text-slate-500">
                    No results
                  </td>
                </tr>
              ) : (
                items.map((item) => {
                  const id = String(item.id ?? item.resource_id ?? "—");
                  const preview =
                    tab === "chunks"
                      ? String(item.text ?? item.content ?? "").slice(0, 120)
                      : JSON.stringify(item.json_data ?? item).slice(0, 120);
                  return (
                    <tr
                      key={id}
                      onClick={() => setSelected(item)}
                      className="cursor-pointer border-b border-slate-900 hover:bg-slate-900/50"
                    >
                      <td className="max-w-[140px] truncate px-4 py-2 font-mono text-xs text-cyan-600">
                        {id}
                      </td>
                      <td className="px-4 py-2 text-slate-400">{preview}</td>
                    </tr>
                  );
                })
              )}
            </tbody>
          </table>
          <div className="flex items-center justify-between border-t border-slate-800 px-4 py-2 text-sm text-slate-500">
            <span>
              {skip + 1}–{Math.min(skip + limit, total)} of {total}
            </span>
            <div className="flex gap-2">
              <button
                type="button"
                disabled={skip === 0}
                onClick={() => setSkip(Math.max(0, skip - limit))}
                className="rounded border border-slate-700 px-2 py-1 disabled:opacity-40"
              >
                Prev
              </button>
              <button
                type="button"
                disabled={skip + limit >= total}
                onClick={() => setSkip(skip + limit)}
                className="rounded border border-slate-700 px-2 py-1 disabled:opacity-40"
              >
                Next
              </button>
            </div>
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
