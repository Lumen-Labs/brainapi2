import { useEffect, useState } from "react";
import { apiFetch } from "../lib/api";

interface VectorItem {
  id: string;
  embeddings?: number[] | null;
  metadata: Record<string, unknown>;
}

interface VectorStore {
  name: string;
  dimension: number;
}

export default function VectorsPage() {
  const [stores, setStores] = useState<VectorStore[]>([]);
  const [store, setStore] = useState("");
  const [vectors, setVectors] = useState<VectorItem[]>([]);
  const [total, setTotal] = useState(0);
  const [skip, setSkip] = useState(0);
  const [limit] = useState(20);
  const [includeEmbeddings, setIncludeEmbeddings] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selected, setSelected] = useState<VectorItem | null>(null);

  useEffect(() => {
    apiFetch<{ stores: VectorStore[] }>("/retrieve/vectors/stores")
      .then((res) => {
        setStores(res.stores ?? []);
        if (res.stores?.[0]) setStore(res.stores[0].name);
      })
      .catch((err) => setError(err instanceof Error ? err.message : "Failed to load stores"));
  }, []);

  useEffect(() => {
    if (!store) return;
    let cancelled = false;
    async function load() {
      setLoading(true);
      setError(null);
      try {
        const params = new URLSearchParams({
          limit: String(limit),
          skip: String(skip),
          include_embeddings: String(includeEmbeddings),
        });
        const res = await apiFetch<{
          vectors: VectorItem[];
          total: number;
        }>(`/retrieve/vectors/${store}?${params}`);
        if (!cancelled) {
          setVectors(res.vectors ?? []);
          setTotal(res.total ?? 0);
        }
      } catch (err) {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : "Failed to load vectors");
        }
      } finally {
        if (!cancelled) setLoading(false);
      }
    }
    load();
    return () => {
      cancelled = true;
    };
  }, [store, skip, limit, includeEmbeddings]);

  return (
    <div>
      <h1 className="mb-4 text-2xl font-semibold">Vectors</h1>
      <div className="mb-4 flex flex-wrap items-center gap-3">
        <select
          value={store}
          onChange={(e) => {
            setStore(e.target.value);
            setSkip(0);
          }}
          className="rounded-lg border border-slate-700 bg-slate-900 px-3 py-2 text-sm"
        >
          {stores.map((s) => (
            <option key={s.name} value={s.name}>
              {s.name} (dim {s.dimension})
            </option>
          ))}
        </select>
        <label className="flex items-center gap-2 text-sm text-slate-400">
          <input
            type="checkbox"
            checked={includeEmbeddings}
            onChange={(e) => setIncludeEmbeddings(e.target.checked)}
          />
          Include embeddings
        </label>
        <span className="text-sm text-slate-500">{total} total</span>
      </div>

      {error && <p className="mb-2 text-sm text-red-400">{error}</p>}

      <div className="flex gap-4">
        <div className="flex-1 overflow-hidden rounded-xl border border-slate-800">
          <table className="w-full text-left text-sm">
            <thead className="border-b border-slate-800 bg-slate-950 text-slate-500">
              <tr>
                <th className="px-4 py-2">ID</th>
                <th className="px-4 py-2">Metadata</th>
                <th className="px-4 py-2">Dim</th>
              </tr>
            </thead>
            <tbody>
              {loading ? (
                <tr>
                  <td colSpan={3} className="px-4 py-8 text-center text-slate-500">
                    Loading…
                  </td>
                </tr>
              ) : vectors.length === 0 ? (
                <tr>
                  <td colSpan={3} className="px-4 py-8 text-center text-slate-500">
                    No vectors
                  </td>
                </tr>
              ) : (
                vectors.map((v) => (
                  <tr
                    key={v.id}
                    onClick={() => setSelected(v)}
                    className="cursor-pointer border-b border-slate-900 hover:bg-slate-900/50"
                  >
                    <td className="max-w-[160px] truncate px-4 py-2 font-mono text-xs text-cyan-600">
                      {v.id}
                    </td>
                    <td className="max-w-md truncate px-4 py-2 text-slate-400">
                      {JSON.stringify(v.metadata).slice(0, 80)}
                    </td>
                    <td className="px-4 py-2 text-slate-500">
                      {v.embeddings?.length ?? "—"}
                    </td>
                  </tr>
                ))
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
            {JSON.stringify(
              {
                ...selected,
                embeddings: selected.embeddings
                  ? [
                      ...selected.embeddings.slice(0, 16),
                      ...(selected.embeddings.length > 16 ? ["…"] : []),
                    ]
                  : null,
              },
              null,
              2,
            )}
          </pre>
        )}
      </div>
    </div>
  );
}
