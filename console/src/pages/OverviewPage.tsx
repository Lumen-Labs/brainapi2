import { useEffect, useState } from "react";
import { apiFetch, getSession } from "../lib/api";

interface StatCard {
  label: string;
  value: string | number;
  loading: boolean;
}

export default function OverviewPage() {
  const session = getSession();
  const [stats, setStats] = useState<StatCard[]>([
    { label: "Entities", value: "—", loading: true },
    { label: "Relationships", value: "—", loading: true },
    { label: "Text chunks", value: "—", loading: true },
    { label: "Observations", value: "—", loading: true },
    { label: "Tasks", value: "—", loading: true },
    { label: "Vectors", value: "—", loading: true },
  ]);

  useEffect(() => {
    let cancelled = false;

    async function load() {
      const results = [...stats];

      try {
        const entities = await apiFetch<{ total?: number; entities?: unknown[] }>(
          "/retrieve/entities?limit=1",
        );
        results[0] = { label: "Entities", value: entities.total ?? 0, loading: false };
      } catch {
        results[0] = { label: "Entities", value: "?", loading: false };
      }

      try {
        const rels = await apiFetch<{ total?: number }>(
          "/retrieve/relationships?limit=1",
        );
        results[1] = { label: "Relationships", value: rels.total ?? 0, loading: false };
      } catch {
        results[1] = { label: "Relationships", value: "?", loading: false };
      }

      try {
        const chunks = await apiFetch<{ total?: number }>(
          "/retrieve/text-chunks?limit=1",
        );
        results[2] = { label: "Text chunks", value: chunks.total ?? 0, loading: false };
      } catch {
        results[2] = { label: "Text chunks", value: "?", loading: false };
      }

      try {
        const obs = await apiFetch<{ count?: number; observations?: unknown[] }>(
          "/retrieve/observations?limit=1",
        );
        results[3] = {
          label: "Observations",
          value: obs.count ?? obs.observations?.length ?? 0,
          loading: false,
        };
      } catch {
        results[3] = { label: "Observations", value: "?", loading: false };
      }

      try {
        const tasks = await apiFetch<{ tasks?: unknown[] }>("/tasks/");
        results[4] = {
          label: "Tasks",
          value: tasks.tasks?.length ?? 0,
          loading: false,
        };
      } catch {
        results[4] = { label: "Tasks", value: "?", loading: false };
      }

      try {
        const stores = await apiFetch<{ stores: { name: string }[] }>(
          "/retrieve/vectors/stores",
        );
        let vectorTotal = 0;
        for (const store of stores.stores) {
          const res = await apiFetch<{ total?: number }>(
            `/retrieve/vectors/${store.name}?limit=1`,
          );
          vectorTotal += res.total ?? 0;
        }
        results[5] = { label: "Vectors", value: vectorTotal, loading: false };
      } catch {
        results[5] = { label: "Vectors", value: "?", loading: false };
      }

      if (!cancelled) setStats(results);
    }

    load();
    return () => {
      cancelled = true;
    };
  }, []);

  return (
    <div>
      <h1 className="mb-1 text-2xl font-semibold">Overview</h1>
      <p className="mb-6 text-sm text-slate-500">
        Brain <span className="text-cyan-400">{session?.brainId}</span> on{" "}
        <span className="text-slate-400">{session?.apiBaseUrl}</span>
      </p>
      <div className="grid grid-cols-2 gap-4 md:grid-cols-3">
        {stats.map((stat) => (
          <div
            key={stat.label}
            className="rounded-xl border border-slate-800 bg-slate-950 p-5"
          >
            <div className="text-sm text-slate-500">{stat.label}</div>
            <div className="mt-1 text-3xl font-semibold text-cyan-400">
              {stat.loading ? "…" : stat.value}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
