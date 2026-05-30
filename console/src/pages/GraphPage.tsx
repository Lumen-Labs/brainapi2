import { useCallback, useEffect, useMemo, useState } from "react";
import GraphCanvas from "../components/GraphCanvas";
import { apiFetch } from "../lib/api";
import { colorForLabel, primaryLabel } from "../lib/graphColors";
import {
  filterRelationshipsByEntities,
  filterRelationshipsByLabel,
  filterRelationshipsByQuery,
  mergeGraphData,
  normalizeTriples,
  relationshipsToEdges,
  type GraphEntity,
  type GraphRelationship,
} from "../lib/graphModel";

interface NeighborEntry {
  neighbor: GraphEntity;
  relationship: { name: string; uuid?: string };
}

interface GraphFilters {
  label: string;
  query: string;
  limit: number;
}

const DEFAULT_FILTERS: GraphFilters = { label: "", query: "", limit: 250 };

function mergeEntities(
  current: GraphEntity[],
  incoming: GraphEntity[],
): GraphEntity[] {
  const map = new Map(current.map((e) => [e.uuid, e]));
  for (const e of incoming) map.set(e.uuid, e);
  return [...map.values()];
}

function mergeRelationships(
  current: GraphRelationship[],
  incoming: GraphRelationship[],
): GraphRelationship[] {
  const map = new Map(
    current.map((r) => [
      `${r.subject.uuid}-${r.object.uuid}-${r.predicate.name}`,
      r,
    ]),
  );
  for (const r of incoming) {
    map.set(`${r.subject.uuid}-${r.object.uuid}-${r.predicate.name}`, r);
  }
  return [...map.values()];
}

function applyGraphFilters(
  rels: GraphRelationship[],
  entities: GraphEntity[],
  filters: GraphFilters,
): GraphRelationship[] {
  const hasFilters = !!(filters.label || filters.query);
  if (!hasFilters) return rels;

  if (filters.label && filters.query) {
    return filterRelationshipsByQuery(
      filterRelationshipsByLabel(rels, filters.label),
      filters.query,
    );
  }
  if (filters.label) {
    return filterRelationshipsByLabel(rels, filters.label);
  }
  if (entities.length > 0) {
    return filterRelationshipsByEntities(rels, entities);
  }
  return filterRelationshipsByQuery(rels, filters.query);
}

export default function GraphPage() {
  const [entities, setEntities] = useState<GraphEntity[]>([]);
  const [relationships, setRelationships] = useState<GraphRelationship[]>([]);
  const [labels, setLabels] = useState<string[]>([]);
  const [draftLabel, setDraftLabel] = useState("");
  const [draftQuery, setDraftQuery] = useState("");
  const [draftLimit, setDraftLimit] = useState(DEFAULT_FILTERS.limit);
  const [selectedNode, setSelectedNode] = useState<GraphEntity | null>(null);
  const [loading, setLoading] = useState(true);
  const [expanding, setExpanding] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [physics, setPhysics] = useState(true);
  const [activeFilters, setActiveFilters] = useState<GraphFilters>(DEFAULT_FILTERS);

  const loadGraph = useCallback(async (filters: GraphFilters) => {
    setLoading(true);
    setError(null);
    setSelectedNode(null);
    setActiveFilters(filters);

    const hasFilters = !!(filters.label || filters.query);

    try {
      const relParams = new URLSearchParams({
        limit: String(filters.limit),
        skip: "0",
      });
      if (filters.query) relParams.set("query_text", filters.query);

      const entityParams = new URLSearchParams({
        limit: String(filters.limit),
        skip: "0",
      });
      if (filters.label) entityParams.set("node_labels", filters.label);
      if (filters.query) entityParams.set("query_text", filters.query);

      const [relRes, labelRes, entityRes] = await Promise.all([
        apiFetch<{ relationships: unknown[]; total?: number }>(
          `/retrieve/relationships?${relParams}`,
        ),
        apiFetch<string[]>("/meta/entity-labels"),
        hasFilters
          ? apiFetch<{ entities: GraphEntity[]; total?: number }>(
              `/retrieve/entities?${entityParams}`,
            )
          : Promise.resolve({ entities: [] as GraphEntity[], total: 0 }),
      ]);

      const normalizedRels = normalizeTriples(relRes.relationships ?? []);
      const entityList = entityRes.entities ?? [];

      let filteredRels = applyGraphFilters(
        normalizedRels,
        entityList,
        filters,
      );

      const merged = mergeGraphData(entityList, filteredRels);

      if (!hasFilters) {
        if (merged.entities.length === 0) {
          const fallback = await apiFetch<{ entities: GraphEntity[] }>(
            `/retrieve/entities?limit=${filters.limit}&skip=0`,
          );
          setEntities(
            mergeGraphData(fallback.entities ?? [], normalizedRels).entities,
          );
          setRelationships(normalizedRels);
        } else {
          setEntities(merged.entities);
          setRelationships(merged.relationships);
        }
      } else {
        setEntities(merged.entities);
        setRelationships(merged.relationships);
      }

      setLabels(Array.isArray(labelRes) ? labelRes : []);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load graph");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadGraph(DEFAULT_FILTERS);
  }, [loadGraph]);

  function runQuery() {
    loadGraph({
      label: draftLabel,
      query: draftQuery.trim(),
      limit: draftLimit,
    });
  }

  function clearFilters() {
    setDraftLabel("");
    setDraftQuery("");
    setDraftLimit(DEFAULT_FILTERS.limit);
    loadGraph(DEFAULT_FILTERS);
  }

  const nodeIds = useMemo(
    () => new Set(entities.map((e) => e.uuid)),
    [entities],
  );

  const edges = useMemo(
    () =>
      relationshipsToEdges(relationships).filter(
        (e) => nodeIds.has(e.from) && nodeIds.has(e.to),
      ),
    [relationships, nodeIds],
  );

  const legendLabels = useMemo(() => {
    const set = new Set<string>();
    for (const e of entities) set.add(primaryLabel(e.labels));
    return [...set].sort();
  }, [entities]);

  const hasActiveFilters = !!(activeFilters.label || activeFilters.query);

  async function expandNode(center: GraphEntity) {
    setExpanding(true);
    try {
      const res = await apiFetch<{
        main_node: GraphEntity;
        neighbors?: NeighborEntry[];
      }>(
        `/retrieve/entities/neighbors?uuid=${encodeURIComponent(center.uuid)}&limit=40`,
      );

      const centerNode = asEntityOrFallback(res.main_node, center);
      const neighbors = res.neighbors ?? [];
      const newEntities: GraphEntity[] = [centerNode];
      const newRels: GraphRelationship[] = [];

      for (const entry of neighbors) {
        const neighbor = asEntityOrFallback(entry.neighbor, null);
        if (!neighbor) continue;
        newEntities.push(neighbor);
        newRels.push({
          subject: centerNode,
          object: neighbor,
          predicate: {
            name: entry.relationship?.name?.trim() || "RELATED",
            uuid: entry.relationship?.uuid,
          },
        });
      }

      setEntities((prev) => mergeEntities(prev, newEntities));
      setRelationships((prev) => mergeRelationships(prev, newRels));
      setSelectedNode(centerNode);
    } catch {
      /* ignore */
    } finally {
      setExpanding(false);
    }
  }

  return (
    <div className="flex h-[calc(100vh-3rem)] flex-col">
      <div className="mb-3 flex flex-wrap items-end gap-2">
        <div className="mr-2">
          <h1 className="text-2xl font-semibold">Graph Explorer</h1>
          <p className="text-xs text-slate-500">
            {entities.length} nodes · {edges.length} edges · {relationships.length}{" "}
            relationships loaded · double-click to expand
          </p>
        </div>
        <select
          value={draftLabel}
          onChange={(e) => setDraftLabel(e.target.value)}
          className="rounded-lg border border-slate-700 bg-slate-900 px-3 py-1.5 text-sm"
        >
          <option value="">All labels</option>
          {labels.map((l) => (
            <option key={l} value={l}>
              {l}
            </option>
          ))}
        </select>
        <input
          type="search"
          placeholder="Search nodes…"
          value={draftQuery}
          onChange={(e) => setDraftQuery(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && runQuery()}
          className="min-w-[180px] rounded-lg border border-slate-700 bg-slate-900 px-3 py-1.5 text-sm"
        />
        <select
          value={draftLimit}
          onChange={(e) => setDraftLimit(Number(e.target.value))}
          className="rounded-lg border border-slate-700 bg-slate-900 px-3 py-1.5 text-sm"
        >
          {[100, 250, 500].map((n) => (
            <option key={n} value={n}>
              Limit {n}
            </option>
          ))}
        </select>
        <button
          type="button"
          onClick={runQuery}
          disabled={loading}
          className="rounded-lg bg-cyan-800 px-3 py-1.5 text-sm hover:bg-cyan-700 disabled:opacity-50"
        >
          {loading ? "Running…" : "Run"}
        </button>
        {hasActiveFilters && (
          <button
            type="button"
            onClick={clearFilters}
            disabled={loading}
            className="rounded-lg border border-slate-700 px-3 py-1.5 text-sm text-slate-300 hover:border-slate-500 disabled:opacity-50"
          >
            Clear
          </button>
        )}
        <button
          type="button"
          onClick={() => setPhysics((p) => !p)}
          className={`rounded-lg border px-3 py-1.5 text-sm ${
            physics
              ? "border-cyan-700 text-cyan-400"
              : "border-slate-700 text-slate-400"
          }`}
        >
          Physics {physics ? "on" : "off"}
        </button>
      </div>

      {error && <p className="mb-2 text-sm text-red-400">{error}</p>}

      <div className="relative flex min-h-0 flex-1 gap-3">
        <div className="relative min-w-0 flex-1 overflow-hidden rounded-xl border border-slate-800">
          {loading ? (
            <div className="flex h-full items-center justify-center bg-[#1a1d23] text-slate-500">
              Loading graph…
            </div>
          ) : entities.length === 0 ? (
            <div className="flex h-full items-center justify-center bg-[#1a1d23] text-slate-500">
              {hasActiveFilters
                ? "No matches — try a different label or search term"
                : "No nodes — ingest data or change filters"}
            </div>
          ) : (
            <GraphCanvas
              entities={entities}
              edges={edges}
              selectedId={selectedNode?.uuid ?? null}
              physics={physics}
              onSelect={setSelectedNode}
              onExpand={expandNode}
            />
          )}

          {legendLabels.length > 0 && !loading && (
            <div className="pointer-events-none absolute bottom-3 left-3 max-h-40 overflow-auto rounded-lg border border-slate-700/80 bg-slate-950/90 p-2 text-xs shadow-lg">
              <div className="mb-1 font-medium text-slate-400">Labels</div>
              {legendLabels.map((label) => (
                <div key={label} className="flex items-center gap-2 py-0.5">
                  <span
                    className="inline-block h-2.5 w-2.5 rounded-full"
                    style={{ backgroundColor: colorForLabel(label) }}
                  />
                  <span className="text-slate-300">{label}</span>
                </div>
              ))}
            </div>
          )}
        </div>

        <div className="w-80 shrink-0 overflow-auto rounded-xl border border-slate-800 bg-slate-950 p-4">
          {selectedNode ? (
            <>
              <div className="mb-3 flex items-start justify-between gap-2">
                <div>
                  <h2 className="text-lg font-semibold text-cyan-400">
                    {selectedNode.name}
                  </h2>
                  <p className="mt-0.5 font-mono text-[10px] text-slate-600">
                    {selectedNode.uuid}
                  </p>
                </div>
                <span
                  className="mt-1 h-4 w-4 shrink-0 rounded-full"
                  style={{
                    backgroundColor: colorForLabel(
                      primaryLabel(selectedNode.labels),
                    ),
                  }}
                />
              </div>
              {selectedNode.labels?.length > 0 && (
                <div className="mb-3 flex flex-wrap gap-1">
                  {selectedNode.labels.map((l) => (
                    <span
                      key={l}
                      className="rounded-full px-2 py-0.5 text-xs"
                      style={{
                        backgroundColor: `${colorForLabel(l)}33`,
                        color: colorForLabel(l),
                      }}
                    >
                      {l}
                    </span>
                  ))}
                </div>
              )}
              {selectedNode.description && (
                <p className="mb-3 text-sm leading-relaxed text-slate-400">
                  {selectedNode.description}
                </p>
              )}
              <button
                type="button"
                disabled={expanding}
                onClick={() => expandNode(selectedNode)}
                className="mb-3 w-full rounded-lg border border-slate-600 py-2 text-sm text-slate-200 hover:border-cyan-600 hover:text-cyan-300 disabled:opacity-50"
              >
                {expanding ? "Expanding…" : "Expand neighbors"}
              </button>
              <div className="text-xs font-medium uppercase tracking-wide text-slate-500">
                Properties
              </div>
              <pre className="mt-2 overflow-auto rounded-lg bg-slate-900 p-3 text-xs text-slate-400">
                {JSON.stringify(selectedNode.properties ?? {}, null, 2)}
              </pre>
            </>
          ) : (
            <div className="text-sm text-slate-500">
              <p className="mb-2 font-medium text-slate-400">Node details</p>
              <p>Click a node to inspect it.</p>
              <p className="mt-2">
                Double-click to expand its neighborhood.
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function asEntityOrFallback(
  value: GraphEntity | null | undefined,
  fallback: GraphEntity | null,
): GraphEntity {
  if (value?.uuid) return value;
  if (fallback) return fallback;
  throw new Error("missing entity");
}
