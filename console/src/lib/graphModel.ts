export interface GraphEntity {
  uuid: string;
  name: string;
  labels: string[];
  description?: string;
  properties?: Record<string, unknown>;
}

export interface GraphRelationship {
  subject: GraphEntity;
  object: GraphEntity;
  predicate: { name: string; uuid?: string };
}

function asEntity(value: unknown): GraphEntity | null {
  if (!value || typeof value !== "object") return null;
  const v = value as Record<string, unknown>;
  const uuid = typeof v.uuid === "string" ? v.uuid : "";
  if (!uuid) return null;
  return {
    uuid,
    name: typeof v.name === "string" ? v.name : uuid,
    labels: Array.isArray(v.labels) ? (v.labels as string[]) : [],
    description: typeof v.description === "string" ? v.description : undefined,
    properties:
      v.properties && typeof v.properties === "object"
        ? (v.properties as Record<string, unknown>)
        : undefined,
  };
}

export function normalizeTriple(raw: unknown): GraphRelationship | null {
  if (!raw || typeof raw !== "object") return null;
  const row = raw as Record<string, unknown>;
  const subject = asEntity(row.subject);
  const object = asEntity(row.object);
  if (!subject || !object) return null;
  const predicateRaw = row.predicate;
  let predName = "RELATED";
  let predUuid: string | undefined;
  if (predicateRaw && typeof predicateRaw === "object") {
    const p = predicateRaw as Record<string, unknown>;
    if (typeof p.name === "string" && p.name.trim()) predName = p.name;
    if (typeof p.uuid === "string") predUuid = p.uuid;
  }
  return {
    subject,
    object,
    predicate: { name: predName, uuid: predUuid },
  };
}

export function normalizeTriples(raw: unknown[]): GraphRelationship[] {
  return raw.map(normalizeTriple).filter((r): r is GraphRelationship => r !== null);
}

export function mergeGraphData(
  entities: GraphEntity[],
  relationships: GraphRelationship[],
): { entities: GraphEntity[]; relationships: GraphRelationship[] } {
  const entityMap = new Map<string, GraphEntity>();
  for (const entity of entities) {
    entityMap.set(entity.uuid, entity);
  }
  for (const rel of relationships) {
    entityMap.set(rel.subject.uuid, rel.subject);
    entityMap.set(rel.object.uuid, rel.object);
  }
  return {
    entities: [...entityMap.values()],
    relationships,
  };
}

export interface GraphEdge {
  id: string;
  from: string;
  to: string;
  label: string;
}

export function normalizeLabel(label: string): string {
  return label
    .replace(/ /g, "_")
    .toUpperCase()
    .replace(/-/g, "_")
    .replace(/\./g, "_")
    .replace(/,/g, "_")
    .replace(/:/g, "_")
    .replace(/;/g, "_")
    .replace(/\(/g, "_")
    .replace(/\)/g, "_")
    .replace(/\[/g, "_")
    .replace(/\]/g, "_")
    .replace(/\{/g, "_")
    .replace(/\}/g, "_")
    .replace(/'/g, "_");
}

function entityMatchesLabel(entity: GraphEntity, label: string): boolean {
  const wanted = normalizeLabel(label);
  return entity.labels.some((value) => normalizeLabel(value) === wanted);
}

export function filterRelationshipsByEntities(
  rels: GraphRelationship[],
  entities: GraphEntity[],
): GraphRelationship[] {
  if (entities.length === 0) return rels;
  const ids = new Set(entities.map((e) => e.uuid));
  return rels.filter(
    (r) => ids.has(r.subject.uuid) || ids.has(r.object.uuid),
  );
}

export function filterRelationshipsByLabel(
  rels: GraphRelationship[],
  label: string,
): GraphRelationship[] {
  return rels.filter(
    (r) =>
      entityMatchesLabel(r.subject, label) || entityMatchesLabel(r.object, label),
  );
}

export function filterRelationshipsByQuery(
  rels: GraphRelationship[],
  query: string,
): GraphRelationship[] {
  const q = query.trim().toLowerCase();
  if (!q) return rels;
  return rels.filter(
    (r) =>
      r.subject.name.toLowerCase().includes(q) ||
      r.object.name.toLowerCase().includes(q) ||
      r.predicate.name.toLowerCase().includes(q),
  );
}

export function relationshipsToEdges(
  rels: GraphRelationship[],
): GraphEdge[] {
  const seen = new Map<string, number>();
  const edges: GraphEdge[] = [];
  for (const rel of rels) {
    const from = rel.subject.uuid;
    const to = rel.object.uuid;
    if (!from || !to) continue;
    const label = rel.predicate?.name?.trim() || "RELATED";
    const key = `${from}|${to}|${label}`;
    const count = seen.get(key) ?? 0;
    seen.set(key, count + 1);
    edges.push({
      id: `${from}-${to}-${label}-${count}`,
      from,
      to,
      label,
    });
  }
  return edges;
}
