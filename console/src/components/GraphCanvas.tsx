import { useEffect, useRef } from "react";
import { DataSet, Network } from "vis-network/standalone";
import "vis-network/styles/vis-network.min.css";
import { colorForLabel, primaryLabel } from "../lib/graphColors";
import type { GraphEdge, GraphEntity } from "../lib/graphModel";

interface GraphCanvasProps {
  entities: GraphEntity[];
  edges: GraphEdge[];
  selectedId: string | null;
  physics: boolean;
  onSelect: (entity: GraphEntity | null) => void;
  onExpand: (entity: GraphEntity) => void;
}

function truncate(text: string, max: number): string {
  if (text.length <= max) return text;
  return `${text.slice(0, max - 1)}…`;
}

function buildNode(entity: GraphEntity, selected: boolean) {
  const label = primaryLabel(entity.labels);
  const color = colorForLabel(label);
  return {
    id: entity.uuid,
    label: truncate(entity.name || entity.uuid, 28),
    title: [entity.name, label, entity.description ?? "", entity.uuid]
      .filter(Boolean)
      .join("\n"),
    shape: "dot" as const,
    size: selected ? 28 : 22,
    color: {
      background: color,
      border: selected ? "#ffffff" : "#1e293b",
      highlight: { background: color, border: "#ffffff" },
      hover: { background: color, border: "#e2e8f0" },
    },
    font: {
      color: "#e2e8f0",
      size: 13,
      face: "Inter, system-ui, sans-serif",
    },
    borderWidth: selected ? 3 : 1.5,
    entity,
  };
}

function buildEdgeItem(e: GraphEdge) {
  return {
    id: e.id,
    from: e.from,
    to: e.to,
    label: truncate(e.label, 28),
    title: e.label,
    arrows: { to: { enabled: true, scaleFactor: 0.8 } },
    color: {
      color: "#94a3b8",
      highlight: "#e2e8f0",
      hover: "#f1f5f9",
      opacity: 0.95,
    },
    font: {
      color: "#cbd5e1",
      size: 12,
      strokeWidth: 0,
      align: "horizontal" as const,
      background: "#1a1d23",
    },
    width: 2,
    smooth: { enabled: true, type: "continuous", roundness: 0.35 },
  };
}

export default function GraphCanvas({
  entities,
  edges,
  selectedId,
  physics,
  onSelect,
  onExpand,
}: GraphCanvasProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const networkRef = useRef<Network | null>(null);
  const nodesRef = useRef<DataSet<ReturnType<typeof buildNode>> | null>(null);
  const edgesRef = useRef<DataSet<ReturnType<typeof buildEdgeItem>> | null>(null);
  const entitiesRef = useRef(entities);
  entitiesRef.current = entities;

  useEffect(() => {
    if (!containerRef.current) return;

    const nodes = new DataSet(entities.map((e) => buildNode(e, false)));
    const edgeSet = new DataSet(edges.map(buildEdgeItem));
    nodesRef.current = nodes;
    edgesRef.current = edgeSet;

    const network = new Network(
      containerRef.current,
      { nodes, edges: edgeSet },
      {
        autoResize: true,
        interaction: {
          hover: true,
          tooltipDelay: 120,
          multiselect: false,
          navigationButtons: true,
          keyboard: true,
        },
        physics: {
          enabled: physics,
          solver: "forceAtlas2Based",
          forceAtlas2Based: {
            gravitationalConstant: -50,
            centralGravity: 0.01,
            springLength: 160,
            springConstant: 0.08,
            damping: 0.4,
            avoidOverlap: 0.5,
          },
          stabilization: { enabled: true, iterations: 200, updateInterval: 25 },
        },
        layout: { improvedLayout: true },
        nodes: {
          shadow: {
            enabled: true,
            size: 8,
            x: 0,
            y: 2,
            color: "rgba(0,0,0,0.35)",
          },
        },
        edges: {
          width: 2,
          selectionWidth: 3,
          smooth: { enabled: true, type: "continuous", roundness: 0.35 },
        },
      },
    );

    network.on("click", (params) => {
      if (!params.nodes.length) {
        onSelect(null);
        return;
      }
      const id = String(params.nodes[0]);
      onSelect(entitiesRef.current.find((e) => e.uuid === id) ?? null);
    });

    network.on("doubleClick", (params) => {
      if (!params.nodes.length) return;
      const entity = entitiesRef.current.find(
        (e) => e.uuid === String(params.nodes[0]),
      );
      if (entity) onExpand(entity);
    });

    network.once("stabilizationIterationsDone", () => {
      network.fit({ animation: { duration: 350, easingFunction: "easeInOutQuad" } });
    });

    networkRef.current = network;

    return () => {
      network.destroy();
      networkRef.current = null;
      nodesRef.current = null;
      edgesRef.current = null;
    };
  }, []);

  useEffect(() => {
    const nodes = nodesRef.current;
    const edgeSet = edgesRef.current;
    if (!nodes || !edgeSet) return;

    const nodeIds = new Set(entities.map((e) => e.uuid));
    for (const id of nodes.getIds().map(String)) {
      if (!nodeIds.has(id)) nodes.remove(id);
    }
    for (const entity of entities) {
      const item = buildNode(entity, entity.uuid === selectedId);
      if (nodes.get(entity.uuid)) nodes.update(item);
      else nodes.add(item);
    }

    const nodeIdSet = new Set(nodes.getIds().map(String));
    const validEdges = edges.filter(
      (e) => nodeIdSet.has(e.from) && nodeIdSet.has(e.to),
    );
    const edgeIds = new Set(validEdges.map((e) => e.id));
    for (const id of edgeSet.getIds().map(String)) {
      if (!edgeIds.has(id)) edgeSet.remove(id);
    }
    for (const e of validEdges) {
      const item = buildEdgeItem(e);
      if (edgeSet.get(e.id)) edgeSet.update(item);
      else edgeSet.add(item);
    }
  }, [entities, edges, selectedId]);

  useEffect(() => {
    networkRef.current?.setOptions({ physics: { enabled: physics } });
  }, [physics]);

  return (
    <div ref={containerRef} className="h-full w-full rounded-xl bg-[#1a1d23]" />
  );
}

export type { GraphEntity, GraphEdge };
