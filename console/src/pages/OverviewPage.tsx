import { useEffect, useState } from "react";
import { Link as RouterLink } from "react-router-dom";
import {
  Inline,
  SectionBand,
  SectionBandContent,
  SectionBandDescription,
  SectionBandEyebrow,
  SectionBandHeader,
  SectionBandTitle,
  SectionStack,
  Stat,
  StatLabel,
  StatValue,
  StatusIndicator,
} from "lumen-ui-kit";
import { apiFetch, getSession } from "../lib/api";
import { PageFrame } from "../components/Workbench";

interface StatCard {
  label: string;
  value: string | number;
  loading: boolean;
  failed: boolean;
  to: string;
}

const shortcuts = [
  {
    title: "Inspect graph",
    description: "Explore entities and relationships for the active brain.",
    to: "/graph",
  },
  {
    title: "Browse memory",
    description: "Search text chunks, structured records, and observations.",
    to: "/data",
  },
  {
    title: "Ingest text",
    description: "Submit new content and track the resulting task pipeline.",
    to: "/ingest",
  },
  {
    title: "Source sync",
    description: "Webhook and scheduled help-center sync, including last synced.",
    to: "/sources",
  },
];

export default function OverviewPage() {
  const session = getSession();
  const [stats, setStats] = useState<StatCard[]>([
    { label: "Entities", value: "—", loading: true, failed: false, to: "/graph" },
    {
      label: "Relationships",
      value: "—",
      loading: true,
      failed: false,
      to: "/graph",
    },
    {
      label: "Text chunks",
      value: "—",
      loading: true,
      failed: false,
      to: "/data",
    },
    {
      label: "Observations",
      value: "—",
      loading: true,
      failed: false,
      to: "/observations",
    },
    { label: "Tasks", value: "—", loading: true, failed: false, to: "/tasks" },
    {
      label: "Vectors",
      value: "—",
      loading: true,
      failed: false,
      to: "/vectors",
    },
  ]);

  useEffect(() => {
    let cancelled = false;

    async function load() {
      const results = [...stats];

      try {
        const entities = await apiFetch<{ total?: number }>(
          "/retrieve/entities?limit=1",
        );
        results[0] = {
          ...results[0],
          value: entities.total ?? 0,
          loading: false,
          failed: false,
        };
      } catch {
        results[0] = { ...results[0], value: "?", loading: false, failed: true };
      }

      try {
        const rels = await apiFetch<{ total?: number }>(
          "/retrieve/relationships?limit=1",
        );
        results[1] = {
          ...results[1],
          value: rels.total ?? 0,
          loading: false,
          failed: false,
        };
      } catch {
        results[1] = { ...results[1], value: "?", loading: false, failed: true };
      }

      try {
        const chunks = await apiFetch<{ total?: number }>(
          "/retrieve/text-chunks?limit=1",
        );
        results[2] = {
          ...results[2],
          value: chunks.total ?? 0,
          loading: false,
          failed: false,
        };
      } catch {
        results[2] = { ...results[2], value: "?", loading: false, failed: true };
      }

      try {
        const obs = await apiFetch<{ count?: number; observations?: unknown[] }>(
          "/retrieve/observations?limit=1",
        );
        results[3] = {
          ...results[3],
          value: obs.count ?? obs.observations?.length ?? 0,
          loading: false,
          failed: false,
        };
      } catch {
        results[3] = { ...results[3], value: "?", loading: false, failed: true };
      }

      try {
        const tasks = await apiFetch<{ tasks?: unknown[] }>("/tasks/");
        results[4] = {
          ...results[4],
          value: tasks.tasks?.length ?? 0,
          loading: false,
          failed: false,
        };
      } catch {
        results[4] = { ...results[4], value: "?", loading: false, failed: true };
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
        results[5] = {
          ...results[5],
          value: vectorTotal,
          loading: false,
          failed: false,
        };
      } catch {
        results[5] = { ...results[5], value: "?", loading: false, failed: true };
      }

      if (!cancelled) setStats(results);
    }

    load();
    return () => {
      cancelled = true;
    };
  }, []);

  const healthy = stats.filter((s) => !s.loading && !s.failed).length;
  const failed = stats.filter((s) => s.failed).length;

  return (
    <PageFrame className="overflow-auto">
      <SectionStack className="border border-lumen-border">
        <SectionBand tone="accent">
          <SectionBandHeader>
            <SectionBandEyebrow>Operations</SectionBandEyebrow>
            <SectionBandTitle>Brain health</SectionBandTitle>
            <SectionBandDescription>
              Live inventory for{" "}
              <span className="font-mono text-lumen-foreground">
                {session?.brainId}
              </span>{" "}
              on {session?.apiBaseUrl}
            </SectionBandDescription>
          </SectionBandHeader>
          <SectionBandContent>
            <Inline gap="md" className="flex-wrap items-center">
              <StatusIndicator status="success">
                {healthy} endpoints responding
              </StatusIndicator>
              {failed > 0 ? (
                <StatusIndicator status="warning">
                  {failed} unavailable
                </StatusIndicator>
              ) : (
                <StatusIndicator status="neutral">
                  All sampled counts available
                </StatusIndicator>
              )}
              <RouterLink
                to="/ingest"
                className="inline-flex h-9 items-center border border-lumen-control-border bg-lumen-action-secondary px-3 text-sm font-medium text-lumen-on-action-secondary"
              >
                Ingest data
              </RouterLink>
            </Inline>
          </SectionBandContent>
        </SectionBand>

        <SectionBand>
          <SectionBandHeader>
            <SectionBandEyebrow>Inventory</SectionBandEyebrow>
            <SectionBandTitle>Memory footprint</SectionBandTitle>
            <SectionBandDescription>
              Counts are sampled with limit=1 queries against retrieve and task
              endpoints.
            </SectionBandDescription>
          </SectionBandHeader>
          <SectionBandContent className="p-0">
            <div className="grid grid-cols-2 border border-lumen-border md:grid-cols-3">
              {stats.map((stat) => (
                <RouterLink
                  key={stat.label}
                  to={stat.to}
                  className="border-b border-r border-lumen-border p-5 transition-colors last:border-r-0 md:[&:nth-child(3n)]:border-r-0 hover:bg-lumen-muted focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-[-2px] focus-visible:outline-lumen-primary"
                >
                  <Stat>
                    <StatLabel className="text-sm text-lumen-muted-foreground">
                      {stat.label}
                    </StatLabel>
                    <StatValue className="mt-1 text-3xl font-semibold text-lumen-foreground">
                      {stat.loading ? "…" : stat.value}
                    </StatValue>
                    {stat.loading ? (
                      <StatusIndicator status="neutral" className="mt-2">
                        Loading
                      </StatusIndicator>
                    ) : stat.failed ? (
                      <StatusIndicator status="warning" className="mt-2">
                        Unavailable
                      </StatusIndicator>
                    ) : (
                      <StatusIndicator status="success" className="mt-2">
                        Open {stat.label.toLowerCase()}
                      </StatusIndicator>
                    )}
                  </Stat>
                </RouterLink>
              ))}
            </div>
          </SectionBandContent>
        </SectionBand>

        <SectionBand tone="muted">
          <SectionBandHeader>
            <SectionBandEyebrow>Workflows</SectionBandEyebrow>
            <SectionBandTitle>Common console paths</SectionBandTitle>
            <SectionBandDescription>
              Jump into the densest operational views for this brain.
            </SectionBandDescription>
          </SectionBandHeader>
          <SectionBandContent>
            <div className="grid gap-0 border border-lumen-border md:grid-cols-3">
              {shortcuts.map((item) => (
                <RouterLink
                  key={item.to}
                  to={item.to}
                  className="border-b border-lumen-border p-4 last:border-b-0 md:border-b-0 md:border-r md:last:border-r-0"
                >
                  <div className="text-sm font-semibold text-lumen-foreground">
                    {item.title}
                  </div>
                  <p className="mt-1 text-sm text-lumen-muted-foreground">
                    {item.description}
                  </p>
                </RouterLink>
              ))}
            </div>
          </SectionBandContent>
        </SectionBand>
      </SectionStack>
    </PageFrame>
  );
}
