import { useEffect, useState } from "react";
import {
  Alert,
  Button,
  EmptyState,
  Field,
  FieldLabel,
  Select,
  StatusIndicator,
  Table,
  TableBody,
  TableCell,
  TableEmptyState,
  TableHead,
  TableHeader,
  TableRow,
  TextField,
} from "lumen-ui-kit";
import { RestartIcon, Icon } from "lumen-ui-kit/icons";
import { apiFetch, getSession } from "../lib/api";
import { JsonInspector, Workbench } from "../components/Workbench";

interface SourceItem {
  id: string;
  name: string;
  source_type: string;
  sync_mode: string;
  last_synced: string | null;
  poll_interval_seconds?: number;
  base_url?: string | null;
  webhook_secret_env?: string | null;
  api_token_env?: string | null;
  webhook_path?: string;
  skip_enrichment?: boolean;
}

function statusFor(lastSynced: string | null): "success" | "warning" | "neutral" {
  if (!lastSynced) return "neutral";
  const parsed = Date.parse(lastSynced);
  if (Number.isNaN(parsed)) return "warning";
  const ageMs = Date.now() - parsed;
  if (ageMs < 24 * 60 * 60 * 1000) return "success";
  return "warning";
}

export default function SourcesPage() {
  const session = getSession();
  const [sources, setSources] = useState<SourceItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selected, setSelected] = useState<SourceItem | null>(null);
  const [name, setName] = useState("Help center");
  const [syncMode, setSyncMode] = useState("webhook");
  const [baseUrl, setBaseUrl] = useState("");
  const [secretEnv, setSecretEnv] = useState("ZENDESK_GUIDE_WEBHOOK_SECRET");
  const [tokenEnv, setTokenEnv] = useState("");
  const [saving, setSaving] = useState(false);
  const [rechecking, setRechecking] = useState<string | null>(null);

  async function loadSources() {
    try {
      const res = await apiFetch<{ sources: SourceItem[] }>("/source-sync/sources");
      setSources(res.sources ?? []);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load sources");
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    loadSources();
  }, []);

  async function createSource(e: React.FormEvent) {
    e.preventDefault();
    setSaving(true);
    setError(null);
    try {
      await apiFetch<SourceItem>("/source-sync/sources", {
        method: "POST",
        body: JSON.stringify({
          name,
          source_type: "zendesk_guide",
          sync_mode: syncMode,
          base_url: baseUrl || null,
          webhook_secret_env: secretEnv || null,
          api_token_env: tokenEnv || null,
        }),
      });
      await loadSources();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to create source");
    } finally {
      setSaving(false);
    }
  }

  async function recheck(id: string) {
    setRechecking(id);
    try {
      await apiFetch(`/source-sync/sources/${id}/recheck`, { method: "POST" });
      await loadSources();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Recheck failed");
    } finally {
      setRechecking(null);
    }
  }

  const webhookUrl = selected?.webhook_path
    ? `${session?.apiBaseUrl ?? ""}${selected.webhook_path}`
    : null;

  return (
    <Workbench
      flush
      title="Sources"
      description="Webhook and scheduled help-center sync. Last synced is the time the connector last applied a revision."
      actions={
        <Button type="button" variant="secondary" size="small" onClick={loadSources}>
          <Icon source={RestartIcon} />
          Refresh
        </Button>
      }
      toolbar={
        error ? (
          <Alert variant="danger" title="Source sync">
            {error}
          </Alert>
        ) : undefined
      }
      inspector={
        selected ? (
          <JsonInspector
            title={selected.name}
            subtitle={webhookUrl ?? selected.id}
            value={selected}
            onClose={() => setSelected(null)}
          />
        ) : undefined
      }
    >
      <form
        onSubmit={createSource}
        className="grid gap-3 border-b border-lumen-border p-4 sm:grid-cols-2 lg:grid-cols-3"
      >
        <TextField
          id="source-name"
          label="Name"
          value={name}
          onChange={(e) => setName(e.target.value)}
          required
        />
        <Field>
          <FieldLabel htmlFor="source-mode">Sync mode</FieldLabel>
          <Select
            id="source-mode"
            value={syncMode}
            onChange={(e) => setSyncMode(e.target.value)}
          >
            <option value="webhook">Webhook</option>
            <option value="poll">Scheduled re-check</option>
          </Select>
        </Field>
        <TextField
          id="source-base"
          label="Help center URL"
          value={baseUrl}
          onChange={(e) => setBaseUrl(e.target.value)}
          placeholder="https://example.zendesk.com"
        />
        <TextField
          id="source-secret"
          label="Webhook secret env"
          value={secretEnv}
          onChange={(e) => setSecretEnv(e.target.value)}
        />
        <TextField
          id="source-token"
          label="API token env"
          value={tokenEnv}
          onChange={(e) => setTokenEnv(e.target.value)}
          placeholder="ZENDESK_GUIDE_API_TOKEN"
        />
        <div className="flex items-end">
          <Button type="submit" size="small" disabled={saving}>
            {saving ? "Saving…" : "Add Zendesk Guide source"}
          </Button>
        </div>
      </form>
      {loading ? (
        <p className="p-6 text-lumen-muted-foreground">Loading…</p>
      ) : sources.length === 0 ? (
        <div className="p-6">
          <EmptyState
            title="No sources yet"
            description="Register a Zendesk Guide source to receive webhooks or poll for article changes."
          />
        </div>
      ) : (
        <Table containerProps={{ "aria-label": "Synced sources" }}>
          <TableHeader>
            <TableRow>
              <TableHead>Name</TableHead>
              <TableHead>Type</TableHead>
              <TableHead>Mode</TableHead>
              <TableHead>Last synced</TableHead>
              <TableHead>Actions</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {sources.length === 0 ? (
              <TableEmptyState colSpan={5} title="No sources" />
            ) : (
              sources.map((source) => {
                const isSelected = selected?.id === source.id;
                return (
                  <TableRow
                    key={source.id}
                    onClick={() => setSelected(source)}
                    aria-selected={isSelected}
                    className={
                      isSelected
                        ? "cursor-pointer bg-lumen-muted"
                        : "cursor-pointer"
                    }
                  >
                    <TableCell className="font-medium">{source.name}</TableCell>
                    <TableCell className="font-mono text-xs">
                      {source.source_type}
                    </TableCell>
                    <TableCell>{source.sync_mode}</TableCell>
                    <TableCell>
                      <StatusIndicator status={statusFor(source.last_synced)}>
                        {source.last_synced ?? "Never"}
                      </StatusIndicator>
                    </TableCell>
                    <TableCell>
                      <Button
                        type="button"
                        size="small"
                        variant="secondary"
                        disabled={rechecking === source.id}
                        onClick={(e) => {
                          e.stopPropagation();
                          recheck(source.id);
                        }}
                      >
                        {rechecking === source.id ? "Queuing…" : "Re-check"}
                      </Button>
                    </TableCell>
                  </TableRow>
                );
              })
            )}
          </TableBody>
        </Table>
      )}
    </Workbench>
  );
}
