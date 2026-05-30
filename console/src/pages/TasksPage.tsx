import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { apiFetch } from "../lib/api";

interface TaskItem {
  id?: string;
  task_id?: string;
  status: string;
  result?: unknown;
  data?: unknown;
  created_at?: number;
}

const TERMINAL = new Set(["completed", "failed", "error"]);

function statusColor(status: string): string {
  switch (status) {
    case "completed":
      return "text-emerald-400";
    case "failed":
    case "error":
      return "text-red-400";
    case "started":
    case "queued":
      return "text-amber-400";
    default:
      return "text-slate-400";
  }
}

export default function TasksPage() {
  const [tasks, setTasks] = useState<TaskItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [expanded, setExpanded] = useState<string | null>(null);
  const [detail, setDetail] = useState<TaskItem | null>(null);

  async function loadTasks() {
    try {
      const res = await apiFetch<{ tasks: TaskItem[] }>("/tasks/");
      setTasks(res.tasks ?? []);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load tasks");
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    loadTasks();
    const interval = setInterval(() => {
      const hasActive = tasks.some((t) => !TERMINAL.has(t.status));
      if (hasActive || tasks.length === 0) loadTasks();
    }, 5000);
    return () => clearInterval(interval);
  }, []);

  async function expandTask(taskId: string) {
    if (expanded === taskId) {
      setExpanded(null);
      setDetail(null);
      return;
    }
    setExpanded(taskId);
    try {
      const res = await apiFetch<TaskItem>(`/tasks/${taskId}`);
      setDetail(res);
    } catch {
      setDetail(null);
    }
  }

  return (
    <div>
      <div className="mb-4 flex items-center justify-between">
        <h1 className="text-2xl font-semibold">Tasks</h1>
        <button
          type="button"
          onClick={loadTasks}
          className="rounded-lg border border-slate-700 px-3 py-1.5 text-sm hover:border-cyan-700"
        >
          Refresh
        </button>
      </div>

      {error && <p className="mb-2 text-sm text-red-400">{error}</p>}

      {loading ? (
        <p className="text-slate-500">Loading…</p>
      ) : tasks.length === 0 ? (
        <p className="text-slate-500">
          No tasks yet.{" "}
          <Link to="/ingest" className="text-cyan-500 hover:underline">
            Ingest some data
          </Link>
        </p>
      ) : (
        <div className="space-y-2">
          {tasks.map((task) => {
            const id = task.id ?? task.task_id ?? "unknown";
            return (
              <div
                key={id}
                className="rounded-xl border border-slate-800 bg-slate-950"
              >
                <button
                  type="button"
                  onClick={() => expandTask(id)}
                  className="flex w-full items-center justify-between px-4 py-3 text-left"
                >
                  <span className="font-mono text-xs text-cyan-600">{id}</span>
                  <span className={`text-sm font-medium ${statusColor(task.status)}`}>
                    {task.status}
                  </span>
                </button>
                {expanded === id && detail && (
                  <pre className="overflow-auto border-t border-slate-800 p-4 text-xs text-slate-400">
                    {JSON.stringify(detail, null, 2)}
                  </pre>
                )}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
