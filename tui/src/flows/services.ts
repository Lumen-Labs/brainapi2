import * as p from "@clack/prompts";
import { isPromptBack, pickOne, type PromptBack } from "../lib/prompts.js";
import type { DbChoices } from "../types.js";

export async function askDatabases(opts?: {
  allowBack?: false;
  initial?: Partial<DbChoices>;
}): Promise<DbChoices>;
export async function askDatabases(opts: {
  allowBack: true;
  backHint?: string;
  initial?: Partial<DbChoices>;
}): Promise<DbChoices | PromptBack>;
export async function askDatabases(opts?: {
  allowBack?: boolean;
  backHint?: string;
  initial?: Partial<DbChoices>;
}): Promise<DbChoices | PromptBack> {
  p.log.step("Pick your databases");

  let vectorDb = opts?.initial?.vectorDb ?? "postgresql";
  let dataDb = opts?.initial?.dataDb ?? "postgresql";
  let graphDb = opts?.initial?.graphDb ?? "networkx";
  let step = 0;

  while (step < 3) {
    if (step === 0) {
      const picked = opts?.allowBack
        ? await pickOne<"postgresql" | "milvus">({
            message: "Vector database",
            options: [
              { value: "postgresql", label: "PostgreSQL + pgvector (default)", hint: "self-hosted, lightweight" },
              { value: "milvus", label: "Milvus", hint: "dedicated vector DB" },
            ],
            initialValue: vectorDb,
            allowBack: true,
            backHint: opts.backHint,
          })
        : await pickOne<"postgresql" | "milvus">({
            message: "Vector database",
            options: [
              { value: "postgresql", label: "PostgreSQL + pgvector (default)", hint: "self-hosted, lightweight" },
              { value: "milvus", label: "Milvus", hint: "dedicated vector DB" },
            ],
            initialValue: vectorDb,
          });
      if (isPromptBack(picked)) return picked;
      vectorDb = picked;
      step = 1;
      continue;
    }
    if (step === 1) {
      const picked = await pickOne<"postgresql" | "mongo">({
        message: "Data database (text chunks, observations, structured data)",
        options: [
          { value: "postgresql", label: "PostgreSQL (default)", hint: "reuses the same Postgres if picked above" },
          { value: "mongo", label: "MongoDB" },
        ],
        initialValue: dataDb,
        allowBack: true,
        backHint: "Previous question",
      });
      if (isPromptBack(picked)) {
        step = 0;
        continue;
      }
      dataDb = picked;
      step = 2;
      continue;
    }
    const picked = await pickOne<"networkx" | "neo4j">({
      message: "Graph database",
      options: [
        { value: "networkx", label: "NetworkX on PostgreSQL (default)", hint: "no extra DB server" },
        { value: "neo4j", label: "Neo4j", hint: "dedicated graph DB" },
      ],
      initialValue: graphDb,
      allowBack: true,
      backHint: "Previous question",
    });
    if (isPromptBack(picked)) {
      step = 1;
      continue;
    }
    graphDb = picked;
    return { vectorDb, dataDb, graphDb };
  }

  return { vectorDb, dataDb, graphDb };
}
