import { randomBytes } from "node:crypto";
import * as p from "@clack/prompts";
import pc from "picocolors";
import { askPassword, isPromptBack, pickOne, type PromptBack } from "../lib/prompts.js";
import type { AuthChoices } from "../types.js";

function generateToken(): string {
  return `brainpat_${randomBytes(24).toString("hex")}`;
}

export async function askAuth(opts?: { allowBack?: false }): Promise<AuthChoices>;
export async function askAuth(opts: {
  allowBack: true;
  backHint?: string;
}): Promise<AuthChoices | PromptBack>;
export async function askAuth(opts?: {
  allowBack?: boolean;
  backHint?: string;
}): Promise<AuthChoices | PromptBack> {
  p.log.step("Authentication token");

  const action = opts?.allowBack
    ? await pickOne<"generate" | "paste">({
        message: "BRAINPAT_TOKEN (used by the API for personal auth)",
        options: [
          { value: "generate", label: "Generate one for me" },
          { value: "paste", label: "Paste my own" },
        ],
        initialValue: "generate",
        allowBack: true,
        backHint: opts.backHint,
      })
    : await pickOne<"generate" | "paste">({
        message: "BRAINPAT_TOKEN (used by the API for personal auth)",
        options: [
          { value: "generate", label: "Generate one for me" },
          { value: "paste", label: "Paste my own" },
        ],
        initialValue: "generate",
      });
  if (isPromptBack(action)) {
    return action;
  }

  if (action === "generate") {
    const token = generateToken();
    p.note(token, "Generated token (saved to .env)");
    p.log.info(pc.dim("Keep this somewhere safe — it grants API access."));
    return { brainpatToken: token };
  }

  const pasted = await askPassword({
    message: "Paste your BRAINPAT_TOKEN",
    validate: (value) => (value.trim().length === 0 ? "Token is required" : undefined),
  });
  return { brainpatToken: pasted.trim() };
}
