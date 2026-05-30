import * as p from "@clack/prompts";

export const PROMPT_BACK = "__back__" as const;
export type PromptBack = typeof PROMPT_BACK;

export function isPromptBack(value: unknown): value is PromptBack {
  return value === PROMPT_BACK;
}

export interface PickOption<T extends string> {
  value: T;
  label: string;
  hint?: string;
}

type PickOneBase<T extends string> = {
  message: string;
  options: PickOption<T>[];
  initialValue?: T;
};

export async function pickOne<T extends string>(
  opts: PickOneBase<T> & { allowBack?: false },
): Promise<T>;
export async function pickOne<T extends string>(
  opts: PickOneBase<T> & {
    allowBack: true;
    backLabel?: string;
    backHint?: string;
  },
): Promise<T | PromptBack>;
export async function pickOne<T extends string>(
  opts: PickOneBase<T> & {
    allowBack?: boolean;
    backLabel?: string;
    backHint?: string;
  },
): Promise<T | PromptBack> {
  const options = opts.allowBack
    ? [
        {
          value: PROMPT_BACK as T,
          label: opts.backLabel ?? "Back",
          hint: opts.backHint,
        },
        ...opts.options,
      ]
    : opts.options;
  const result = await (p.select as unknown as (o: {
    message: string;
    options: PickOption<T>[];
    initialValue?: T;
  }) => Promise<T | symbol>)({
    message: opts.message,
    options,
    initialValue: opts.initialValue,
  });
  if (p.isCancel(result)) {
    p.cancel("Setup cancelled.");
    process.exit(1);
  }
  if (isPromptBack(result)) {
    return PROMPT_BACK;
  }
  return result as T;
}

export async function askConfirmOrBack(opts: {
  message: string;
  initialValue?: boolean;
  backHint?: string;
}): Promise<boolean | PromptBack> {
  const initial = opts.initialValue === false ? "no" : "yes";
  const choice = await pickOne<"yes" | "no">({
    message: opts.message,
    allowBack: true,
    backHint: opts.backHint,
    initialValue: initial,
    options: [
      { value: "yes", label: "Yes" },
      { value: "no", label: "No" },
    ],
  });
  if (isPromptBack(choice)) {
    return PROMPT_BACK;
  }
  return choice === "yes";
}

export async function askConfirm(opts: {
  message: string;
  initialValue?: boolean;
}): Promise<boolean> {
  const result = await p.confirm(opts);
  if (p.isCancel(result)) {
    p.cancel("Setup cancelled.");
    process.exit(1);
  }
  return result as boolean;
}

export async function askText(opts: {
  message: string;
  placeholder?: string;
  defaultValue?: string;
  initialValue?: string;
  validate?: (value: string) => string | undefined;
}): Promise<string> {
  const result = await p.text(opts);
  if (p.isCancel(result)) {
    p.cancel("Setup cancelled.");
    process.exit(1);
  }
  return result as string;
}

export async function askPassword(opts: {
  message: string;
  validate?: (value: string) => string | undefined;
}): Promise<string> {
  const result = await p.password(opts);
  if (p.isCancel(result)) {
    p.cancel("Setup cancelled.");
    process.exit(1);
  }
  return result as string;
}
