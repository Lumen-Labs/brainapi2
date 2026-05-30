import * as p from "@clack/prompts";
import pc from "picocolors";
import { isPromptBack, pickOne, type PromptBack } from "../lib/prompts.js";
import type { OcrMode, PipelineChoices } from "../types.js";

export async function askPipeline(options?: {
  prechosenOcrMode?: OcrMode;
  allowBack?: false;
  initialOcrMode?: OcrMode;
}): Promise<PipelineChoices>;
export async function askPipeline(options: {
  prechosenOcrMode?: OcrMode;
  allowBack: true;
  backHint?: string;
  initialOcrMode?: OcrMode;
}): Promise<PipelineChoices | PromptBack>;
export async function askPipeline(options?: {
  prechosenOcrMode?: OcrMode;
  allowBack?: boolean;
  backHint?: string;
  initialOcrMode?: OcrMode;
}): Promise<PipelineChoices | PromptBack> {
  if (options?.prechosenOcrMode) {
    return { ocrMode: options.prechosenOcrMode };
  }

  p.log.step("Document ingestion");
  const ocrMode = options?.allowBack
    ? await pickOne<OcrMode>({
        message: "OCR engine for document parsing",
        options: [
          {
            value: "docparser",
            label: "Remote DocParser (default)",
            hint: "lightweight install, calls Lumen DocParser API",
          },
          {
            value: "docling",
            label: "Local Docling",
            hint: "adds ~2 GB ML deps (docling, accelerate)",
          },
        ],
        initialValue: options.initialOcrMode ?? "docparser",
        allowBack: true,
        backHint: options.backHint,
      })
    : await pickOne<OcrMode>({
        message: "OCR engine for document parsing",
        options: [
          {
            value: "docparser",
            label: "Remote DocParser (default)",
            hint: "lightweight install, calls Lumen DocParser API",
          },
          {
            value: "docling",
            label: "Local Docling",
            hint: "adds ~2 GB ML deps (docling, accelerate)",
          },
        ],
        initialValue: options?.initialOcrMode ?? "docparser",
      });
  if (isPromptBack(ocrMode)) {
    return ocrMode;
  }

  if (ocrMode === "docling") {
    p.log.info(
      pc.dim(
        "Docling and accelerate will be installed into the venv after this flow.",
      ),
    );
  }

  return { ocrMode };
}

export function ocrToExtras(ocrMode: OcrMode): string[] {
  return ocrMode === "docling" ? ["docling-ocr"] : [];
}

export function postgresBackendExtras(dbs: {
  vectorDb: string;
  dataDb: string;
  graphDb: string;
}): string[] {
  const needs =
    dbs.vectorDb === "postgresql" ||
    dbs.dataDb === "postgresql" ||
    dbs.graphDb === "networkx";
  return needs ? ["postgresql-backend"] : [];
}
