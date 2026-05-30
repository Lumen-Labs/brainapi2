#!/usr/bin/env node
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(here, "..", "..");
const sourcePath = resolve(repoRoot, "scripts", "install_extras.py");
const targetPath = resolve(here, "..", "src", "embedded", "install-extras-script.ts");

const script = readFileSync(sourcePath, "utf8");
mkdirSync(dirname(targetPath), { recursive: true });
const out = `// Auto-generated from scripts/install_extras.py — do not edit by hand.
// Regenerate with: npm run sync:install-extras (also runs as prebuild).

export const INSTALL_EXTRAS_SCRIPT = ${JSON.stringify(script)};
`;
writeFileSync(targetPath, out, "utf8");
console.log(`Synced ${sourcePath} → ${targetPath} (${script.length} bytes)`);
