/**
 * BAG modular system-prompt loader.
 *
 * Reads `src/prompts/principles.md` + `src/prompts/tactics/*.md`, parses each
 * file's YAML frontmatter (with a tiny regex — no `js-yaml` dep), drops
 * tactics whose `status` is not `active`, concatenates the active bodies in
 * `order:` order, and injects them at the `${TACTICS}` placeholder inside
 * principles.
 *
 * Two placeholder substitutions are performed:
 *   - `${TACTICS}`         — replaced once by the joined active-tactic bodies.
 *   - `${SUBMIT_SENTINEL}` — replaced everywhere by the per-call sentinel.
 *
 * An attestation footer
 *     [Tactics loaded: N — auditable in src/prompts/tactics/]
 * is appended so an observer reading the prompt can audit which rules were
 * active at synthesis time.
 *
 * The loader is generic and contains no BAG-specific tactic content. To add a
 * new tactic, drop a markdown file under `src/prompts/tactics/`; to retire
 * one, flip its frontmatter `status` to `deprecated`.
 */

import { existsSync, readFileSync, readdirSync } from "node:fs";
import { dirname, isAbsolute, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

export type TacticStatus = "active" | "deprecated";

export type TacticFrontmatter = {
  id: string;
  status: TacticStatus;
  order?: number;
  incident?: string;
  introduced?: string;
  review_by?: string;
  trigger?: string;
  merged_into?: string;
  /** Captures any extra keys we don't recognise — useful for future evolution. */
  extra: Record<string, string>;
};

export type Tactic = {
  id: string;
  status: TacticStatus;
  body: string;
  frontmatter: TacticFrontmatter;
  /** Absolute path the tactic was loaded from. */
  path: string;
};

const TACTICS_PLACEHOLDER = "${TACTICS}";
const SENTINEL_PLACEHOLDER = "${SUBMIT_SENTINEL}";

const FRONTMATTER_RE = /^---\r?\n([\s\S]*?)\r?\n---\r?\n?/;

const moduleDir = (): string => {
  // Resolve `src/prompts/` regardless of whether we're running compiled or
  // via tsx. import.meta.url is `file://...src/prompts/loader.ts`.
  return dirname(fileURLToPath(import.meta.url));
};

const defaultRepoRoot = (): string => {
  // <repoRoot>/src/prompts/loader.ts → up two levels.
  return resolve(moduleDir(), "..", "..");
};

const promptsDir = (repoRoot?: string): string => {
  const root = repoRoot ?? defaultRepoRoot();
  // Direct hit when caller already pointed at the prompts dir.
  const direct = join(root, "src", "prompts");
  if (existsSync(direct)) return direct;
  // Fallback: assume the caller's `repoRoot` IS already the prompts dir.
  return root;
};

/**
 * Parse YAML frontmatter using a permissive line-by-line splitter. We
 * deliberately avoid `js-yaml`/`yaml` to keep the runtime bundle small and
 * dep-free. The schema is shallow (string scalars and one integer), so a
 * regex-based parser suffices.
 *
 * Returns `null` when the frontmatter is malformed (caller skips the file).
 */
export const parseFrontmatter = (
  raw: string,
): { frontmatter: TacticFrontmatter; body: string } | null => {
  const match = FRONTMATTER_RE.exec(raw);
  if (match == null) return null;
  const fmText = match[1] ?? "";
  const body = raw.slice(match[0].length);
  const fields: Record<string, string> = {};
  for (const line of fmText.split(/\r?\n/)) {
    if (line.trim().length === 0) continue;
    // Allow comments inside frontmatter.
    if (line.trim().startsWith("#")) continue;
    const colonIdx = line.indexOf(":");
    if (colonIdx < 0) {
      // Malformed line — treat the whole frontmatter as malformed.
      return null;
    }
    const key = line.slice(0, colonIdx).trim();
    let value = line.slice(colonIdx + 1).trim();
    // Strip matching quotes (single or double).
    if (
      (value.startsWith('"') && value.endsWith('"') && value.length >= 2) ||
      (value.startsWith("'") && value.endsWith("'") && value.length >= 2)
    ) {
      value = value.slice(1, -1);
    }
    fields[key] = value;
  }
  const id = fields["id"];
  const status = fields["status"];
  if (id == null || id.length === 0) return null;
  if (status !== "active" && status !== "deprecated") return null;
  const known = new Set([
    "id",
    "status",
    "order",
    "incident",
    "introduced",
    "review_by",
    "trigger",
    "merged_into",
  ]);
  const extra: Record<string, string> = {};
  for (const [k, v] of Object.entries(fields)) {
    if (!known.has(k)) extra[k] = v;
  }
  const orderRaw = fields["order"];
  const orderNum = orderRaw != null && orderRaw.length > 0 ? Number(orderRaw) : undefined;
  const fm: TacticFrontmatter = {
    id,
    status,
    extra,
    ...(orderNum !== undefined && Number.isFinite(orderNum) ? { order: orderNum } : {}),
    ...(fields["incident"] !== undefined ? { incident: fields["incident"] } : {}),
    ...(fields["introduced"] !== undefined ? { introduced: fields["introduced"] } : {}),
    ...(fields["review_by"] !== undefined ? { review_by: fields["review_by"] } : {}),
    ...(fields["trigger"] !== undefined ? { trigger: fields["trigger"] } : {}),
    ...(fields["merged_into"] !== undefined ? { merged_into: fields["merged_into"] } : {}),
  };
  return { frontmatter: fm, body };
};

const stripFrontmatter = (raw: string): string => {
  const m = FRONTMATTER_RE.exec(raw);
  return m == null ? raw : raw.slice(m[0].length);
};

const sortByOrder = (a: Tactic, b: Tactic): number => {
  const ao = a.frontmatter.order ?? Number.POSITIVE_INFINITY;
  const bo = b.frontmatter.order ?? Number.POSITIVE_INFINITY;
  if (ao !== bo) return ao - bo;
  // Stable tiebreaker by id so output is deterministic across filesystems.
  if (a.frontmatter.id < b.frontmatter.id) return -1;
  if (a.frontmatter.id > b.frontmatter.id) return 1;
  return 0;
};

/**
 * Load every tactic markdown file from `<repoRoot>/src/prompts/tactics/`.
 * Malformed frontmatter is logged to stderr and the file is skipped — never
 * a hard failure (the prompt must keep loading even if one tactic file is
 * broken).
 */
export const loadAllTactics = (repoRoot?: string): Tactic[] => {
  const dir = join(promptsDir(repoRoot), "tactics");
  if (!existsSync(dir)) return [];
  let entries: string[];
  try {
    entries = readdirSync(dir);
  } catch {
    return [];
  }
  const tactics: Tactic[] = [];
  for (const entry of entries.sort()) {
    if (!entry.endsWith(".md")) continue;
    const path = join(dir, entry);
    let raw: string;
    try {
      raw = readFileSync(path, "utf8");
    } catch (err) {
      // eslint-disable-next-line no-console
      console.warn(`[bag/prompts] cannot read tactic ${path}: ${(err as Error).message}`);
      continue;
    }
    const parsed = parseFrontmatter(raw);
    if (parsed == null) {
      // eslint-disable-next-line no-console
      console.warn(`[bag/prompts] malformed frontmatter in ${path} — skipping`);
      continue;
    }
    tactics.push({
      id: parsed.frontmatter.id,
      status: parsed.frontmatter.status,
      body: parsed.body,
      frontmatter: parsed.frontmatter,
      path,
    });
  }
  return tactics;
};

export const loadActiveTactics = (repoRoot?: string): Tactic[] => {
  return loadAllTactics(repoRoot)
    .filter((t) => t.status === "active")
    .sort(sortByOrder);
};

const loadPrinciples = (repoRoot?: string): { body: string; loaded: boolean } => {
  const file = join(promptsDir(repoRoot), "principles.md");
  if (!existsSync(file)) return { body: "", loaded: false };
  let raw: string;
  try {
    raw = readFileSync(file, "utf8");
  } catch (err) {
    // eslint-disable-next-line no-console
    console.warn(`[bag/prompts] cannot read principles ${file}: ${(err as Error).message}`);
    return { body: "", loaded: false };
  }
  return { body: stripFrontmatter(raw), loaded: true };
};

const trimTrailingNewlines = (s: string): string => s.replace(/\n+$/u, "");

const renderAttestation = (count: number, sourceLabel: string): string => {
  return `[Tactics loaded: ${count} — auditable in ${sourceLabel}]`;
};

export type BuildSystemPromptOptions = {
  sentinel: string;
  /** Override active tactics (e.g. tests). When undefined, loads from disk. */
  tactics?: Tactic[];
  /** Override principles body (e.g. tests). When undefined, loads from disk. */
  principles?: string;
  /** Override the repo root used for default loading. */
  repoRoot?: string;
  /** Footer label printed in the attestation. Defaults to the canonical path. */
  attestationLabel?: string;
  /**
   * Extra `${KEY}` placeholders to substitute in the assembled prompt. The
   * sentinel is always substituted via `${SUBMIT_SENTINEL}`; pass any
   * additional path-profile placeholders here (e.g. `SCRATCH`, `PATH_JOINED`).
   */
  placeholders?: Record<string, string>;
};

/**
 * Assemble the BAG executor system prompt.
 *
 * Contract:
 *  - Pure: same inputs ⇒ same output. No I/O when both `tactics` and
 *    `principles` are supplied.
 *  - Generic: knows nothing about BAG-specific clauses. Tactics are opaque
 *    bodies sorted by `order:` and pasted into the placeholder.
 *  - Byte-stable: when the previous monolithic prompt is migrated cleanly
 *    into principles.md + tactics/*.md, the output is byte-equivalent (modulo
 *    the appended attestation footer).
 */
export const buildSystemPrompt = (opts: BuildSystemPromptOptions): string => {
  const sentinel = opts.sentinel;
  if (typeof sentinel !== "string" || sentinel.length === 0) {
    throw new Error("buildSystemPrompt: sentinel must be a non-empty string");
  }
  const principlesBody = opts.principles ?? loadPrinciples(opts.repoRoot).body;
  const tactics = opts.tactics ?? loadActiveTactics(opts.repoRoot);

  const joined = tactics.map((t) => trimTrailingNewlines(t.body)).join("\n");
  let prompt: string;
  if (principlesBody.includes(TACTICS_PLACEHOLDER)) {
    prompt = principlesBody.replace(TACTICS_PLACEHOLDER, joined);
  } else {
    // No placeholder — append tactics after the principles body with a
    // single newline separator. Keeps the loader useful even with a
    // placeholder-less skeleton.
    prompt = `${principlesBody}${joined.length > 0 ? `\n${joined}` : ""}`;
  }
  // Substitute the sentinel everywhere it appears, plus any caller-provided
  // placeholders (e.g. PathProfile derived strings).
  const allPlaceholders: Record<string, string> = {
    SUBMIT_SENTINEL: sentinel,
    ...(opts.placeholders ?? {}),
  };
  for (const [key, value] of Object.entries(allPlaceholders)) {
    prompt = prompt.split(`\${${key}}`).join(value);
  }
  // Append attestation footer (always present, even when zero tactics — the
  // observer needs to know the count is zero).
  const label = opts.attestationLabel ?? "src/prompts/tactics/";
  const footer = renderAttestation(tactics.length, label);
  return `${prompt}\n${footer}\n`;
};

export const PROMPT_SENTINEL_PLACEHOLDER = SENTINEL_PLACEHOLDER;
export const PROMPT_TACTICS_PLACEHOLDER = TACTICS_PLACEHOLDER;

// Internal helper exported for tests only.
export const __test_only = {
  promptsDir,
  defaultRepoRoot,
  stripFrontmatter,
  trimTrailingNewlines,
  renderAttestation,
  isAbsolute,
};
