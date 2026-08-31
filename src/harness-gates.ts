/**
 * Central harness-gating module — read-once env-var resolution for the
 * BAG ablation study (see `docs/bag-harness-ablation-study.md`).
 *
 * Every BAG harness gate (probe extractor, self-check auditor, snapshot/
 * restore around probes, view_image tool, code_search tool, Best-of-N
 * retry path, failure-cluster matcher) is independently controllable via
 * an env var. Set the env var to `0` to disable. Defaults are ON so
 * existing behavior is unchanged for callers that do nothing.
 *
 * In addition to the boolean gates the module also exposes the per-turn
 * EDIT-STRATEGY selector (`BAG_EDIT_STRATEGY`). It defaults to
 * `shell-heredoc` so the BAG-full path is byte-for-byte unchanged; setting
 * the env var to one of the other registered strategy ids switches the
 * autonomous-coding-turn over to a structured edit tool. See
 * `src/edit-strategies/registry.ts` and `docs/bag-edit-strategy-study.md`.
 *
 * Two presets are exposed for the A/B/C ablation harness:
 *   - `BAG_MODE_BARE_ENV`    — gates OFF, multi-tool ON
 *   - `BAG_MODE_MINIMAL_ENV` — everything OFF (single-tool only)
 *
 * The "BAG-full" mode is "all defaults" (no env override needed).
 *
 * Design notes:
 *   - This module reads the process env eagerly when `loadHarnessGates()` is
 *     called. Call it once per turn at the top of `runAutonomousCodingTurn`
 *     (and once at `buildVerifierFromInstruction`) and pass the result down
 *     — DON'T spread `process.env` reads through every code path.
 *   - The env-var contract is part of the bag-runtime ABI: the harness
 *     driver in `bench/ablation/run_ablation.sh` writes these vars into the
 *     container env via `--agent-kwarg env=…` so the gating travels with
 *     the BAG runtime bundle.
 */

/**
 * The full set of harness gates exposed to the autonomous coding turn and
 * the instruction-verifier helpers. Each field is `true` when the gate is
 * ENABLED (the default) and `false` when the matching `BAG_*` env var is
 * set to `"0"`.
 */
export type HarnessGates = {
  /** When false, `buildVerifierFromInstruction` returns undefined: no probes ever run. */
  probeExtractor: boolean;
  /** When false, `runSelfCheckGate` is skipped end-to-end. */
  selfCheck: boolean;
  /** When false, snapshot/restore around probes is skipped (probes still run if probeExtractor=true). */
  snapshotRestore: boolean;
  /** When false, the `view_image` tool is removed from the tool list entirely. */
  viewImage: boolean;
  /** When false, the `code_search` tool is removed from the tool list entirely. */
  codeSearch: boolean;
  /** When false, no Best-of-N retry — first verifier failure ends the attempt. */
  retryPath: boolean;
  /** When false, failure-cluster hint injection is skipped on retry (verifier-signature-library may still fire). */
  clusterMatcher: boolean;
  /**
   * Selected edit strategy id. Default = `"shell-heredoc"` (current behavior:
   * the model edits files via the bash tool only). Other values switch the
   * autonomous-coding-turn over to a structured edit tool from the registry —
   * see `src/edit-strategies/registry.ts`. Read from `BAG_EDIT_STRATEGY`;
   * unknown values fall back to `"shell-heredoc"` with a warning so a typo
   * never silently degrades a model that lacks the structured tool.
   */
  editStrategy:
    | "shell-heredoc"
    | "fs-write-whole-file"
    | "edit-tool-stringreplace"
    | "apply-patch-unified"
    | "edit-diff-blocks";
};

const ENV_VAR_NAMES = {
  probeExtractor: "BAG_GATE_PROBE_EXTRACTOR",
  selfCheck: "BAG_GATE_SELF_CHECK",
  snapshotRestore: "BAG_GATE_SNAPSHOT_RESTORE",
  viewImage: "BAG_TOOL_VIEW_IMAGE",
  codeSearch: "BAG_TOOL_CODE_SEARCH",
  retryPath: "BAG_GATE_RETRY",
  clusterMatcher: "BAG_GATE_CLUSTER_MATCHER",
  editStrategy: "BAG_EDIT_STRATEGY",
} as const satisfies Record<keyof HarnessGates, string>;

/**
 * Source of truth for the registered edit-strategy ids. Mirrors the runtime
 * factory in `src/edit-strategies/registry.ts`; we duplicate it here so the
 * harness gate module stays leaf-level (no import of runtime apply code) and
 * its tests can be run in isolation.
 */
const EDIT_STRATEGY_VALUES = [
  "shell-heredoc",
  "fs-write-whole-file",
  "edit-tool-stringreplace",
  "apply-patch-unified",
  "edit-diff-blocks",
] as const satisfies readonly HarnessGates["editStrategy"][];

const DEFAULT_EDIT_STRATEGY: HarnessGates["editStrategy"] = "shell-heredoc";

const resolveEditStrategy = (raw: string | undefined): HarnessGates["editStrategy"] => {
  if (raw === undefined || raw === "") return DEFAULT_EDIT_STRATEGY;
  if ((EDIT_STRATEGY_VALUES as readonly string[]).includes(raw)) {
    return raw as HarnessGates["editStrategy"];
  }
  // eslint-disable-next-line no-console
  console.warn(
    `[harness-gates] BAG_EDIT_STRATEGY='${raw}' is not a registered strategy; falling back to '${DEFAULT_EDIT_STRATEGY}'. Valid values: ${EDIT_STRATEGY_VALUES.join(", ")}`,
  );
  return DEFAULT_EDIT_STRATEGY;
};

/**
 * Backwards-compat alias: the existing `BAG_CODE_SEARCH` env var was already
 * threaded through `runAutonomousCodingTurn` before this ablation harness
 * landed. We honor BOTH `BAG_TOOL_CODE_SEARCH=0` (the new generic gate name)
 * AND the legacy `BAG_CODE_SEARCH=0` so existing A/B harnesses don't break.
 */
const LEGACY_CODE_SEARCH_ENV = "BAG_CODE_SEARCH";

const isDisabled = (value: string | undefined): boolean => value === "0";

/**
 * Read the current process env once and produce a `HarnessGates` snapshot.
 * Pure function of `process.env` — call once per turn, then thread the
 * result through helper APIs. Safe to call multiple times (it just re-reads
 * the env each time).
 */
export const loadHarnessGates = (): HarnessGates => {
  const codeSearchDisabled =
    isDisabled(process.env[ENV_VAR_NAMES.codeSearch]) ||
    isDisabled(process.env[LEGACY_CODE_SEARCH_ENV]);
  return {
    probeExtractor: !isDisabled(process.env[ENV_VAR_NAMES.probeExtractor]),
    selfCheck: !isDisabled(process.env[ENV_VAR_NAMES.selfCheck]),
    snapshotRestore: !isDisabled(process.env[ENV_VAR_NAMES.snapshotRestore]),
    viewImage: !isDisabled(process.env[ENV_VAR_NAMES.viewImage]),
    codeSearch: !codeSearchDisabled,
    retryPath: !isDisabled(process.env[ENV_VAR_NAMES.retryPath]),
    clusterMatcher: !isDisabled(process.env[ENV_VAR_NAMES.clusterMatcher]),
    editStrategy: resolveEditStrategy(process.env[ENV_VAR_NAMES.editStrategy]),
  };
};

/**
 * Exported for tests so they can assert the registered strategy id list and
 * the default. DO NOT use this from production code — read via
 * `loadHarnessGates()`.
 */
export const HARNESS_EDIT_STRATEGIES = {
  values: EDIT_STRATEGY_VALUES,
  default: DEFAULT_EDIT_STRATEGY,
} as const;

/**
 * Env preset for the BAG-bare ablation cell:
 *   - All LLM gates OFF (no probe extraction, no self-check, no snapshot/restore,
 *     no Best-of-N retry, no cluster-matcher hint).
 *   - Multi-tool surface still enabled (bash + view_image + code_search) so we
 *     can isolate the "harness scaffolding" cost from the "tool richness" cost.
 *
 * Use as: `Object.assign(env, BAG_MODE_BARE_ENV)` before launching the harness.
 */
export const BAG_MODE_BARE_ENV: Record<string, string> = {
  [ENV_VAR_NAMES.probeExtractor]: "0",
  [ENV_VAR_NAMES.selfCheck]: "0",
  [ENV_VAR_NAMES.snapshotRestore]: "0",
  [ENV_VAR_NAMES.retryPath]: "0",
  [ENV_VAR_NAMES.clusterMatcher]: "0",
  // viewImage + codeSearch deliberately left ON
};

/**
 * Env preset for the BAG-minimal ablation cell:
 *   - Everything OFF — only the bash tool, no LLM gates, no rich tooling.
 *   - This is the "mini-swe-agent baseline" condition: a tool-use loop with
 *     a single bash tool and no scaffolding. BAG-full vs BAG-minimal isolates
 *     the FULL contribution of the harness over a baseline tool-use loop.
 *
 * Note: BAG-minimal disables view_image, so any task that requires native
 * vision (e.g. chess-best-move) is expected to fail at the input layer.
 * That is the intended signal — chess pass/fail tells you whether the model
 * had to "see" the board.
 */
export const BAG_MODE_MINIMAL_ENV: Record<string, string> = {
  ...BAG_MODE_BARE_ENV,
  [ENV_VAR_NAMES.viewImage]: "0",
  [ENV_VAR_NAMES.codeSearch]: "0",
  // Also set the legacy alias so older code paths see the disable.
  [LEGACY_CODE_SEARCH_ENV]: "0",
};

/**
 * Exposed for tests so we can assert the env-var contract is stable.
 * DO NOT use this from production code — read the gates via `loadHarnessGates()`.
 */
export const HARNESS_GATE_ENV_VARS = ENV_VAR_NAMES;
