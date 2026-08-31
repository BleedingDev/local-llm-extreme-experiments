/**
 * Tests for src/harness-gates.ts — env-var contract for the BAG ablation
 * harness. We mutate process.env directly (and restore on teardown) because
 * `loadHarnessGates` is the only function that reads it; no other module
 * caches the result during a test.
 */
import { afterEach, beforeEach, describe, expect, test } from "bun:test";
import {
  BAG_MODE_BARE_ENV,
  BAG_MODE_MINIMAL_ENV,
  HARNESS_GATE_ENV_VARS,
  loadHarnessGates,
  type HarnessGates,
} from "../src/harness-gates";

const ALL_GATE_VARS = Object.values(HARNESS_GATE_ENV_VARS).concat(["BAG_CODE_SEARCH"]);

const snapshotEnv = (): Record<string, string | undefined> => {
  const snap: Record<string, string | undefined> = {};
  for (const v of ALL_GATE_VARS) snap[v] = process.env[v];
  return snap;
};

const restoreEnv = (snap: Record<string, string | undefined>): void => {
  for (const [k, v] of Object.entries(snap)) {
    if (v === undefined) delete process.env[k];
    else process.env[k] = v;
  }
};

describe("harness-gates / loadHarnessGates", () => {
  let saved: Record<string, string | undefined>;
  beforeEach(() => {
    saved = snapshotEnv();
    for (const v of ALL_GATE_VARS) delete process.env[v];
  });
  afterEach(() => {
    restoreEnv(saved);
  });

  test("all gates default to true when no env vars are set", () => {
    const gates = loadHarnessGates();
    expect(gates).toEqual({
      probeExtractor: true,
      selfCheck: true,
      snapshotRestore: true,
      viewImage: true,
      codeSearch: true,
      retryPath: true,
      clusterMatcher: true,
      // editStrategy defaults to the registry's "shell-heredoc" (current behavior)
      editStrategy: "shell-heredoc",
    } satisfies HarnessGates);
  });

  test("each boolean gate can be flipped off independently via its env var", () => {
    const cases: Array<[Exclude<keyof HarnessGates, "editStrategy">, string]> = [
      ["probeExtractor", "BAG_GATE_PROBE_EXTRACTOR"],
      ["selfCheck", "BAG_GATE_SELF_CHECK"],
      ["snapshotRestore", "BAG_GATE_SNAPSHOT_RESTORE"],
      ["viewImage", "BAG_TOOL_VIEW_IMAGE"],
      ["codeSearch", "BAG_TOOL_CODE_SEARCH"],
      ["retryPath", "BAG_GATE_RETRY"],
      ["clusterMatcher", "BAG_GATE_CLUSTER_MATCHER"],
    ];
    for (const [field, envVar] of cases) {
      // start clean
      for (const v of ALL_GATE_VARS) delete process.env[v];
      process.env[envVar] = "0";
      const g = loadHarnessGates();
      expect(g[field]).toBe(false);
      // sibling gates remain enabled
      for (const [other] of cases) {
        if (other === field) continue;
        expect(g[other]).toBe(true);
      }
    }
  });

  test("non-zero values do NOT disable a gate (only literal '0' is the off-switch)", () => {
    process.env[HARNESS_GATE_ENV_VARS.selfCheck] = "1";
    expect(loadHarnessGates().selfCheck).toBe(true);
    process.env[HARNESS_GATE_ENV_VARS.selfCheck] = "true";
    expect(loadHarnessGates().selfCheck).toBe(true);
    process.env[HARNESS_GATE_ENV_VARS.selfCheck] = "";
    expect(loadHarnessGates().selfCheck).toBe(true);
    process.env[HARNESS_GATE_ENV_VARS.selfCheck] = "0";
    expect(loadHarnessGates().selfCheck).toBe(false);
  });

  test("legacy BAG_CODE_SEARCH=0 still disables codeSearch (back-compat)", () => {
    process.env.BAG_CODE_SEARCH = "0";
    expect(loadHarnessGates().codeSearch).toBe(false);
  });

  test("BAG_TOOL_CODE_SEARCH=0 disables codeSearch (new generic name)", () => {
    process.env[HARNESS_GATE_ENV_VARS.codeSearch] = "0";
    expect(loadHarnessGates().codeSearch).toBe(false);
  });
});

describe("harness-gates / presets", () => {
  let saved: Record<string, string | undefined>;
  beforeEach(() => {
    saved = snapshotEnv();
    for (const v of ALL_GATE_VARS) delete process.env[v];
  });
  afterEach(() => {
    restoreEnv(saved);
  });

  test("BAG_MODE_BARE_ENV: gates off, multi-tool ON", () => {
    Object.assign(process.env, BAG_MODE_BARE_ENV);
    const g = loadHarnessGates();
    expect(g.probeExtractor).toBe(false);
    expect(g.selfCheck).toBe(false);
    expect(g.snapshotRestore).toBe(false);
    expect(g.retryPath).toBe(false);
    expect(g.clusterMatcher).toBe(false);
    // Multi-tool surface remains:
    expect(g.viewImage).toBe(true);
    expect(g.codeSearch).toBe(true);
  });

  test("BAG_MODE_MINIMAL_ENV: every boolean gate AND every non-bash tool OFF", () => {
    Object.assign(process.env, BAG_MODE_MINIMAL_ENV);
    const g = loadHarnessGates();
    // All seven boolean gates must be false in MINIMAL mode.
    expect(g.probeExtractor).toBe(false);
    expect(g.selfCheck).toBe(false);
    expect(g.snapshotRestore).toBe(false);
    expect(g.viewImage).toBe(false);
    expect(g.codeSearch).toBe(false);
    expect(g.retryPath).toBe(false);
    expect(g.clusterMatcher).toBe(false);
    // editStrategy is a categorical, not a boolean; minimal mode keeps the
    // shell-heredoc default so the BAG-minimal cell looks like a vanilla
    // mini-swe-agent baseline (single bash tool, no structured edit tool).
    expect(g.editStrategy).toBe("shell-heredoc");
  });

  test("MINIMAL preset is a strict superset of BARE preset", () => {
    for (const k of Object.keys(BAG_MODE_BARE_ENV)) {
      expect(BAG_MODE_MINIMAL_ENV[k]).toBe(BAG_MODE_BARE_ENV[k]!);
    }
  });

  test("HARNESS_GATE_ENV_VARS exposes one env var per HarnessGates field", () => {
    const fields: Array<keyof HarnessGates> = [
      "probeExtractor",
      "selfCheck",
      "snapshotRestore",
      "viewImage",
      "codeSearch",
      "retryPath",
      "clusterMatcher",
    ];
    for (const f of fields) {
      expect(typeof HARNESS_GATE_ENV_VARS[f]).toBe("string");
      expect(HARNESS_GATE_ENV_VARS[f].length).toBeGreaterThan(0);
    }
  });
});
