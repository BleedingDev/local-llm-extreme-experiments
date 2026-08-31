import { afterEach, beforeEach, describe, expect, test } from "bun:test";
import {
  HARNESS_EDIT_STRATEGIES,
  HARNESS_GATE_ENV_VARS,
  loadHarnessGates,
} from "../../src/harness-gates";

const ENV_KEY = HARNESS_GATE_ENV_VARS.editStrategy;

const captureEnv = () => {
  const original = process.env[ENV_KEY];
  return () => {
    if (original === undefined) {
      delete process.env[ENV_KEY];
    } else {
      process.env[ENV_KEY] = original;
    }
  };
};

describe("BAG_EDIT_STRATEGY env-var selection", () => {
  let restore: () => void;

  beforeEach(() => {
    restore = captureEnv();
  });
  afterEach(() => restore());

  test("default is shell-heredoc when env-var is unset", () => {
    delete process.env[ENV_KEY];
    expect(loadHarnessGates().editStrategy).toBe("shell-heredoc");
    expect(HARNESS_EDIT_STRATEGIES.default).toBe("shell-heredoc");
  });

  test("env-var picks the registered strategy", () => {
    for (const id of HARNESS_EDIT_STRATEGIES.values) {
      process.env[ENV_KEY] = id;
      expect(loadHarnessGates().editStrategy).toBe(id);
    }
  });

  test("invalid env-var falls back to shell-heredoc with a warning", () => {
    process.env[ENV_KEY] = "not-a-real-strategy";
    const warnings: unknown[][] = [];
    const original = console.warn;
    console.warn = ((...args: unknown[]) => {
      warnings.push(args);
    }) as unknown as typeof console.warn;
    try {
      const gates = loadHarnessGates();
      expect(gates.editStrategy).toBe("shell-heredoc");
      expect(warnings.length).toBeGreaterThan(0);
      const firstArg = (warnings[0]?.[0] ?? "") as string;
      expect(firstArg).toContain("BAG_EDIT_STRATEGY");
      expect(firstArg).toContain("not-a-real-strategy");
    } finally {
      console.warn = original;
    }
  });

  test("empty env-var keeps the default", () => {
    process.env[ENV_KEY] = "";
    expect(loadHarnessGates().editStrategy).toBe("shell-heredoc");
  });

  test("env-var name matches BAG_EDIT_STRATEGY (collision check)", () => {
    expect(HARNESS_GATE_ENV_VARS.editStrategy).toBe("BAG_EDIT_STRATEGY");
    // Make sure none of the harness ablation gate names collide with the
    // edit-strategy slot — the parallel ablation agent owns BAG_GATE_* and
    // BAG_TOOL_* exclusively.
    const otherEnvNames = Object.entries(HARNESS_GATE_ENV_VARS)
      .filter(([key]) => key !== "editStrategy")
      .map(([, value]) => value);
    expect(otherEnvNames).not.toContain("BAG_EDIT_STRATEGY");
  });
});
