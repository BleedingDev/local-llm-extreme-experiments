/// <reference path="../../types/bun-test.d.ts" />

import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { describe, expect, test } from "bun:test";
import {
  OPTIMIZER_GATE_SUITE_PATH,
  loadOptimizerGateSuiteStatus,
  type OptimizerGateSuite,
} from "./gate-suite";

const now = "2026-05-04T11:18:00Z";

const baseSuite = (overrides: Partial<OptimizerGateSuite> = {}): OptimizerGateSuite => ({
  schemaVersion: "local-evidence-optimizer-gate-suite.v1",
  optimizerGateSuiteId: "optimizer-gate-suite.test",
  graphId: "local-evidence-flywheel-v1",
  generatedAt: now,
  sourceEvidenceIndex: ".bag/evidence/index.jsonl",
  sourceScorecardSuite: ".bag/evidence/scorecards/index.json",
  contracts: [
    {
      contractId: "optimizer-input-slices.test",
      jsonPath: ".bag/evidence/optimizer/input-slices.json",
      markdownPath: "docs/local-evidence-optimizer-input-slices.md",
      primaryUse: "candidate/dev/hidden-holdout/monitor/live optimizer visibility rules",
    },
  ],
  currentDecision: {
    candidateGeneration: "allowed_as_scoped_dry_run",
    autoPromotion: "allowed",
    promotionReady: true,
    blockingReasons: [],
  },
  mustFailClosedOn: [
    "schema quality failure",
    "missing rollback checkpoint",
  ],
  policySeparation: {
    dimensions: [
      "modelProfileId",
      "codebaseProfileId",
      "modelCodebasePolicyId",
    ],
    principle: "Promotion applies only to the exact evaluated model/codebase policy tuple.",
  },
  ...overrides,
});

const withTempCwd = (fn: (cwd: string) => void): void => {
  const cwd = mkdtempSync(join(tmpdir(), "optimizer-gate-suite-"));
  try {
    fn(cwd);
  } finally {
    rmSync(cwd, { recursive: true, force: true });
  }
};

const writeSuite = (cwd: string, value: unknown): void => {
  const path = join(cwd, OPTIMIZER_GATE_SUITE_PATH);
  mkdirSync(dirname(path), { recursive: true });
  writeFileSync(path, `${typeof value === "string" ? value : JSON.stringify(value, null, 2)}\n`);
};

describe("optimizer gate suite status", () => {
  test("loads a valid promotion-ready suite", () => {
    withTempCwd((cwd) => {
      writeSuite(cwd, baseSuite());

      const status = loadOptimizerGateSuiteStatus({ cwd });

      expect(status.suiteLoaded).toBe(true);
      expect(status.state).toBe("promotion_ready");
      expect(status.promotionAllowed).toBe(true);
      expect(status.autoPromotionAllowed).toBe(true);
      expect(status.errors).toEqual([]);
      expect(status.suite?.optimizerGateSuiteId).toBe("optimizer-gate-suite.test");
      expect(status.mustFailClosedOn).toContain("schema quality failure");
    });
  });

  test("fails closed when the suite is missing", () => {
    withTempCwd((cwd) => {
      const status = loadOptimizerGateSuiteStatus({ cwd });

      expect(status.suiteLoaded).toBe(false);
      expect(status.state).toBe("fail_closed");
      expect(status.promotionAllowed).toBe(false);
      expect(status.autoPromotionAllowed).toBe(false);
      expect(status.errors).toMatchObject([{ kind: "missing" }]);
      expect(status.blockingReasons[0]).toContain("missing");
    });
  });

  test("fails closed for invalid JSON and schema-ish shape", () => {
    withTempCwd((cwd) => {
      writeSuite(cwd, "{ not-json");
      const invalidJson = loadOptimizerGateSuiteStatus({ cwd });

      expect(invalidJson.state).toBe("fail_closed");
      expect(invalidJson.errors).toMatchObject([{ kind: "parse_error" }]);
    });

    withTempCwd((cwd) => {
      writeSuite(cwd, {
        schemaVersion: "local-evidence-optimizer-gate-suite.v1",
        optimizerGateSuiteId: "optimizer-gate-suite.incomplete",
        currentDecision: {
          autoPromotion: "allowed",
          promotionReady: true,
          blockingReasons: [],
        },
      });
      const invalidShape = loadOptimizerGateSuiteStatus({ cwd });

      expect(invalidShape.state).toBe("fail_closed");
      expect(invalidShape.errors).toMatchObject([{ kind: "validation_error" }]);
      expect(invalidShape.errors[0]?.message).toContain("graphId");
    });
  });

  test("fails closed when the loaded suite blocks promotion", () => {
    withTempCwd((cwd) => {
      writeSuite(cwd, baseSuite({
        currentDecision: {
          candidateGeneration: "allowed_as_scoped_dry_run",
          autoPromotion: "blocked",
          promotionReady: false,
          blockingReasons: [
            "operator approval and rollback checkpoint are required",
          ],
        },
      }));

      const status = loadOptimizerGateSuiteStatus({ cwd });

      expect(status.suiteLoaded).toBe(true);
      expect(status.state).toBe("fail_closed");
      expect(status.errors).toEqual([]);
      expect(status.promotionAllowed).toBe(false);
      expect(status.autoPromotionAllowed).toBe(false);
      expect(status.blockingReasons).toContain("Auto-promotion is blocked.");
      expect(status.blockingReasons).toContain("Promotion readiness is false.");
      expect(status.blockingReasons).toContain("operator approval and rollback checkpoint are required");
    });
  });
});
