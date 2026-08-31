/// <reference path="../../types/bun-test.d.ts" />

import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { describe, expect, test } from "bun:test";
import {
  OPTIMIZER_GATE_SUITE_PATH,
  type OptimizerGateSuite,
} from "./gate-suite";
import { evaluateOptimizerRuntimeReadiness } from "./runtime-readiness";

const now = "2026-05-04T12:00:00.000Z";

const baseSuite = (overrides: Partial<OptimizerGateSuite> = {}): OptimizerGateSuite => ({
  schemaVersion: "local-evidence-optimizer-gate-suite.v1",
  optimizerGateSuiteId: "optimizer-gate-suite.runtime-test",
  graphId: "self-evolving-runtime-gates-v1",
  generatedAt: now,
  sourceEvidenceIndex: ".bag/evidence/index.jsonl",
  sourceScorecardSuite: ".bag/evidence/scorecards/index.json",
  contracts: [
    {
      contractId: "optimizer-runtime-readiness.test",
      jsonPath: ".bag/evidence/optimizer/runtime-readiness.json",
      markdownPath: "docs/local-evidence-optimizer-runtime-readiness.md",
      primaryUse: "runtime promotion readiness enforcement",
    },
  ],
  currentDecision: {
    candidateGeneration: "allowed_as_scoped_dry_run",
    autoPromotion: "allowed",
    promotionReady: true,
    blockingReasons: [],
  },
  mustFailClosedOn: [
    "missing optimizer gate suite",
    "invalid optimizer gate suite",
    "blocking optimizer gate suite decision",
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
  const cwd = mkdtempSync(join(tmpdir(), "optimizer-runtime-readiness-"));
  try {
    fn(cwd);
  } finally {
    rmSync(cwd, { recursive: true, force: true });
  }
};

const writeSuite = (cwd: string, value: unknown): void => {
  const path = join(cwd, OPTIMIZER_GATE_SUITE_PATH);
  mkdirSync(dirname(path), { recursive: true });
  writeFileSync(path, `${typeof value === "string" ? value : JSON.stringify(value, null, 2)}\n`, "utf8");
};

describe("optimizer runtime readiness", () => {
  test("allows auto-promotion only when the gate suite is promotion ready", () => {
    withTempCwd((cwd) => {
      writeSuite(cwd, baseSuite());

      const decision = evaluateOptimizerRuntimeReadiness({ cwd, checkedAt: now });

      expect(decision.decision).toBe("allow");
      expect(decision.allowed).toBe(true);
      expect(decision.failClosed).toBe(false);
      expect(decision.requiredCapability).toBe("auto_promotion");
      expect(decision.reasons).toEqual(["Optimizer gate suite allows runtime auto-promotion."]);
      expect(decision.gateSuite).toMatchObject({
        suiteLoaded: true,
        suiteId: "optimizer-gate-suite.runtime-test",
        graphId: "self-evolving-runtime-gates-v1",
        promotionAllowed: true,
        candidateGeneration: "allowed_as_scoped_dry_run",
        contractIds: ["optimizer-runtime-readiness.test"],
      });
    });
  });

  test("fails closed when the gate suite is missing", () => {
    withTempCwd((cwd) => {
      const decision = evaluateOptimizerRuntimeReadiness({ cwd, checkedAt: now });

      expect(decision.decision).toBe("block");
      expect(decision.allowed).toBe(false);
      expect(decision.failClosed).toBe(true);
      expect(decision.gateSuite.suiteLoaded).toBe(false);
      expect(decision.gateSuite.errors).toMatchObject([{ kind: "missing" }]);
      expect(decision.reasons[0]).toContain("missing");
    });
  });

  test("fails closed when the gate suite is invalid", () => {
    withTempCwd((cwd) => {
      writeSuite(cwd, "{ not-json");

      const decision = evaluateOptimizerRuntimeReadiness({ cwd, checkedAt: now });

      expect(decision.decision).toBe("block");
      expect(decision.gateSuite.errors).toMatchObject([{ kind: "parse_error" }]);
      expect(decision.reasons[0]).toContain("parse_error");
    });
  });

  test("fails closed when the gate suite blocks auto-promotion", () => {
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

      const decision = evaluateOptimizerRuntimeReadiness({ cwd, checkedAt: now });

      expect(decision.decision).toBe("block");
      expect(decision.allowed).toBe(false);
      expect(decision.gateSuite.suiteLoaded).toBe(true);
      expect(decision.gateSuite.promotionAllowed).toBe(false);
      expect(decision.reasons).toEqual([
        "Auto-promotion is blocked.",
        "Promotion readiness is false.",
        "operator approval and rollback checkpoint are required",
      ]);
    });
  });

  test("allows scheduler candidate generation when auto-promotion remains blocked", () => {
    withTempCwd((cwd) => {
      writeSuite(cwd, baseSuite({
        currentDecision: {
          candidateGeneration: "allowed_as_scoped_dry_run",
          autoPromotion: "blocked",
          promotionReady: false,
          blockingReasons: [
            "operator approval is required before promotion",
          ],
        },
      }));

      const decision = evaluateOptimizerRuntimeReadiness({
        cwd,
        checkedAt: now,
        requiredCapability: "candidate_generation",
      });

      expect(decision.decision).toBe("allow");
      expect(decision.allowed).toBe(true);
      expect(decision.failClosed).toBe(false);
      expect(decision.gateSuite.promotionAllowed).toBe(false);
      expect(decision.reasons).toEqual([
        "Optimizer gate suite allows candidate generation: allowed_as_scoped_dry_run.",
      ]);
    });
  });

  test("includes registry and resolved policy evidence and blocks registry load errors", () => {
    withTempCwd((cwd) => {
      writeSuite(cwd, baseSuite());

      const decision = evaluateOptimizerRuntimeReadiness({
        cwd,
        checkedAt: now,
        registry: {
          root: join(cwd, ".bag", "optimizer"),
          invalidRecords: [
            {
              path: join(cwd, ".bag", "optimizer", "records", "bad.json"),
              kind: "validation_error",
              message: "recordKind is invalid",
            },
          ],
          errors: [
            {
              path: join(cwd, ".bag", "optimizer", "records", "bad.json"),
              kind: "validation_error",
              message: "recordKind is invalid",
            },
          ],
        },
        resolvedPolicy: {
          source: "seed",
          modelProfileId: "model.local",
          codebaseProfileId: "codebase.test",
          codebaseRootFingerprint: "sha256:test",
          policyId: "policy.test",
        },
      });

      expect(decision.decision).toBe("block");
      expect(decision.registry).toMatchObject({
        errorCount: 1,
        invalidRecordCount: 1,
      });
      expect(decision.resolvedPolicy).toMatchObject({
        source: "seed",
        policyId: "policy.test",
      });
      expect(decision.reasons).toContain("Optimizer registry has 1 load error(s) and 1 invalid record(s).");
    });
  });
});
