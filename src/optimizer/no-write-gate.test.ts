import { describe, expect, test } from "bun:test";
import type { NoWriteValidationInput } from "../replay/no-write-validation";
import { evaluateNoWritePromotionGate } from "./no-write-gate";

const mutationCase: NoWriteValidationInput = {
  recordId: "record.gate.no-write",
  taskId: "task.gate.no-write",
  routeSelectedMode: "coding",
  expectedMutation: "edit_existing",
  expectedSideEffect: "mutation",
  changedFiles: [],
  fsWriteCount: 0,
  terminalCreateCount: 0,
  terminalExitCount: 0,
  terminalCommandCount: 0,
  stopReason: "end_turn",
  editStrategyFamily: "none",
  verifierStatus: "failed",
  evidenceRefs: [
    ".bag/evidence/scorecards/tool-routing.json",
    ".bag/replay-corpus/real-acp-runs/real-acp-run.headless-visible-20260504/real-acp-run.headless-visible-20260504.manifest.json",
  ],
};

describe("no-write promotion gate", () => {
  test("blocks promotion when a mutation-expected ACP coding case makes no progress", () => {
    const decision = evaluateNoWritePromotionGate({
      cases: [mutationCase],
      requireEvidence: true,
    });

    expect(decision).toMatchObject({
      gateId: "acp-no-write-progress",
      status: "block",
      passed: false,
      blocking: true,
      checkedRecordIds: ["record.gate.no-write"],
      blockedRecordIds: ["record.gate.no-write"],
      resultCounts: {
        total: 1,
        blocked: 1,
      },
    });
    expect(decision.evidenceRefs).toContain(".bag/evidence/scorecards/tool-routing.json");
    expect(decision.reasons.join("\n")).toContain("record.gate.no-write");
  });

  test("warns for explicit verifier skip justification without blocking promotion", () => {
    const decision = evaluateNoWritePromotionGate({
      cases: [{
        ...mutationCase,
        recordId: "record.gate.verifier-skip",
        verifierStatus: "skipped",
        verifierSkippedJustification: {
          present: true,
          policy: "allowed_to_skip",
          reason: "docs update has no executable verifier",
        },
      }],
      requireEvidence: true,
    });

    expect(decision).toMatchObject({
      status: "warn",
      passed: true,
      blocking: false,
      warnedRecordIds: ["record.gate.verifier-skip"],
      resultCounts: {
        total: 1,
        passed: 1,
        warned: 1,
      },
    });
  });

  test("passes when write or terminal progress evidence is present", () => {
    const decision = evaluateNoWritePromotionGate({
      cases: [{
        ...mutationCase,
        recordId: "record.gate.progress",
        changedFiles: ["src/greeter.ts"],
        fsWriteCount: 1,
        terminalCreateCount: 1,
        terminalExitCount: 1,
        terminalCommandCount: 1,
        editStrategyFamily: "whole_file",
        verifierStatus: "passed",
      }],
      requireEvidence: true,
    });

    expect(decision).toMatchObject({
      status: "pass",
      passed: true,
      blocking: false,
      blockedRecordIds: [],
      warnedRecordIds: [],
      resultCounts: {
        total: 1,
        passed: 1,
        blocked: 0,
        warned: 0,
      },
    });
  });

  test("can require at least one no-write validation slice", () => {
    const decision = evaluateNoWritePromotionGate({
      cases: [],
      requireEvidence: true,
    });

    expect(decision).toMatchObject({
      status: "block",
      passed: false,
      blocking: true,
      resultCounts: {
        total: 0,
      },
    });
  });
});
