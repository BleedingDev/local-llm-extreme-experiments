import { describe, expect, test } from "bun:test";
import {
  createEditAttemptRecord,
  editAttemptRecordTargetHash,
} from "../../acp/edit-attempt-record";
import {
  failureSignalsForEditAttemptRecord,
  projectEditAttemptRecordsToScorecard,
} from "./edit-attempt-scorecard-projection";

const createdAt = "2026-05-04T12:00:00.000Z";

const baseInput = (id: string) => ({
  editAttemptRecordId: `edit-record.${id}`,
  editAttemptId: `edit-attempt.${id}`,
  runId: `run.${id}`,
  traceId: `trace.${id}`,
  editStrategyId: "edit.hash-range.experimental.v1",
  renderedEditContractVersion: "rendered-edit-contract.v1",
  modelProfileId: "model.scorecard.local",
  codebaseProfileId: "codebase.scorecard.fixture",
  policyId: "policy.scorecard.fixture",
  targetPaths: [`src/${id}.ts`],
  createdAt,
});

const changedHash = (id: string) =>
  editAttemptRecordTargetHash({
    path: `src/${id}.ts`,
    beforeHash: `sha256:${id}:before`,
    afterHash: `sha256:${id}:after`,
  });

const beforeOnlyHash = (id: string) =>
  editAttemptRecordTargetHash({
    path: `src/${id}.ts`,
    beforeHash: `sha256:${id}:before`,
  });

describe("edit attempt scorecard projection", () => {
  test("aggregates first-class edit attempt records by outcome and failure signal set", () => {
    const success = createEditAttemptRecord({
      ...baseInput("success"),
      targetHashes: [changedHash("success")],
      phases: {
        preview: { status: "passed" },
        apply: { status: "passed" },
        write: { status: "passed" },
        verify: { status: "passed", artifactRefs: ["artifact://success/verify.log"] },
      },
      artifactRefs: ["artifact://success/diff.patch", "terminal://looks-like-a-shell-failure.log"],
    });
    const noWrite = createEditAttemptRecord({
      ...baseInput("no-write"),
      targetHashes: [beforeOnlyHash("no-write")],
      phases: {
        preview: { status: "skipped", skipJustification: "model produced no edit operations" },
        apply: { status: "skipped", skipJustification: "model produced no edit operations" },
        write: { status: "skipped", skipJustification: "no write was produced" },
        verify: { status: "skipped", skipJustification: "no write was produced" },
      },
      artifactRefs: ["artifact://no-write/record.json"],
    });
    const appliedBroken = createEditAttemptRecord({
      ...baseInput("applied-broken"),
      targetHashes: [changedHash("applied-broken")],
      phases: {
        preview: { status: "passed" },
        apply: { status: "passed" },
        write: { status: "passed" },
        verify: { status: "failed", errorCode: "verifier_error", artifactRefs: ["artifact://applied-broken/typecheck.log"] },
      },
      signals: {
        staleContext: { status: "fresh" },
        appliedButBroken: {
          detected: true,
          status: "inconsistent",
          evidenceRefs: ["artifact://applied-broken/post-apply.json"],
        },
      },
    });
    const rollback = createEditAttemptRecord({
      ...baseInput("rollback"),
      targetHashes: [changedHash("rollback")],
      phases: {
        preview: { status: "passed" },
        apply: { status: "passed" },
        write: { status: "passed" },
        verify: { status: "failed", errorCode: "verifier_error", artifactRefs: ["artifact://rollback/verify.log"] },
        repair: { status: "failed", errorCode: "self_detected_regression" },
        rollback: { status: "passed", artifactRefs: ["artifact://rollback/result.json"] },
      },
      signals: {
        staleContext: { status: "fresh" },
        selfDetectedRegression: {
          status: "confirmed",
          evidenceRefs: ["artifact://rollback/self-check.json"],
        },
      },
    });
    const verifierSkipped = createEditAttemptRecord({
      ...baseInput("verifier-skipped"),
      targetHashes: [changedHash("verifier-skipped")],
      phases: {
        preview: { status: "passed" },
        apply: { status: "passed" },
        write: { status: "passed" },
        verify: { status: "skipped", skipJustification: "docs-only change has no verifier command" },
      },
      artifactRefs: ["artifact://verifier-skipped/diff.patch"],
    });

    const projection = projectEditAttemptRecordsToScorecard({
      graphId: "self-evolving-runtime-gates-v1",
      generatedAt: "2026-05-04T12:30:00.000Z",
      records: [verifierSkipped, rollback, appliedBroken, noWrite, success],
    });
    const reorderedProjection = projectEditAttemptRecordsToScorecard({
      graphId: "self-evolving-runtime-gates-v1",
      generatedAt: "2026-05-04T12:30:00.000Z",
      records: [success, noWrite, appliedBroken, rollback, verifierSkipped],
    });

    expect(reorderedProjection).toEqual(projection);
    expect(projection.sourceBasis).toBe("edit_attempt_records");
    expect(projection.sourceRecordCount).toBe(5);
    expect(projection.totals.byFinalOutcome).toEqual({
      applied_but_broken: 1,
      no_write: 1,
      rolled_back: 1,
      success: 2,
    });
    expect(projection.totals.byFailureSignal).toMatchObject({
      applied_but_broken: 1,
      no_write: 1,
      repair_failed: 1,
      rolled_back: 1,
      self_detected_regression: 1,
      verifier_failed: 2,
      verifier_skipped: 2,
    });
    expect(projection.groups).toHaveLength(5);
    expect(projection.evidenceRefs).toContain("artifact://rollback/self-check.json");
    expect(projection.evidenceRefs).toContain("artifact://applied-broken/post-apply.json");

    const plainSuccess = projection.groups.find((group) =>
      group.dimensions.finalOutcome === "success" && group.dimensions.failureSignals.length === 0
    );
    expect(plainSuccess).toMatchObject({
      attemptCount: 1,
      evidenceRefs: ["artifact://success/diff.patch", "artifact://success/verify.log", "terminal://looks-like-a-shell-failure.log"],
    });
    expect(plainSuccess?.sourceRecords[0]?.editAttemptRecordId).toBe("edit-record.success");
    expect(plainSuccess?.dimensions.failureSignals).not.toContain("verifier_failed");

    const skippedGroup = projection.groups.find((group) =>
      group.dimensions.finalOutcome === "success" && group.dimensions.failureSignals.includes("verifier_skipped")
    );
    expect(skippedGroup).toMatchObject({
      attemptCount: 1,
      verificationStatuses: { skipped: 1 },
    });
  });

  test("normalizes failure signals from edit attempt record fields only", () => {
    const record = createEditAttemptRecord({
      ...baseInput("terminal-noise"),
      targetHashes: [changedHash("terminal-noise")],
      phases: {
        preview: { status: "passed" },
        apply: { status: "passed" },
        write: { status: "passed" },
        verify: { status: "passed" },
      },
      artifactRefs: ["terminal://bun-test-exit-1.log"],
    });

    expect(failureSignalsForEditAttemptRecord(record)).toEqual([]);
  });
});
