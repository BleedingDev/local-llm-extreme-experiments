import { describe, expect, test } from "bun:test";
import {
  EDIT_ATTEMPT_RECORD_SCHEMA_VERSION,
  EditAttemptRecordSchema,
  createEditAttemptRecord,
  editAttemptRecordTargetHash,
} from "./edit-attempt-record";

const createdAt = "2026-05-04T12:00:00.000Z";

const baseInput = () => ({
  editAttemptRecordId: "edit-record.test",
  editAttemptId: "edit.attempt.test",
  runId: "run.edit-attempt-record",
  traceId: "trace.edit-attempt-record",
  editStrategyId: "edit.hash-range.experimental.v1",
  renderedEditToolContractId: "rendered.edit.hash-range.test",
  renderedEditContractVersion: "rendered-edit-contract.v1",
  modelProfileId: "model.qwen36.local",
  codebaseProfileId: "codebase.bleeding-agent",
  policyId: "policy.qwen36.bleeding-agent",
  targetPaths: ["src/example.ts"],
  createdAt,
});

describe("edit attempt record", () => {
  test("records a successful edit with strategy profile ids phases and target hashes", () => {
    const record = createEditAttemptRecord({
      ...baseInput(),
      targetHashes: [
        editAttemptRecordTargetHash({
          path: "src/example.ts",
          beforeHash: "sha256:before",
          afterHash: "sha256:after",
        }),
      ],
      phases: {
        preview: { status: "passed" },
        apply: { status: "passed" },
        write: { status: "passed" },
        verify: { status: "passed", artifactRefs: ["artifact://verify/bun-test.log"] },
        repair: { status: "skipped", skipJustification: "repair was not needed" },
        rollback: { status: "skipped", skipJustification: "rollback was not needed" },
      },
      signals: {
        staleContext: { status: "fresh" },
      },
      artifactRefs: ["artifact://edit/diff.patch"],
    });

    expect(record.schemaVersion).toBe(EDIT_ATTEMPT_RECORD_SCHEMA_VERSION);
    expect(record.editStrategyId).toBe("edit.hash-range.experimental.v1");
    expect(record.renderedEditContractVersion).toBe("rendered-edit-contract.v1");
    expect(record.modelProfileId).toBe("model.qwen36.local");
    expect(record.codebaseProfileId).toBe("codebase.bleeding-agent");
    expect(record.policyId).toBe("policy.qwen36.bleeding-agent");
    expect(record.targetPaths).toEqual(["src/example.ts"]);
    expect(record.targetHashes).toEqual([
      {
        path: "src/example.ts",
        beforeHash: "sha256:before",
        afterHash: "sha256:after",
        hashAlgorithm: "sha256",
      },
    ]);
    expect(record.verificationStatus).toBe("passed");
    expect(record.finalOutcome).toBe("success");
  });

  test("records stale context rejection without write output hashes", () => {
    const record = createEditAttemptRecord({
      ...baseInput(),
      targetHashes: [
        editAttemptRecordTargetHash({
          path: "src/example.ts",
          beforeHash: "sha256:before",
        }),
      ],
      phases: {
        preview: { status: "failed", errorCode: "hash_mismatch", message: "expected hash did not match" },
        apply: { status: "failed", errorCode: "hash_mismatch" },
        write: { status: "not_started" },
        verify: { status: "skipped", skipJustification: "preview rejected stale context before verification" },
      },
      signals: {
        staleContext: {
          status: "stale",
          errorCode: "hash_mismatch",
          evidenceRefs: ["artifact://edit/preview.json"],
        },
      },
    });

    expect(record.staleContextStatus).toBe("stale");
    expect(record.phases.write.status).toBe("not_started");
    expect(record.targetHashes[0]).not.toHaveProperty("afterHash");
    expect(record.finalOutcome).toBe("stale_context_rejected");
  });

  test("records applied-but-broken self-detected regression evidence", () => {
    const record = createEditAttemptRecord({
      ...baseInput(),
      editAttemptRecordId: "edit-record.applied-broken",
      targetHashes: [
        editAttemptRecordTargetHash({
          path: "src/example.ts",
          beforeHash: "sha256:before",
          afterHash: "sha256:after",
        }),
      ],
      phases: {
        preview: { status: "passed" },
        apply: { status: "passed" },
        write: { status: "passed" },
        verify: {
          status: "failed",
          errorCode: "verifier_error",
          artifactRefs: ["artifact://verify/typecheck.log"],
        },
        repair: { status: "skipped", skipJustification: "repair was not attempted in this unit case" },
        rollback: { status: "skipped", skipJustification: "rollback was not attempted in this unit case" },
      },
      signals: {
        staleContext: { status: "fresh" },
        syntaxBreakage: {
          detected: true,
          errorCode: "post_apply_syntax_failure",
          evidenceRefs: ["artifact://verify/typecheck.log"],
        },
        appliedButBroken: {
          detected: true,
          status: "inconsistent",
          evidenceRefs: ["artifact://post-apply/syntax.json"],
        },
        selfDetectedRegression: {
          status: "confirmed",
          evidenceRefs: ["artifact://self-check/regression.json"],
        },
      },
    });

    expect(record.signals.appliedButBroken.detected).toBe(true);
    expect(record.signals.selfDetectedRegression.status).toBe("confirmed");
    expect(record.signals.syntaxBreakage.detected).toBe(true);
    expect(record.finalOutcome).toBe("syntax_breakage");
  });

  test("records rollback outcome as the final edit attempt result", () => {
    const record = createEditAttemptRecord({
      ...baseInput(),
      editAttemptRecordId: "edit-record.rollback",
      targetHashes: [
        editAttemptRecordTargetHash({
          path: "src/example.ts",
          beforeHash: "sha256:before",
          afterHash: "sha256:after",
        }),
      ],
      phases: {
        preview: { status: "passed" },
        apply: { status: "passed" },
        write: { status: "passed" },
        verify: { status: "failed", errorCode: "verifier_error" },
        repair: { status: "failed", errorCode: "self_detected_regression" },
        rollback: { status: "passed", artifactRefs: ["artifact://rollback/result.json"] },
      },
      signals: {
        staleContext: { status: "fresh" },
        selfDetectedRegression: {
          status: "confirmed",
          evidenceRefs: ["artifact://verify/bun-test.log"],
        },
      },
    });

    expect(record.repairOutcome).toBe("failed");
    expect(record.rollbackOutcome).toBe("succeeded");
    expect(record.phases.rollback.status).toBe("passed");
    expect(record.finalOutcome).toBe("rolled_back");
  });

  test("requires verifier skipped records to include a justification", () => {
    const skipped = createEditAttemptRecord({
      ...baseInput(),
      editAttemptRecordId: "edit-record.verify-skipped",
      targetHashes: [
        editAttemptRecordTargetHash({
          path: "src/example.ts",
          beforeHash: "sha256:before",
          afterHash: "sha256:after",
        }),
      ],
      phases: {
        preview: { status: "passed" },
        apply: { status: "passed" },
        write: { status: "passed" },
        verify: { status: "skipped", skipJustification: "docs-only change has no verifier command" },
      },
    });
    const missingJustification = EditAttemptRecordSchema.safeParse({
      ...skipped,
      phases: {
        ...skipped.phases,
        verify: { status: "skipped" },
      },
    });

    expect(skipped.verificationStatus).toBe("skipped");
    expect(skipped.phases.verify.skipJustification).toBe("docs-only change has no verifier command");
    expect(skipped.finalOutcome).toBe("success");
    expect(missingJustification.success).toBe(false);
  });

  test("records no-write as an explicit final outcome", () => {
    const record = createEditAttemptRecord({
      ...baseInput(),
      editAttemptRecordId: "edit-record.no-write",
      targetHashes: [
        editAttemptRecordTargetHash({
          path: "src/example.ts",
          beforeHash: "sha256:before",
        }),
      ],
      phases: {
        preview: { status: "skipped", skipJustification: "model produced no edit operations" },
        apply: { status: "skipped", skipJustification: "model produced no edit operations" },
        write: { status: "skipped", skipJustification: "no write was produced" },
        verify: { status: "skipped", skipJustification: "no write was produced" },
      },
      signals: {
        staleContext: { status: "not_checked" },
      },
    });

    expect(record.targetHashes.some((hash) => hash.afterHash !== undefined)).toBe(false);
    expect(record.phases.write.status).toBe("skipped");
    expect(record.finalOutcome).toBe("no_write");
  });
});
