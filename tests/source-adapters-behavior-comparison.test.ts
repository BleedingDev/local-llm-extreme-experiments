import { existsSync } from "node:fs";
import { readFile } from "node:fs/promises";
import { resolve } from "node:path";
import { describe, expect, test } from "bun:test";
import {
  createAdapterBehaviorComparisonScorecards,
  summarizeAdapterBehaviorComparisonScorecards,
  type AdapterBehaviorManifest,
} from "../src/source-adapters/behavior-comparison";
import type { ReplayEvalCaseSkeleton } from "../src/replay";

const replayCase = (overrides: Partial<ReplayEvalCaseSkeleton> = {}): ReplayEvalCaseSkeleton => ({
  evalCaseId: "replay.eval.source-adapter.synthetic",
  schemaVersion: "replay-eval-case.v1",
  split: "dev",
  splitAssignment: {
    split: "dev",
    assignedBy: "manual",
    rationale: "Synthetic behavior-comparison test split.",
  },
  title: "Synthetic observed source-adapter behavior",
  task: "Compare observed external behavior to BAG policy behavior.",
  captureId: "capture.source-adapter.synthetic",
  sourceSessionId: "claude-session-1",
  sourceTraceIds: ["trace-1"],
  sourceRefs: [{
    sourceKind: "capture",
    captureId: "capture.source-adapter.synthetic",
    redactionStatus: "redacted",
  }],
  redaction: {
    status: "redacted",
    needsReview: false,
    needsReviewRecordIds: [],
    recordStatuses: [],
  },
  oracle: {
    strength: "weak",
    expectedBehavior: {
      summary: "Observed baseline evidence, not gold.",
      assertions: [],
      notes: [],
    },
  },
  routing: {
    promptRecordIds: ["record.prompt"],
    routingRecordIds: [],
  },
  observedFailures: [
    {
      failureKind: "tool_call",
      recordId: "record.tool.malformed",
      status: "malformed_args",
      errorCode: "malformed_args",
      artifactRefs: [],
    },
    {
      failureKind: "terminal_command",
      recordId: "record.terminal.permission",
      status: "permission_denied",
      errorCode: "permission_denied",
      artifactRefs: [],
    },
    {
      failureKind: "edit_attempt",
      recordId: "record.edit.non-unique",
      status: "failed",
      phase: "apply",
      errorCode: "non_unique_edit_string",
      artifactRefs: [],
    },
    {
      failureKind: "tool_call",
      recordId: "record.tool.cancelled",
      status: "failed",
      errorCode: "cancellation",
      artifactRefs: [],
    },
  ],
  tags: [
    "replay",
    "source-adapter",
    "observed-baseline",
    "cc-session-jsonl-v2",
    "malformed_args",
    "permission_denied",
    "non_unique_edit_string",
    "cancellation",
  ],
  timeoutMs: 120000,
  ...overrides,
});

describe("source adapter behavior comparison", () => {
  test("labels external traces as observed baselines and queues missing BAG outcomes", () => {
    const scorecards = createAdapterBehaviorComparisonScorecards({
      manifest: {
        exportedSessions: [{
          sourceType: "cc-session-jsonl-v2",
          sourceSessionId: "claude-session-1",
          split: "dev",
          captureId: "capture.source-adapter.synthetic",
          evalCaseId: "replay.eval.source-adapter.synthetic",
        }],
      },
      replayCases: [replayCase()],
      defaultPolicyId: "policy.bleeding-agent.candidate",
    });

    expect(scorecards).toHaveLength(1);
    const scorecard = scorecards[0];
    expect(scorecard?.baseline).toMatchObject({
      role: "observed_baseline",
      gold: false,
      sourceSystem: "claude",
      sourceType: "cc-session-jsonl-v2",
      sessionKind: "cc",
      split: "dev",
    });
    expect(scorecard?.gold).toBe(false);
    expect(scorecard?.comparisonStatus).toBe("needs_bag_run");
    expect(scorecard?.bag).toMatchObject({
      policyId: "policy.bleeding-agent.candidate",
      status: "needs_bag_run",
      outcomeSource: "bleeding_agent_policy",
    });
    expect(scorecard?.dimensions.failureKinds.observed).toEqual([
      "cancellation",
      "malformed_args",
      "non_unique_edit_string",
      "permission_denied",
    ]);
    expect(scorecard?.dimensions.toolFailures.observed).toBe(2);
    expect(scorecard?.dimensions.terminalFailures.observed).toBe(1);
    expect(scorecard?.dimensions.editFailures.observed).toBe(1);
    expect(scorecard?.dimensions.permissionEvents.observed).toBe(1);
    expect(scorecard?.dimensions.cancellationEvents.observed).toBe(1);
    expect(scorecard?.dimensions.toolFailures.status).toBe("needs_bag_run");
    expect(scorecard?.alignment).toEqual({
      comparableDimensionCount: 0,
      matchedDimensionCount: 0,
    });
  });

  test("compares BAG policy outcomes without converting observed behavior into gold pass/fail", () => {
    const [scorecard] = createAdapterBehaviorComparisonScorecards({
      manifest: {
        exportedSessions: [{
          sourceType: "codex-session-jsonl",
          sourceSessionId: "codex-session-1",
          split: "holdout",
          captureId: "capture.source-adapter.synthetic",
          evalCaseId: "replay.eval.source-adapter.synthetic",
        }],
      },
      replayCases: [replayCase({
        split: "holdout",
        splitAssignment: {
          split: "holdout",
          assignedBy: "manual",
          rationale: "Synthetic hidden split.",
        },
      })],
      bagOutcomes: [{
        evalCaseId: "replay.eval.source-adapter.synthetic",
        captureId: "capture.source-adapter.synthetic",
        policyId: "policy.bleeding-agent.candidate",
        status: "failed",
        failureKinds: ["cancellation", "malformed_args", "non_unique_edit_string", "permission_denied"],
        failures: [
          { failureKind: "edit_attempt", errorCode: "non_unique_edit_string", status: "failed" },
          { failureKind: "terminal_command", errorCode: "permission_denied", status: "permission_denied" },
          { failureKind: "tool_call", errorCode: "malformed_args", status: "malformed_args" },
          { failureKind: "tool_call", errorCode: "cancellation", status: "failed" },
        ],
        toolFailures: 2,
        terminalFailures: 0,
        editFailures: 1,
        fileReadFailures: 0,
        permissionEvents: 1,
        cancellationEvents: 1,
        sourceType: "codex-session-jsonl",
        sessionKind: "codex",
        split: "holdout",
      }],
    });

    expect(scorecard?.baseline).toMatchObject({
      role: "observed_baseline",
      gold: false,
      sourceSystem: "codex",
    });
    expect(scorecard?.comparisonStatus).toBe("compared");
    expect(scorecard?.bag.status).toBe("failed");
    expect(scorecard?.dimensions.failureKinds.status).toBe("matches_observed");
    expect(scorecard?.dimensions.terminalFailures.status).toBe("differs_from_observed");
    expect(scorecard?.dimensions.policyOutcome).toEqual({
      observed: "observed_baseline",
      bag: "failed",
      status: "observed_only",
    });
    expect(scorecard?.gold).toBe(false);
    expect(scorecard?.alignment.comparableDimensionCount).toBeGreaterThan(0);
    expect(scorecard?.alignment.score).toBeLessThan(1);
  });

  test("summarizes deterministic scorecards by split, source, and comparison status", () => {
    const scorecards = createAdapterBehaviorComparisonScorecards({
      manifest: {
        exportedSessions: [
          {
            sourceType: "cc-session-jsonl-v2",
            sourceSessionId: "claude-session-1",
            split: "dev",
            captureId: "capture.source-adapter.b",
            evalCaseId: "replay.eval.source-adapter.b",
          },
          {
            sourceType: "codex-session-jsonl",
            sourceSessionId: "codex-session-1",
            split: "train",
            captureId: "capture.source-adapter.a",
            evalCaseId: "replay.eval.source-adapter.a",
          },
        ],
      },
      replayCases: [
        replayCase({
          evalCaseId: "replay.eval.source-adapter.b",
          captureId: "capture.source-adapter.b",
          split: "dev",
        }),
        replayCase({
          evalCaseId: "replay.eval.source-adapter.a",
          captureId: "capture.source-adapter.a",
          split: "train",
          splitAssignment: {
            split: "train",
            assignedBy: "manual",
            rationale: "Synthetic train split.",
          },
        }),
      ],
    });

    expect(scorecards.map((scorecard) => scorecard.evalCaseId)).toEqual([
      "replay.eval.source-adapter.a",
      "replay.eval.source-adapter.b",
    ]);
    expect(createAdapterBehaviorComparisonScorecards({
      replayCases: scorecards.map((scorecard) => replayCase({
        evalCaseId: scorecard.evalCaseId,
        captureId: scorecard.captureId,
        split: scorecard.split,
      })),
    }).map((scorecard) => scorecard.evalCaseId)).toEqual([
      "replay.eval.source-adapter.a",
      "replay.eval.source-adapter.b",
    ]);

    const summary = summarizeAdapterBehaviorComparisonScorecards(scorecards);
    expect(summary).toMatchObject({
      scorecardCount: 2,
      needsBagRunCount: 2,
      comparedCount: 0,
      bySplit: {
        dev: 1,
        holdout: 0,
        train: 1,
      },
      bySourceKind: {
        "cc-session-jsonl-v2": 1,
        "codex-session-jsonl": 1,
      },
      byObservedSystem: {
        bag: 0,
        claude: 1,
        codex: 1,
        unknown: 0,
      },
    });
  });
});

const exportRoot = resolve(".bag/replay-corpus/source-adapters/adapter-replay-export");
const manifestPath = resolve(exportRoot, "manifest.json");
const testWithLocalExport = existsSync(manifestPath) ? test : test.skip;

testWithLocalExport("builds deterministic needs_bag_run scorecards for the local 50-session export", async () => {
  const manifest = JSON.parse(await readFile(manifestPath, "utf8")) as AdapterBehaviorManifest;
  const replayCases = await Promise.all(
    manifest.exportedSessions.map(async (session) =>
      JSON.parse(await readFile(resolve(session.replayCasePath ?? ""), "utf8")) as ReplayEvalCaseSkeleton),
  );

  const first = createAdapterBehaviorComparisonScorecards({
    manifest,
    replayCases,
    defaultPolicyId: "policy.bleeding-agent.pending",
  });
  const second = createAdapterBehaviorComparisonScorecards({
    manifest,
    replayCases,
    defaultPolicyId: "policy.bleeding-agent.pending",
  });
  const summary = summarizeAdapterBehaviorComparisonScorecards(first);

  expect(JSON.stringify(first)).toBe(JSON.stringify(second));
  expect(first).toHaveLength(manifest.exportedSessions.length);
  expect(first.every((scorecard) =>
    scorecard.baseline.role === "observed_baseline" &&
    scorecard.baseline.gold === false &&
    scorecard.gold === false &&
    scorecard.comparisonStatus === "needs_bag_run" &&
    scorecard.bag.status === "needs_bag_run"
  )).toBe(true);
  expect(summary.scorecardCount).toBe(manifest.exportedSessions.length);
  expect(summary.needsBagRunCount).toBe(manifest.exportedSessions.length);
  expect(summary.bySplit).toEqual({
    dev: 10,
    holdout: 10,
    train: 30,
  });
});
