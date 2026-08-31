import { describe, expect, test } from "bun:test";
import {
  buildParallelMergeVerificationPlan,
  buildParallelOrchestrationPlan,
  detectParallelLaneConflicts,
  parallelOrchestrationFeedbackToEvidenceBundle,
  resolveParallelConcurrencyPolicy,
  type ParallelLaneContract,
} from "../src/parallel-orchestration";

const lanes: ParallelLaneContract[] = [
  {
    laneId: "lane.explore",
    title: "Read the target package",
    laneKind: "exploration",
    sideEffectPolicy: "read_only",
    readPaths: ["src"],
  },
  {
    laneId: "lane.impl",
    title: "Patch the runtime contract",
    laneKind: "implementation",
    sideEffectPolicy: "writes_allowed",
    targetPaths: ["src/runtime.ts"],
    dependsOnLaneIds: ["lane.explore"],
  },
  {
    laneId: "lane.verify",
    title: "Run focused verification",
    laneKind: "verification",
    sideEffectPolicy: "terminal_allowed",
    dependsOnLaneIds: ["lane.impl"],
  },
];

describe("parallel orchestration primitives", () => {
  test("builds ACP-visible lane contracts with trace lineage and conservative isolation", () => {
    const plan = buildParallelOrchestrationPlan({
      planId: "parallel.plan.basic",
      lanes,
      parentTraceId: "trace.parent",
      policyId: "policy.qwen36.bleeding-agent",
      modelProfileId: "model.qwen36.local",
      codebaseProfileId: "codebase.bleeding-agent",
      maxLaneConcurrency: 6,
      taskRisk: "medium",
      editConflictRisk: "low",
    });

    expect(plan.conflicts).toEqual([]);
    expect(plan.isolationByLaneId).toMatchObject({
      "lane.explore": "shared_read_only",
      "lane.impl": "patch_queue",
      "lane.verify": "dry_run_apply_layer",
    });
    expect(plan.concurrency.recommendedLaneConcurrency).toBe(3);
    expect(plan.acpProgressLabels).toEqual([
      "exploration:lane.explore:pending",
      "implementation:lane.impl:pending",
      "verification:lane.verify:pending",
    ]);
    expect(plan.traceLineage).toMatchObject({
      parentTraceId: "trace.parent",
      policyId: "policy.qwen36.bleeding-agent",
    });
  });

  test("detects write conflicts and isolates conflicting implementation lanes", () => {
    const conflictingLanes: ParallelLaneContract[] = [
      {
        laneId: "lane.a",
        title: "Edit shared runtime",
        laneKind: "implementation",
        sideEffectPolicy: "writes_allowed",
        targetPaths: ["src/runtime.ts"],
      },
      {
        laneId: "lane.b",
        title: "Edit shared runtime differently",
        laneKind: "implementation",
        sideEffectPolicy: "writes_allowed",
        targetPaths: ["./src/runtime.ts"],
      },
    ];

    const conflicts = detectParallelLaneConflicts(conflictingLanes);
    const plan = buildParallelOrchestrationPlan({
      lanes: conflictingLanes,
      maxLaneConcurrency: 8,
      editConflictRisk: "high",
    });

    expect(conflicts).toHaveLength(1);
    expect(conflicts[0]).toMatchObject({
      laneIds: ["lane.a", "lane.b"],
      path: "src/runtime.ts",
      severity: "blocking",
    });
    expect(plan.isolationByLaneId).toMatchObject({
      "lane.a": "temp_workspace",
      "lane.b": "temp_workspace",
    });
    expect(plan.concurrency.recommendedLaneConcurrency).toBe(1);
  });

  test("ties concurrency to model throughput, policy, task risk, tool failures, and user mode", () => {
    const yolo = resolveParallelConcurrencyPolicy({
      maxLaneConcurrency: 16,
      modelProfile: {
        measuredMaxConcurrentRequests: 10,
        measuredConcurrentThroughputTokensPerSecond: 420,
      },
      modelCodebasePolicy: {
        maxConcurrentEvaluations: 6,
        riskTolerance: "medium",
      },
      taskRisk: "low",
      editConflictRisk: "low",
      toolFailureRate: 0,
      userMode: "yolo",
    });
    const safe = resolveParallelConcurrencyPolicy({
      maxLaneConcurrency: 16,
      modelProfile: {
        measuredMaxConcurrentRequests: 10,
        measuredConcurrentThroughputTokensPerSecond: 420,
      },
      modelCodebasePolicy: {
        maxConcurrentEvaluations: 6,
        riskTolerance: "low",
      },
      taskRisk: "high",
      editConflictRisk: "medium",
      toolFailureRate: 0.3,
      userMode: "safe",
    });

    expect(yolo).toMatchObject({
      hardLaneConcurrencyCap: 6,
      recommendedLaneConcurrency: 6,
    });
    expect(safe.recommendedLaneConcurrency).toBe(1);
    expect(safe.reasons.join("\n")).toContain("safe mode keeps orchestration serial");
    expect(safe.reasons.join("\n")).toContain("recent tool failure rate throttles parallelism");
  });

  test("requires merge verification with codebase commands and rollback on implementation lanes", () => {
    const plan = buildParallelOrchestrationPlan({ lanes, maxLaneConcurrency: 4 });
    const verification = buildParallelMergeVerificationPlan({
      orchestrationPlan: plan,
      codebaseProfile: {
        typecheckCommands: [{ commandId: "typecheck", command: ["npm", "run", "typecheck"], required: true }],
        testCommands: [{ commandId: "test", command: ["npm", "test"], required: true }],
        lintCommands: [],
      },
    });

    expect(verification).toMatchObject({
      verificationRequired: true,
      expectedChangedPaths: ["src/runtime.ts"],
      rollbackRequiredOnFailure: true,
      blockingConflictIds: [],
    });
    expect(verification.commands.map((command) => command.commandId)).toEqual(["typecheck", "test"]);
  });

  test("converts lane outcomes into optimizer evidence without runtime source rewriting", () => {
    const evidence = parallelOrchestrationFeedbackToEvidenceBundle({
      runId: "parallel.run.1",
      createdAt: "2026-05-01T00:00:00.000Z",
      lineage: {
        modelProfileIds: ["model.qwen36.local"],
        codebaseProfileIds: ["codebase.bleeding-agent"],
        policyIds: ["policy.qwen36.bleeding-agent"],
      },
      outcomes: [
        {
          laneId: "lane.impl",
          outcome: "merge_conflict",
          summary: "Two lanes produced incompatible edits to src/runtime.ts.",
          durationMs: 1200,
          tokenCost: 400,
        },
        {
          laneId: "lane.verify",
          outcome: "successful_speedup",
          summary: "Parallel verification finished faster than serial baseline.",
          durationMs: 800,
        },
      ],
    });

    expect(evidence.evidenceBundleId).toBe("parallel-evidence.parallel.run.1");
    expect(evidence.observations.map((observation) => observation.title)).toEqual([
      "Parallel lane merge_conflict: lane.impl",
      "Parallel lane successful_speedup: lane.verify",
    ]);
    expect(evidence.lineage.policyIds).toEqual(["policy.qwen36.bleeding-agent"]);
  });
});
