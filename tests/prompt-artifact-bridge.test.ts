/// <reference types="node" />
/// <reference path="../types/bun-test.d.ts" />
import { afterEach, beforeEach, describe, expect, test } from "bun:test";
import { mkdtempSync, readFileSync, rmSync, lstatSync, readlinkSync } from "node:fs";
import { tmpdir } from "node:os";
import { basename, join } from "node:path";
import { env } from "node:process";

import { materializePromotedPromptArtifact } from "../src/optimizer/prompt-artifact-bridge";
import { loadOptimizedExecutorPrompt } from "../src/optimized-prompt-loader";
import type { CandidatePromotionResult } from "../src/optimizer/promotion";
import type { CandidatePatch } from "../src/optimizer/types";

const ENV_FLAG = "BAG_USE_OPTIMIZED_PROMPT";
const ROOT_OVERRIDE = "BAG_REPO_ROOT";

const fakePromotion = (candidatePatchId: string): CandidatePromotionResult => ({
  promoted: true,
  candidatePatchId,
  decision: {
    promotionDecisionId: "promotion.test.smoke.run.0001",
    decision: "promote",
    policyId: "policy.test.smoke",
    candidatePatchId,
    modelProfileId: "model.test",
    codebaseProfileId: "codebase.test",
    canonicalToolVersion: "v1",
    renderedToolVersion: "v1",
    resultStyleVersion: "v1",
    verificationPolicyVersion: "v1",
    reason: "test promotion for prompt artifact bridge",
    decidedAt: new Date().toISOString(),
    decidedBy: "deterministic_gate",
    appliesToNewSessionsOnly: true,
  },
  registryRecordIds: [],
});

const fakeCandidate = (candidatePatchId: string): CandidatePatch => ({
  candidatePatchId,
  policyId: "policy.test.smoke",
  modelProfileId: "model.test",
  codebaseProfileId: "codebase.test",
  scope: {
    artifactKind: "rendered_tool_contract",
    artifactId: "prompt.autonomous-coding-turn.system",
    allowedJsonPointers: ["/promptFragments/0"],
  },
  operations: [
    {
      op: "replace",
      path: "/promptFragments/0",
      value: "Promoted prompt text from bridge test.",
    },
  ],
  rationale: "Test rationale for the bridge smoke test.",
  createdAt: new Date().toISOString(),
  sourceTraceIds: [],
});

describe("materializePromotedPromptArtifact", () => {
  let tempRoot: string | null = null;
  let prevFlag: string | undefined;
  let prevRoot: string | undefined;

  beforeEach(() => {
    tempRoot = mkdtempSync(join(tmpdir(), "bag-prompt-bridge-"));
    prevFlag = env[ENV_FLAG];
    prevRoot = env[ROOT_OVERRIDE];
  });

  afterEach(() => {
    if (tempRoot != null) {
      try {
        rmSync(tempRoot, { recursive: true, force: true });
      } catch {
        // best-effort
      }
      tempRoot = null;
    }
    if (prevFlag === undefined) delete env[ENV_FLAG];
    else env[ENV_FLAG] = prevFlag;
    if (prevRoot === undefined) delete env[ROOT_OVERRIDE];
    else env[ROOT_OVERRIDE] = prevRoot;
  });

  test("writes best_candidate.json and rotates the latest symlink", () => {
    if (tempRoot == null) throw new Error("tempRoot unset");
    const promotion = fakePromotion("candidate.test.smoke.0001");
    const candidate = fakeCandidate("candidate.test.smoke.0001");
    const resolved = "Promoted prompt text from bridge test.";

    const result = materializePromotedPromptArtifact({
      promotion,
      candidate,
      resolvedPromptText: resolved,
      cwd: tempRoot,
    });

    const raw = readFileSync(result.artifactPath, "utf8");
    const parsed = JSON.parse(raw) as Record<string, unknown>;
    expect(typeof parsed.system).toBe("string");
    expect(parsed.system).toBe(resolved);
    expect(parsed.runId).toBe(result.runId);
    expect(parsed.candidatePatchId).toBe("candidate.test.smoke.0001");

    const linkStat = lstatSync(result.latestSymlink);
    expect(linkStat.isSymbolicLink()).toBe(true);
    const target = readlinkSync(result.latestSymlink);
    expect(basename(target.replace(/\/+$/, ""))).toBe(result.runId);

    env[ENV_FLAG] = "1";
    env[ROOT_OVERRIDE] = tempRoot;
    const loaded = loadOptimizedExecutorPrompt();
    expect(loaded).not.toBeNull();
    expect(loaded?.system).toBe(resolved);
    expect(loaded?.runId).toBe(result.runId);
  });

  test("re-rotates latest symlink on second materialize call", () => {
    if (tempRoot == null) throw new Error("tempRoot unset");
    const first = materializePromotedPromptArtifact({
      promotion: fakePromotion("candidate.first"),
      candidate: fakeCandidate("candidate.first"),
      resolvedPromptText: "first",
      cwd: tempRoot,
      runId: "run-aaaa",
    });
    const second = materializePromotedPromptArtifact({
      promotion: fakePromotion("candidate.second"),
      candidate: fakeCandidate("candidate.second"),
      resolvedPromptText: "second",
      cwd: tempRoot,
      runId: "run-bbbb",
    });
    const target = readlinkSync(second.latestSymlink);
    expect(basename(target.replace(/\/+$/, ""))).toBe("run-bbbb");
    // First artifact should still exist on disk.
    const firstRaw = readFileSync(first.artifactPath, "utf8");
    expect(JSON.parse(firstRaw).system).toBe("first");
  });
});
