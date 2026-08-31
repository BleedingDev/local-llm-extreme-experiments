/// <reference types="node" />
/// <reference path="../types/bun-test.d.ts" />
import { afterEach, beforeEach, describe, expect, test } from "bun:test";
import { mkdirSync, mkdtempSync, rmSync, symlinkSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { env } from "node:process";
import { loadOptimizedExecutorPrompt } from "../src/optimized-prompt-loader";
import { lookupSimilarSituation } from "../src/trace-rag-shim";

const DISABLE_FLAG = "BAG_DISABLE_OPTIMIZED_PROMPT";
const ENABLE_FLAG = "BAG_USE_OPTIMIZED_PROMPT";
const ROOT_OVERRIDE = "BAG_REPO_ROOT";
const TRACE_RAG_ENABLE_FLAG = "BAG_USE_TRACE_RAG";

const setupRoot = (): string => {
  const root = mkdtempSync(join(tmpdir(), "bag-opt-prompt-"));
  return root;
};

const writeArtifact = (root: string, runId: string, payload: unknown): void => {
  const runDir = join(root, "artifacts", "optimized-prompts", runId);
  mkdirSync(runDir, { recursive: true });
  writeFileSync(join(runDir, "best_candidate.json"), JSON.stringify(payload), "utf8");
  const latest = join(root, "artifacts", "optimized-prompts", "latest");
  symlinkSync(runDir, latest, "dir");
};

describe("loadOptimizedExecutorPrompt", () => {
  let prevDisable: string | undefined;
  let prevEnable: string | undefined;
  let prevRoot: string | undefined;
  let prevTraceRagEnable: string | undefined;
  let tempRoots: string[] = [];

  beforeEach(() => {
    prevDisable = env[DISABLE_FLAG];
    prevEnable = env[ENABLE_FLAG];
    prevRoot = env[ROOT_OVERRIDE];
    prevTraceRagEnable = env[TRACE_RAG_ENABLE_FLAG];
    delete env[DISABLE_FLAG];
    delete env[ENABLE_FLAG];
    delete env[ROOT_OVERRIDE];
    delete env[TRACE_RAG_ENABLE_FLAG];
    tempRoots = [];
  });

  afterEach(() => {
    if (prevDisable === undefined) delete env[DISABLE_FLAG];
    else env[DISABLE_FLAG] = prevDisable;
    if (prevEnable === undefined) delete env[ENABLE_FLAG];
    else env[ENABLE_FLAG] = prevEnable;
    if (prevRoot === undefined) delete env[ROOT_OVERRIDE];
    else env[ROOT_OVERRIDE] = prevRoot;
    if (prevTraceRagEnable === undefined) delete env[TRACE_RAG_ENABLE_FLAG];
    else env[TRACE_RAG_ENABLE_FLAG] = prevTraceRagEnable;
    for (const r of tempRoots) {
      try {
        rmSync(r, { recursive: true, force: true });
      } catch {
        // best-effort cleanup
      }
    }
  });

  test("returns null when no artefact exists (missing-file fallthrough)", () => {
    const root = setupRoot();
    tempRoots.push(root);
    env[ENABLE_FLAG] = "1";
    env[ROOT_OVERRIDE] = root;
    expect(loadOptimizedExecutorPrompt()).toBeNull();
  });

  test("returns null by default when artefact exists but opt-in flag is unset", () => {
    const root = setupRoot();
    tempRoots.push(root);
    writeArtifact(root, "20260501T123045Z", {
      system: "Optimized executor system prompt body.",
    });
    env[ROOT_OVERRIDE] = root;
    expect(loadOptimizedExecutorPrompt()).toBeNull();
  });

  test("returns {system, runId} when BAG_USE_OPTIMIZED_PROMPT=1 and artefact exists", () => {
    const root = setupRoot();
    tempRoots.push(root);
    const runId = "20260501T123045Z";
    writeArtifact(root, runId, {
      system: "Optimized executor system prompt body.",
    });
    env[ENABLE_FLAG] = "1";
    env[ROOT_OVERRIDE] = root;
    const out = loadOptimizedExecutorPrompt();
    expect(out).not.toBeNull();
    expect(out?.system).toBe("Optimized executor system prompt body.");
    expect(out?.runId).toBe(runId);
  });

  test("returns null when BAG_DISABLE_OPTIMIZED_PROMPT=1 even if artefact exists", () => {
    const root = setupRoot();
    tempRoots.push(root);
    writeArtifact(root, "20260501T000000Z", { system: "ignored" });
    env[ENABLE_FLAG] = "1";
    env[DISABLE_FLAG] = "1";
    env[ROOT_OVERRIDE] = root;
    expect(loadOptimizedExecutorPrompt()).toBeNull();
  });

  test("returns null when BAG_DISABLE_OPTIMIZED_PROMPT=true even if artefact exists", () => {
    const root = setupRoot();
    tempRoots.push(root);
    writeArtifact(root, "20260501T000000Z", { system: "ignored" });
    env[ENABLE_FLAG] = "1";
    env[DISABLE_FLAG] = "true";
    env[ROOT_OVERRIDE] = root;
    expect(loadOptimizedExecutorPrompt()).toBeNull();
  });

  test("trace-RAG lookup is disabled unless BAG_USE_TRACE_RAG is set", async () => {
    expect(await lookupSimilarSituation("similar failing context")).toEqual([]);
  });
});
