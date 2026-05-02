/// <reference types="node" />
/// <reference path="../types/bun-test.d.ts" />
import { afterEach, beforeEach, describe, expect, test } from "bun:test";
import { mkdirSync, mkdtempSync, rmSync, symlinkSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { env } from "node:process";
import { loadOptimizedExecutorPrompt } from "../src/optimized-prompt-loader";

const DISABLE_FLAG = "BAG_DISABLE_OPTIMIZED_PROMPT";
const ROOT_OVERRIDE = "BAG_REPO_ROOT";

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
  let prevRoot: string | undefined;
  let tempRoots: string[] = [];

  beforeEach(() => {
    prevDisable = env[DISABLE_FLAG];
    prevRoot = env[ROOT_OVERRIDE];
    delete env[DISABLE_FLAG];
    delete env[ROOT_OVERRIDE];
    tempRoots = [];
  });

  afterEach(() => {
    if (prevDisable === undefined) delete env[DISABLE_FLAG];
    else env[DISABLE_FLAG] = prevDisable;
    if (prevRoot === undefined) delete env[ROOT_OVERRIDE];
    else env[ROOT_OVERRIDE] = prevRoot;
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
    env[ROOT_OVERRIDE] = root;
    expect(loadOptimizedExecutorPrompt()).toBeNull();
  });

  test("returns {system, runId} by default when artefact exists (no env set)", () => {
    const root = setupRoot();
    tempRoots.push(root);
    const runId = "20260501T123045Z";
    writeArtifact(root, runId, {
      system: "Optimized executor system prompt body.",
    });
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
    env[DISABLE_FLAG] = "1";
    env[ROOT_OVERRIDE] = root;
    expect(loadOptimizedExecutorPrompt()).toBeNull();
  });

  test("returns null when BAG_DISABLE_OPTIMIZED_PROMPT=true even if artefact exists", () => {
    const root = setupRoot();
    tempRoots.push(root);
    writeArtifact(root, "20260501T000000Z", { system: "ignored" });
    env[DISABLE_FLAG] = "true";
    env[ROOT_OVERRIDE] = root;
    expect(loadOptimizedExecutorPrompt()).toBeNull();
  });
});
