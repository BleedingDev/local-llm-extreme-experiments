import { existsSync, mkdirSync, mkdtempSync, readdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { describe, expect, test } from "bun:test";
import { defaultConfig } from "../src/config";
import {
  gepaRunnerCheckpointsRoot,
  latestGepaRunnerCheckpoint,
  listGepaRunnerCheckpoints,
  loadGepaRunnerCheckpoint,
  loadLatestGepaRunnerCheckpoint,
  saveGepaRunnerCheckpoint,
} from "../src/optimizer/gepa-checkpoints";
import { buildGepaFeedbackBundle, type GepaFeedbackBundle } from "../src/optimizer/gepa-feedback";
import { runGepaOptimizer, type GepaRunnerState } from "../src/optimizer/gepa-runner";

const now = "2026-04-30T00:00:00.000Z";
const later = "2026-04-30T00:05:00.000Z";

const lineage = {
  modelProfileId: "model.qwen36.local",
  codebaseProfileId: "codebase.bleeding-agent",
  policyId: "policy.qwen36.bleeding-agent",
};

const withTempCwd = (fn: (cwd: string) => void): void => {
  const cwd = mkdtempSync(join(tmpdir(), "bag-gepa-checkpoints-"));
  try {
    fn(cwd);
  } finally {
    rmSync(cwd, { recursive: true, force: true });
  }
};

const feedbackBundle = (): GepaFeedbackBundle =>
  buildGepaFeedbackBundle({
    feedbackBundleId: "gepa.bundle.checkpoints",
    testOutputs: [
      {
        id: "typecheck",
        text: "TS2322: bad tool result shape",
        ...lineage,
      },
      {
        id: "unit",
        text: "expected repo write result to include changed file",
        ...lineage,
      },
    ],
  });

const oneIterationState = (): GepaRunnerState =>
  runGepaOptimizer({
    feedbackBundle: feedbackBundle(),
    runId: "gepa.run.checkpoints",
    createdAt: now,
    maxIterations: 1,
    maxFeedbackRecordsPerIteration: 1,
  });

describe("GEPA runner checkpoints", () => {
  test("saves and loads runner state under the GEPA checkpoint namespace", () => {
    withTempCwd((cwd) => {
      const config = defaultConfig();
      const state = oneIterationState();

      const saved = saveGepaRunnerCheckpoint({
        config,
        cwd,
        state,
        savedAt: now,
      });
      const loaded = loadGepaRunnerCheckpoint(saved.path);

      expect(saved.path).toContain(join(".bag", "optimizer", "checkpoints", "gepa-runs"));
      expect(saved.checkpoint).toMatchObject({
        schemaVersion: "gepa-checkpoint.v1",
        checkpointKind: "gepa-runner-state",
        runId: "gepa.run.checkpoints",
        feedbackBundleId: "gepa.bundle.checkpoints",
        savedAt: now,
        iterationCount: 1,
        candidateCount: 1,
        exhausted: false,
      });
      expect(loaded.errors).toEqual([]);
      expect(loaded.state).toEqual(state);
    });
  });

  test("uses stable filenames and atomic-ish writes without leaked temp files", () => {
    withTempCwd((cwd) => {
      const config = defaultConfig();
      const state = oneIterationState();
      const first = saveGepaRunnerCheckpoint({ config, cwd, state, savedAt: now });
      const second = saveGepaRunnerCheckpoint({ config, cwd, state, savedAt: now });

      expect(second.path).toBe(first.path);
      expect(JSON.parse(readFileSync(first.path, "utf8"))).toMatchObject({
        checkpointId: first.checkpoint.checkpointId,
        state: {
          runId: state.runId,
        },
      });
      expect(readdirSync(dirname(first.path)).filter((entry) => entry.includes(".tmp"))).toEqual([]);
    });
  });

  test("lists checkpoints by run id and returns the latest checkpoint deterministically", () => {
    withTempCwd((cwd) => {
      const config = defaultConfig();
      const firstState = oneIterationState();
      const secondState = runGepaOptimizer({
        feedbackBundle: feedbackBundle(),
        initialState: firstState,
        createdAt: later,
        maxIterations: 1,
        maxFeedbackRecordsPerIteration: 1,
      });
      saveGepaRunnerCheckpoint({ config, cwd, state: secondState, savedAt: later });
      saveGepaRunnerCheckpoint({ config, cwd, state: firstState, savedAt: now });

      const listed = listGepaRunnerCheckpoints({ config, cwd, runId: "gepa.run.checkpoints" });
      const latest = latestGepaRunnerCheckpoint({ config, cwd, runId: "gepa.run.checkpoints" });
      const loadedLatest = loadLatestGepaRunnerCheckpoint({ config, cwd, runId: "gepa.run.checkpoints" });

      expect(listed.invalidCheckpoints).toEqual([]);
      expect(listed.checkpoints.map((checkpoint) => checkpoint.savedAt)).toEqual([now, later]);
      expect(latest?.iterationCount).toBe(2);
      expect(loadedLatest?.state).toEqual(secondState);
      expect(listGepaRunnerCheckpoints({ config, cwd, runId: "gepa.run.missing" }).checkpoints).toEqual([]);
    });
  });

  test("reports invalid JSON and schema failures without throwing", () => {
    withTempCwd((cwd) => {
      const config = defaultConfig();
      const root = gepaRunnerCheckpointsRoot(config, cwd);
      const runDir = join(root, "gepa.run.invalid");
      const invalidJson = join(runDir, "2026-04-30T00:00:00.000Z.invalid-json.json");
      const invalidSchema = join(runDir, "2026-04-30T00:01:00.000Z.invalid-schema.json");
      const promotionStyleCheckpoint = join(cwd, ".bag", "optimizer", "checkpoints", "2026-04-30.candidate.json");
      mkdirSync(runDir, { recursive: true });
      mkdirSync(dirname(promotionStyleCheckpoint), { recursive: true });
      writeFileSync(invalidJson, "{ nope", "utf8");
      writeFileSync(invalidSchema, JSON.stringify({ candidatePatchId: "candidate.rollback", createdAt: now }), "utf8");
      writeFileSync(promotionStyleCheckpoint, JSON.stringify({ candidatePatchId: "candidate.rollback", createdAt: now }), "utf8");

      const listed = listGepaRunnerCheckpoints({ config, cwd });

      expect(listed.checkpoints).toEqual([]);
      expect(listed.invalidCheckpoints).toEqual([
        expect.objectContaining({ kind: "parse_error", path: invalidJson }),
        expect.objectContaining({ kind: "validation_error", path: invalidSchema }),
      ]);
      expect(listed.invalidCheckpoints.map((error) => error.path)).not.toContain(promotionStyleCheckpoint);
      expect(loadGepaRunnerCheckpoint(invalidJson).errors[0]).toMatchObject({
        kind: "parse_error",
        path: invalidJson,
      });
    });
  });

  test("hands loaded state back to the runner for resume", () => {
    withTempCwd((cwd) => {
      const config = defaultConfig();
      const firstState = oneIterationState();
      const saved = saveGepaRunnerCheckpoint({ config, cwd, state: firstState, savedAt: now });
      const loaded = loadGepaRunnerCheckpoint(saved.path);

      expect(loaded.state).toBeDefined();
      const resumed = runGepaOptimizer({
        feedbackBundle: feedbackBundle(),
        initialState: loaded.state,
        createdAt: later,
        maxIterations: 1,
        maxFeedbackRecordsPerIteration: 1,
      });

      expect(resumed.iterationCount).toBe(2);
      expect(resumed.processedFeedbackIds).toEqual([
        "gepa.test_output.typecheck",
        "gepa.test_output.unit",
      ]);
      expect(resumed.exhausted).toBe(true);
    });
  });

  test("detects checkpoint state content hash corruption", () => {
    withTempCwd((cwd) => {
      const config = defaultConfig();
      const saved = saveGepaRunnerCheckpoint({
        config,
        cwd,
        state: oneIterationState(),
        savedAt: now,
      });
      const checkpoint = JSON.parse(readFileSync(saved.path, "utf8")) as {
        state: { iterationCount: number };
      };
      checkpoint.state.iterationCount = 99;
      writeFileSync(saved.path, `${JSON.stringify(checkpoint, null, 2)}\n`, "utf8");

      expect(existsSync(saved.path)).toBe(true);
      expect(loadGepaRunnerCheckpoint(saved.path).errors).toEqual([
        expect.objectContaining({
          kind: "validation_error",
          message: expect.stringContaining("stateContentHash mismatch"),
        }),
      ]);
    });
  });
});
