import { describe, expect, test } from "bun:test";
import { mkdtempSync, mkdirSync, rmSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";
import { defaultConfig } from "../src/config";
import {
  loadActiveOptimizerPointer,
  loadOptimizerRegistry,
  optimizerRegistryRecordsDir,
  saveActiveOptimizerPointer,
  saveOptimizerRegistryRecord,
  seedOptimizerRegistry,
} from "../src/optimizer/registry";

const withTempCwd = (fn: (cwd: string) => void): void => {
  const cwd = mkdtempSync(join(tmpdir(), "bag-registry-"));
  try {
    fn(cwd);
  } finally {
    rmSync(cwd, { recursive: true, force: true });
  }
};

describe("optimizer registry", () => {
  test("missing optimizer artifacts return conservative seed records", () => {
    withTempCwd((cwd) => {
      const config = defaultConfig();
      const result = loadOptimizerRegistry(config, cwd);
      const modelProfiles = result.records.filter((record) => record.recordKind === "model_profile");
      const policies = result.records.filter((record) => record.recordKind === "model_codebase_policy");

      expect(result.invalidRecords).toEqual([]);
      expect(result.persistedRecords).toEqual([]);
      expect(result.records.length).toBeGreaterThanOrEqual(3);
      expect(result.records.some((record) => record.recordKind === "model_profile")).toBe(true);
      expect(result.records.some((record) => record.recordKind === "codebase_profile")).toBe(true);
      expect(result.records.some((record) => record.recordKind === "model_codebase_policy")).toBe(true);
      expect(modelProfiles.map((record) => record.payload.modelRole).sort()).toEqual([
        "critic",
        "executor",
        "fast_scout",
        "local",
        "local_batch_executor",
        "master",
        "planner",
        "summarizer",
        "verifier",
      ]);
      expect(policies).toHaveLength(modelProfiles.length);
      expect(policies.every((record) => record.payload.codebaseRootFingerprint?.startsWith("sha256:"))).toBe(true);
      expect(result.activePointer).toBeUndefined();
    });
  });

  test("seeds distinct model profiles for roles sharing the same provider model", () => {
    withTempCwd((cwd) => {
      const config = defaultConfig();
      const records = seedOptimizerRegistry(config, cwd);
      const planner = records.find((record) => record.recordKind === "model_profile" && record.payload.modelRole === "planner");
      const critic = records.find((record) => record.recordKind === "model_profile" && record.payload.modelRole === "critic");
      const executor = records.find((record) => record.recordKind === "model_profile" && record.payload.modelRole === "executor");
      const batchExecutor = records.find((record) =>
        record.recordKind === "model_profile" && record.payload.modelRole === "local_batch_executor"
      );

      expect(planner?.recordKind).toBe("model_profile");
      expect(critic?.recordKind).toBe("model_profile");
      expect(executor?.recordKind).toBe("model_profile");
      expect(batchExecutor?.recordKind).toBe("model_profile");
      expect(planner!.payload.model).toBe(config.master.model);
      expect(critic!.payload.model).toBe(config.master.model);
      expect(planner!.payload.modelProfileId).not.toBe(critic!.payload.modelProfileId);
      expect(planner!.payload.providerConfigRole).toBe("master");
      expect(planner!.payload.fallbackModelRole).toBe("local");
      expect(planner!.payload.baseUrl).toBe(config.master.baseUrl);
      expect(planner!.payload.endpointKind).toBe("chat_completions");
      expect(planner!.payload.modelServerId).toMatch(/^server\.master\.openai\.[a-f0-9]{12}$/);
      expect(planner!.payload.modelServerProfileId).toMatch(/^server-profile\.master\.[a-f0-9]{12}$/);
      expect(planner!.payload.contextWindowTokens).toBe(Math.max(config.master.maxTokens, 8192));
      expect(planner!.payload.contextWindowSource).toBe("deterministic_floor");
      expect(planner!.payload.maxOutputTokens).toBe(config.master.maxTokens);
      expect(critic!.payload.modelServerProfileId).toBe(planner!.payload.modelServerProfileId);
      expect(executor!.payload.providerConfigRole).toBe("local");
      expect(executor!.payload.fallbackModelRole).toBe("master");
      expect(batchExecutor!.payload.providerConfigRole).toBe("local");
      expect(batchExecutor!.payload.fallbackModelRole).toBe("executor");
      expect(batchExecutor!.payload.modelServerProfileId).toBe(executor!.payload.modelServerProfileId);
    });
  });

  test("saves and loads a registry record roundtrip", () => {
    withTempCwd((cwd) => {
      const config = defaultConfig();
      const seedRecord = seedOptimizerRegistry(config, cwd)[0];
      expect(seedRecord).toBeDefined();

      const saved = saveOptimizerRegistryRecord(config, seedRecord!, cwd);
      const result = loadOptimizerRegistry(config, cwd);
      const loaded = result.persistedRecords.find((record) => record.registryRecordId === saved.registryRecordId);

      expect(loaded).toEqual(saved);
      expect(saved.contentHash).toStartWith("sha256:");
      expect(result.invalidRecords).toEqual([]);
    });
  });

  test("invalid JSON and invalid records are reported without preventing defaults", () => {
    withTempCwd((cwd) => {
      const config = defaultConfig();
      const recordsDir = optimizerRegistryRecordsDir(config, cwd);
      mkdirSync(join(recordsDir, "model_profile"), { recursive: true });
      writeFileSync(join(recordsDir, "model_profile", "broken.json"), "{");
      writeFileSync(
        join(recordsDir, "model_profile", "invalid.json"),
        `${JSON.stringify({ registryRecordId: "registry.invalid", recordKind: "model_profile" })}\n`,
      );

      const result = loadOptimizerRegistry(config, cwd);

      expect(result.invalidRecords).toHaveLength(2);
      expect(result.invalidRecords.map((entry) => entry.kind).sort()).toEqual(["parse_error", "validation_error"]);
      expect(result.records.some((record) => record.recordKind === "model_profile")).toBe(true);
      expect(result.records.some((record) => record.recordKind === "codebase_profile")).toBe(true);
      expect(result.errors).toHaveLength(2);
    });
  });

  test("saves and loads active pointer roundtrip", () => {
    withTempCwd((cwd) => {
      const config = defaultConfig();
      const records = seedOptimizerRegistry(config, cwd);
      const model = records.find((record) => record.recordKind === "model_profile");
      const codebase = records.find((record) => record.recordKind === "codebase_profile");
      const policy = records.find((record) => record.recordKind === "model_codebase_policy");

      expect(model?.recordKind).toBe("model_profile");
      expect(codebase?.recordKind).toBe("codebase_profile");
      expect(policy?.recordKind).toBe("model_codebase_policy");

      const saved = saveActiveOptimizerPointer(
        config,
        {
          schemaVersion: "optimizer-active.v1",
          activeModelProfileId: model!.payload.modelProfileId,
          activeCodebaseProfileId: codebase!.payload.codebaseProfileId,
          activeCodebaseRootFingerprint: codebase!.payload.rootFingerprint,
          activePolicyId: policy!.payload.policyId,
          promotedAt: "2026-04-30T00:00:00.000Z",
        },
        cwd,
      );
      const loaded = loadActiveOptimizerPointer(config, cwd);

      expect(loaded.errors).toEqual([]);
      expect(loaded.pointer).toEqual(saved);
      expect(loaded.pointer?.activeCodebaseRootFingerprint).toBe(codebase!.payload.rootFingerprint);
      expect(loaded.pointer?.contentHash).toStartWith("sha256:");
    });
  });
});
