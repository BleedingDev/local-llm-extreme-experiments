import { describe, expect, test } from "bun:test";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { defaultConfig } from "../src/config";
import {
  loadOptimizerRegistry,
  saveActiveOptimizerPointer,
  saveOptimizerRegistryRecord,
  seedOptimizerRegistry,
} from "../src/optimizer/registry";
import { resolveLoadedOptimizerPolicy } from "../src/optimizer/policy-resolver";
import type { OptimizerRegistryRecord } from "../src/optimizer/types";

const withTempCwd = (fn: (cwd: string) => void): void => {
  const cwd = mkdtempSync(join(tmpdir(), "bag-policy-resolver-"));
  try {
    fn(cwd);
  } finally {
    rmSync(cwd, { recursive: true, force: true });
  }
};

const firstRecord = <Kind extends OptimizerRegistryRecord["recordKind"]>(
  records: OptimizerRegistryRecord[],
  recordKind: Kind,
  predicate: (record: Extract<OptimizerRegistryRecord, { recordKind: Kind }>) => boolean = () => true,
): Extract<OptimizerRegistryRecord, { recordKind: Kind }> => {
  const record = records.find((entry): entry is Extract<OptimizerRegistryRecord, { recordKind: Kind }> =>
    entry.recordKind === recordKind && predicate(entry as Extract<OptimizerRegistryRecord, { recordKind: Kind }>)
  );
  if (record === undefined) {
    throw new Error(`missing ${recordKind} test fixture`);
  }
  return record;
};

describe("optimizer policy resolver", () => {
  test("resolves the default local seed policy", () => {
    withTempCwd((cwd) => {
      const config = defaultConfig();
      const registry = loadOptimizerRegistry(config, cwd);
      const resolved = resolveLoadedOptimizerPolicy(registry, { modelRole: "local" });

      expect(resolved.source).toBe("seed");
      expect(resolved.modelProfile.model).toBe(config.local.model);
      expect(resolved.provider).toBe(config.local.provider);
      expect(resolved.endpointKind).toBe("chat_completions");
      expect(resolved.modelServerId).toBe(resolved.modelProfile.modelServerId);
      expect(resolved.modelServerProfileId).toBe(resolved.modelProfile.modelServerProfileId);
      expect(resolved.contextWindowTokens).toBe(resolved.modelProfile.contextWindowTokens);
      expect(resolved.maxOutputTokens).toBe(config.local.maxTokens);
      expect(resolved.modelProfileId).toBe(resolved.policy.modelProfileId);
      expect(resolved.codebaseProfileId).toBe(resolved.policy.codebaseProfileId);
      expect(resolved.canonicalToolVersion).toBe("canonical-tools.v1");
      expect(resolved.renderedToolVersion).toBe("rendered-tools.v1");
      expect(resolved.resultStyleVersion).toBe("result-style.v1");
      expect(resolved.verificationPolicyVersion).toBe("verification.v1");
      expect(resolved.editStrategyVersion).toBe("edit-strategy.v1");
      expect(resolved.renderedEditContractVersion).toBe("rendered-edit-contract.v1");
      expect(resolved.editFallbackPolicyVersion).toBe("edit-fallback.v1");
      expect(resolved.editRepairPolicyVersion).toBe("edit-repair.v1");
      expect(resolved.editVerifierPolicyVersion).toBe("edit-verifier.v1");
      expect(resolved.editObjectiveSetId).toBe("edit-objectives.default.v1");
    });
  });

  test("resolves the master seed policy", () => {
    withTempCwd((cwd) => {
      const config = defaultConfig();
      const registry = loadOptimizerRegistry(config, cwd);
      const resolved = resolveLoadedOptimizerPolicy(registry, { modelRole: "master" });

      expect(resolved.source).toBe("seed");
      expect(resolved.modelProfile.model).toBe(config.master.model);
      expect(resolved.modelProfileId).toBe(resolved.policy.modelProfileId);
      expect(resolved.codebaseProfileId).toBe(resolved.policy.codebaseProfileId);
    });
  });

  test("resolves role-specific seed policies without collapsing same-model roles", () => {
    withTempCwd((cwd) => {
      const config = defaultConfig();
      const registry = loadOptimizerRegistry(config, cwd);
      const planner = resolveLoadedOptimizerPolicy(registry, { modelRole: "planner" });
      const critic = resolveLoadedOptimizerPolicy(registry, { modelRole: "critic" });
      const executor = resolveLoadedOptimizerPolicy(registry, { modelRole: "executor" });
      const batchExecutor = resolveLoadedOptimizerPolicy(registry, { modelRole: "local_batch_executor" });

      expect(planner.source).toBe("seed");
      expect(planner.modelRole).toBe("planner");
      expect(planner.providerConfigRole).toBe("master");
      expect(planner.fallbackModelRole).toBe("local");
      expect(planner.modelProfile.model).toBe(config.master.model);
      expect(planner.modelServerProfileId).toBe(planner.modelProfile.modelServerProfileId);
      expect(critic.modelRole).toBe("critic");
      expect(critic.providerConfigRole).toBe("master");
      expect(critic.modelProfile.model).toBe(config.master.model);
      expect(critic.modelProfileId).not.toBe(planner.modelProfileId);
      expect(critic.policyId).not.toBe(planner.policyId);
      expect(critic.modelServerProfileId).toBe(planner.modelServerProfileId);
      expect(executor.modelRole).toBe("executor");
      expect(executor.providerConfigRole).toBe("local");
      expect(executor.fallbackModelRole).toBe("master");
      expect(batchExecutor.modelRole).toBe("local_batch_executor");
      expect(batchExecutor.providerConfigRole).toBe("local");
      expect(batchExecutor.fallbackModelRole).toBe("executor");
      expect(batchExecutor.modelProfileId).not.toBe(executor.modelProfileId);
      expect(batchExecutor.modelServerProfileId).toBe(executor.modelServerProfileId);
    });
  });

  test("keeps an existing role target stable while a different role becomes active for new resolutions", () => {
    withTempCwd((cwd) => {
      const config = defaultConfig();
      const registryBeforePointer = loadOptimizerRegistry(config, cwd);
      const plannerPin = resolveLoadedOptimizerPolicy(registryBeforePointer, { modelRole: "planner" });
      const seedRecords = seedOptimizerRegistry(config, cwd);
      const criticModel = firstRecord(seedRecords, "model_profile", (record) => record.payload.modelRole === "critic");
      const codebase = firstRecord(seedRecords, "codebase_profile");
      const criticPolicy = firstRecord(seedRecords, "model_codebase_policy", (record) =>
        record.payload.modelProfileId === criticModel.payload.modelProfileId
      );

      saveActiveOptimizerPointer(
        config,
        {
          schemaVersion: "optimizer-active.v1",
          activeModelProfileId: criticModel.payload.modelProfileId,
          activeCodebaseProfileId: codebase.payload.codebaseProfileId,
          activePolicyId: criticPolicy.payload.policyId,
          promotedAt: "2026-04-30T00:00:00.000Z",
        },
        cwd,
      );

      const criticNewSession = resolveLoadedOptimizerPolicy(loadOptimizerRegistry(config, cwd), { modelRole: "critic" });
      const replayTarget = {
        policyId: plannerPin.policyId,
        modelProfileId: plannerPin.modelProfileId,
        codebaseProfileId: plannerPin.codebaseProfileId,
        modelServerId: plannerPin.modelServerId,
        modelServerProfileId: plannerPin.modelServerProfileId,
        canonicalToolVersion: plannerPin.canonicalToolVersion,
        renderedToolVersion: plannerPin.renderedToolVersion,
        resultStyleVersion: plannerPin.resultStyleVersion,
        verificationPolicyVersion: plannerPin.verificationPolicyVersion,
      };

      expect(plannerPin.modelRole).toBe("planner");
      expect(plannerPin.source).toBe("seed");
      expect(criticNewSession.modelRole).toBe("critic");
      expect(criticNewSession.source).toBe("active_pointer");
      expect(criticNewSession.modelProfileId).not.toBe(plannerPin.modelProfileId);
      expect(criticNewSession.policyId).not.toBe(plannerPin.policyId);
      expect(replayTarget).toEqual({
        policyId: plannerPin.policyId,
        modelProfileId: plannerPin.modelProfileId,
        codebaseProfileId: plannerPin.codebaseProfileId,
        modelServerId: plannerPin.modelServerId,
        modelServerProfileId: plannerPin.modelServerProfileId,
        canonicalToolVersion: "canonical-tools.v1",
        renderedToolVersion: "rendered-tools.v1",
        resultStyleVersion: "result-style.v1",
        verificationPolicyVersion: "verification.v1",
      });
    });
  });

  test("uses an active pointer exact match", () => {
    withTempCwd((cwd) => {
      const config = defaultConfig();
      const seedRecords = seedOptimizerRegistry(config, cwd);
      const model = firstRecord(seedRecords, "model_profile", (record) => record.labels.includes("local"));
      const codebase = firstRecord(seedRecords, "codebase_profile");
      const policy = firstRecord(seedRecords, "model_codebase_policy", (record) =>
        record.payload.modelProfileId === model.payload.modelProfileId &&
        record.payload.codebaseProfileId === codebase.payload.codebaseProfileId
      );

      saveActiveOptimizerPointer(
        config,
        {
          schemaVersion: "optimizer-active.v1",
          activeModelProfileId: model.payload.modelProfileId,
          activeCodebaseProfileId: codebase.payload.codebaseProfileId,
          activePolicyId: policy.payload.policyId,
          promotedAt: "2026-04-30T00:00:00.000Z",
        },
        cwd,
      );

      const resolved = resolveLoadedOptimizerPolicy(loadOptimizerRegistry(config, cwd), { modelRole: "local" });

      expect(resolved.source).toBe("active_pointer");
      expect(resolved.modelProfileId).toBe(model.payload.modelProfileId);
      expect(resolved.codebaseProfileId).toBe(codebase.payload.codebaseProfileId);
      expect(resolved.codebaseRootFingerprint).toBe(codebase.payload.rootFingerprint);
      expect(resolved.policyId).toBe(policy.payload.policyId);
    });
  });

  test("fails closed when an active pointer fingerprint does not match the codebase profile", () => {
    withTempCwd((cwd) => {
      const config = defaultConfig();
      const seedRecords = seedOptimizerRegistry(config, cwd);
      const model = firstRecord(seedRecords, "model_profile", (record) => record.labels.includes("local"));
      const codebase = firstRecord(seedRecords, "codebase_profile");
      const policy = firstRecord(seedRecords, "model_codebase_policy", (record) =>
        record.payload.modelProfileId === model.payload.modelProfileId &&
        record.payload.codebaseProfileId === codebase.payload.codebaseProfileId
      );

      saveActiveOptimizerPointer(
        config,
        {
          schemaVersion: "optimizer-active.v1",
          activeModelProfileId: model.payload.modelProfileId,
          activeCodebaseProfileId: codebase.payload.codebaseProfileId,
          activeCodebaseRootFingerprint: "sha256:stale",
          activePolicyId: policy.payload.policyId,
          promotedAt: "2026-04-30T00:00:00.000Z",
        },
        cwd,
      );

      const resolved = resolveLoadedOptimizerPolicy(loadOptimizerRegistry(config, cwd), { modelRole: "local" });

      expect(resolved.source).toBe("seed");
      expect(resolved.policyId).toBe(policy.payload.policyId);
      expect(resolved.codebaseRootFingerprint).toBe(codebase.payload.rootFingerprint);
    });
  });

  test("falls back when an active pointer is mismatched", () => {
    withTempCwd((cwd) => {
      const config = defaultConfig();
      const seedRecords = seedOptimizerRegistry(config, cwd);
      const localModel = firstRecord(seedRecords, "model_profile", (record) => record.labels.includes("local"));
      const masterModel = firstRecord(seedRecords, "model_profile", (record) => record.labels.includes("master"));
      const codebase = firstRecord(seedRecords, "codebase_profile");
      const masterPolicy = firstRecord(seedRecords, "model_codebase_policy", (record) =>
        record.payload.modelProfileId === masterModel.payload.modelProfileId
      );

      saveActiveOptimizerPointer(
        config,
        {
          schemaVersion: "optimizer-active.v1",
          activeModelProfileId: localModel.payload.modelProfileId,
          activeCodebaseProfileId: codebase.payload.codebaseProfileId,
          activePolicyId: masterPolicy.payload.policyId,
          promotedAt: "2026-04-30T00:00:00.000Z",
        },
        cwd,
      );

      const resolved = resolveLoadedOptimizerPolicy(loadOptimizerRegistry(config, cwd), { modelRole: "local" });

      expect(resolved.source).toBe("seed");
      expect(resolved.modelProfileId).toBe(localModel.payload.modelProfileId);
      expect(resolved.policy.modelProfileId).toBe(localModel.payload.modelProfileId);
      expect(resolved.policyId).not.toBe(masterPolicy.payload.policyId);
    });
  });

  test("does not reuse an active pointer across different roles with the same model", () => {
    withTempCwd((cwd) => {
      const config = defaultConfig();
      const seedRecords = seedOptimizerRegistry(config, cwd);
      const plannerModel = firstRecord(seedRecords, "model_profile", (record) => record.payload.modelRole === "planner");
      const criticModel = firstRecord(seedRecords, "model_profile", (record) => record.payload.modelRole === "critic");
      const codebase = firstRecord(seedRecords, "codebase_profile");
      const criticPolicy = firstRecord(seedRecords, "model_codebase_policy", (record) =>
        record.payload.modelProfileId === criticModel.payload.modelProfileId
      );

      expect(plannerModel.payload.model).toBe(criticModel.payload.model);

      saveActiveOptimizerPointer(
        config,
        {
          schemaVersion: "optimizer-active.v1",
          activeModelProfileId: criticModel.payload.modelProfileId,
          activeCodebaseProfileId: codebase.payload.codebaseProfileId,
          activePolicyId: criticPolicy.payload.policyId,
          promotedAt: "2026-04-30T00:00:00.000Z",
        },
        cwd,
      );

      const resolved = resolveLoadedOptimizerPolicy(loadOptimizerRegistry(config, cwd), { modelRole: "planner" });

      expect(resolved.source).toBe("seed");
      expect(resolved.modelRole).toBe("planner");
      expect(resolved.modelProfileId).toBe(plannerModel.payload.modelProfileId);
      expect(resolved.policy.modelProfileId).toBe(plannerModel.payload.modelProfileId);
      expect(resolved.policyId).not.toBe(criticPolicy.payload.policyId);
    });
  });

  test("prefers a promoted persisted matching policy over the seed policy", () => {
    withTempCwd((cwd) => {
      const config = defaultConfig();
      const seedRecords = seedOptimizerRegistry(config, cwd);
      const localModel = firstRecord(seedRecords, "model_profile", (record) => record.labels.includes("local"));
      const codebase = firstRecord(seedRecords, "codebase_profile");
      const seedPolicy = firstRecord(seedRecords, "model_codebase_policy", (record) =>
        record.payload.modelProfileId === localModel.payload.modelProfileId
      );

      saveOptimizerRegistryRecord(
        config,
        {
          ...seedPolicy,
          registryRecordId: "registry.policy.local.promoted-override",
          status: "promoted",
          updatedAt: "2026-04-30T00:00:00.000Z",
          supersedesRecordId: seedPolicy.registryRecordId,
          payload: {
            ...seedPolicy.payload,
            policyId: "policy.local.promoted-override",
            modelProfileId: localModel.payload.modelProfileId,
            codebaseProfileId: codebase.payload.codebaseProfileId,
            canonicalToolVersion: "canonical-tools.v2",
            renderedToolVersion: "rendered-tools.v2",
            resultStyleVersion: "result-style.v2",
            verificationPolicyVersion: "verification.v2",
            editStrategyVersion: "edit-strategy.v2",
            renderedEditContractVersion: "rendered-edit-contract.v2",
            editFallbackPolicyVersion: "edit-fallback.v2",
            editRepairPolicyVersion: "edit-repair.v2",
            editVerifierPolicyVersion: "edit-verifier.v2",
            editObjectiveSetId: "edit-objectives.promoted.v2",
          },
        },
        cwd,
      );

      const resolved = resolveLoadedOptimizerPolicy(loadOptimizerRegistry(config, cwd), { modelRole: "local" });

      expect(resolved.source).toBe("registry");
      expect(resolved.policyId).toBe("policy.local.promoted-override");
      expect(resolved.canonicalToolVersion).toBe("canonical-tools.v2");
      expect(resolved.renderedToolVersion).toBe("rendered-tools.v2");
      expect(resolved.resultStyleVersion).toBe("result-style.v2");
      expect(resolved.verificationPolicyVersion).toBe("verification.v2");
      expect(resolved.editStrategyVersion).toBe("edit-strategy.v2");
      expect(resolved.renderedEditContractVersion).toBe("rendered-edit-contract.v2");
      expect(resolved.editFallbackPolicyVersion).toBe("edit-fallback.v2");
      expect(resolved.editRepairPolicyVersion).toBe("edit-repair.v2");
      expect(resolved.editVerifierPolicyVersion).toBe("edit-verifier.v2");
      expect(resolved.editObjectiveSetId).toBe("edit-objectives.promoted.v2");
    });
  });

  test("ignores promoted policies pinned to a stale codebase fingerprint", () => {
    withTempCwd((cwd) => {
      const config = defaultConfig();
      const seedRecords = seedOptimizerRegistry(config, cwd);
      const localModel = firstRecord(seedRecords, "model_profile", (record) => record.labels.includes("local"));
      const codebase = firstRecord(seedRecords, "codebase_profile");
      const seedPolicy = firstRecord(seedRecords, "model_codebase_policy", (record) =>
        record.payload.modelProfileId === localModel.payload.modelProfileId
      );

      saveOptimizerRegistryRecord(
        config,
        {
          ...seedPolicy,
          registryRecordId: "registry.policy.local.stale-fingerprint",
          status: "promoted",
          updatedAt: "2026-04-30T00:00:00.000Z",
          supersedesRecordId: seedPolicy.registryRecordId,
          payload: {
            ...seedPolicy.payload,
            policyId: "policy.local.stale-fingerprint",
            modelProfileId: localModel.payload.modelProfileId,
            codebaseProfileId: codebase.payload.codebaseProfileId,
            codebaseRootFingerprint: "sha256:stale",
            canonicalToolVersion: "canonical-tools.stale",
          },
        },
        cwd,
      );

      const resolved = resolveLoadedOptimizerPolicy(loadOptimizerRegistry(config, cwd), { modelRole: "local" });

      expect(resolved.source).toBe("seed");
      expect(resolved.policyId).toBe(seedPolicy.payload.policyId);
      expect(resolved.canonicalToolVersion).toBe("canonical-tools.v1");
      expect(resolved.policy.codebaseRootFingerprint).toBe(codebase.payload.rootFingerprint);
    });
  });

  test("prefers a promoted persisted profile over the seed profile with the same payload id", () => {
    withTempCwd((cwd) => {
      const config = defaultConfig();
      const seedRecords = seedOptimizerRegistry(config, cwd);
      const localModel = firstRecord(seedRecords, "model_profile", (record) => record.labels.includes("local"));

      saveOptimizerRegistryRecord(
        config,
        {
          ...localModel,
          registryRecordId: "registry.model.local.promoted-override",
          status: "promoted",
          updatedAt: "2026-04-30T00:00:00.000Z",
          supersedesRecordId: localModel.registryRecordId,
          payload: {
            ...localModel.payload,
            maxOutputTokens: localModel.payload.maxOutputTokens + 1,
          },
        },
        cwd,
      );

      const resolved = resolveLoadedOptimizerPolicy(loadOptimizerRegistry(config, cwd), { modelRole: "local" });

      expect(resolved.recordIds.modelProfileRecordId).toBe("registry.model.local.promoted-override");
      expect(resolved.modelProfile.maxOutputTokens).toBe(localModel.payload.maxOutputTokens + 1);
      expect(resolved.policy.modelProfileId).toBe(localModel.payload.modelProfileId);
    });
  });
});
