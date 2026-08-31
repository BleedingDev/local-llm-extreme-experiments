import { z } from "zod";
import {
  loadOptimizerGateSuiteStatus,
  type OptimizerGateSuiteLoadError,
  type OptimizerGateSuiteStatus,
} from "./gate-suite";
import type { ResolvedOptimizerPolicy } from "./policy-resolver";
import type { OptimizerRegistryLoadResult, RegistryArtifactError } from "./registry";

export const OPTIMIZER_RUNTIME_READINESS_SCHEMA_VERSION = "optimizer-runtime-readiness.v1";

export const OptimizerRuntimeReadinessCapabilitySchema = z.enum([
  "candidate_generation",
  "auto_promotion",
]);
export type OptimizerRuntimeReadinessCapability = z.infer<typeof OptimizerRuntimeReadinessCapabilitySchema>;

const OptimizerRuntimeReadinessErrorSchema = z.object({
  path: z.string().min(1),
  kind: z.string().min(1),
  message: z.string().min(1),
}).strict();

export const OptimizerRuntimeGateSuiteEvidenceSchema = z.object({
  suitePath: z.string().min(1),
  suiteLoaded: z.boolean(),
  state: z.enum(["promotion_ready", "fail_closed"]),
  promotionAllowed: z.boolean(),
  autoPromotionAllowed: z.boolean(),
  candidateGeneration: z.string().min(1),
  blockingReasons: z.array(z.string().min(1)),
  mustFailClosedOn: z.array(z.string().min(1)),
  errors: z.array(OptimizerRuntimeReadinessErrorSchema),
  suiteId: z.string().min(1).optional(),
  graphId: z.string().min(1).optional(),
  generatedAt: z.string().min(1).optional(),
  sourceEvidenceIndex: z.string().min(1).optional(),
  sourceScorecardSuite: z.string().min(1).optional(),
  contractIds: z.array(z.string().min(1)).optional(),
  policySeparationDimensions: z.array(z.string().min(1)).optional(),
}).strict();
export type OptimizerRuntimeGateSuiteEvidence = z.infer<typeof OptimizerRuntimeGateSuiteEvidenceSchema>;

export const OptimizerRuntimeRegistryEvidenceSchema = z.object({
  root: z.string().min(1),
  errorCount: z.number().int().nonnegative(),
  invalidRecordCount: z.number().int().nonnegative(),
  errors: z.array(OptimizerRuntimeReadinessErrorSchema),
}).strict();
export type OptimizerRuntimeRegistryEvidence = z.infer<typeof OptimizerRuntimeRegistryEvidenceSchema>;

export const OptimizerRuntimePolicyEvidenceSchema = z.object({
  source: z.enum(["active_pointer", "registry", "seed"]),
  modelProfileId: z.string().min(1),
  codebaseProfileId: z.string().min(1),
  codebaseRootFingerprint: z.string().min(1),
  policyId: z.string().min(1),
}).strict();
export type OptimizerRuntimePolicyEvidence = z.infer<typeof OptimizerRuntimePolicyEvidenceSchema>;

export const OptimizerRuntimeReadinessDecisionSchema = z.object({
  schemaVersion: z.literal(OPTIMIZER_RUNTIME_READINESS_SCHEMA_VERSION),
  checkedAt: z.string().min(1),
  requiredCapability: OptimizerRuntimeReadinessCapabilitySchema,
  decision: z.enum(["allow", "block"]),
  allowed: z.boolean(),
  failClosed: z.boolean(),
  reasons: z.array(z.string().min(1)),
  gateSuite: OptimizerRuntimeGateSuiteEvidenceSchema,
  registry: OptimizerRuntimeRegistryEvidenceSchema.optional(),
  resolvedPolicy: OptimizerRuntimePolicyEvidenceSchema.optional(),
}).strict();
export type OptimizerRuntimeReadinessDecision = z.infer<typeof OptimizerRuntimeReadinessDecisionSchema>;

export type EvaluateOptimizerRuntimeReadinessInput = {
  cwd?: string;
  suitePath?: string;
  checkedAt?: string;
  requiredCapability?: OptimizerRuntimeReadinessCapability;
  gateSuiteStatus?: OptimizerGateSuiteStatus;
  registry?: Pick<OptimizerRegistryLoadResult, "root" | "errors" | "invalidRecords">;
  resolvedPolicy?: Pick<
    ResolvedOptimizerPolicy,
    "source" | "modelProfileId" | "codebaseProfileId" | "codebaseRootFingerprint" | "policyId"
  >;
};

export const evaluateOptimizerRuntimeReadiness = (
  input: EvaluateOptimizerRuntimeReadinessInput = {},
): OptimizerRuntimeReadinessDecision => {
  const requiredCapability = OptimizerRuntimeReadinessCapabilitySchema.parse(input.requiredCapability ?? "auto_promotion");
  const gateSuiteStatus = input.gateSuiteStatus ?? loadOptimizerGateSuiteStatus({
    ...(input.cwd === undefined ? {} : { cwd: input.cwd }),
    ...(input.suitePath === undefined ? {} : { suitePath: input.suitePath }),
  });
  const gateSuite = gateSuiteEvidence(gateSuiteStatus);
  const registry = input.registry === undefined ? undefined : registryEvidence(input.registry);
  const resolvedPolicy = input.resolvedPolicy === undefined ? undefined : OptimizerRuntimePolicyEvidenceSchema.parse(input.resolvedPolicy);
  const gateSuiteAllowed = capabilityAllowed(requiredCapability, gateSuiteStatus);
  const registryAllowsRuntime = registry === undefined || registry.errorCount === 0;
  const allowed = gateSuiteAllowed && registryAllowsRuntime;
  const reasons = readinessReasons({
    requiredCapability,
    gateSuiteStatus,
    gateSuiteAllowed,
    allowed,
    ...(registry === undefined ? {} : { registry }),
  });

  return OptimizerRuntimeReadinessDecisionSchema.parse({
    schemaVersion: OPTIMIZER_RUNTIME_READINESS_SCHEMA_VERSION,
    checkedAt: input.checkedAt ?? new Date().toISOString(),
    requiredCapability,
    decision: allowed ? "allow" : "block",
    allowed,
    failClosed: !allowed,
    reasons,
    gateSuite,
    ...(registry === undefined ? {} : { registry }),
    ...(resolvedPolicy === undefined ? {} : { resolvedPolicy }),
  });
};

const capabilityAllowed = (
  requiredCapability: OptimizerRuntimeReadinessCapability,
  gateSuiteStatus: OptimizerGateSuiteStatus,
): boolean => {
  if (requiredCapability === "candidate_generation") {
    return gateSuiteStatus.suiteLoaded &&
      gateSuiteStatus.errors.length === 0 &&
      gateSuiteStatus.candidateGeneration.startsWith("allowed");
  }
  return gateSuiteStatus.promotionAllowed;
};

const gateSuiteEvidence = (status: OptimizerGateSuiteStatus): OptimizerRuntimeGateSuiteEvidence =>
  OptimizerRuntimeGateSuiteEvidenceSchema.parse({
    suitePath: status.suitePath,
    suiteLoaded: status.suiteLoaded,
    state: status.state,
    promotionAllowed: status.promotionAllowed,
    autoPromotionAllowed: status.autoPromotionAllowed,
    candidateGeneration: status.candidateGeneration,
    blockingReasons: status.blockingReasons,
    mustFailClosedOn: status.mustFailClosedOn,
    errors: status.errors.map(readinessError),
    ...(status.suite === undefined ? {} : {
      suiteId: status.suite.optimizerGateSuiteId,
      graphId: status.suite.graphId,
      generatedAt: status.suite.generatedAt,
      sourceEvidenceIndex: status.suite.sourceEvidenceIndex,
      sourceScorecardSuite: status.suite.sourceScorecardSuite,
      contractIds: status.suite.contracts.map((contract) => contract.contractId),
      policySeparationDimensions: status.suite.policySeparation.dimensions,
    }),
  });

const registryEvidence = (
  registry: Pick<OptimizerRegistryLoadResult, "root" | "errors" | "invalidRecords">,
): OptimizerRuntimeRegistryEvidence =>
  OptimizerRuntimeRegistryEvidenceSchema.parse({
    root: registry.root,
    errorCount: registry.errors.length,
    invalidRecordCount: registry.invalidRecords.length,
    errors: registry.errors.map(readinessError),
  });

const readinessReasons = (input: {
  requiredCapability: OptimizerRuntimeReadinessCapability;
  gateSuiteStatus: OptimizerGateSuiteStatus;
  gateSuiteAllowed: boolean;
  registry?: OptimizerRuntimeRegistryEvidence;
  allowed: boolean;
}): string[] => {
  if (input.allowed) {
    return [
      input.requiredCapability === "candidate_generation"
        ? `Optimizer gate suite allows candidate generation: ${input.gateSuiteStatus.candidateGeneration}.`
        : "Optimizer gate suite allows runtime auto-promotion.",
    ];
  }

  return uniqueReasons([
    ...gateSuiteBlockReasons(input.requiredCapability, input.gateSuiteStatus, input.gateSuiteAllowed),
    ...(input.registry === undefined || input.registry.errorCount === 0
      ? []
      : [
          `Optimizer registry has ${input.registry.errorCount} load error(s) and ${input.registry.invalidRecordCount} invalid record(s).`,
          ...input.registry.errors.map((error) => `${error.kind} at ${error.path}: ${error.message}`),
        ]),
  ]);
};

const gateSuiteBlockReasons = (
  requiredCapability: OptimizerRuntimeReadinessCapability,
  status: OptimizerGateSuiteStatus,
  gateSuiteAllowed: boolean,
): string[] => {
  if (gateSuiteAllowed) {
    return [];
  }
  if (status.errors.length > 0) {
    return status.errors.map((error) => `${error.kind} at ${error.path}: ${error.message}`);
  }
  if (requiredCapability === "candidate_generation") {
    return [
      `Candidate generation is not explicitly allowed by optimizer gate suite: ${status.candidateGeneration}.`,
      ...status.blockingReasons,
    ];
  }
  return status.blockingReasons.length > 0
    ? status.blockingReasons
    : ["Optimizer gate suite did not allow runtime auto-promotion."];
};

const readinessError = (
  error: OptimizerGateSuiteLoadError | RegistryArtifactError,
): z.infer<typeof OptimizerRuntimeReadinessErrorSchema> => ({
  path: error.path,
  kind: error.kind,
  message: error.message,
});

const uniqueReasons = (reasons: readonly string[]): string[] => [...new Set(reasons.filter((reason) => reason.length > 0))];
