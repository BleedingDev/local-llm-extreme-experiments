import { existsSync, readFileSync } from "node:fs";
import { resolve } from "node:path";
import { z } from "zod";

export const OPTIMIZER_GATE_SUITE_PATH = ".bag/evidence/optimizer/index.json";
export const OPTIMIZER_GATE_SUITE_SCHEMA_VERSION = "local-evidence-optimizer-gate-suite.v1";
export const OPTIMIZER_GATE_SUITE_STATUS_SCHEMA_VERSION = "optimizer-gate-suite-status.v1";

const NonEmptyStringSchema = z.string().min(1);

export const OptimizerGateContractSchema = z.object({
  contractId: NonEmptyStringSchema,
  jsonPath: NonEmptyStringSchema,
  markdownPath: NonEmptyStringSchema,
  primaryUse: NonEmptyStringSchema,
}).strict();
export type OptimizerGateContract = z.infer<typeof OptimizerGateContractSchema>;

export const OptimizerGateSuiteDecisionSchema = z.object({
  candidateGeneration: NonEmptyStringSchema,
  autoPromotion: z.enum(["allowed", "blocked"]),
  promotionReady: z.boolean(),
  blockingReasons: z.array(NonEmptyStringSchema).default([]),
}).strict();
export type OptimizerGateSuiteDecision = z.infer<typeof OptimizerGateSuiteDecisionSchema>;

export const OptimizerGateSuitePolicySeparationSchema = z.object({
  dimensions: z.array(z.enum([
    "modelProfileId",
    "codebaseProfileId",
    "modelCodebasePolicyId",
  ])).min(1),
  principle: NonEmptyStringSchema,
}).strict();
export type OptimizerGateSuitePolicySeparation = z.infer<typeof OptimizerGateSuitePolicySeparationSchema>;

export const OptimizerGateSuiteSchema = z.object({
  schemaVersion: z.literal(OPTIMIZER_GATE_SUITE_SCHEMA_VERSION),
  optimizerGateSuiteId: NonEmptyStringSchema,
  graphId: NonEmptyStringSchema,
  generatedAt: NonEmptyStringSchema,
  sourceEvidenceIndex: NonEmptyStringSchema,
  sourceScorecardSuite: NonEmptyStringSchema,
  contracts: z.array(OptimizerGateContractSchema).min(1),
  currentDecision: OptimizerGateSuiteDecisionSchema,
  mustFailClosedOn: z.array(NonEmptyStringSchema).min(1),
  policySeparation: OptimizerGateSuitePolicySeparationSchema,
}).strict();
export type OptimizerGateSuite = z.infer<typeof OptimizerGateSuiteSchema>;

export type OptimizerGateSuiteErrorKind = "missing" | "read_error" | "parse_error" | "validation_error";

export type OptimizerGateSuiteLoadError = {
  kind: OptimizerGateSuiteErrorKind;
  path: string;
  message: string;
};

export type OptimizerGateSuiteState = "promotion_ready" | "fail_closed";

export type OptimizerGateSuiteStatus = {
  schemaVersion: typeof OPTIMIZER_GATE_SUITE_STATUS_SCHEMA_VERSION;
  state: OptimizerGateSuiteState;
  suitePath: string;
  suiteLoaded: boolean;
  promotionAllowed: boolean;
  autoPromotionAllowed: boolean;
  candidateGeneration: string;
  blockingReasons: string[];
  mustFailClosedOn: string[];
  errors: OptimizerGateSuiteLoadError[];
  suite?: OptimizerGateSuite;
};

export type LoadOptimizerGateSuiteStatusInput = {
  cwd?: string;
  suitePath?: string;
};

export const loadOptimizerGateSuiteStatus = (
  input: LoadOptimizerGateSuiteStatusInput = {},
): OptimizerGateSuiteStatus => {
  const suitePath = resolve(input.cwd ?? process.cwd(), input.suitePath ?? OPTIMIZER_GATE_SUITE_PATH);
  const loaded = readOptimizerGateSuite(suitePath);
  if ("error" in loaded) {
    return failClosedStatus({
      suitePath,
      errors: [loaded.error],
      blockingReasons: [loaded.error.message],
    });
  }

  const decision = loaded.suite.currentDecision;
  const promotionAllowed = decision.promotionReady &&
    decision.autoPromotion === "allowed" &&
    decision.blockingReasons.length === 0;
  const blockingReasons = promotionAllowed
    ? []
    : normalizedBlockingReasons(decision);

  return {
    schemaVersion: OPTIMIZER_GATE_SUITE_STATUS_SCHEMA_VERSION,
    state: promotionAllowed ? "promotion_ready" : "fail_closed",
    suitePath,
    suiteLoaded: true,
    promotionAllowed,
    autoPromotionAllowed: promotionAllowed,
    candidateGeneration: decision.candidateGeneration,
    blockingReasons,
    mustFailClosedOn: loaded.suite.mustFailClosedOn,
    errors: [],
    suite: loaded.suite,
  };
};

const readOptimizerGateSuite = (
  path: string,
): { suite: OptimizerGateSuite } | { error: OptimizerGateSuiteLoadError } => {
  if (!existsSync(path)) {
    return {
      error: {
        kind: "missing",
        path,
        message: "Optimizer gate suite is missing; runtime optimizer promotions fail closed.",
      },
    };
  }

  let raw: string;
  try {
    raw = readFileSync(path, "utf8");
  } catch (error) {
    return {
      error: {
        kind: "read_error",
        path,
        message: errorMessage(error),
      },
    };
  }

  let value: unknown;
  try {
    value = JSON.parse(raw) as unknown;
  } catch (error) {
    return {
      error: {
        kind: "parse_error",
        path,
        message: `Optimizer gate suite JSON is invalid: ${errorMessage(error)}`,
      },
    };
  }

  const parsed = OptimizerGateSuiteSchema.safeParse(value);
  if (!parsed.success) {
    return {
      error: {
        kind: "validation_error",
        path,
        message: `Optimizer gate suite shape is invalid: ${zodErrorMessage(parsed.error)}`,
      },
    };
  }

  return { suite: parsed.data };
};

const failClosedStatus = (input: {
  suitePath: string;
  errors: OptimizerGateSuiteLoadError[];
  blockingReasons: string[];
}): OptimizerGateSuiteStatus => ({
  schemaVersion: OPTIMIZER_GATE_SUITE_STATUS_SCHEMA_VERSION,
  state: "fail_closed",
  suitePath: input.suitePath,
  suiteLoaded: false,
  promotionAllowed: false,
  autoPromotionAllowed: false,
  candidateGeneration: "blocked",
  blockingReasons: input.blockingReasons,
  mustFailClosedOn: [],
  errors: input.errors,
});

const normalizedBlockingReasons = (decision: OptimizerGateSuiteDecision): string[] => {
  const reasons = [...decision.blockingReasons];
  if (!decision.promotionReady) {
    reasons.unshift("Promotion readiness is false.");
  }
  if (decision.autoPromotion === "blocked") {
    reasons.unshift("Auto-promotion is blocked.");
  }
  return [...new Set(reasons)];
};

const zodErrorMessage = (error: z.ZodError): string =>
  error.issues.map((issue) => `${issue.path.join(".") || "<root>"}: ${issue.message}`).join("; ");

const errorMessage = (error: unknown): string =>
  error instanceof Error ? error.message : String(error);
