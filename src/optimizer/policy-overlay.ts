import { z } from "zod";
import { EditStrategyFamilySchema } from "../edit-strategy/types";
import {
  JsonValueSchema,
  OptimizerIdSchema,
  OptimizerVersionSchema,
  ToolResultStyleSchema,
} from "./types";

export const POLICY_OVERLAY_SCHEMA_VERSION = "optimizer.policy-overlay.v1";

export const PolicyOverlayIdentitySchema = z.object({
  modelProfileId: OptimizerIdSchema,
  codebaseProfileId: OptimizerIdSchema,
  policyId: OptimizerIdSchema,
}).strict();
export type PolicyOverlayIdentity = z.infer<typeof PolicyOverlayIdentitySchema>;

export const PolicyOverlayPromptFragmentTargetSchema = z.enum([
  "generic",
  "tool_routing",
  "edit_routing",
  "verifier",
  "recovery",
  "result_style",
]);
export type PolicyOverlayPromptFragmentTarget = z.infer<typeof PolicyOverlayPromptFragmentTargetSchema>;

export const PolicyOverlayPromptFragmentSchema = z.object({
  fragmentId: OptimizerIdSchema,
  target: PolicyOverlayPromptFragmentTargetSchema,
  text: z.string().min(1),
  reason: z.string().min(1).optional(),
}).strict();
export type PolicyOverlayPromptFragment = z.infer<typeof PolicyOverlayPromptFragmentSchema>;

export const PolicyOverlayToolContractSchema = z.object({
  overlayToolContractId: OptimizerIdSchema,
  canonicalToolId: OptimizerIdSchema.optional(),
  renderedToolId: OptimizerIdSchema.optional(),
  description: z.string().min(1).optional(),
  contractVersion: OptimizerVersionSchema.optional(),
  inputSchemaHints: z.record(z.string(), JsonValueSchema).optional(),
  resultStyle: ToolResultStyleSchema.optional(),
  promptFragments: z.array(PolicyOverlayPromptFragmentSchema).default([]),
  recoveryHintIds: z.array(OptimizerIdSchema).default([]),
}).strict().refine((contract) => contract.canonicalToolId !== undefined || contract.renderedToolId !== undefined, {
  message: "tool contract overlays require canonicalToolId or renderedToolId",
});
export type PolicyOverlayToolContract = z.infer<typeof PolicyOverlayToolContractSchema>;

export const PolicyOverlayEditContractSchema = z.object({
  overlayEditContractId: OptimizerIdSchema,
  editStrategyId: OptimizerIdSchema,
  editStrategyFamily: EditStrategyFamilySchema.optional(),
  editStrategyVersion: OptimizerVersionSchema.optional(),
  renderedEditContractVersion: OptimizerVersionSchema.optional(),
  editFallbackPolicyVersion: OptimizerVersionSchema.optional(),
  editRepairPolicyVersion: OptimizerVersionSchema.optional(),
  editVerifierPolicyVersion: OptimizerVersionSchema.optional(),
  contractFragments: z.array(PolicyOverlayPromptFragmentSchema).default([]),
  recoveryHintIds: z.array(OptimizerIdSchema).default([]),
}).strict();
export type PolicyOverlayEditContract = z.infer<typeof PolicyOverlayEditContractSchema>;

export const PolicyOverlayVerifierTacticSchema = z.object({
  verifierTacticId: OptimizerIdSchema,
  tacticVersion: OptimizerVersionSchema.default("v1"),
  commandIds: z.array(OptimizerIdSchema).default([]),
  appliesToFailureCodes: z.array(OptimizerIdSchema).default([]),
  promptFragments: z.array(PolicyOverlayPromptFragmentSchema).default([]),
  required: z.boolean().default(false),
}).strict();
export type PolicyOverlayVerifierTactic = z.infer<typeof PolicyOverlayVerifierTacticSchema>;

export const PolicyOverlayRecoveryHintSchema = z.object({
  recoveryHintId: OptimizerIdSchema,
  failureClass: OptimizerIdSchema,
  hint: z.string().min(1),
  stopAfterDeterministicFailure: z.boolean().default(true),
  promptFragments: z.array(PolicyOverlayPromptFragmentSchema).default([]),
}).strict();
export type PolicyOverlayRecoveryHint = z.infer<typeof PolicyOverlayRecoveryHintSchema>;

export const PolicyOverlayResultStyleSchema = z.object({
  resultStyleOverlayId: OptimizerIdSchema,
  resultStyleVersion: OptimizerVersionSchema,
  defaultToolResultStyle: ToolResultStyleSchema.optional(),
  toolResultStyles: z.array(z.object({
    canonicalToolId: OptimizerIdSchema.optional(),
    renderedToolId: OptimizerIdSchema.optional(),
    resultStyle: ToolResultStyleSchema,
    promptFragments: z.array(PolicyOverlayPromptFragmentSchema).default([]),
  }).strict().refine((entry) => entry.canonicalToolId !== undefined || entry.renderedToolId !== undefined, {
    message: "tool result style overlays require canonicalToolId or renderedToolId",
  })).default([]),
  promptFragments: z.array(PolicyOverlayPromptFragmentSchema).default([]),
}).strict();
export type PolicyOverlayResultStyle = z.infer<typeof PolicyOverlayResultStyleSchema>;

export const PolicyOverlayContractSchema = z.object({
  schemaVersion: z.literal(POLICY_OVERLAY_SCHEMA_VERSION).default(POLICY_OVERLAY_SCHEMA_VERSION),
  overlayId: OptimizerIdSchema,
  overlayVersion: OptimizerVersionSchema,
  modelProfileId: OptimizerIdSchema,
  codebaseProfileId: OptimizerIdSchema,
  policyId: OptimizerIdSchema,
  status: z.enum(["draft", "evaluating", "promoted", "retired", "rejected"]).default("draft"),
  createdAt: z.string().optional(),
  updatedAt: z.string().optional(),
  sourceEvidenceRefs: z.array(OptimizerIdSchema).default([]),
  toolContracts: z.array(PolicyOverlayToolContractSchema).default([]),
  editContracts: z.array(PolicyOverlayEditContractSchema).default([]),
  routingPromptFragments: z.array(PolicyOverlayPromptFragmentSchema).default([]),
  verifierTactics: z.array(PolicyOverlayVerifierTacticSchema).default([]),
  recoveryHints: z.array(PolicyOverlayRecoveryHintSchema).default([]),
  resultStyle: PolicyOverlayResultStyleSchema.optional(),
  promptFragments: z.array(PolicyOverlayPromptFragmentSchema).default([]),
  notes: z.array(z.string().min(1)).default([]),
}).strict();
export type PolicyOverlayContract = z.infer<typeof PolicyOverlayContractSchema>;

export type PolicyOverlayTupleField = keyof PolicyOverlayIdentity;

export type PolicyOverlayIdentityCheck =
  | {
    matches: true;
    identityKey: string;
  }
  | {
    matches: false;
    identityKey: string;
    expected: PolicyOverlayIdentity;
    actual: PolicyOverlayIdentity;
    mismatches: PolicyOverlayTupleField[];
  };

export interface PolicyOverlaySummary {
  overlayId: string;
  overlayVersion: string;
  identityKey: string;
  modelProfileId: string;
  codebaseProfileId: string;
  policyId: string;
  counts: {
    toolContracts: number;
    editContracts: number;
    routingPromptFragments: number;
    verifierTactics: number;
    recoveryHints: number;
    resultStyle: number;
    promptFragments: number;
  };
}

const identityFields: readonly PolicyOverlayTupleField[] = [
  "modelProfileId",
  "codebaseProfileId",
  "policyId",
];

export const policyOverlayIdentity = (
  input: PolicyOverlayIdentity | Pick<PolicyOverlayContract, PolicyOverlayTupleField>,
): PolicyOverlayIdentity =>
  PolicyOverlayIdentitySchema.parse({
    modelProfileId: input.modelProfileId,
    codebaseProfileId: input.codebaseProfileId,
    policyId: input.policyId,
  });

export const policyOverlayIdentityKey = (
  input: PolicyOverlayIdentity | Pick<PolicyOverlayContract, PolicyOverlayTupleField>,
): string => {
  const identity = policyOverlayIdentity(input);
  return [
    `modelProfileId=${identity.modelProfileId}`,
    `codebaseProfileId=${identity.codebaseProfileId}`,
    `policyId=${identity.policyId}`,
  ].join("|");
};

export const checkPolicyOverlayIdentity = (
  overlay: PolicyOverlayIdentity | Pick<PolicyOverlayContract, PolicyOverlayTupleField>,
  expected: PolicyOverlayIdentity | Pick<PolicyOverlayContract, PolicyOverlayTupleField>,
): PolicyOverlayIdentityCheck => {
  const actualIdentity = policyOverlayIdentity(overlay);
  const expectedIdentity = policyOverlayIdentity(expected);
  const mismatches = identityFields.filter((field) => actualIdentity[field] !== expectedIdentity[field]);
  const identityKey = policyOverlayIdentityKey(actualIdentity);

  return mismatches.length === 0
    ? { matches: true, identityKey }
    : {
        matches: false,
        identityKey,
        expected: expectedIdentity,
        actual: actualIdentity,
        mismatches,
      };
};

export const policyOverlayAppliesTo = (
  overlay: PolicyOverlayContract,
  expected: PolicyOverlayIdentity | Pick<PolicyOverlayContract, PolicyOverlayTupleField>,
): boolean =>
  checkPolicyOverlayIdentity(overlay, expected).matches;

export const filterApplicablePolicyOverlays = (
  overlays: readonly PolicyOverlayContract[],
  expected: PolicyOverlayIdentity | Pick<PolicyOverlayContract, PolicyOverlayTupleField>,
): PolicyOverlayContract[] =>
  overlays
    .map((overlay) => PolicyOverlayContractSchema.parse(overlay))
    .filter((overlay) => policyOverlayAppliesTo(overlay, expected));

const uniqueStrings = (values: readonly string[]): string[] =>
  [...new Set(values)].sort((left, right) => left.localeCompare(right));

const replaceById = <T extends Record<string, unknown>>(
  values: readonly T[],
  key: keyof T,
): T[] => {
  const byId = new Map<string, T>();
  for (const value of values) {
    const id = value[key];
    if (typeof id !== "string") {
      throw new Error(`policy overlay merge key ${String(key)} must be a string`);
    }
    byId.set(id, value);
  }
  return [...byId.values()].sort((left, right) => String(left[key]).localeCompare(String(right[key])));
};

export const mergePolicyOverlays = (
  overlays: readonly PolicyOverlayContract[],
): PolicyOverlayContract => {
  if (overlays.length === 0) {
    throw new Error("mergePolicyOverlays requires at least one overlay");
  }

  const parsed = overlays.map((overlay) => PolicyOverlayContractSchema.parse(overlay));
  const base = parsed[0]!;
  for (const overlay of parsed.slice(1)) {
    const check = checkPolicyOverlayIdentity(overlay, base);
    if (!check.matches) {
      throw new Error(`policy overlay tuple mismatch: ${check.mismatches.join(", ")}`);
    }
  }

  const resultStyle = [...parsed].reverse().find((overlay) => overlay.resultStyle !== undefined)?.resultStyle;

  return PolicyOverlayContractSchema.parse({
    schemaVersion: POLICY_OVERLAY_SCHEMA_VERSION,
    overlayId: base.overlayId,
    overlayVersion: base.overlayVersion,
    modelProfileId: base.modelProfileId,
    codebaseProfileId: base.codebaseProfileId,
    policyId: base.policyId,
    status: base.status,
    createdAt: base.createdAt,
    updatedAt: parsed[parsed.length - 1]?.updatedAt ?? base.updatedAt,
    sourceEvidenceRefs: uniqueStrings(parsed.flatMap((overlay) => overlay.sourceEvidenceRefs)),
    toolContracts: replaceById(parsed.flatMap((overlay) => overlay.toolContracts), "overlayToolContractId"),
    editContracts: replaceById(parsed.flatMap((overlay) => overlay.editContracts), "overlayEditContractId"),
    routingPromptFragments: replaceById(parsed.flatMap((overlay) => overlay.routingPromptFragments), "fragmentId"),
    verifierTactics: replaceById(parsed.flatMap((overlay) => overlay.verifierTactics), "verifierTacticId"),
    recoveryHints: replaceById(parsed.flatMap((overlay) => overlay.recoveryHints), "recoveryHintId"),
    ...(resultStyle === undefined ? {} : { resultStyle }),
    promptFragments: replaceById(parsed.flatMap((overlay) => overlay.promptFragments), "fragmentId"),
    notes: uniqueStrings(parsed.flatMap((overlay) => overlay.notes)),
  });
};

export const summarizePolicyOverlay = (overlay: PolicyOverlayContract): PolicyOverlaySummary => {
  const parsed = PolicyOverlayContractSchema.parse(overlay);
  return {
    overlayId: parsed.overlayId,
    overlayVersion: parsed.overlayVersion,
    identityKey: policyOverlayIdentityKey(parsed),
    modelProfileId: parsed.modelProfileId,
    codebaseProfileId: parsed.codebaseProfileId,
    policyId: parsed.policyId,
    counts: {
      toolContracts: parsed.toolContracts.length,
      editContracts: parsed.editContracts.length,
      routingPromptFragments: parsed.routingPromptFragments.length,
      verifierTactics: parsed.verifierTactics.length,
      recoveryHints: parsed.recoveryHints.length,
      resultStyle: parsed.resultStyle === undefined ? 0 : 1,
      promptFragments: parsed.promptFragments.length,
    },
  };
};
