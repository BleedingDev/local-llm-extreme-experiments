import { describe, expect, test } from "bun:test";
import {
  POLICY_OVERLAY_SCHEMA_VERSION,
  PolicyOverlayContractSchema,
  checkPolicyOverlayIdentity,
  filterApplicablePolicyOverlays,
  mergePolicyOverlays,
  policyOverlayIdentityKey,
  summarizePolicyOverlay,
  type PolicyOverlayContract,
} from "./policy-overlay";

const identity = {
  modelProfileId: "model.qwen36.local",
  codebaseProfileId: "codebase.bleeding-agent",
  policyId: "policy.qwen36.bleeding-agent",
};

const overlay = (overrides: Partial<PolicyOverlayContract> = {}): PolicyOverlayContract =>
  PolicyOverlayContractSchema.parse({
    schemaVersion: POLICY_OVERLAY_SCHEMA_VERSION,
    overlayId: "overlay.qwen36.bleeding-agent.policy",
    overlayVersion: "overlay.v1",
    ...identity,
    status: "promoted",
    sourceEvidenceRefs: ["evidence.tool-routing"],
    toolContracts: [
      {
        overlayToolContractId: "overlay.tool.repo-read",
        canonicalToolId: "tool.repo.read",
        description: "Read exactly one repository file; never invent missing paths.",
        contractVersion: "rendered-tools.v2",
        promptFragments: [
          {
            fragmentId: "fragment.tool.repo-read.arguments",
            target: "tool_routing",
            text: "Before calling repo_read, validate that path is repository-relative.",
          },
        ],
      },
    ],
    editContracts: [
      {
        overlayEditContractId: "overlay.edit.apply-patch",
        editStrategyId: "edit.apply-patch.v1",
        editStrategyFamily: "apply_patch",
        renderedEditContractVersion: "rendered-edit-contract.v2",
        contractFragments: [
          {
            fragmentId: "fragment.edit.apply-patch.contract",
            target: "edit_routing",
            text: "Use apply_patch only for deterministic hunks with fresh context.",
          },
        ],
      },
    ],
    routingPromptFragments: [
      {
        fragmentId: "fragment.router.no-write",
        target: "tool_routing",
        text: "Do not route no-write requests to mutating tools.",
      },
    ],
    verifierTactics: [
      {
        verifierTacticId: "verifier.focused-first",
        commandIds: ["typecheck"],
        promptFragments: [
          {
            fragmentId: "fragment.verifier.focused-first",
            target: "verifier",
            text: "Run focused verification before broad suites when the codebase profile provides it.",
          },
        ],
      },
    ],
    recoveryHints: [
      {
        recoveryHintId: "recovery.invalid-tool-args",
        failureClass: "invalid_tool_args",
        hint: "Repair arguments once from the schema, then stop on deterministic schema failure.",
      },
    ],
    resultStyle: {
      resultStyleOverlayId: "result-style.structured-errors",
      resultStyleVersion: "result-style.v2",
      defaultToolResultStyle: "structured_error",
    },
    promptFragments: [
      {
        fragmentId: "fragment.generic.scoped",
        target: "generic",
        text: "This prompt fragment is scoped to the exact model/codebase/policy tuple.",
      },
    ],
    notes: ["Representative policy overlay fixture."],
    ...overrides,
  });

describe("optimizer policy overlay", () => {
  test("parses a strict versioned overlay across independently optimizable dimensions", () => {
    const parsed = overlay();
    const summary = summarizePolicyOverlay(parsed);

    expect(parsed.schemaVersion).toBe(POLICY_OVERLAY_SCHEMA_VERSION);
    expect(summary.identityKey).toBe(
      "modelProfileId=model.qwen36.local|codebaseProfileId=codebase.bleeding-agent|policyId=policy.qwen36.bleeding-agent",
    );
    expect(summary.counts).toEqual({
      toolContracts: 1,
      editContracts: 1,
      routingPromptFragments: 1,
      verifierTactics: 1,
      recoveryHints: 1,
      resultStyle: 1,
      promptFragments: 1,
    });
  });

  test("rejects non-contract fields instead of permitting global overlays", () => {
    const result = PolicyOverlayContractSchema.safeParse({
      ...overlay(),
      global: true,
    });

    expect(result.success).toBe(false);
  });

  test("stable identity and mismatch helper require exact model codebase policy tuple", () => {
    const parsed = overlay();

    expect(policyOverlayIdentityKey(parsed)).toBe(policyOverlayIdentityKey(identity));
    expect(checkPolicyOverlayIdentity(parsed, identity)).toEqual({
      matches: true,
      identityKey: policyOverlayIdentityKey(identity),
    });
    expect(checkPolicyOverlayIdentity(parsed, {
      ...identity,
      modelProfileId: "model.gpt55.master",
      codebaseProfileId: "codebase.other",
      policyId: "policy.other",
    })).toMatchObject({
      matches: false,
      mismatches: ["modelProfileId", "codebaseProfileId", "policyId"],
    });
  });

  test("filters by exact tuple with no model-only codebase-only or policy-only transfer", () => {
    const localOverlay = overlay();
    const otherModel = overlay({
      overlayId: "overlay.other-model",
      modelProfileId: "model.gpt55.master",
      policyId: "policy.gpt55.bleeding-agent",
    });
    const otherCodebase = overlay({
      overlayId: "overlay.other-codebase",
      codebaseProfileId: "codebase.other",
      policyId: "policy.qwen36.other",
    });
    const otherPolicy = overlay({
      overlayId: "overlay.other-policy",
      policyId: "policy.qwen36.bleeding-agent.experimental",
    });

    expect(filterApplicablePolicyOverlays(
      [otherModel, otherCodebase, otherPolicy, localOverlay],
      identity,
    ).map((entry) => entry.overlayId)).toEqual(["overlay.qwen36.bleeding-agent.policy"]);
  });

  test("merge is allowed only inside the same tuple and keeps scoped dimensions", () => {
    const first = overlay();
    const second = overlay({
      overlayId: "overlay.qwen36.bleeding-agent.policy.extra",
      overlayVersion: "overlay.v2",
      sourceEvidenceRefs: ["evidence.recovery"],
      toolContracts: [
        {
          overlayToolContractId: "overlay.tool.repo-write",
          canonicalToolId: "tool.repo.write",
          description: "Write only after the scoped edit strategy selected a mutating route.",
          promptFragments: [],
          recoveryHintIds: [],
        },
      ],
      promptFragments: [
        {
          fragmentId: "fragment.generic.verify-before-final",
          target: "generic",
          text: "Report verification status from the current codebase profile before final response.",
        },
      ],
    });

    const merged = mergePolicyOverlays([first, second]);

    expect(merged.modelProfileId).toBe(identity.modelProfileId);
    expect(merged.codebaseProfileId).toBe(identity.codebaseProfileId);
    expect(merged.policyId).toBe(identity.policyId);
    expect(merged.sourceEvidenceRefs).toEqual(["evidence.recovery", "evidence.tool-routing"]);
    expect(merged.toolContracts.map((contract) => contract.overlayToolContractId)).toEqual([
      "overlay.tool.repo-read",
      "overlay.tool.repo-write",
    ]);
    expect(merged.promptFragments.map((fragment) => fragment.fragmentId)).toEqual([
      "fragment.generic.scoped",
      "fragment.generic.verify-before-final",
    ]);

    expect(() => mergePolicyOverlays([
      first,
      overlay({
        overlayId: "overlay.mismatched-codebase",
        codebaseProfileId: "codebase.other",
        policyId: "policy.qwen36.other",
      }),
    ])).toThrow("policy overlay tuple mismatch: codebaseProfileId, policyId");
  });
});
