import { describe, expect, test } from "bun:test";
import {
  CANONICAL_EDIT_STRATEGY_DEFINITIONS,
  canonicalEditStrategyDefinitionsByFamily,
  CanonicalEditStrategyDefinitionSchema,
  initialExperimentalEditStrategyIds,
  parseCanonicalEditStrategyDefinitions,
} from "../src/edit-strategy/taxonomy";
import { EditStrategyFamilySchema } from "../src/edit-strategy/types";

describe("edit strategy taxonomy", () => {
  test("parses every canonical strategy definition", () => {
    const definitions = parseCanonicalEditStrategyDefinitions();

    expect(definitions).toHaveLength(CANONICAL_EDIT_STRATEGY_DEFINITIONS.length);
    expect(definitions.every((definition) => definition.traceRequirements.includes("post_apply_consistency"))).toBe(true);
    expect(definitions.every((definition) => definition.traceRequirements.includes("verification_status"))).toBe(true);
  });

  test("covers every concrete strategy family without choosing a global default", () => {
    const byFamily = canonicalEditStrategyDefinitionsByFamily();
    const missingFamilies = EditStrategyFamilySchema.options
      .filter((family) => family !== "custom")
      .filter((family) => !byFamily.has(family));

    expect(missingFamilies).toEqual([]);
    expect("globalDefaultStrategyId" in { definitions: CANONICAL_EDIT_STRATEGY_DEFINITIONS }).toBe(false);
  });

  test("keeps initial experiment candidates small and explicitly experimental", () => {
    expect(initialExperimentalEditStrategyIds()).toEqual([
      "edit.apply-patch.v1",
      "edit.exact-replace.v1",
      "edit.hash-range.experimental.v1",
      "edit.unified-diff.v1",
      "edit.whole-file.acp-write.v1",
    ]);
  });

  test("future-gated strategies are not initial experiment candidates", () => {
    const futureGated = parseCanonicalEditStrategyDefinitions()
      .filter((definition) => definition.maturity === "future_gate");

    expect(futureGated.map((definition) => definition.family).sort()).toEqual([
      "ast_structured",
      "range_native",
    ]);
    expect(futureGated.every((definition) => !definition.initialExperimentCandidate)).toBe(true);
  });

  test("rejects future gates on non-future strategies and future strategies in the initial set", () => {
    expect(
      CanonicalEditStrategyDefinitionSchema.safeParse({
        ...CANONICAL_EDIT_STRATEGY_DEFINITIONS[0],
        futureGate: "lsp_explicit_approval_required",
      }).success,
    ).toBe(false);

    expect(
      CanonicalEditStrategyDefinitionSchema.safeParse({
        ...CANONICAL_EDIT_STRATEGY_DEFINITIONS.at(-1),
        initialExperimentCandidate: true,
      }).success,
    ).toBe(false);
  });

  test("does not encode model-specific strategy routing", () => {
    const serialized = JSON.stringify(CANONICAL_EDIT_STRATEGY_DEFINITIONS).toLowerCase();

    expect(serialized.includes("qwen")).toBe(false);
    expect(serialized.includes("gpt")).toBe(false);
    expect(serialized.includes("claude")).toBe(false);
    expect(serialized.includes("gemini")).toBe(false);
  });
});
