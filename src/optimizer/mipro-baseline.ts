import { z } from "zod";
import { JsonValueSchema, OptimizerIdSchema, type JsonValue } from "./types";

const DEFAULT_MAX_DEMOS = 8;
const MAX_DEMOS = 64;

export const MiproBaselineDiagnosticSchema = z.object({
  code: z.enum([
    "mipro_baseline_disabled",
    "mipro_no_eligible_demos",
    "mipro_demo_limit_reached",
  ]),
  severity: z.enum(["info", "warning", "error"]),
  reason: z.string().min(1),
}).strict();
export type MiproBaselineDiagnostic = z.infer<typeof MiproBaselineDiagnosticSchema>;

export const MiproBaselineSidecarCommandSchema = z.object({
  sidecarId: OptimizerIdSchema,
  kind: z.literal("dspy_mipro_v2"),
  command: z.array(z.string().min(1)).min(1),
  cwd: z.string().min(1).optional(),
  env: z.record(z.string(), z.string()).default({}),
  timeoutMs: z.number().int().positive().optional(),
  notes: z.array(z.string().min(1)).default([]),
}).strict();
export type MiproBaselineSidecarCommand = z.infer<typeof MiproBaselineSidecarCommandSchema>;

export const MiproDemoSplitSchema = z.enum(["train", "dev", "holdout", "manual"]);
export type MiproDemoSplit = z.infer<typeof MiproDemoSplitSchema>;

export const MiproBaselineDemoSchema = z.object({
  demoId: OptimizerIdSchema,
  input: JsonValueSchema,
  expectedOutput: JsonValueSchema.optional(),
  tags: z.array(z.string().min(1)).default([]),
  split: MiproDemoSplitSchema.default("train"),
  score: z.number().finite().default(0),
  sourceRef: z.string().min(1).optional(),
}).strict();
export type MiproBaselineDemo = z.infer<typeof MiproBaselineDemoSchema>;

export const MiproBaselineSelectorConfigSchema = z.object({
  maxDemos: z.number().int().positive().max(MAX_DEMOS).default(DEFAULT_MAX_DEMOS),
  requiredTags: z.array(z.string().min(1)).default([]),
  excludeSplits: z.array(MiproDemoSplitSchema).default(["holdout"]),
  includeSplits: z.array(MiproDemoSplitSchema).optional(),
}).strict();
export type MiproBaselineSelectorConfig = z.infer<typeof MiproBaselineSelectorConfigSchema>;

export const MiproBaselineConfigSchema = z.object({
  enabled: z.boolean().default(false),
  purpose: z.enum(["offline_baseline", "few_shot_demo_selection"]).default("few_shot_demo_selection"),
  selector: MiproBaselineSelectorConfigSchema.default({
    maxDemos: DEFAULT_MAX_DEMOS,
    requiredTags: [],
    excludeSplits: ["holdout"],
  }),
  sidecar: MiproBaselineSidecarCommandSchema.optional(),
}).strict();
export type MiproBaselineConfig = z.infer<typeof MiproBaselineConfigSchema>;

export const MiproBaselineSelectionResultSchema = z.object({
  enabled: z.boolean(),
  purpose: MiproBaselineConfigSchema.shape.purpose,
  demos: z.array(MiproBaselineDemoSchema),
  sidecar: MiproBaselineSidecarCommandSchema.optional(),
  diagnostics: z.array(MiproBaselineDiagnosticSchema).default([]),
}).strict();
export type MiproBaselineSelectionResult = z.infer<typeof MiproBaselineSelectionResultSchema>;

export type SelectMiproBaselineDemosInput = {
  config?: z.input<typeof MiproBaselineConfigSchema>;
  demos?: readonly MiproBaselineDemo[];
  taskTags?: readonly string[];
  taskText?: string;
};

export const selectMiproBaselineDemos = (input: SelectMiproBaselineDemosInput = {}): MiproBaselineSelectionResult => {
  const config = MiproBaselineConfigSchema.parse(input.config ?? {});
  if (!config.enabled) {
    return MiproBaselineSelectionResultSchema.parse({
      enabled: false,
      purpose: config.purpose,
      demos: [],
      diagnostics: [
        {
          code: "mipro_baseline_disabled",
          severity: "info",
          reason: "MiPRO baseline is disabled by default and requires explicit offline opt-in.",
        },
      ],
    });
  }

  const parsedDemos = (input.demos ?? []).map((demo) => MiproBaselineDemoSchema.parse(demo));
  const taskTags = [...new Set(input.taskTags ?? [])].sort((left, right) => left.localeCompare(right));
  const selector = config.selector;
  const selected = parsedDemos
    .filter((demo) => isDemoEligible(demo, selector))
    .filter((demo) => selector.requiredTags.every((tag) => demo.tags.includes(tag)))
    .map((demo) => ({ demo, rank: rankDemo(demo, taskTags, input.taskText ?? "") }))
    .sort(compareRankedDemos)
    .slice(0, selector.maxDemos)
    .map(({ demo }) => demo);

  const diagnostics: MiproBaselineDiagnostic[] = [];
  if (selected.length === 0) {
    diagnostics.push({
      code: "mipro_no_eligible_demos",
      severity: "warning",
      reason: "No eligible MiPRO baseline demos matched the explicit selector.",
    });
  } else if (parsedDemos.length > selected.length && selected.length >= selector.maxDemos) {
    diagnostics.push({
      code: "mipro_demo_limit_reached",
      severity: "info",
      reason: `MiPRO baseline demo selection was capped at ${selector.maxDemos} demos.`,
    });
  }

  return MiproBaselineSelectionResultSchema.parse({
    enabled: true,
    purpose: config.purpose,
    demos: selected,
    diagnostics,
    ...(config.sidecar == null ? {} : { sidecar: config.sidecar }),
  });
};

type RankedDemo = {
  demo: MiproBaselineDemo;
  rank: {
    score: number;
    tagMatches: number;
    textMatches: number;
  };
};

const isDemoEligible = (demo: MiproBaselineDemo, selector: MiproBaselineSelectorConfig): boolean => {
  if (selector.includeSplits != null && !selector.includeSplits.includes(demo.split)) {
    return false;
  }
  return !selector.excludeSplits.includes(demo.split);
};

const rankDemo = (
  demo: MiproBaselineDemo,
  taskTags: readonly string[],
  taskText: string,
): RankedDemo["rank"] => {
  const tagMatches = taskTags.filter((tag) => demo.tags.includes(tag)).length;
  const normalizedText = taskText.toLowerCase();
  const textMatches = demo.tags.filter((tag) => normalizedText.includes(tag.toLowerCase())).length;
  return {
    score: demo.score,
    tagMatches,
    textMatches,
  };
};

const compareRankedDemos = (left: RankedDemo, right: RankedDemo): number =>
  right.rank.score - left.rank.score
  || right.rank.tagMatches - left.rank.tagMatches
  || right.rank.textMatches - left.rank.textMatches
  || left.demo.demoId.localeCompare(right.demo.demoId);

export const miproDemo = (input: {
  demoId: string;
  input: JsonValue;
  expectedOutput?: JsonValue;
  tags?: readonly string[];
  split?: MiproDemoSplit;
  score?: number;
  sourceRef?: string;
}): MiproBaselineDemo =>
  MiproBaselineDemoSchema.parse({
    demoId: input.demoId,
    input: input.input,
    ...(input.expectedOutput === undefined ? {} : { expectedOutput: input.expectedOutput }),
    tags: [...(input.tags ?? [])],
    ...(input.split == null ? {} : { split: input.split }),
    ...(input.score == null ? {} : { score: input.score }),
    ...(input.sourceRef == null ? {} : { sourceRef: input.sourceRef }),
  });
