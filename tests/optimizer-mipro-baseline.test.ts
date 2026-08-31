import { readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, test } from "bun:test";
import {
  miproDemo,
  selectMiproBaselineDemos,
  type MiproBaselineConfig,
} from "../src/optimizer/mipro-baseline";

const demos = [
  miproDemo({
    demoId: "demo.tool-error.high",
    input: { prompt: "fix tool call" },
    expectedOutput: { action: "tighten tool schema" },
    tags: ["tool-call", "typescript"],
    score: 0.92,
  }),
  miproDemo({
    demoId: "demo.plan.medium",
    input: { prompt: "make a coding plan" },
    expectedOutput: { action: "write plan" },
    tags: ["planning"],
    score: 0.7,
  }),
  miproDemo({
    demoId: "demo.tool-error.tie",
    input: { prompt: "repair malformed args" },
    expectedOutput: { action: "validate args" },
    tags: ["tool-call"],
    score: 0.92,
  }),
  miproDemo({
    demoId: "demo.hidden.best",
    input: { prompt: "holdout answer" },
    expectedOutput: { action: "must stay hidden" },
    tags: ["tool-call", "typescript"],
    split: "holdout",
    score: 1,
  }),
];

describe("MiPRO baseline boundary", () => {
  test("is disabled by default and returns an explicit diagnostic", () => {
    const result = selectMiproBaselineDemos({ demos });

    expect(result.enabled).toBe(false);
    expect(result.demos).toEqual([]);
    expect(result.sidecar).toBeUndefined();
    expect(result.diagnostics).toEqual([
      {
        code: "mipro_baseline_disabled",
        severity: "info",
        reason: "MiPRO baseline is disabled by default and requires explicit offline opt-in.",
      },
    ]);
  });

  test("accepts explicit offline sidecar config as data only", () => {
    const config: MiproBaselineConfig = {
      enabled: true,
      purpose: "offline_baseline",
      selector: {
        maxDemos: 2,
        requiredTags: ["tool-call"],
        excludeSplits: ["holdout"],
      },
      sidecar: {
        sidecarId: "sidecar.mipro.offline",
        kind: "dspy_mipro_v2",
        command: ["python", "-m", "dspy.teleprompt.mipro_optimizer"],
        env: { DSPY_CACHE: ".bag/optimizer/mipro-cache" },
        timeoutMs: 600_000,
        notes: ["Opt-in offline baseline only; never part of normal ACP runtime."],
      },
    };

    const result = selectMiproBaselineDemos({
      config,
      demos,
      taskTags: ["tool-call"],
    });

    expect(result.enabled).toBe(true);
    expect(result.purpose).toBe("offline_baseline");
    expect(result.sidecar).toEqual(config.sidecar);
    expect(result.demos.map((demo) => demo.demoId)).toEqual([
      "demo.tool-error.high",
      "demo.tool-error.tie",
    ]);
  });

  test("does not provide any runtime command execution path", () => {
    const result = selectMiproBaselineDemos({
      config: {
        enabled: true,
        purpose: "offline_baseline",
        selector: {
          maxDemos: 1,
          requiredTags: [],
          excludeSplits: ["holdout"],
        },
        sidecar: {
          sidecarId: "sidecar.mipro.explicit",
          kind: "dspy_mipro_v2",
          command: ["python", "offline-mipro.py"],
        },
      },
      demos,
    });

    expect(result.sidecar?.command).toEqual(["python", "offline-mipro.py"]);
    expect(result).not.toHaveProperty("process");
    expect(result).not.toHaveProperty("exitCode");

    const source = readFileSync(join(import.meta.dir, "..", "src", "optimizer", "mipro-baseline.ts"), "utf8");
    expect(source).not.toContain("node:child_process");
    expect(source).not.toContain("child_process");
    expect(source).not.toContain("execFile(");
    expect(source).not.toContain("spawn(");
  });

  test("selects demos deterministically without exposing holdout by default", () => {
    const first = selectMiproBaselineDemos({
      config: {
        enabled: true,
        purpose: "few_shot_demo_selection",
        selector: {
          maxDemos: 3,
          requiredTags: [],
          excludeSplits: ["holdout"],
        },
      },
      demos,
      taskTags: ["tool-call"],
      taskText: "repair a TypeScript tool-call failure",
    });
    const second = selectMiproBaselineDemos({
      config: {
        enabled: true,
        purpose: "few_shot_demo_selection",
        selector: {
          maxDemos: 3,
          requiredTags: [],
          excludeSplits: ["holdout"],
        },
      },
      demos: [...demos].reverse(),
      taskTags: ["tool-call"],
      taskText: "repair a TypeScript tool-call failure",
    });

    expect(first).toEqual(second);
    expect(first.demos.map((demo) => demo.demoId)).toEqual([
      "demo.tool-error.high",
      "demo.tool-error.tie",
      "demo.plan.medium",
    ]);
    expect(first.demos.map((demo) => demo.split)).not.toContain("holdout");
  });
});
