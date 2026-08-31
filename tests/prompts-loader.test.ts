/// <reference types="node" />
/// <reference path="../types/bun-test.d.ts" />
import { afterEach, beforeEach, describe, expect, test } from "bun:test";
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import {
  buildSystemPrompt,
  loadActiveTactics,
  loadAllTactics,
  parseFrontmatter,
  type Tactic,
} from "../src/prompts/loader";
import { SYSTEM_PROMPT_DEFAULT } from "../src/autonomous-coding-turn";

const tempRoots: string[] = [];

const mkRepo = (): string => {
  const root = mkdtempSync(join(tmpdir(), "bag-prompts-"));
  mkdirSync(join(root, "src", "prompts", "tactics"), { recursive: true });
  tempRoots.push(root);
  return root;
};

const writePrinciples = (root: string, body: string): void => {
  writeFileSync(join(root, "src", "prompts", "principles.md"), body, "utf8");
};

const writeTactic = (root: string, slug: string, content: string): void => {
  writeFileSync(join(root, "src", "prompts", "tactics", `${slug}.md`), content, "utf8");
};

describe("prompts loader", () => {
  beforeEach(() => {
    // No-op; each test mints its own root.
  });

  afterEach(() => {
    while (tempRoots.length > 0) {
      const r = tempRoots.pop();
      if (!r) continue;
      try {
        rmSync(r, { recursive: true, force: true });
      } catch {
        // best-effort
      }
    }
  });

  test("1. loader reads frontmatter correctly", () => {
    const raw = `---
id: my-tactic
status: active
order: 7
incident: foo R#1 — example incident description
introduced: 2026-05-02
review_by: 2026-08-02
trigger: "applies when X happens"
---
Body line one.
Body line two.
`;
    const parsed = parseFrontmatter(raw);
    expect(parsed).not.toBeNull();
    expect(parsed?.frontmatter.id).toBe("my-tactic");
    expect(parsed?.frontmatter.status).toBe("active");
    expect(parsed?.frontmatter.order).toBe(7);
    expect(parsed?.frontmatter.incident).toContain("foo R#1");
    expect(parsed?.frontmatter.trigger).toBe("applies when X happens");
    expect(parsed?.body).toBe("Body line one.\nBody line two.\n");
  });

  test("2. inactive tactics are NOT included", () => {
    const root = mkRepo();
    writePrinciples(root, "P\n${TACTICS}\nFOOTER\n");
    writeTactic(
      root,
      "active-one",
      `---
id: active-one
status: active
order: 1
---
ACTIVE\n`,
    );
    writeTactic(
      root,
      "deprecated-one",
      `---
id: deprecated-one
status: deprecated
order: 2
---
DEPRECATED\n`,
    );
    const tactics = loadActiveTactics(root);
    expect(tactics.length).toBe(1);
    expect(tactics[0]?.id).toBe("active-one");
    // loadAllTactics should still see both.
    const all = loadAllTactics(root);
    expect(all.length).toBe(2);
    const built = buildSystemPrompt({ sentinel: "X", repoRoot: root });
    expect(built).toContain("ACTIVE");
    expect(built).not.toContain("DEPRECATED");
  });

  test("3. sentinel placeholder is replaced", () => {
    const root = mkRepo();
    writePrinciples(
      root,
      "Run echo ${SUBMIT_SENTINEL} to submit.\nAnd ${SUBMIT_SENTINEL} again.\n",
    );
    const built = buildSystemPrompt({ sentinel: "MY_SENTINEL", repoRoot: root });
    expect(built).toContain("Run echo MY_SENTINEL to submit.");
    expect(built).toContain("And MY_SENTINEL again.");
    expect(built).not.toContain("${SUBMIT_SENTINEL}");
  });

  test("4. attestation footer is appended", () => {
    const root = mkRepo();
    writePrinciples(root, "Body.\n");
    writeTactic(
      root,
      "t1",
      `---
id: t1
status: active
order: 1
---
T1\n`,
    );
    writeTactic(
      root,
      "t2",
      `---
id: t2
status: active
order: 2
---
T2\n`,
    );
    const built = buildSystemPrompt({ sentinel: "X", repoRoot: root });
    expect(built.endsWith("[Tactics loaded: 2 — auditable in src/prompts/tactics/]\n")).toBe(true);
  });

  test("5. tactics are loaded from the tactics subdir in deterministic order", () => {
    const root = mkRepo();
    writePrinciples(root, "P\n${TACTICS}\nQ\n");
    writeTactic(
      root,
      "z-last",
      `---
id: z-last
status: active
order: 5
---
ZBODY\n`,
    );
    writeTactic(
      root,
      "a-first",
      `---
id: a-first
status: active
order: 1
---
ABODY\n`,
    );
    writeTactic(
      root,
      "m-middle",
      `---
id: m-middle
status: active
order: 3
---
MBODY\n`,
    );
    const built = buildSystemPrompt({ sentinel: "X", repoRoot: root });
    const aIdx = built.indexOf("ABODY");
    const mIdx = built.indexOf("MBODY");
    const zIdx = built.indexOf("ZBODY");
    expect(aIdx).toBeGreaterThan(-1);
    expect(mIdx).toBeGreaterThan(aIdx);
    expect(zIdx).toBeGreaterThan(mIdx);
  });

  test("6. empty tactics dir yields principles-only prompt", () => {
    const root = mkRepo();
    writePrinciples(root, "Just principles, no tactics.\n");
    const built = buildSystemPrompt({ sentinel: "X", repoRoot: root });
    expect(built).toContain("Just principles, no tactics.");
    expect(built).toContain("[Tactics loaded: 0 —");
  });

  test("7. malformed frontmatter does not crash the loader (logs + skips)", () => {
    const root = mkRepo();
    writePrinciples(root, "P\n${TACTICS}\n");
    writeTactic(
      root,
      "broken",
      `---
this is not valid frontmatter
no colon at all
---
broken body
`,
    );
    writeTactic(
      root,
      "good",
      `---
id: good
status: active
order: 1
---
GOOD\n`,
    );
    // Capture console.warn so the test output stays clean.
    const originalWarn = console.warn;
    const warnings: string[] = [];
    console.warn = (...args: unknown[]) => {
      warnings.push(args.map(String).join(" "));
    };
    try {
      const tactics = loadActiveTactics(root);
      expect(tactics.length).toBe(1);
      expect(tactics[0]?.id).toBe("good");
      expect(warnings.some((w) => w.includes("malformed frontmatter"))).toBe(true);
      const built = buildSystemPrompt({ sentinel: "X", repoRoot: root });
      expect(built).toContain("GOOD");
      expect(built).not.toContain("broken body");
    } finally {
      console.warn = originalWarn;
    }
  });

  test("8. full migration prompt is structurally equivalent to the previous monolithic constant", () => {
    // SYSTEM_PROMPT_DEFAULT is now built via the modular loader, so we anchor
    // on (a) length within ±5% of the historical 4707-byte string and (b)
    // presence of every key forensic phrase.
    const built = SYSTEM_PROMPT_DEFAULT;
    const HISTORICAL_LENGTH = 4707;
    const lower = Math.floor(HISTORICAL_LENGTH * 0.95);
    const upper = Math.ceil(HISTORICAL_LENGTH * 1.05);
    expect(built.length).toBeGreaterThanOrEqual(lower);
    expect(built.length).toBeLessThanOrEqual(upper);

    const requiredFragments = [
      "You are BleedingAgent in autonomous coding mode",
      "Tool selection guide:",
      "Workflow:",
      "1. Read the task carefully.",
      "Reproduce the failure or required behaviour",
      "COMPILED-LANGUAGE GATE",
      "SCRATCH-DIR HYGIENE",
      "CRITICAL — pre-submit final-check pass",
      "(a) Re-read the original task instruction",
      "(b) For end-to-end flows",
      "(c) Confirm every output the task specified",
      "(d) **SUBPROCESS-PATH GATE**",
      "If any check disagrees, fix it BEFORE submitting",
      "echo BAG_TASK_COMPLETE",
      "Hard rules:",
      "Each bash call runs in a NEW subshell",
      "Available tools: `bash(command, timeout_sec?)`",
      "[Tactics loaded:",
    ];
    for (const fragment of requiredFragments) {
      expect(built).toContain(fragment);
    }

    // tactics injected MUST come from the on-disk tactics dir.
    const tactics = loadActiveTactics();
    const ids = tactics.map((t) => t.id);
    expect(ids).toContain("cleanup-before-submit");
    expect(ids).toContain("subprocess-path-gate");
    expect(ids).toContain("enumerate-deliverables");
    expect(ids).toContain("no-tmp-leak");
    expect(ids).toContain("pre-submit-final-check");
  });

  test("buildSystemPrompt rejects empty sentinel", () => {
    expect(() =>
      buildSystemPrompt({
        sentinel: "",
        principles: "Hello\n",
        tactics: [],
      }),
    ).toThrow();
  });

  test("buildSystemPrompt accepts injected tactics override (no I/O)", () => {
    const tactic: Tactic = {
      id: "injected",
      status: "active",
      body: "INJECTED-BODY\n",
      frontmatter: { id: "injected", status: "active", extra: {} },
      path: "/virtual/injected.md",
    };
    const built = buildSystemPrompt({
      sentinel: "Z",
      principles: "Top\n${TACTICS}\nBot\n",
      tactics: [tactic],
    });
    expect(built).toContain("Top");
    expect(built).toContain("INJECTED-BODY");
    expect(built).toContain("Bot");
    expect(built).toContain("[Tactics loaded: 1 —");
  });
});
