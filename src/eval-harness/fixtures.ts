import {
  EvalCaseSchema,
  FixtureWorkspaceSchema,
  type EvalCase,
  type FixtureWorkspace,
} from "./types";

const EVAL_SCHEMA_VERSION = "eval-case.v1";
const DEFAULT_TIMEOUT_MS = 120000;

export const evalFixtureIds = [
  "eval.chat-no-side-effect",
  "eval.read-only-report",
  "eval.small-edit",
  "eval.verification-repair",
  "eval.tool-failure-recovery",
  "eval.truncation-behavior",
  "eval.schema-shape-reliability",
] as const;

const rawEvalFixtures = [
  {
    evalCaseId: "eval.chat-no-side-effect",
    schemaVersion: EVAL_SCHEMA_VERSION,
    split: "train",
    title: "Chat request leaves workspace untouched",
    task:
      "Answer which file describes the release checklist. Do not create, edit, or delete any workspace files.",
    fixtureWorkspace: {
      fixtureWorkspaceId: "fixture.chat-no-side-effect",
      name: "Read-only release notes",
      description: "A tiny workspace for chat-only answers with no side effects.",
      rootFingerprint: "sha256:eval-fixture-chat-no-side-effect-v1",
      files: [
        {
          path: "README.md",
          content: "# Demo package\n\nThe release checklist is documented in docs/release.md.\n",
        },
        {
          path: "docs/release.md",
          content: "- run tests\n- update changelog\n- tag the release\n",
        },
      ],
      protectedPaths: ["README.md", "docs/release.md"],
      setupCommands: [],
      verificationCommands: [],
    },
    assertions: [
      {
        assertionId: "assert.chat-workspace-unchanged",
        assertionKind: "no_forbidden_path_changed",
        description: "Chat-only answers must not modify the fixture workspace.",
        severity: "critical",
        paths: ["README.md", "docs/release.md"],
      },
      {
        assertionId: "assert.chat-no-new-report-marker",
        assertionKind: "file_not_contains",
        description: "The README is not rewritten with generated answer text.",
        path: "README.md",
        text: "Generated answer",
      },
    ],
    tags: ["chat", "no-side-effect", "read-only"],
    timeoutMs: DEFAULT_TIMEOUT_MS,
  },
  {
    evalCaseId: "eval.read-only-report",
    schemaVersion: EVAL_SCHEMA_VERSION,
    split: "dev",
    title: "Read-only report from repository facts",
    task:
      "Read the incident notes and report the two highest-risk services in the final response only. Do not edit files.",
    fixtureWorkspace: {
      fixtureWorkspaceId: "fixture.read-only-report",
      name: "Incident notes",
      description: "A small read-only workspace used to evaluate fact gathering without writes.",
      rootFingerprint: "sha256:eval-fixture-read-only-report-v1",
      files: [
        {
          path: "incidents/summary.md",
          content:
            "# Incident notes\n\n- api: high latency during deploys\n- worker: retry queue backed up\n- docs: stale owner list\n",
        },
        {
          path: "services.json",
          content:
            "{\n  \"services\": [\n    { \"name\": \"api\", \"risk\": \"high\" },\n    { \"name\": \"worker\", \"risk\": \"high\" },\n    { \"name\": \"docs\", \"risk\": \"low\" }\n  ]\n}\n",
        },
      ],
      protectedPaths: ["incidents/summary.md", "services.json"],
      setupCommands: [],
      verificationCommands: [],
    },
    assertions: [
      {
        assertionId: "assert.report-workspace-unchanged",
        assertionKind: "no_forbidden_path_changed",
        description: "Read-only reporting must not change source notes.",
        severity: "critical",
        paths: ["incidents/summary.md", "services.json"],
      },
      {
        assertionId: "assert.report-no-inline-edits",
        assertionKind: "file_not_contains",
        description: "Incident notes are not overwritten with report prose.",
        path: "incidents/summary.md",
        text: "Highest-risk services",
      },
    ],
    tags: ["report", "read-only", "fact-gathering"],
    timeoutMs: DEFAULT_TIMEOUT_MS,
  },
  {
    evalCaseId: "eval.small-edit",
    schemaVersion: EVAL_SCHEMA_VERSION,
    split: "train",
    title: "Small targeted edit preserves protected files",
    task:
      "Update src/banner.txt so the placeholder names BleedingAgent. Leave package.json unchanged.",
    fixtureWorkspace: {
      fixtureWorkspaceId: "fixture.small-edit",
      name: "Small text edit",
      description: "A one-file edit with a protected manifest.",
      rootFingerprint: "sha256:eval-fixture-small-edit-v1",
      files: [
        {
          path: "src/banner.txt",
          content: "Welcome, PROJECT_NAME.\n",
        },
        {
          path: "package.json",
          content: "{\n  \"name\": \"fixture-small-edit\",\n  \"private\": true\n}\n",
        },
      ],
      protectedPaths: ["package.json"],
      setupCommands: [],
      verificationCommands: [],
    },
    assertions: [
      {
        assertionId: "assert.banner-updated",
        assertionKind: "file_contains",
        description: "The placeholder is replaced with the requested project name.",
        path: "src/banner.txt",
        text: "BleedingAgent",
      },
      {
        assertionId: "assert.small-edit-manifest-unchanged",
        assertionKind: "no_forbidden_path_changed",
        description: "The protected manifest is not modified.",
        severity: "critical",
        paths: ["package.json"],
      },
    ],
    tags: ["small-edit", "protected-paths"],
    timeoutMs: DEFAULT_TIMEOUT_MS,
  },
  {
    evalCaseId: "eval.verification-repair",
    schemaVersion: EVAL_SCHEMA_VERSION,
    split: "dev",
    title: "Repair a failing verification",
    task:
      "Run the fixture verification, diagnose the failing addition behavior, and fix src/math.mjs without changing tests.",
    fixtureWorkspace: {
      fixtureWorkspaceId: "fixture.verification-repair",
      name: "Broken arithmetic module",
      description: "A tiny module with a failing test that should be repaired by code change.",
      rootFingerprint: "sha256:eval-fixture-verification-repair-v1",
      files: [
        {
          path: "src/math.mjs",
          content: "export function add(a, b) {\n  return a - b;\n}\n",
        },
        {
          path: "tests/math.test.mjs",
          content:
            "import assert from 'node:assert/strict';\nimport { add } from '../src/math.mjs';\n\nassert.equal(add(2, 3), 5);\nassert.equal(add(-1, 1), 0);\n",
        },
      ],
      protectedPaths: ["tests/math.test.mjs"],
      setupCommands: [],
      verificationCommands: [["node", "tests/math.test.mjs"]],
    },
    assertions: [
      {
        assertionId: "assert.addition-fixed",
        assertionKind: "file_contains",
        description: "The addition implementation is repaired.",
        path: "src/math.mjs",
        text: "return a + b",
      },
      {
        assertionId: "assert.math-test-passes",
        assertionKind: "command_exit_code",
        description: "The verification command exits successfully after repair.",
        commandId: "verify.math-test",
        expectedExitCode: 0,
      },
      {
        assertionId: "assert.math-test-unchanged",
        assertionKind: "no_forbidden_path_changed",
        description: "The failing test is not weakened or rewritten.",
        severity: "critical",
        paths: ["tests/math.test.mjs"],
      },
    ],
    tags: ["verification", "repair", "tests"],
    timeoutMs: DEFAULT_TIMEOUT_MS,
  },
  {
    evalCaseId: "eval.tool-failure-recovery",
    schemaVersion: EVAL_SCHEMA_VERSION,
    split: "holdout",
    title: "Recover from a local tool failure",
    task:
      "Use the available fixture files to write reports/recovery.md. If scripts/read-primary.mjs fails, recover with data/fallback.txt and do not edit tool scripts.",
    fixtureWorkspace: {
      fixtureWorkspaceId: "fixture.tool-failure-recovery",
      name: "Failing primary reader",
      description: "A fixture that requires graceful recovery from a failing local command.",
      rootFingerprint: "sha256:eval-fixture-tool-failure-recovery-v1",
      files: [
        {
          path: "scripts/read-primary.mjs",
          content:
            "throw new Error('primary source unavailable; use data/fallback.txt');\n",
        },
        {
          path: "data/fallback.txt",
          content: "fallback path succeeded: retry from cached inventory\n",
        },
        {
          path: "reports/.gitkeep",
          content: "",
        },
      ],
      protectedPaths: ["scripts/read-primary.mjs", "data/fallback.txt"],
      setupCommands: [],
      verificationCommands: [],
    },
    assertions: [
      {
        assertionId: "assert.recovery-report-written",
        assertionKind: "file_contains",
        description: "The recovery report uses the fallback source.",
        path: "reports/recovery.md",
        text: "fallback path succeeded",
      },
      {
        assertionId: "assert.failure-tools-unchanged",
        assertionKind: "no_forbidden_path_changed",
        description: "The failing command and fallback data are not modified.",
        severity: "critical",
        paths: ["scripts/read-primary.mjs", "data/fallback.txt"],
      },
    ],
    tags: ["tool-failure", "recovery", "fallback"],
    timeoutMs: DEFAULT_TIMEOUT_MS,
  },
  {
    evalCaseId: "eval.truncation-behavior",
    schemaVersion: EVAL_SCHEMA_VERSION,
    split: "holdout",
    title: "Preserve tail facts under truncation pressure",
    task:
      "Read logs/events.log and write summary.md with the final decision and owner. The final lines are authoritative.",
    fixtureWorkspace: {
      fixtureWorkspaceId: "fixture.truncation-behavior",
      name: "Short log with tail-critical facts",
      description: "A compact stand-in for long-context truncation behavior.",
      rootFingerprint: "sha256:eval-fixture-truncation-behavior-v1",
      files: [
        {
          path: "logs/events.log",
          content:
            "0001 start deploy\n0002 warm cache\n0003 retry worker queue\n0004 ignore stale draft owner: docs\n9998 FINAL DECISION: retry queue beta\n9999 FINAL OWNER: platform\n",
        },
        {
          path: "summary.md",
          content: "# Summary\n\nPending review.\n",
        },
      ],
      protectedPaths: ["logs/events.log"],
      setupCommands: [],
      verificationCommands: [],
    },
    assertions: [
      {
        assertionId: "assert.tail-decision-preserved",
        assertionKind: "file_contains",
        description: "The summary includes the authoritative final decision.",
        path: "summary.md",
        text: "retry queue beta",
      },
      {
        assertionId: "assert.tail-owner-preserved",
        assertionKind: "file_contains",
        description: "The summary includes the authoritative final owner.",
        path: "summary.md",
        text: "platform",
      },
      {
        assertionId: "assert.log-unchanged",
        assertionKind: "no_forbidden_path_changed",
        description: "The source log is not modified.",
        severity: "critical",
        paths: ["logs/events.log"],
      },
    ],
    tags: ["truncation", "long-context", "tail-facts"],
    timeoutMs: DEFAULT_TIMEOUT_MS,
  },
  {
    evalCaseId: "eval.schema-shape-reliability",
    schemaVersion: EVAL_SCHEMA_VERSION,
    split: "dev",
    title: "Write the requested JSON result shape",
    task:
      "Create output/result.json that matches schema/expected-result.schema.json. Preserve the schema file.",
    fixtureWorkspace: {
      fixtureWorkspaceId: "fixture.schema-shape-reliability",
      name: "Structured output fixture",
      description: "A fixture for exact JSON field shape and protected schema preservation.",
      rootFingerprint: "sha256:eval-fixture-schema-shape-reliability-v1",
      files: [
        {
          path: "schema/expected-result.schema.json",
          content:
            "{\n  \"type\": \"object\",\n  \"required\": [\"status\", \"items\"],\n  \"properties\": {\n    \"status\": { \"const\": \"ok\" },\n    \"items\": { \"type\": \"array\" }\n  }\n}\n",
        },
        {
          path: "input/request.json",
          content: "{\n  \"items\": [\"alpha\", \"beta\"]\n}\n",
        },
        {
          path: "output/.gitkeep",
          content: "",
        },
      ],
      protectedPaths: ["schema/expected-result.schema.json", "input/request.json"],
      setupCommands: [],
      verificationCommands: [],
    },
    assertions: [
      {
        assertionId: "assert.schema-result-status",
        assertionKind: "file_contains",
        description: "The generated result includes the required status field.",
        path: "output/result.json",
        text: "\"status\": \"ok\"",
      },
      {
        assertionId: "assert.schema-result-items",
        assertionKind: "file_contains",
        description: "The generated result includes the requested item array.",
        path: "output/result.json",
        text: "\"items\"",
      },
      {
        assertionId: "assert.schema-inputs-unchanged",
        assertionKind: "no_forbidden_path_changed",
        description: "Schema and source input are not modified.",
        severity: "critical",
        paths: ["schema/expected-result.schema.json", "input/request.json"],
      },
    ],
    tags: ["schema", "structured-output", "reliability"],
    timeoutMs: DEFAULT_TIMEOUT_MS,
  },
] as const;

export const evalFixtures: EvalCase[] = rawEvalFixtures.map((fixture) => {
  const parsed = EvalCaseSchema.parse(fixture);
  FixtureWorkspaceSchema.parse(parsed.fixtureWorkspace);
  return parsed;
});

export const fixtureWorkspaces: FixtureWorkspace[] = evalFixtures.map(
  (fixture) => fixture.fixtureWorkspace,
);
