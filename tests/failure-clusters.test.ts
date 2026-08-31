import { mkdtempSync, mkdirSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, test } from "bun:test";
import {
  loadFailureClusters,
  matchClusterByVerifierOutput,
  __test,
  type FailureClustersDocument,
} from "../src/optimizer/failure-clusters";

const buildSyntheticDocument = (): FailureClustersDocument => ({
  generatedAt: "2026-05-01T12:00:00Z",
  totalFailures: 22,
  clusters: [
    {
      id: "polyglot-cmain-leftover",
      name: "polyglot-cmain-leftover",
      size: 5,
      trialIds: ["polyglot-c-py__a", "polyglot-c-py__b", "polyglot-c-py__c", "polyglot-c-py__d", "polyglot-c-py__e"],
      signature: "assert ['main.py.c', 'cmain'] == ['main.py.c']",
      tasks: ["polyglot-c-py"],
      firstSeen: "2026-04-30__10-00-00",
      lastSeen: "2026-05-01__09-30-00",
      exemplarVerifierExcerpt: "E       AssertionError\nE       assert ['main.py.c', 'cmain'] == ['main.py.c']\n",
    },
    {
      id: "missing-app-artifact",
      name: "missing-app-artifact",
      size: 13,
      trialIds: Array.from({ length: 13 }, (_, i) => `mixed-task__${i}`),
      signature: "FileNotFoundError: [Errno 2] No such file or directory: '/app/report.jsonl'",
      tasks: ["build-cython-ext", "fix-code-vulnerability", "polyglot-c-py"],
      firstSeen: "2026-04-29__08-00-00",
      lastSeen: "2026-05-01__22-00-00",
      exemplarVerifierExcerpt: "E       FileNotFoundError: [Errno 2] No such file or directory: '/app/report.jsonl'\n",
    },
    {
      id: "git-webserver-http-000",
      name: "git-webserver-http-000",
      size: 4,
      trialIds: ["configure-git-webserver__a", "configure-git-webserver__b"],
      signature: "assert 'TEST PASSED' in 'Web server returned HTTP 000'",
      tasks: ["configure-git-webserver"],
      firstSeen: "2026-04-30__18-00-00",
      lastSeen: "2026-05-01__20-00-00",
      exemplarVerifierExcerpt: "E       AssertionError: Did not pass test\n",
    },
  ],
});

describe("failure-clusters: matchClusterByVerifierOutput", () => {
  test("matches a verifier output that mirrors a known cluster signature", () => {
    const doc = buildSyntheticDocument();
    const verifier = `
=================================== FAILURES ===================================
___________________________ test_fibonacci_polyglot ____________________________

>       assert sorted(polyglot_files) == ['main.py.c']
E       AssertionError
E       assert ['main.py.c', 'cmain'] == ['main.py.c']
E         Left contains one more item: 'cmain'

/tests/test_outputs.py:18: AssertionError
=========================== short test summary info ============================
FAILED ../tests/test_outputs.py::test_fibonacci_polyglot - AssertionError
============================== 1 failed in 0.04s ===============================
`;
    const match = matchClusterByVerifierOutput(doc, verifier);
    expect(match).not.toBeNull();
    expect(match?.id).toBe("polyglot-cmain-leftover");
  });

  test("matches the FileNotFoundError cluster across different missing artifacts", () => {
    const doc = buildSyntheticDocument();
    const verifier = `
>       polyglot_files = os.listdir("/app/polyglot")
E       FileNotFoundError: [Errno 2] No such file or directory: '/app/polyglot'

/tests/test_outputs.py:17: FileNotFoundError
FAILED ../tests/test_outputs.py::test_fibonacci_polyglot - FileNotFoundError
`;
    const match = matchClusterByVerifierOutput(doc, verifier);
    expect(match).not.toBeNull();
    expect(match?.id).toBe("missing-app-artifact");
  });

  test("returns null on totally unrelated text", () => {
    const doc = buildSyntheticDocument();
    const match = matchClusterByVerifierOutput(
      doc,
      "the cat sat on the mat and watched birds outside the window",
    );
    expect(match).toBeNull();
  });

  test("returns null when document has no clusters", () => {
    const empty: FailureClustersDocument = {
      generatedAt: "x",
      totalFailures: 0,
      clusters: [],
    };
    const verifier = "E   AssertionError: anything";
    expect(matchClusterByVerifierOutput(empty, verifier)).toBeNull();
  });

  test("returns null when verifier output is blank", () => {
    expect(matchClusterByVerifierOutput(buildSyntheticDocument(), "")).toBeNull();
    expect(matchClusterByVerifierOutput(buildSyntheticDocument(), "\n   \n")).toBeNull();
  });
});

describe("failure-clusters: extractSignature heuristic", () => {
  test("prefers the exception-header E line over continuation hints", () => {
    const sig = __test.extractSignature(
      [
        "_______________ test_move_correct ________________",
        "E       AssertionError: File is wrong",
        "E       assert ['e2e4'] == ['e2e4', 'g2g4']",
        "E         Right contains one more item: 'g2g4'",
        "E         Use -v to get more diff",
        "/tests/test_outputs.py:25: AssertionError",
        "FAILED ../tests/test_outputs.py::test_move_correct - AssertionError: File is ...",
      ].join("\n"),
    );
    // Both 'AssertionError: File is wrong' and the assert-detail line match
    // the header regex; the heuristic picks the LAST header-matching E line
    // because it carries the concrete diff, not the bare exception name.
    expect(sig).toContain("e2e4");
    expect(sig).not.toContain("Use -v");
  });

  test("falls back to the FAILED line when no E lines exist", () => {
    const sig = __test.extractSignature(
      "rootdir: /tests\ncollected 1 item\nFAILED ../tests/test_outputs.py::test_repo_cloned - AssertionError: missing\n",
    );
    expect(sig).toContain("FAILED");
    expect(sig).toContain("test_repo_cloned");
  });
});

describe("failure-clusters: loadFailureClusters", () => {
  test("returns null when the file does not exist", () => {
    const tmp = mkdtempSync(join(tmpdir(), "fc-"));
    try {
      expect(loadFailureClusters(tmp)).toBeNull();
    } finally {
      rmSync(tmp, { recursive: true, force: true });
    }
  });

  test("loads and normalizes a real on-disk document", () => {
    const tmp = mkdtempSync(join(tmpdir(), "fc-"));
    try {
      const dir = join(tmp, "bench", ".bag", "optimizer");
      mkdirSync(dir, { recursive: true });
      const raw = {
        generated_at: "2026-05-01T00:00:00Z",
        total_failures: 1,
        clusters: [
          {
            id: "x",
            name: "x",
            size: 1,
            trial_ids: ["t1"],
            signature: "E   AssertionError: x",
            tasks: ["taskA"],
            first_seen: "2026-05-01",
            last_seen: "2026-05-01",
            exemplar_verifier_excerpt: "...",
          },
        ],
      };
      writeFileSync(join(dir, "failure-clusters.json"), JSON.stringify(raw));
      const doc = loadFailureClusters(tmp);
      expect(doc).not.toBeNull();
      expect(doc?.clusters[0]?.id).toBe("x");
      expect(doc?.clusters[0]?.trialIds).toEqual(["t1"]);
      expect(doc?.totalFailures).toBe(1);
    } finally {
      rmSync(tmp, { recursive: true, force: true });
    }
  });
});
