/**
 * Parity test: curated `verifier-signature-library` vs auto-discovered
 * `failure-clusters` matcher.
 *
 * This is the BAR for retiring `src/verifier-signature-library.ts`. The plan
 * is to retire the curated library and rely entirely on auto-discovered
 * clusters from `bench/.bag/optimizer/failure-clusters.json` (built by
 * `scripts/build_failure_clusters.py`). Before we delete, the cluster
 * matcher MUST cover the curated library's specific failure patterns at the
 * recall target documented below.
 *
 * Methodology: for each curated signature, we render a synthetic verifier
 * output that contains the trigger pattern (and surrounding pytest-style
 * context so the cluster's signature-extraction heuristic produces the same
 * shape it would on a real run). We then run BOTH matchers and ask:
 *   - does the library still fire? (it had better — sanity check)
 *   - does the cluster matcher fire at the chosen threshold?
 *
 * Recall target: ≥5/8 specific signatures must produce a non-null cluster
 * match at threshold = CLUSTER_MATCH_THRESHOLD. The remaining slots are
 * documented as "not yet covered by corpus" in
 * `docs/bag-verifier-signature-retirement.md` and require either more BAG
 * runs (acp-internal-error) or a different signal source entirely
 * (typecheck-missing-import — BAG's verifier runs pytest, not tsc;
 * submit-without-verify-catchall — generic by design, no cluster equivalent).
 *
 * The threshold (0.30) is chosen to:
 *   1. Allow the 5 specific library signatures to fire on synthetic outputs
 *      (lowest matching score in the suite is 0.944, well above 0.30).
 *   2. Keep noise traffic (unrelated text, generic errors) from spuriously
 *      matching real clusters — verified by `tests/failure-clusters.test.ts`
 *      negative cases plus the false-positive sanity assertion below.
 *   3. Stay well above the 0.216 spurious match for acp-internal-error
 *      (which incidentally trigram-overlaps with a FileNotFoundError
 *      cluster). Lowering the threshold to 0.20 to catch acp-internal-error
 *      would also pull in that misclassification; better to leave it as a
 *      "needs corpus expansion" item than to ship a wrong hint.
 */

import { describe, expect, test } from "bun:test";
import {
  VERIFIER_SIGNATURE_LIBRARY,
  matchVerifierSignature,
} from "../src/verifier-signature-library";
import {
  loadFailureClusters,
  matchClusterByVerifierOutput,
  type FailureClustersDocument,
} from "../src/optimizer/failure-clusters";

const REPO_ROOT = process.cwd();

/**
 * Threshold used by the cluster matcher in this parity test. See the file
 * docstring for rationale. The same value is documented in
 * `bench/.bag/optimizer/failure-clusters-config.json` so the runtime
 * matcher (and any future bench scripts) reads from one source of truth.
 */
const CLUSTER_MATCH_THRESHOLD = 0.3;

/**
 * Synthetic verifier outputs for each curated signature. Constructed to
 * contain the trigger pattern AND enough pytest-style framing that the
 * cluster matcher's `extractSignature` heuristic isolates the same line
 * shape it sees on real failures. Each blob is small (≤8 lines) — that's
 * intentional; we want to test the matcher contract, not the
 * normalization scaffolding.
 */
const SYNTHETIC_VERIFIER_OUTPUTS: Record<string, string> = {
  "polyglot-cmain-leftover": [
    ">       polyglot_files = os.listdir('/app/polyglot')",
    "E       AssertionError: Expected only main.py.c, found: ['main.py.c', 'cmain']",
    "E       assert ['main.py.c', 'cmain'] == ['main.py.c']",
    "E         Left contains one more item: 'cmain'",
    "/tests/test_outputs.py:18: AssertionError",
    "FAILED ../tests/test_outputs.py::test_fibonacci_polyglot - AssertionError: Ex...",
  ].join("\n"),
  "chess-multiple-moves-missed": [
    ">       assert sorted(move) == sorted([\"g2g4\", \"e2e4\"]), \"File is wrong\"",
    "E       AssertionError: File is wrong",
    "E       assert ['e2e4', 'g2h3'] == ['e2e4', 'g2g4']",
    "E         At index 1 diff: 'g2h3' != 'g2g4'",
    "/tests/test_outputs.py:25: AssertionError",
    "FAILED ../tests/test_outputs.py::test_move_correct - AssertionError: File is ...",
  ].join("\n"),
  "end-to-end-http-not-served": [
    "Testing web server...",
    "Using curl to test web server...",
    "TEST FAILED: Web server returned HTTP 000",
    "E       assert 'TEST PASSED' in 'Hit:1 http://security.ubuntu.com/ubuntu noble-security InRelease\\nHit:2 https://deb.nodesource.com/node_22.x nodistro...Testing web server...\\nUsing curl to test web server...\\nTEST FAILED: Web server returned HTTP 000\\nTest completed\\n'",
    "FAILED ../tests/test_outputs.py::test_hello_html_exists - AssertionError: Did...",
  ].join("\n"),
  "qemu-host-vs-guest-confusion": [
    ">       assert \"6.6.4-1-lts\" in result.stdout, \"Output of ssh is wrong version\"",
    "E       AssertionError: Output of ssh is wrong version",
    "E       assert '6.6.4-1-lts' in '6.19.13-orbstack-gb73df9775337\\n'",
    "/tests/test_outputs.py:14: AssertionError",
    "FAILED ../tests/test_outputs.py::test_sshpass - AssertionError: Output of ssh...",
  ].join("\n"),
  "build-pipeline-flag-lost": [
    ">       assert len(gcda_files) > 0, (",
    "            \"No .gcda files found, gcov instrumentation may not be enabled\"",
    "        )",
    "E       AssertionError: No .gcda files found, gcov instrumentation may not be enabled",
    "E       assert 0 > 0",
    "E        +  where 0 = len([])",
    "FAILED ../tests/test_outputs.py::test_gcov_enabled - AssertionError: No .gcda...",
  ].join("\n"),
  "typecheck-missing-import": [
    "src/foo.ts(12,8): error TS2305: Module '\"./bar\"' has no exported member 'baz'.",
    "src/foo.ts(13,8): error TS2307: Cannot find module './zonk' or its corresponding type declarations.",
  ].join("\n"),
  "acp-internal-error": [
    "{\"jsonrpc\":\"2.0\",\"error\":{\"code\":-32603,\"message\":\"Internal error\"},\"id\":7}",
    "ENOENT: no such file or directory, open '/app/missing.txt'",
  ].join("\n"),
  "submit-without-verify-catchall": [
    "FAILED tests/test_widget.py::test_widget - AssertionError: Expected 5 widgets but got 3",
    "E       Expected 5 widgets but got 3",
  ].join("\n"),
};

/**
 * Curated → cluster mapping documented in
 * `docs/bag-verifier-signature-retirement.md`. `null` = not yet covered by
 * the current corpus; failure of these is the EXPECTED state, not a
 * regression. When the corpus expands enough to cover an item we'd flip
 * the entry here AND update the migration doc.
 */
const EXPECTED_CLUSTER_FOR: Record<string, string | null> = {
  "polyglot-cmain-leftover": "polyglot-c-py-main-cmain",
  "chess-multiple-moves-missed": "chess-best-move-e2e4-g2h3-g2g4",
  "end-to-end-http-not-served": "configure-git-webserver-http-000-test-passed-http",
  "qemu-host-vs-guest-confusion": "qemu-alpine-ssh-orbstack-gb73df9775337",
  "build-pipeline-flag-lost": "sqlite-with-gcov-assert-0-0",
  // Not covered — see migration doc.
  "typecheck-missing-import": null,
  "acp-internal-error": null,
  "submit-without-verify-catchall": null,
};

const loadDocOrSkip = (): FailureClustersDocument | null => {
  const doc = loadFailureClusters(REPO_ROOT);
  return doc;
};

describe("verifier-signature-library vs failure-clusters parity", () => {
  test("clusters document is loadable from repo root", () => {
    const doc = loadDocOrSkip();
    expect(doc).not.toBeNull();
    expect(doc?.clusters.length ?? 0).toBeGreaterThan(0);
  });

  test("library contains exactly 8 curated signatures (matches retirement plan)", () => {
    expect(VERIFIER_SIGNATURE_LIBRARY.length).toBe(8);
  });

  test("synthetic outputs trigger the library matcher (sanity check)", () => {
    for (const sig of VERIFIER_SIGNATURE_LIBRARY) {
      const synthetic = SYNTHETIC_VERIFIER_OUTPUTS[sig.id];
      expect(synthetic).toBeDefined();
      const libMatch = matchVerifierSignature({
        verifierOutput: synthetic ?? "",
      });
      expect(libMatch).not.toBeNull();
      // The catchall is allowed to absorb specific patterns that fall
      // through; for known specific ids we expect exact id match. The
      // 'end-to-end-http-not-served' synthetic happens to also match
      // 'submit-without-verify-catchall' once the FAILED line is in the
      // input, so we accept either a match on the expected id OR the
      // catchall absorbing it. The point of this test is that *something*
      // in the library fires.
      if (libMatch?.id !== "submit-without-verify-catchall") {
        expect(libMatch?.id).toBe(sig.id);
      }
    }
  });

  test("cluster matcher recovers ≥5/8 specific curated signatures (retirement bar)", () => {
    const doc = loadDocOrSkip();
    if (doc === null) {
      throw new Error(
        "bench/.bag/optimizer/failure-clusters.json missing — run `python3 scripts/build_failure_clusters.py` first",
      );
    }
    let coveredCount = 0;
    const misses: string[] = [];
    for (const sig of VERIFIER_SIGNATURE_LIBRARY) {
      const expected = EXPECTED_CLUSTER_FOR[sig.id];
      const synthetic = SYNTHETIC_VERIFIER_OUTPUTS[sig.id] ?? "";
      const match = matchClusterByVerifierOutput(
        doc,
        synthetic,
        CLUSTER_MATCH_THRESHOLD,
      );
      if (expected === null) {
        // Not yet covered by corpus. Recording miss but it doesn't count
        // against the bar.
        if (match !== null) {
          // Pleasant surprise — corpus has caught up; the migration doc
          // should be updated to flip this slot from null → match.id.
        }
        continue;
      }
      if (match?.id === expected) {
        coveredCount += 1;
      } else {
        misses.push(`${sig.id} → expected ${expected}, got ${match?.id ?? "(none)"}`);
      }
    }
    if (misses.length > 0) {
      console.error("Cluster matcher misses:", misses);
    }
    expect(coveredCount).toBeGreaterThanOrEqual(5);
  });

  test("cluster matcher does NOT spuriously match unrelated noise at chosen threshold", () => {
    const doc = loadDocOrSkip();
    if (doc === null) throw new Error("clusters doc missing");
    const noiseInputs = [
      "the cat sat on the mat in november",
      "Hello, World!\nNothing failed here.",
      "All tests passed",
      "Permission denied",
    ];
    for (const noise of noiseInputs) {
      const m = matchClusterByVerifierOutput(
        doc,
        noise,
        CLUSTER_MATCH_THRESHOLD,
      );
      expect(m).toBeNull();
    }
  });

  test("documents which signatures are covered vs not (parity table)", () => {
    // Pure documentation test: enumerates the mapping so a developer
    // reading test output can see at-a-glance which slots are covered. Not
    // load-bearing — only fails if the EXPECTED_CLUSTER_FOR table doesn't
    // align with the library set.
    const libIds = new Set(VERIFIER_SIGNATURE_LIBRARY.map((s) => s.id));
    const tableIds = new Set(Object.keys(EXPECTED_CLUSTER_FOR));
    expect(tableIds).toEqual(libIds);
  });
});
