import { describe, expect, test } from "bun:test";
import {
  matchVerifierSignature,
  renderHintForRetry,
  VERIFIER_SIGNATURE_LIBRARY,
} from "../src/verifier-signature-library";

describe("VERIFIER_SIGNATURE_LIBRARY", () => {
  test("has at least 8 curated signatures", () => {
    expect(VERIFIER_SIGNATURE_LIBRARY.length).toBeGreaterThanOrEqual(8);
  });

  test("every signature has non-empty patterns and fixHint", () => {
    for (const sig of VERIFIER_SIGNATURE_LIBRARY) {
      expect(sig.patterns.length).toBeGreaterThan(0);
      expect(sig.fixHint.trim().length).toBeGreaterThan(20);
      expect(sig.id.trim().length).toBeGreaterThan(0);
    }
  });

  test("ids are unique", () => {
    const ids = VERIFIER_SIGNATURE_LIBRARY.map((s) => s.id);
    expect(new Set(ids).size).toBe(ids.length);
  });
});

describe("matchVerifierSignature", () => {
  test("matches polyglot cmain leftover from real run #5 verifier output", () => {
    const verifierOutput =
      "AssertionError: Expected only main.py.c, found: ['main.py.c', 'cmain']\nE       assert ['main.py.c', 'cmain'] == ['main.py.c']";
    const sig = matchVerifierSignature({ verifierOutput });
    expect(sig).not.toBeNull();
    expect(sig?.id).toBe("polyglot-cmain-leftover");
  });

  test("matches chess multiple-moves miss from real run #8", () => {
    const verifierOutput =
      ">       assert sorted(move) == sorted([\"g2g4\", \"e2e4\"]), \"File is wrong\"\nE       AssertionError: File is wrong";
    const sig = matchVerifierSignature({ verifierOutput });
    expect(sig).not.toBeNull();
    expect(sig?.id).toBe("chess-multiple-moves-missed");
  });

  test("matches end-to-end HTTP 404 from configure-git", () => {
    const verifierOutput =
      "Testing web server...\nUsing curl to test web server...\n❌ TEST FAILED: Web server returned HTTP 404\nTest completed";
    const sig = matchVerifierSignature({ verifierOutput });
    expect(sig).not.toBeNull();
    expect(sig?.id).toBe("end-to-end-http-not-served");
  });

  test("matches qemu host-vs-guest kernel mismatch", () => {
    const verifierOutput =
      "expected guest kernel 6.6.4-1-lts but got: Linux 6.19.13-orbstack-foo";
    const sig = matchVerifierSignature({ verifierOutput });
    expect(sig).not.toBeNull();
    expect(sig?.id).toBe("qemu-host-vs-guest-confusion");
  });

  test("matches build-pipeline gcov flag loss", () => {
    const verifierOutput = "gcov: cannot find /tmp/build/foo.gcda";
    const sig = matchVerifierSignature({ verifierOutput });
    expect(sig).not.toBeNull();
    expect(sig?.id).toBe("build-pipeline-flag-lost");
  });

  test("falls through to submit-without-verify catchall on generic AssertionError", () => {
    const verifierOutput =
      "FAILED tests/test_outputs.py::test_widget - AssertionError: Expected 5 widgets but got 3";
    const sig = matchVerifierSignature({ verifierOutput });
    expect(sig).not.toBeNull();
    expect(sig?.id).toBe("submit-without-verify-catchall");
  });

  test("returns null on totally unrelated text", () => {
    const sig = matchVerifierSignature({
      verifierOutput: "the cat sat on the mat in november",
    });
    expect(sig).toBeNull();
  });

  test("returns null on empty input", () => {
    const sig = matchVerifierSignature({});
    expect(sig).toBeNull();
  });
});

describe("renderHintForRetry", () => {
  test("includes signature id and fix hint", () => {
    const sig = VERIFIER_SIGNATURE_LIBRARY[0];
    if (!sig) throw new Error("library empty");
    const rendered = renderHintForRetry(sig);
    expect(rendered).toContain(sig.id);
    expect(rendered).toContain(sig.fixHint);
    expect(rendered).toContain("BAG history library");
  });
});
