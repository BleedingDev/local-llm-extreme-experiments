/**
 * @deprecated SOFT-DEPRECATED 2026-05-02 — slated for removal after the
 * auto-discovered failure-cluster matcher (`src/optimizer/failure-clusters.ts`)
 * proves out at parity in production.
 *
 * Verifier-Signature Library — static curated mapping of (verifier complaint
 * pattern) → (fix hint), distilled from `docs/bag-failure-pattern-digest.md`,
 * `docs/polyglot-success-postmortem.md`, and `docs/bag-vs-opus-direct.md`.
 *
 * Retirement plan: see `docs/bag-verifier-signature-retirement.md`. As of
 * 2026-05-02 the cluster matcher already covers 5/8 of these signatures with
 * the 143-failure corpus; the remaining 3 (typecheck-missing-import,
 * acp-internal-error, submit-without-verify-catchall) are either out of the
 * pytest-shaped corpus's reach or are intentional generic catch-alls.
 *
 * Runtime ordering: as of 2026-05-02, `autonomous-coding-turn.ts` calls the
 * cluster matcher FIRST and falls back to this library. The `retry_hint`
 * trace entry records `source: "cluster" | "library" | "both" | "none"` so
 * we can audit retirement readiness — when `library` hit-rate stays <5% of
 * total retry-hint fires over 30 BAG runs, this file can be deleted.
 *
 * TODO: delete after 30 runs of cluster-primary, if hit-rate stays equivalent.
 */

export type VerifierSignatureMatchScope =
  | "task-name"
  | "verifier-output"
  | "instruction"
  | "any";

export type VerifierSignature = {
  /** Stable identifier; used in telemetry. */
  id: string;
  /** Patterns to match. Each is a RegExp. ANY match means hit. */
  patterns: ReadonlyArray<RegExp>;
  /** Where to look in the matching context. */
  scope: VerifierSignatureMatchScope;
  /** Human-readable fix hint to inject into the retry prompt. */
  fixHint: string;
  /** Provenance — which trial / report seeded this entry. */
  evidence: {
    source: string;
    trialIds: ReadonlyArray<string>;
  };
};

export const VERIFIER_SIGNATURE_LIBRARY: ReadonlyArray<VerifierSignature> = [
  {
    id: "polyglot-cmain-leftover",
    patterns: [
      /Expected only main\.py\.c, found \[.*['"]cmain['"].*\]/i,
      /AssertionError:\s*Expected only/i,
    ],
    scope: "verifier-output",
    fixHint:
      "The verifier asserts an exact file list. Run `rm -f /app/polyglot/cmain` (and any other compiled binary you produced during testing) before submitting. Verify with `ls /app/polyglot` — it must contain only `main.py.c`.",
    evidence: {
      source: "docs/polyglot-success-postmortem.md + docs/bag-failure-pattern-digest.md",
      trialIds: [
        "polyglot-c-py__dveNqWQ",
        "polyglot-c-py__CegYrmJ",
        "polyglot-c-py__RHtUSKX",
        "polyglot-c-py__qyjazbq",
      ],
    },
  },
  {
    id: "chess-multiple-moves-missed",
    patterns: [
      /assert sorted\(\[.*?\]\) == sorted\(\[.*?\]\)/,
      /File is wrong/,
      /\bmove\.txt\b.*assert/i,
    ],
    scope: "verifier-output",
    fixHint:
      "The task requires ALL winning moves on separate lines, not just one. Re-analyze the position more carefully (use stockfish via `apt-get install -y stockfish && python3 -c 'import chess.engine; ...'` to enumerate every move that delivers mate-in-1 or mate-in-N). Write each winning move on its own line in `/app/move.txt`. Then re-verify by reading the file and counting lines.",
    evidence: {
      source: "docs/bag-vs-opus-direct.md (run #8 chess regression)",
      trialIds: ["chess-best-move__3hGaYQK"],
    },
  },
  {
    id: "end-to-end-http-not-served",
    patterns: [
      /TEST FAILED: Web server returned HTTP 404/i,
      /TEST FAILED: Web server returned HTTP 5\d\d/i,
      /curl: \(7\) Failed to connect/i,
      /assert "TEST PASSED" in/i,
    ],
    scope: "verifier-output",
    fixHint:
      "The HTTP server is up but isn't serving the right content for the verifier's literal curl. Trace the request path end-to-end: (1) check the post-receive git hook copies pushed content to the served directory, (2) verify the document root maps the URL the verifier curls, (3) literally run the verifier's command (e.g. `curl -s http://localhost:8080/hello.html`) and inspect the body before submitting. Do not submit on `service is up` alone — the verifier checks RESPONSE BODY.",
    evidence: {
      source: "docs/bag-failure-pattern-digest.md (configure-git-webserver)",
      trialIds: ["configure-git-webserver__V8v2hGJ"],
    },
  },
  {
    id: "qemu-host-vs-guest-confusion",
    patterns: [
      /uname.*orbstack/i,
      /6\.\d+\.\d+-orbstack/i,
      /kernel.*does not match expected/i,
    ],
    scope: "verifier-output",
    fixHint:
      "You're connected to the orbstack host kernel, not the QEMU guest. The SSH session must reach the Alpine VM, not the docker host. Verify port forwarding is correct (likely a localhost tunnel into qemu-system-x86_64), and confirm `ssh user@localhost -p <vm-port> uname -a` reports the GUEST kernel (e.g. `6.6.4-1-lts`), not orbstack. Re-check the VM is actually booting before authenticating.",
    evidence: {
      source: "docs/bag-failure-pattern-digest.md (qemu-alpine-ssh)",
      trialIds: ["qemu-alpine-ssh__p6NuDvN"],
    },
  },
  {
    id: "build-pipeline-flag-lost",
    patterns: [
      /gcov: cannot find/i,
      /\.gcda.*not found/i,
      /enable-gcov.*not.*set/i,
      /flag .* not propagat/i,
    ],
    scope: "verifier-output",
    fixHint:
      "Build flags must propagate from configure to compile and stay in the installed binary. Use a SINGLE chained bash invocation: `cd /src && ./configure --enable-gcov && make -j && make install` — separate calls in different bash subshells lose state. After install, confirm flags survived: `nm /usr/local/bin/<binary> | grep gcov` or run a coverage smoke. Treat this as monolithic, not three independent steps.",
    evidence: {
      source: "docs/bag-failure-pattern-digest.md (sqlite-with-gcov misroute)",
      trialIds: ["sqlite-with-gcov__qqr3VjQ"],
    },
  },
  {
    id: "typecheck-missing-import",
    patterns: [
      /Cannot find module ['"]\.\//,
      /is not exported by/,
      /TS2305:/,
      /TS2307:/,
    ],
    scope: "verifier-output",
    fixHint:
      "TypeScript can't resolve an import. Run `node_modules/.bin/tsc -p tsconfig.json --noEmit 2>&1 | head -5` to see the unresolved name, then either (a) add the missing `export` to the source module, or (b) fix the import path. Watch for `exactOptionalPropertyTypes` mismatches when assigning `undefined` to optional fields.",
    evidence: {
      source: "Codex iterations on src/llm.ts + acp-agent.ts",
      trialIds: [],
    },
  },
  {
    id: "acp-internal-error",
    patterns: [
      /error:Internal error/i,
      /code: -32603/,
      /ENOENT.*\/app\//i,
    ],
    scope: "verifier-output",
    fixHint:
      "The ACP session crashed mid-flight. Reproduce locally with `tsx scripts/bag_acp_run.ts \"<task>\" --workdir <X>` and capture the stderr trace before retrying. Most common cause: a tool callback threw because a referenced file path didn't exist or a permission gate fired.",
    evidence: {
      source: "docs/bag-failure-pattern-digest.md (5/50 ACP crashes)",
      trialIds: [],
    },
  },
  {
    id: "submit-without-verify-catchall",
    patterns: [
      /AssertionError(?!.*Expected only main\.py\.c)/,
      /FAILED .*\.py::test_/,
      /\bExpected\b.*\bgot\b/,
      /returned exit code (?!0)/,
    ],
    scope: "verifier-output",
    fixHint:
      "Verifier rejected your submission. Before retry: (1) re-read the original task instruction line by line, watching for plurals (\"all moves\"), edge cases (\"if multiple X\"), and end-to-end flows (\"then `curl http://...` returns Y\"). (2) Literally run the verification command from the task description (curl, cat, diff, python -c, etc.) and inspect output. (3) Fix any disagreement BEFORE submitting again. 70% of past failures (35/50 in run-history) submitted without running the literal verifier.",
    evidence: {
      source: "docs/bag-failure-pattern-digest.md (dominant 70% failure cluster)",
      trialIds: [],
    },
  },
];

/**
 * Match a verifier failure context against the library and return the
 * highest-priority match. Library order = priority (specific signatures first;
 * the catch-all `submit-without-verify-catchall` is intentionally last).
 */
export const matchVerifierSignature = (context: {
  taskName?: string;
  verifierOutput?: string;
  instructionText?: string;
}): VerifierSignature | null => {
  for (const sig of VERIFIER_SIGNATURE_LIBRARY) {
    const haystack =
      sig.scope === "task-name"
        ? context.taskName ?? ""
        : sig.scope === "instruction"
          ? context.instructionText ?? ""
          : sig.scope === "any"
            ? `${context.taskName ?? ""}\n${context.instructionText ?? ""}\n${context.verifierOutput ?? ""}`
            : context.verifierOutput ?? "";
    if (haystack.length === 0) continue;
    for (const pattern of sig.patterns) {
      if (pattern.test(haystack)) {
        return sig;
      }
    }
  }
  return null;
};

/**
 * Render a retry hint message body. The autonomous-coding-turn retry path
 * prepends this before the verifier output dump.
 */
export const renderHintForRetry = (
  signature: VerifierSignature,
  _context?: {
    taskName?: string;
    verifierOutput?: string;
  },
): string =>
  [
    "[BAG history library — past trials of similar tasks failed for this reason]:",
    `  signature: ${signature.id}`,
    `  fix: ${signature.fixHint}`,
    `  evidence: ${signature.evidence.source}`,
  ].join("\n");
