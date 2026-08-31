/**
 * Pre-submit self-check — generic instruction-comprehension audit.
 *
 * Motivation: the agent declared `BAG_TASK_COMPLETE` but had silently dropped
 * one of the explicit numbered deliverables in the original instruction (e.g.
 * "create /app/report.jsonl"). The existing `verifyAfterSubmit` callback
 * fires only when probes were extracted from the instruction; tasks that
 * miss probe extraction get no second look. This module is the universal
 * fallback: BEFORE we accept a submit sentinel as terminal, we re-read the
 * original instruction and a compact summary of the bash trace via the
 * cheap `local` role and ask "is every requirement met?". If the
 * model returns `complete: false` with a list of `missing` items, the
 * autonomous-coding-turn caller injects a synthetic feedback message and
 * gives the agent another attempt.
 *
 * Design constraints (per BAG memo):
 *   1. Generic only — no task-specific keywords, no regex over instruction
 *      patterns. Pure LLM-driven instruction comprehension.
 *   2. Pass when uncertain — any throw / parse failure / suspicious payload
 *      yields `{complete: true, missing: []}`. We never block a submit on a
 *      non-functional model call.
 *   3. Bounded — the instruction is truncated to the first 3000 chars and
 *      the bash trace tail to the last 8 calls (cmd + first 200 chars of
 *      output + exit code). One local-role call per attempt, max 600 tokens.
 *   4. JSON-only contract — `{complete: bool, missing: string[]}`.
 */

import type { LlmRouter } from "./llm";
import { parseJsonObject } from "./llm";
import { auditScratchHygiene, renderScratchHygieneBlock } from "./scratch-hygiene";
import { detectAnswerWobble, renderWobbleScanBlock } from "./audit/answer-wobble";
import { DEFAULT_PATH_PROFILE, type PathProfile } from "./types";

/** Hard cap on the instruction text fed to the auditor model. */
export const SELF_CHECK_INSTRUCTION_MAX_CHARS = 3000;

/** Hard cap on how many recent bash calls we summarise. */
export const SELF_CHECK_BASH_TAIL_MAX_CALLS = 8;

/** Hard cap on the per-call output excerpt fed to the auditor. */
export const SELF_CHECK_BASH_OUTPUT_MAX_CHARS = 200;

/** Hard cap on how many missing items we propagate (defence-in-depth). */
export const SELF_CHECK_MAX_MISSING_ITEMS = 12;

/** Soft cap on a single missing-item string before we truncate. */
export const SELF_CHECK_MISSING_ITEM_MAX_CHARS = 280;

/**
 * Build the pre-submit self-check auditor system prompt from a PathProfile.
 *
 * The prompt's SUBPROCESS-PATH GATE wording cites the default subprocess
 * `PATH` (colon-joined, used by `bash -c 'unset PATH; PATH=... command -v X'`)
 * — that string is interpolated from `pathProfile.systemPathDirs`. The
 * descriptive comma-list enumeration of "well-known default-PATH locations"
 * (`/usr/local/bin, /usr/bin, /bin, /sbin`) stays literal because it is
 * prose listing common Linux conventions rather than the authoritative
 * subprocess PATH. Deployments that need to override the descriptive list
 * can supply a custom `systemPromptOverride` via the executor config.
 *
 * Exposed for tests so the path-profile flow-through can be asserted
 * without spinning a full coding turn.
 */
export const buildSelfCheckSystemPrompt = (
  pathProfile: PathProfile = DEFAULT_PATH_PROFILE,
): string => {
  const pathJoined = pathProfile.systemPathDirs.join(":");
  return `\
You are auditing whether an autonomous coding agent has completed every \
requirement of a task. Read the original instruction and the bash session \
summary. Return JSON: {"complete": bool, "missing": string[]}.

DELIVERABLE-ENUMERATION DISCIPLINE — before you decide "complete: true", \
mentally enumerate EVERY discrete deliverable named in the instruction \
(numbered list items, imperative sentences, "create / produce / write X" \
clauses, "report file at PATH must contain Y" clauses, side-effects like \
"X must be available in PATH"). For each one, confirm by SPECIFIC EVIDENCE \
in the bash trace that it was satisfied — do not infer from "tests passed" \
or "build succeeded" that all numbered deliverables were produced. \
Tests passing is NECESSARY but not SUFFICIENT: a passing test suite says \
the agent did not break the existing code, NOT that the agent created \
every artifact the instruction demanded. If you cannot point to a bash \
command that produced deliverable D, list D in "missing" with a verbatim \
quote of the instruction text. When in doubt, FLAG — better one false \
positive that gives the agent another iteration than a confident pass on \
a missed step.

Check each of these failure modes and add concrete entries to "missing" when \
any apply:

1. UNMET DELIVERABLE — an explicit step in the instruction (numbered list, \
   imperative sentence, "must produce X") whose artifact is not visible in \
   the bash trace. List the deliverable verbatim.

   Pay extra attention to PLURALS and CONDITIONAL OUTPUTS in the \
   instruction: words like "all", "each", "every", "multiple", "list", "if \
   there are multiple X, output them all", "for each Y do Z". When the \
   instruction has such a quantifier and the agent's deliverable contains a \
   single item, ALWAYS flag this — explicitly ask "did the agent verify \
   they enumerated every case the instruction asked for?". The agent often \
   stops at the first solution they find when the instruction wanted all of \
   them. Quote the plural/conditional fragment from the instruction and \
   compare to what the agent wrote.

2. FORMAT MISMATCH — a deliverable exists but its structure / key names / \
   exact strings do not match what the instruction EXPLICITLY specified. Only \
   fire this when the instruction names the exact key, exact string, exact \
   shape, and the agent's output diverges. DO NOT fire on compiler warnings, \
   stylistic preferences, "could be cleaner", or anything the instruction did \
   not literally pin. Common true positives: instruction says key "file_path" \
   but agent wrote "file"; instruction says array of objects but agent wrote \
   single object; instruction quotes an exact output string but the agent's \
   output differs. Quote the literal instruction text and the literal agent \
   output side by side. If you cannot quote the instruction's exact \
   requirement, do not raise this.

3. STRAY ARTIFACT — the agent created files (compiled binaries, temp dirs, \
   build outputs) that are NOT requested by the instruction and that may \
   violate "expected only X" assertions. The bash trace shows compile / build \
   commands without a matching cleanup. Name the leftover file(s).

4. SHELL-SESSION-ONLY FIX / NON-STANDARD INSTALL PATH — the verifier will \
   spawn FRESH subprocesses (e.g. subprocess.run(['X', ...]) from Python \
   tests) that have ONLY the system default PATH (${pathJoined}) \
   and inherit NONE of the agent's shell state. Fire this aggressively under \
   either condition:

   (a) STATE-MUTATION CONDITION — the trace contains 'export ', 'source ', \
   'alias ', a virtualenv activate, or a working-directory-dependent \
   invocation that succeeded.

   (b) NON-STANDARD INSTALL PATH CONDITION — the instruction says a tool \
   must be 'in PATH' / 'available system-wide' / 'callable as X' / \
   'installed' / 'compiled and ready to use' AND the agent built or placed \
   the binary at a non-default path (anything outside /usr/local/bin, /usr/bin, \
   /bin, /sbin). If the binary is at /app/X/X, /opt/X, /home/.../X, /tmp/X, \
   etc., the verifier's fresh subprocess won't find it.

   Either condition: ALWAYS flag UNLESS the trace also shows a system-wide \
   persistence step — a symlink into /usr/local/bin or /usr/bin (ln -s ... \
   /usr/local/bin/), a copy (cp X /usr/local/bin/), an apt/pip/npm/cargo \
   install of a published package, 'make install' to a system prefix, a file \
   written to /etc/profile or /etc/bash.bashrc, or placement in a directory \
   that was on the default PATH BEFORE the agent's modifications. A \
   successful 'which X' from the agent's bash is NOT sufficient evidence: \
   that bash inherited the agent's session, fresh subprocesses do not. The \
   ground-truth check is: bash -c 'unset PATH; PATH=${pathJoined} \
   command -v X'. If the trace shows that succeeding, the install is real; \
   otherwise flag it.

5. SUBMITTED WITHOUT FULL VERIFICATION — fire under any of these conditions:

   (a) EXPLICIT VERBATIM MISMATCH: the instruction quotes a verification \
   command in backticks or a code-fence block (e.g. \`curl http://server:8080/hello.html\`, \
   \`git clone user@server:/git/server\`, \`pytest -rA\`). The bash trace must \
   contain that EXACT string — same hostname, same port, same path, same \
   protocol, same flags. If the agent ran a SIMILAR-LOOKING but \
   non-equivalent variant (e.g. \`curl http://localhost:8080/\` instead of \
   \`curl http://server:8080/hello.html\`, or \`git clone /git/server\` \
   instead of \`git clone user@server:/git/server\`), FIRE this — the \
   harbor verifier likely uses the literal string from the instruction, \
   and a substitute does not prove the contract holds. Quote both versions \
   side by side: instruction wrote X, agent ran Y.

   (b) IMPLICIT TEST-SUITE MENTION: the instruction mentions a test \
   directory, a test runner, or "tests should pass" / "all tests pass" / \
   "run the test suite" / "the existing tests must still pass" without \
   naming a specific command. In that case the agent must run THE FULL \
   test suite that the project ships (typically "pytest -rA" at repo root, \
   "cargo test", "go test ./...", "npm test", "make test" — pick whichever \
   the project's tooling implies). Running ONE narrow test file or a \
   curated subset is INSUFFICIENT. Flag this when the bash trace shows only \
   targeted/narrow test runs but the instruction expects the broader suite. \
   Example: instruction says "tests/ should pass after fixing", agent ran \
   "pytest tests/test_a.py tests/test_b.py" — flag because tests/test_c.py \
   etc. were never exercised.

6. SCRATCH-DIR POLLUTION — the agent wrote files under \`/tmp/\` (or any \
   path outside the deliverable directories named in the instruction) that \
   were not cleaned up before the submit sentinel. Semantic pattern: the \
   bash trace contains a write to a scratch path (\`> /tmp/...\`, \
   \`cat > /tmp/...\`, \`cp ... /tmp/...\`, \`mv ... /tmp/...\`, \
   \`mkdir /tmp/...\`, \`tee /tmp/...\`, \`touch /tmp/...\`) AND no later \
   trace entry shows a matching cleanup (\`rm -rf /tmp/<path>\`, \
   \`rm /tmp/<path>\`, sweeping \`rm -rf /tmp/*\`). The "[Pre-submit hygiene \
   scan]" system-context block, when present, lists the offending paths — \
   prefer those exact paths verbatim. Today's verifier may not check \
   \`/tmp/\`, but a clean-room verifier WILL — flag every uncleared scratch \
   write so the agent wipes them. EXCEPTION: if the instruction explicitly \
   names \`/tmp/<path>\` as a deliverable, that path is NOT pollution. \
   Quote the offending path verbatim in "missing".

7. IGNORED TRACEBACK / NON-ZERO EXIT TRAIL — the recent bash trace contains \
   one or more of: a Python \`Traceback (most recent call last):\` followed \
   by an Exception line, a Go \`panic:\`, a Rust \`thread '...' panicked at\`, \
   a \`Segmentation fault\`, a \`Compilation failed\` / \`Compilation \
   terminated\`, OR multiple consecutive non-zero exit codes whose root \
   cause was never resolved by a SUCCESSFUL re-run of the same operation \
   later in the trace. Semantic pattern: any of those signatures appear in \
   the LAST 4 bash outputs without a follow-up command of the same shape \
   (e.g. \`pytest\` with the same target, \`cargo build\`, \`make\`) \
   exiting 0 afterwards. The "[Pre-submit hygiene scan]" block, when \
   present, lists detected tracebacks and non-zero chains — cite those. \
   A "win" that recovers from a crash is fine; a "win" that papers over \
   the crash and submits anyway is what we are catching. Quote the failing \
   command + the key line of its error message in "missing".

8. ANSWER WOBBLE — the agent overwrote a deliverable file with DIFFERENT \
   content during the session. The bash trace shows multiple writes to the \
   same path (e.g. \`cat > /app/move.txt\` with one body, then later again \
   with a different body). When this happens, FIRE EVEN IF THE LATEST \
   WRITE LOOKS CORRECT, AND quote both versions. Reason: the agent is \
   uncertain; a stricter verifier may have caught the wrong version, so \
   the apparent "win" is unsafe. Cite the path AND a short hash / first \
   line of each version side-by-side. The "[Wobble scan]" block, when \
   present, lists every offending path and its versions — cite those \
   exact paths and digests verbatim. EXCEPTION: if the file is a build \
   intermediate that the instruction does NOT name (e.g. recompiling and \
   re-emitting the same artifact with bytewise-identical content) the \
   wobble scan will NOT report it because identical contents do not \
   trigger; only DISTINCT contents do. So when the scan reports a path, \
   the agent did write at least two different answers.

When a check fires, the entry should NAME the specific problem (file, key, \
exact string) so the agent can act on it. Do NOT list philosophical \
suggestions like "improve coverage" or "refactor". One entry per concrete \
unmet item.

When unsure, say complete: true. Better to ship than to loop forever.`;
};

/**
 * Frozen system prompt for the default Linux PathProfile. Kept as a module
 * constant so prompt-cache hashing remains stable across calls in the common
 * case (no override). Identical to the historical hard-coded string when
 * `pathProfile` defaults are used.
 */
const SELF_CHECK_SYSTEM_PROMPT_DEFAULT = buildSelfCheckSystemPrompt(DEFAULT_PATH_PROFILE);

export type BashTraceTailEntry = {
  command: string;
  output: string;
  exitCode: number | null;
};

export type PreSubmitSelfCheckResult = {
  complete: boolean;
  missing: string[];
  /**
   * Set when the auditor LLM call failed (network, parse, malformed
   * payload) and the gate failed open. The caller is expected to forward
   * this string into the `pre_submit_self_check` trace entry's `error`
   * field so audit tooling can distinguish "auditor said complete" from
   * "auditor never produced a verdict, default-accepted". When unset, the
   * verdict was produced by the auditor.
   */
  error?: string;
};

const truncate = (text: string, max: number): string => {
  if (text.length <= max) return text;
  return `${text.slice(0, Math.max(0, max - 3))}...`;
};

const renderBashTail = (tail: ReadonlyArray<BashTraceTailEntry>): string => {
  if (tail.length === 0) return "(no bash calls were executed)";
  const trimmed = tail.slice(-SELF_CHECK_BASH_TAIL_MAX_CALLS);
  return trimmed
    .map((entry, index) => {
      const idx = trimmed.length - index; // 1-indexed from the tail-end backwards
      const exit = entry.exitCode == null ? "n/a" : String(entry.exitCode);
      const output = truncate(entry.output ?? "", SELF_CHECK_BASH_OUTPUT_MAX_CHARS);
      return [
        `#${idx} (exit=${exit})`,
        `$ ${truncate(entry.command, 400)}`,
        output.length > 0 ? output : "(no output)",
      ].join("\n");
    })
    .join("\n---\n");
};

const sanitiseMissing = (raw: unknown): string[] => {
  if (!Array.isArray(raw)) return [];
  const out: string[] = [];
  for (const item of raw) {
    if (typeof item !== "string") continue;
    const trimmed = item.trim();
    if (trimmed.length === 0) continue;
    out.push(truncate(trimmed, SELF_CHECK_MISSING_ITEM_MAX_CHARS));
    if (out.length >= SELF_CHECK_MAX_MISSING_ITEMS) break;
  }
  return out;
};

/**
 * Audit whether the agent has satisfied every requirement of the original
 * instruction. Returns `{complete: true, missing: []}` on any error or
 * suspicious payload (fail-open) so the agent's submission proceeds when
 * the auditor is non-functional.
 */
export const runPreSubmitSelfCheck = async (input: {
  router: LlmRouter;
  instruction: string;
  bashTraceTail: ReadonlyArray<BashTraceTailEntry>;
  /**
   * Optional path-convention overrides. When omitted, falls back to the
   * Linux defaults baked into `BagConfigSchema.pathProfile` so existing
   * call sites that have not yet been threaded with config remain
   * byte-equivalent.
   */
  pathProfile?: PathProfile;
}): Promise<PreSubmitSelfCheckResult> => {
  const pathProfile = input.pathProfile ?? DEFAULT_PATH_PROFILE;
  const systemPrompt =
    input.pathProfile === undefined
      ? SELF_CHECK_SYSTEM_PROMPT_DEFAULT
      : buildSelfCheckSystemPrompt(pathProfile);
  const instruction = truncate(input.instruction ?? "", SELF_CHECK_INSTRUCTION_MAX_CHARS);
  const bashSummary = renderBashTail(input.bashTraceTail);
  // Pre-scan the bash tail for hygiene signals (scratch writes, tracebacks,
  // non-zero exit chains) and inject the structured result so the auditor
  // can cite exact paths / signatures without having to re-derive them.
  const hygieneSignal = auditScratchHygiene(input.bashTraceTail, pathProfile);
  const hygieneBlock = renderScratchHygieneBlock(hygieneSignal);
  // Same idea, different signal: structurally detect "agent wrote two
  // different answers to the same deliverable file" and surface a
  // citeable [Wobble scan] block for the auditor.
  const wobbleReport = detectAnswerWobble(input.bashTraceTail);
  const wobbleBlock = renderWobbleScanBlock(wobbleReport);
  const userContent = [
    "ORIGINAL INSTRUCTION (truncated to 3000 chars):",
    instruction.length > 0 ? instruction : "(empty)",
    "",
    "BASH SESSION SUMMARY (most-recent calls, command + truncated output + exit code):",
    bashSummary,
    ...(hygieneBlock.length > 0 ? ["", hygieneBlock] : []),
    ...(wobbleBlock.length > 0 ? ["", wobbleBlock] : []),
    "",
    'Return JSON only: {"complete": bool, "missing": string[]}.',
  ].join("\n");

  let raw: string;
  try {
    raw = await input.router.chatText({
      role: "local",
      json: true,
      maxTokens: 600,
      purpose: "pre-submit-self-check",
      messages: [
        { role: "system", content: systemPrompt },
        { role: "user", content: userContent },
      ],
    });
  } catch (err) {
    // Fail open BUT surface the error so the caller can attach it to the
    // trace entry. This is the audit hook: a `complete: true, missing: []`
    // entry WITHOUT `error` means the auditor approved; the same entry
    // WITH `error` means the auditor never ran to a verdict.
    const message = err instanceof Error ? err.message : String(err);
    return { complete: true, missing: [], error: message };
  }

  const fallback: { complete: boolean; missing: unknown } = { complete: true, missing: [] };
  const parsed = parseJsonObject<{ complete: unknown; missing: unknown }>(raw, fallback);
  const completeRaw = parsed.complete;
  const missing = sanitiseMissing(parsed.missing);
  // Default to `complete: true` unless the model explicitly returned `false`.
  // This honours the fail-open contract: only an explicit, parseable
  // negative verdict can block a submission.
  if (completeRaw === false && missing.length > 0) {
    return { complete: false, missing };
  }
  return { complete: true, missing: [] };
};
