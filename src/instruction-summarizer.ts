/**
 * Instruction summarizer — universal trace-mining lesson #1.
 *
 * Long instruction.md files (e.g. SWE-bench Pro tasks, user-pasted multi-page
 * specs) cause the master loop to re-read the full text on every turn,
 * burning tokens. We pre-summarize the task once via the cheap local role
 * when it exceeds a threshold and forward the compact summary to the coding
 * loop / planner. The original text is preserved on the returned record so
 * verifier-quote use cases can still grep it verbatim.
 */
import type { LlmRouter } from "./llm";

export type InstructionSummary = {
  /** Full original instruction text — preserved for verifier quotes. */
  original: string;
  /** Compact summary forwarded to the coding loop. <=800 chars when summarized. */
  summary: string;
  /** Approx tokens saved (chars/4 heuristic). 0 when passthrough. */
  tokensSaved: number;
  /** False when input was below threshold (no LLM call); true when summarized. */
  summarized: boolean;
};

/** Inputs below this character count skip the LLM call (passthrough). */
export const SUMMARIZE_THRESHOLD_CHARS = 1500;

/** Soft cap on the model summary length. */
export const SUMMARY_MAX_CHARS = 800;

const SUMMARIZER_SYSTEM_PROMPT = `\
You are BAG's instruction-summarizer. The agent will execute the task using \
ONLY this summary; preserve EVERY hard requirement (file paths, exact \
strings, expected outputs, verifier commands). Keep summary <800 chars. Do \
NOT invent details. Do NOT drop "must", "exactly", "if X then Y" clauses. \
If a verification command appears in the task, include it verbatim in the \
summary. Output the summary as plain prose only — no preamble, no fences.`;

/** chars/4 token heuristic — avoids a tokenizer dependency. */
const approxTokens = (text: string): number => Math.ceil(text.length / 4);

const clampSummary = (raw: string): string => {
  const trimmed = raw.trim();
  if (trimmed.length <= SUMMARY_MAX_CHARS) return trimmed;
  return `${trimmed.slice(0, SUMMARY_MAX_CHARS - 3)}...`;
};

/**
 * Summarize the given task text via the cheap local role when it exceeds
 * `SUMMARIZE_THRESHOLD_CHARS`. Below threshold returns a passthrough record
 * with no LLM call. On any model error, falls back to a passthrough record
 * (the caller continues with the full original text — fail-open).
 */
export const maybeSummarizeInstruction = async (input: {
  router: LlmRouter;
  task: string;
}): Promise<InstructionSummary> => {
  const original = input.task;
  if (original.length < SUMMARIZE_THRESHOLD_CHARS) {
    return {
      original,
      summary: original,
      tokensSaved: 0,
      summarized: false,
    };
  }

  try {
    const raw = await input.router.chatText({
      role: "local",
      json: false,
      maxTokens: 600,
      purpose: "instruction-summarizer",
      messages: [
        { role: "system", content: SUMMARIZER_SYSTEM_PROMPT },
        { role: "user", content: original },
      ],
    });
    const summary = clampSummary(raw);
    if (summary.length === 0 || summary.length >= original.length) {
      // Degenerate result — fall back to passthrough so the agent still has the spec.
      return {
        original,
        summary: original,
        tokensSaved: 0,
        summarized: false,
      };
    }
    const tokensSaved = Math.max(0, approxTokens(original) - approxTokens(summary));
    return {
      original,
      summary,
      tokensSaved,
      summarized: true,
    };
  } catch {
    return {
      original,
      summary: original,
      tokensSaved: 0,
      summarized: false,
    };
  }
};
