import { parseJsonObject, type LlmRouter } from "./llm";
import { InterviewTurnSchema, type InterviewTurn } from "./types";

export const fallbackInterview = (task: string): InterviewTurn => ({
  question:
    "What should BleedingAgent change first: architecture/design, implementation, tests, or an existing bug path?",
  rationale:
    "The initial task is broad enough that one clarification reduces wasted coding and keeps the generated PRD grounded.",
  acceptedFacts: [
    `User task: ${task}`,
    "Tool name is BleedingAgent.",
    "BleedingAgent is an ACP coding-agent backend and self-evolving optimization harness.",
    "The bag command is the operator entrypoint for launching ACP and maintenance flows.",
    "Runtime should be TypeScript + Ax + mixed master/local LLMs.",
    "Every run should be measured, monitored, self-evaluated, and fed into self-improvement.",
  ],
  openQuestions: [
    "Which repository should be modified by the first production run?",
    "Should execution be read-only, patch-only, or allowed to run tests and apply edits?",
  ],
  canGeneratePrdNow: true,
  suggestedNextAction: "generate_prd",
});

export const runInterview = async (input: {
  router: LlmRouter;
  task: string;
  repoContext: string;
  knowledge: string;
}): Promise<InterviewTurn> => {
  if (!input.router.masterAvailable) {
    return fallbackInterview(input.task);
  }

  const fallback = fallbackInterview(input.task);
  const raw = await input.router.chatText({
    role: "master",
    json: true,
    purpose: "interview-lead",
    messages: [
      {
        role: "system",
        content:
          'You are BleedingAgent\'s product interview lead. Ask only useful questions, accept explicit facts, and decide whether enough is known to create a PRD. Return JSON ONLY (no prose, no code fences) with exactly these keys: question (string), rationale (string), acceptedFacts (string[]), openQuestions (string[]), canGeneratePrdNow (boolean), suggestedNextAction (must be exactly "continue_interview" OR "generate_prd" — no other values allowed).',
      },
      {
        role: "user",
        content: [
          `Task:\n${input.task}`,
          `Knowledge:\n${input.knowledge.slice(0, 6000)}`,
          `Repo context:\n${input.repoContext.slice(0, 12000)}`,
        ].join("\n\n"),
      },
    ],
  });

  const candidate = parseJsonObject(raw, fallback);
  const result = InterviewTurnSchema.safeParse(candidate);
  return result.success ? result.data : fallback;
};
