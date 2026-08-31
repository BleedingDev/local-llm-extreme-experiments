import { createRunId } from "../artifacts";
import { createLlmRouter, parseJsonObject } from "../llm";
import { RunTelemetry } from "../telemetry";
import type { BagConfig } from "../types";
import type { BagAcpSession } from "./session";
import { renderUserCapabilitySurface } from "./surface";

export type AcpPromptRoute = "chat" | "plan" | "run";

export type AcpPromptRouterDeps = {
  config: BagConfig;
  agentMessage: (sessionId: string, text: string) => Promise<void>;
  throwIfAborted: (signal?: AbortSignal) => void;
};

export const runAcpConversationTurn = async (
  deps: Pick<AcpPromptRouterDeps, "agentMessage">,
  session: BagAcpSession,
): Promise<void> => {
  session.updatedAt = new Date().toISOString();
  await deps.agentMessage(session.id, renderUserCapabilitySurface());
};

export const decideAcpPromptRoute = async (
  deps: AcpPromptRouterDeps,
  input: {
    session: BagAcpSession;
    task: string;
    signal: AbortSignal;
  },
): Promise<AcpPromptRoute> => {
  const { session, task, signal } = input;
  const runId = `acp-route-${createRunId()}`;
  const telemetry = new RunTelemetry(deps.config, runId, session.cwd, session.optimizerPin.telemetry);
  const router = createLlmRouter(deps.config, telemetry);
  const role = router.masterAvailable ? "master" : (await router.localAvailable()) ? "local" : null;

  if (role == null) {
    telemetry.event("acp.route.fallback", {
      sessionId: session.id,
      reason: "no-router-model-available",
    });
    await deps.agentMessage(
      session.id,
      "Auto router model is unavailable, so I am staying in chat mode and will not touch the project.",
    );
    return "chat";
  }

  deps.throwIfAborted(signal);
  const raw = await router.chatText({
    role,
    maxTokens: 180,
    temperature: 0,
    json: true,
    purpose: "prompt-router",
    messages: [
      {
        role: "system",
        content: [
          "You are a side-effect router for an ACP coding agent.",
          "Decide what kind of agent turn is needed for the user's latest message.",
          "Do not use language-specific keyword rules. Infer intent semantically.",
          "Return JSON only with this shape:",
          "{\"route\":\"chat|plan|run\",\"confidence\":0..1,\"rationale\":\"short\"}",
          "Routes:",
          "- chat: conversation, explanation, capabilities, clarification, or no repository access needed.",
          "- plan: repository read/analysis/report/PRD/DAG/evaluation is useful, but file edits and terminal commands are not needed.",
          "- run: file edits, command execution, verification, installation, cleanup, benchmarking, or other project mutation/execution is needed.",
          "If uncertain between chat and project access, choose chat. If uncertain between plan and run, choose plan.",
        ].join("\n"),
      },
      {
        role: "user",
        content: [
          `Current ACP mode: ${session.mode}`,
          `Project root: ${session.cwd}`,
          "User message:",
          task,
        ].join("\n"),
      },
    ],
  });

  const parsed = parseJsonObject<{ route?: unknown; confidence?: unknown; rationale?: unknown }>(raw, {});
  const route = parsed.route === "plan" || parsed.route === "run" || parsed.route === "chat" ? parsed.route : "chat";
  telemetry.event("acp.route.decided", {
    sessionId: session.id,
    route,
    confidence: typeof parsed.confidence === "number" ? parsed.confidence : undefined,
    rationale: typeof parsed.rationale === "string" ? parsed.rationale : undefined,
    modelRole: role,
  });
  await deps.agentMessage(
    session.id,
    `Auto route: ${route}${typeof parsed.rationale === "string" && parsed.rationale.trim() !== "" ? ` (${parsed.rationale.trim()})` : ""}`,
  );
  return route;
};
