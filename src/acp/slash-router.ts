import type { AgentSideConnection as AcpConnection } from "@agentclientprotocol/sdk";
import { resolve } from "node:path";
import { TraceStore } from "../trace-store";
import type { BagConfig } from "../types";
import type { BagAcpSession } from "./session";

export type SkillSummary = {
  name: string;
  description: string;
  path: string;
};

export type AcpSlashRouterDeps = {
  connection: AcpConnection;
  config: BagConfig;
  agentMessage: (sessionId: string, text: string) => Promise<void>;
  listSkills: () => SkillSummary[];
  runWithTemporaryMode: <T>(
    session: BagAcpSession,
    activeMode: "plan" | "run",
    previousMode: BagAcpSession["mode"],
    fn: () => Promise<T>,
  ) => Promise<T>;
  runCodingTurn: (session: BagAcpSession, task: string, signal: AbortSignal) => Promise<void>;
  runPlanningTurn: (session: BagAcpSession, task: string, signal: AbortSignal) => Promise<void>;
  runAutonomousToolUseTurn: (session: BagAcpSession, task: string, signal: AbortSignal) => Promise<void>;
  runDagDrivenToolUseTurn: (session: BagAcpSession, task: string, signal: AbortSignal) => Promise<void>;
  runAdaptiveCodingTurn: (session: BagAcpSession, task: string, signal: AbortSignal) => Promise<void>;
  runMaintenanceCommand: (session: BagAcpSession, task: string) => Promise<void>;
};

export type AcpSlashCommandInput = {
  session: BagAcpSession;
  text: string;
  signal: AbortSignal;
};

export const handleAcpSlashCommand = async (
  deps: AcpSlashRouterDeps,
  input: AcpSlashCommandInput,
): Promise<boolean> => {
  const trimmed = input.text.trim();
  if (!trimmed.startsWith("/")) {
    return false;
  }
  // Splitting on /\s+/ and rejoining with " " collapses newlines INSIDE the
  // task body — that breaks every code fence the user includes (```python\n
  // ...\n``` becomes ```python ... ``` on a single line). Match command +
  // remainder once and keep the remainder verbatim so newlines, indentation,
  // and code fences survive into downstream runners.
  const commandMatch = trimmed.slice(1).match(/^(\S+)(?:\s+([\s\S]*))?$/);
  const rawCommand = commandMatch?.[1] ?? "";
  const command = rawCommand.toLowerCase();
  const task = (commandMatch?.[2] ?? "").trim();
  const previousMode = input.session.mode;

  if (command === "run" || command === "code") {
    await publishMode(deps, input.session, "run");
    if (task === "") {
      await deps.agentMessage(input.session.id, "Run mode enabled. Send a coding task.");
      return true;
    }
    await deps.runWithTemporaryMode(input.session, "run", previousMode, () =>
      deps.runCodingTurn(input.session, task, input.signal),
    );
    return true;
  }

  if (command === "run-tools" || command === "tools" || command === "auto-code") {
    await publishMode(deps, input.session, "run");
    if (task === "") {
      await deps.agentMessage(
        input.session.id,
        "Autonomous tool-use coding mode. Send a task and I will iterate via bash until done.",
      );
      return true;
    }
    await deps.runWithTemporaryMode(input.session, "run", previousMode, () =>
      deps.runAutonomousToolUseTurn(input.session, task, input.signal),
    );
    return true;
  }

  if (command === "run-dag-tools" || command === "dag-tools" || command === "plan-tools") {
    await publishMode(deps, input.session, "run");
    if (task === "") {
      await deps.agentMessage(
        input.session.id,
        "DAG-driven tool-use mode. Send a task; I will plan 1-5 issues, then run a scoped bash loop per issue with verifier gating.",
      );
      return true;
    }
    await deps.runWithTemporaryMode(input.session, "run", previousMode, () =>
      deps.runDagDrivenToolUseTurn(input.session, task, input.signal),
    );
    return true;
  }

  if (command === "run-auto" || command === "auto-tools" || command === "run-adaptive") {
    await publishMode(deps, input.session, "run");
    if (task === "") {
      await deps.agentMessage(
        input.session.id,
        "Adaptive coding mode. Send a task; I will classify its shape and route to either tools or dag-tools automatically.",
      );
      return true;
    }
    await deps.runWithTemporaryMode(input.session, "run", previousMode, () =>
      deps.runAdaptiveCodingTurn(input.session, task, input.signal),
    );
    return true;
  }

  if (command === "plan") {
    await publishMode(deps, input.session, "plan");
    if (task === "") {
      await deps.agentMessage(input.session.id, "Plan mode enabled. Send a planning task.");
      return true;
    }
    await deps.runWithTemporaryMode(input.session, "plan", previousMode, () =>
      deps.runPlanningTurn(input.session, task, input.signal),
    );
    return true;
  }

  if (command === "chat") {
    await publishMode(deps, input.session, "chat");
    await deps.agentMessage(input.session.id, "Chat mode enabled. Plain messages will not read, edit, or run the project.");
    return true;
  }

  if (command === "auto") {
    await publishMode(deps, input.session, "auto");
    await deps.agentMessage(input.session.id, "Auto mode enabled. BleedingAgent will route each prompt as chat, plan, or run.");
    return true;
  }

  if (command === "yolo") {
    input.session.yolo = true;
    await deps.agentMessage(input.session.id, "YOLO mode enabled. File writes and terminal commands will not ask for permission.");
    return true;
  }

  if (command === "safe" || command === "ask") {
    input.session.yolo = false;
    await deps.agentMessage(input.session.id, "Safe mode enabled. File writes and terminal commands will ask for permission.");
    return true;
  }

  if (command === "skills") {
    const skills = deps.listSkills();
    await deps.agentMessage(
      input.session.id,
      skills.length === 0
        ? "No local skills found in the configured skill directories."
        : [
            `Found ${skills.length} local skills:`,
            ...skills.slice(0, 80).map((skill) => `- ${skill.name}: ${skill.description}\n  ${skill.path}`),
          ].join("\n"),
    );
    return true;
  }

  if (command === "mcp") {
    await deps.agentMessage(
      input.session.id,
      input.session.mcpServers.length === 0
        ? "No MCP servers were attached to this ACP session."
        : [
            `Attached MCP servers: ${input.session.mcpServers.length}`,
            ...input.session.mcpServers.map((server) => {
              if ("url" in server) {
                return `- ${server.name}: ${server.type} ${server.url}`;
              }
              return `- ${server.name}: stdio ${server.command} ${server.args.join(" ")}`;
            }),
          ].join("\n"),
    );
    return true;
  }

  if (command === "metrics") {
    await deps.agentMessage(
      input.session.id,
      [
        `Telemetry JSONL: ${resolve(input.session.cwd, deps.config.telemetry.jsonl)}`,
        `Metrics store: ${resolve(input.session.cwd, deps.config.telemetry.metrics)}`,
        `Span traces: ${resolve(input.session.cwd, deps.config.telemetry.spans)}`,
        `Span index: ${resolve(input.session.cwd, `${deps.config.telemetry.spans}.index.jsonl`)}`,
        `Run artifacts: ${resolve(input.session.cwd, deps.config.artifactDir, "runs")}`,
        `Current mode: ${input.session.mode}`,
        `YOLO mode: ${input.session.yolo}`,
        "Active tuning: pinned for this session; detailed maintenance internals are intentionally hidden from the normal coding surface.",
      ].join("\n"),
    );
    return true;
  }

  if (command === "traces") {
    const store = TraceStore.open(deps.config, input.session.cwd);
    const overview = store.getOverview();
    const failing = store.queryTraces({ hasErrors: true }, 8);
    await deps.agentMessage(
      input.session.id,
      [
        "HALO-style trace dataset:",
        `- traces: ${overview.traceCount}`,
        `- spans: ${overview.spanCount}`,
        `- error traces: ${overview.errorTraceCount}`,
        `- error spans: ${overview.errorSpanCount}`,
        `- models: ${overview.models.join(", ") || "none"}`,
        `- observation kinds: ${overview.observationKinds.join(", ") || "none"}`,
        `- sample trace ids: ${overview.sampleTraceIds.join(", ") || "none"}`,
        "",
        "Failing traces:",
        failing.traces.length === 0
          ? "- none"
          : failing.traces
              .map(
                (trace) =>
                  `- ${trace.traceId}: spans=${trace.spanCount} errors=${trace.errorSpanCount} names=${trace.spanNames
                    .slice(0, 5)
                    .join(", ")}`,
              )
              .join("\n"),
      ].join("\n"),
    );
    return true;
  }

  if (command === "maintenance" || command === "maint") {
    await deps.runWithTemporaryMode(input.session, "plan", previousMode, () =>
      deps.runMaintenanceCommand(input.session, task),
    );
    return true;
  }

  await deps.agentMessage(input.session.id, `Unknown command: /${command}`);
  return true;
};

const publishMode = async (
  deps: Pick<AcpSlashRouterDeps, "connection">,
  session: BagAcpSession,
  mode: BagAcpSession["mode"],
): Promise<void> => {
  session.mode = mode;
  await deps.connection.sessionUpdate({
    sessionId: session.id,
    update: { sessionUpdate: "current_mode_update", currentModeId: session.mode },
  });
};
