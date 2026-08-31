import { createHash } from "node:crypto";
import type {
  ClientCapabilities,
  ContentBlock,
  PlanEntry,
  SessionConfigOption,
  SessionModeId,
  SessionModeState,
} from "@agentclientprotocol/sdk";
import type { BagConfig } from "../types";

export type AcpSessionMode = "auto" | "chat" | "plan" | "run";

export type AcpConsumerCompatibilityStatus = "tested-offline" | "setup-documented" | "documented-limitation";

export type AcpNamedConsumerFixture = {
  consumer: string;
  status: AcpConsumerCompatibilityStatus;
  notes: string;
};

export type AcpConsumerCompatibilityCase = {
  id: string;
  flow: string;
  acpContract: AcpConsumerCompatibilityStatus;
  namedConsumerFixtures: AcpNamedConsumerFixture[];
  expectedBehavior: string;
  smokeSignal: string;
};

export type AcpClientCapabilityProfile = {
  fsReadTextFile: boolean;
  fsWriteTextFile: boolean;
  terminal: boolean;
  richDiffContent: boolean;
  richTerminalContent: boolean;
  source: string;
};

export type AcpSessionConfigSource = {
  executorConcurrency: number;
  yolo: boolean;
};

export type TraceEvent = {
  timestamp: string;
  phase: string;
  event: string;
  ok: boolean;
  data: Record<string, unknown>;
};

export const modeState = (currentModeId: SessionModeId = "auto"): SessionModeState => ({
  currentModeId,
  availableModes: [
    {
      id: "auto",
      name: "Auto",
      description: "Model-routed mode. Chooses chat, read-only planning/reporting, or coding run per prompt.",
    },
    {
      id: "chat",
      name: "Chat",
      description: "Conversation and command discovery only. Never reads, edits, or runs the project.",
    },
    {
      id: "plan",
      name: "Plan",
      description: "Read-only analysis, reports, interview, PRD, DAG, and self-evaluation artifacts.",
    },
    {
      id: "run",
      name: "Run",
      description: "Full coding-agent mode: read, edit, verify, trace, and self-evaluate.",
    },
  ],
});

export const sessionConfigOptions = (config: BagConfig, session?: AcpSessionConfigSource): SessionConfigOption[] => [
  {
    id: "executor-concurrency",
    name: "Executor Concurrency",
    category: "model",
    description: "Local executor parallelism used by context scouting and future coding tools.",
    type: "select",
    currentValue: String(session?.executorConcurrency ?? config.policy.executorConcurrency),
    options: [8, 12, 16, 20, 24].map((value) => ({
      value: String(value),
      name: `${value}`,
      description: value > 20 ? "High-throughput batch mode." : "Interactive-safe local parallelism.",
    })),
  },
  {
    id: "yolo",
    name: "YOLO Mode",
    category: "mode",
    description: "When enabled, edits and terminal commands run without ACP permission prompts.",
    type: "boolean",
    currentValue: session?.yolo ?? !config.policy.requirePermissions,
  },
];

export const promptToText = (blocks: ContentBlock[]): string => {
  const parts = blocks.flatMap((block) => {
    if (block.type === "text") {
      return [block.text];
    }
    if (block.type === "resource") {
      const resource = block.resource;
      if ("text" in resource) {
        return [`Embedded resource ${resource.uri}:\n${resource.text}`];
      }
      return [`Embedded binary resource ${resource.uri} (${resource.mimeType ?? "application/octet-stream"})`];
    }
    if (block.type === "resource_link") {
      return [`Resource link ${block.name}: ${block.uri}`];
    }
    if (block.type === "image") {
      return [`Image input (${block.mimeType}) omitted from text-only BleedingAgent ACP mode.`];
    }
    if (block.type === "audio") {
      return [`Audio input (${block.mimeType}) omitted from text-only BleedingAgent ACP mode.`];
    }
    return [];
  });
  return parts.join("\n\n").trim();
};

export const markdownContent = (text: string) => ({
  type: "content" as const,
  content: {
    type: "text" as const,
    text,
  },
});

export const compactJson = (value: unknown): string => JSON.stringify(value, null, 2).slice(0, 12_000);

export const defaultAcpClientCapabilityProfile = (): AcpClientCapabilityProfile => ({
  fsReadTextFile: true,
  fsWriteTextFile: true,
  terminal: true,
  richDiffContent: true,
  richTerminalContent: true,
  source: "not-initialized",
});

export const acpClientCapabilityProfileFromInitialize = (
  capabilities: ClientCapabilities | undefined,
  source: string,
): AcpClientCapabilityProfile => {
  const fsReadTextFile = capabilities?.fs?.readTextFile === true;
  const fsWriteTextFile = capabilities?.fs?.writeTextFile === true;
  const terminal = capabilities?.terminal === true;
  return {
    fsReadTextFile,
    fsWriteTextFile,
    terminal,
    richDiffContent: fsWriteTextFile,
    richTerminalContent: terminal,
    source,
  };
};

export const namedAcpConsumerFixtures = (): AcpNamedConsumerFixture[] => [
  {
    consumer: "Glass",
    status: "setup-documented",
    notes: "Named consumer setup is documented; broad desktop UI automation is not the ACP contract.",
  },
  {
    consumer: "Zed",
    status: "setup-documented",
    notes: "Named consumer setup is documented; broad desktop UI automation is not the ACP contract.",
  },
];

export const acpConsumerCompatibilityMatrix = (): AcpConsumerCompatibilityCase[] => [
  {
    id: "session-start",
    flow: "Session start",
    acpContract: "tested-offline",
    namedConsumerFixtures: namedAcpConsumerFixtures(),
    expectedBehavior: "initialize advertises ACP v1, text context, session list/resume/close, Auto mode, YOLO config, and visible slash commands.",
    smokeSignal: "initialize + session/new + available_commands_update",
  },
  {
    id: "greeting",
    flow: "Greeting",
    acpContract: "tested-offline",
    namedConsumerFixtures: namedAcpConsumerFixtures(),
    expectedBehavior: "Chat stays conversational and does not read, edit, run terminals, or expose maintenance tuning internals.",
    smokeSignal: "chat-mode prompt emits only agent_message_chunk help",
  },
  {
    id: "plan-report",
    flow: "Plan/report",
    acpContract: "tested-offline",
    namedConsumerFixtures: namedAcpConsumerFixtures(),
    expectedBehavior: "Plan mode streams read-only plan/tool updates and writes planning artifacts without file edits or terminal execution.",
    smokeSignal: "plan route updates current_mode_update + plan/tool_call read/think events",
  },
  {
    id: "edit-run",
    flow: "Edit run",
    acpContract: "tested-offline",
    namedConsumerFixtures: namedAcpConsumerFixtures(),
    expectedBehavior: "Run mode reads files, previews diffs, applies writes through ACP, records edit attempts, and leaves artifact links in the trace.",
    smokeSignal: "read/edit/write tool calls with diff content and edit lifecycle spans",
  },
  {
    id: "terminal-verification",
    flow: "Terminal verification",
    acpContract: "tested-offline",
    namedConsumerFixtures: namedAcpConsumerFixtures(),
    expectedBehavior: "Verification commands run in ACP terminals, stream the terminal handle, capture exit/output, and feed repair/rollback if failing.",
    smokeSignal: "execute tool_call + terminal content + command-results artifact",
  },
  {
    id: "permissions",
    flow: "Permissions",
    acpContract: "tested-offline",
    namedConsumerFixtures: namedAcpConsumerFixtures(),
    expectedBehavior: "YOLO bypasses prompts by default; Safe asks before writes and commands; rejection/cancellation is visible and traceable.",
    smokeSignal: "request_permission only after /safe; rejected write/command return failed tool_call_update",
  },
  {
    id: "slash-commands",
    flow: "Slash commands",
    acpContract: "tested-offline",
    namedConsumerFixtures: namedAcpConsumerFixtures(),
    expectedBehavior: "Default command surface stays focused on run, plan, chat, auto, YOLO/Safe, skills, MCP, metrics, and traces.",
    smokeSignal: "available_commands_update excludes maintenance/promote/rollback/optimizer controls",
  },
  {
    id: "cancellation",
    flow: "Cancellation",
    acpContract: "tested-offline",
    namedConsumerFixtures: namedAcpConsumerFixtures(),
    expectedBehavior: "Cancel aborts the active prompt, releases terminal resources where possible, writes a cancellation artifact, and keeps the session reusable.",
    smokeSignal: "session/cancel leads to cancelled stopReason and .bag run cancellation files",
  },
  {
    id: "trace-artifacts",
    flow: "Trace/artifact updates",
    acpContract: "tested-offline",
    namedConsumerFixtures: namedAcpConsumerFixtures(),
    expectedBehavior: "Metrics and traces remain accessible through /metrics and /traces while normal status text stays concise.",
    smokeSignal: "artifact paths in final messages and raw tool outputs, compact user-facing content",
  },
];

export const initialPlan = (): PlanEntry[] => [
  { content: "Load learned guidance and session context", priority: "high", status: "pending" },
  { content: "Scout repository context with local executor pool", priority: "high", status: "pending" },
  { content: "Run interview flow and extract accepted facts", priority: "high", status: "pending" },
  { content: "Generate PRD artifact", priority: "medium", status: "pending" },
  { content: "Generate dependency-aware DAG", priority: "medium", status: "pending" },
  { content: "Run deterministic self-evaluation and capture telemetry learnings", priority: "medium", status: "pending" },
];

export const availableCommands = () => [
  {
    name: "run",
    description: "Execute an explicit coding task with read/edit/verify/repair.",
    input: { hint: "coding task" },
  },
  {
    name: "plan",
    description: "Execute an explicit planning or reporting task and produce interview, PRD, and DAG artifacts.",
    input: { hint: "planning task" },
  },
  {
    name: "chat",
    description: "Force no-side-effects chat mode.",
  },
  {
    name: "auto",
    description: "Switch back to model-routed auto mode.",
  },
  {
    name: "yolo",
    description: "Enable YOLO mode: no permission prompts for file writes or terminal commands.",
  },
  {
    name: "safe",
    description: "Disable YOLO mode: ask for permission before writes and terminal commands.",
  },
  {
    name: "skills",
    description: "List locally available BleedingAgent/Codex skills visible to this agent.",
  },
  {
    name: "mcp",
    description: "Show MCP servers attached to the current ACP session.",
  },
  {
    name: "metrics",
    description: "Show telemetry, metrics, trace, and artifact locations for the current session.",
  },
  {
    name: "traces",
    description: "Show HALO-style trace dataset overview and recent failing trace IDs.",
  },
];

export const renderUserCapabilitySurface = (): string =>
  [
    "Ahoj. Jsem BleedingAgent ACP coding agent.",
    "",
    "V Auto módu rozhoduju pro každý prompt, jestli stačí odpověď, read-only plán/report, nebo coding run. Side effecty řídí aktuální YOLO/Safe nastavení.",
    "",
    "Hlavní práce:",
    "- `/run <task>`: čtení souborů, editace, verifikace, traces",
    "- `/plan <task>`: interview, PRD, DAG, report bez editací",
    "- `/auto`: návrat do model-routed režimu",
    "- `/yolo` / `/safe`: přepnutí approval režimu",
    "- `/skills`, `/mcp`: lokální skilly a MCP servery",
    "- `/metrics`, `/traces`: telemetry, metriky a trace přehled",
  ].join("\n");

export const updatePlanEntry = (entries: PlanEntry[], index: number, status: PlanEntry["status"]): PlanEntry[] =>
  entries.map((entry, entryIndex) => (entryIndex === index ? { ...entry, status } : entry));

export const artifactLocation = (path: string) => ({ path });

export const maintenanceCommandHelp = (): string =>
  [
    "BleedingAgent maintenance commands are hidden from the normal coding surface.",
    "",
    "Use:",
    "- `/maintenance status` or `/maintenance inspect`: read optimizer registry and active session pin",
    "- `/maintenance eval`: summarize configured eval splits without running evals",
    "- `/maintenance optimize`: compute the existing safe optimization report",
    "- `/maintenance promote <candidate-id>`: inspect promotion readiness; no promotion is applied",
    "- `/maintenance rollback [checkpoint]`: inspect rollback target; no rollback is applied",
  ].join("\n");

export const traceEvent = (
  phase: string,
  event: string,
  ok: boolean,
  data: Record<string, unknown> = {},
): TraceEvent => ({
  timestamp: new Date().toISOString(),
  phase,
  event,
  ok,
  data,
});

export const sha256 = (text: string): string => createHash("sha256").update(text).digest("hex");

export const replaySafeId = (value: string): string => {
  const normalized = value.replace(/[^A-Za-z0-9._:-]+/g, "-").replace(/^-+/, "");
  return normalized.length === 0 ? `id.${sha256(value).slice(0, 12)}` : normalized;
};
