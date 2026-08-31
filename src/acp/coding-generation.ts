import { parseJsonObject, type LlmRouter } from "../llm";
import type { BagConfig, ContextScoutFinding } from "../types";
import {
  CodingStructuredImpossibilitySchema,
  type CodingStructuredImpossibility,
} from "./coding-progress-diagnostics";
import { compactJson } from "./surface";
import {
  parseCodingEditOperation,
  renderVerifierResultsForLlm,
  type CodingEditOperation,
  type CodingFileSelection,
  type CodingFileSnapshot,
  type CodingPatch,
  type LiveEditContext,
} from "./coding-types";
import type { TerminalCommandResult } from "./terminal";

export const selectCodingFiles = async (input: {
  router: LlmRouter;
  task: string;
  repoContext: string;
  knowledge: string;
  scoutFindings: ContextScoutFinding[];
}): Promise<CodingFileSelection> => {
  const fallback: CodingFileSelection = {
    approach: "deterministic scout fallback",
    filesToRead: input.scoutFindings.map((finding) => finding.file).slice(0, 6),
    filesToCreate: [],
  };
  const role = input.router.masterAvailable ? "master" : (await input.router.localAvailable()) ? "local" : undefined;
  if (role == null) {
    return fallback;
  }

  try {
    const raw = await input.router.chatText({
      role,
      json: true,
      maxTokens: 1200,
      purpose: "coding-plan",
      messages: [
        {
          role: "system",
          content: [
            "You are BleedingAgent's coding file selector. Return JSON ONLY (no prose, no fences):",
            '{"approach":"short plan","filesToRead":["relative/path.ts"],"filesToCreate":["relative/path.ts"]}',
            "filesToRead: existing files the model needs to inspect before editing.",
            "filesToCreate: NEW files (paths that do not yet exist) the task requires creating from scratch — for example a script.py, output.txt, or a new module. Empty array if no creates needed.",
            "Both arrays may be non-empty in mixed tasks. Do not include directories. Use repository-relative paths.",
          ].join("\n"),
        },
        {
          role: "user",
          content: [
            `Task:\n${input.task}`,
            `Knowledge:\n${input.knowledge.slice(0, 5000)}`,
            `Repository context:\n${input.repoContext.slice(0, 24000)}`,
          ].join("\n\n"),
        },
      ],
    });
    const parsed = parseJsonObject<CodingFileSelection>(raw, fallback);
    return {
      approach: String(parsed.approach ?? fallback.approach),
      filesToRead: Array.isArray(parsed.filesToRead)
        ? parsed.filesToRead.map(String).filter((file) => file.length > 0).slice(0, 8)
        : fallback.filesToRead,
      filesToCreate: Array.isArray(parsed.filesToCreate)
        ? parsed.filesToCreate.map(String).filter((file) => file.length > 0).slice(0, 8)
        : fallback.filesToCreate,
    };
  } catch {
    return fallback;
  }
};

export const generateCodingPatch = async (input: {
  config: BagConfig;
  router: LlmRouter;
  task: string;
  repoContext: string;
  knowledge: string;
  fileSnapshots: CodingFileSnapshot[];
  editContext: LiveEditContext;
  verifierResults?: readonly TerminalCommandResult[];
  postApplyFailures?: readonly { path: string; status: string; reason?: string }[];
  repairRound?: number;
}): Promise<CodingPatch> => {
  const editStrategy = {
    strategyId: input.editContext.decision.selectedStrategyId,
    strategyFamily: input.editContext.decision.selectedStrategyFamily,
    renderedEditToolContractId: input.editContext.renderedContract.renderedToolId,
  };
  const greenfield =
    input.fileSnapshots.length > 0 &&
    input.fileSnapshots.every((file) => file.kind === "create" && file.content.length === 0);
  const hasNoSnapshots = input.fileSnapshots.length === 0;
  const fallback: CodingPatch = {
    summary: "No master or local model is configured; cannot generate edits.",
    editStrategy,
    generation: {
      modelAvailable: false,
      rawEditCount: 0,
      rawCommandCount: 0,
    },
    edits: [],
    commands: [],
    risks: ["No edits were generated because no master or local model is available."],
    parseFailures: [],
  };
  const role = input.router.masterAvailable ? "master" : (await input.router.localAvailable()) ? "local" : undefined;
  if (role == null) {
    return fallback;
  }

  try {
    const raw = await input.router.chatText({
      role,
      json: true,
      maxTokens: Math.max(4096, input.config.master.maxTokens),
      temperature: 0.1,
      purpose: "coding-generation",
      messages: [
        {
          role: "system",
          content: [
            "You are BleedingAgent's code editor. Return strict JSON only.",
            "BleedingAgent has already selected the edit strategy for this turn; do not choose or rename the strategy.",
            `Selected strategy id: ${input.editContext.decision.selectedStrategyId}.`,
            `Selected strategy family: ${input.editContext.decision.selectedStrategyFamily}.`,
            `Rendered edit tool contract id: ${input.editContext.renderedContract.renderedToolId}.`,
            `Contract description: ${input.editContext.renderedContract.description}`,
            "Return JSON in this envelope:",
            "{\"summary\":\"...\",\"edits\":[{\"reason\":\"...\",\"payload\":{}}],\"commands\":[{\"command\":\"python3\",\"args\":[\"-m\",\"py_compile\",\"x.py\"],\"reason\":\"...\"}],\"risks\":[\"...\"]}",
            "If the task is impossible because required information or capabilities are genuinely unavailable, return no edits and include structuredImpossibility: {\"reason\":\"specific blocker\",\"evidenceRefs\":[]}. Do not use structuredImpossibility for ordinary uncertainty or verifier failures.",
            "Each edit.payload must match this JSON schema exactly:",
            compactJson(input.editContext.renderedContract.inputSchema),
            "Use repository-relative paths exactly as shown in the file headers. Preserve existing style, imports, and behavior. Do not edit files you have not seen unless creating a clearly requested new file.",
            greenfield
              ? "GREENFIELD MODE: the workspace has no pre-existing source files relevant to this task. PROPOSE NEW FILES TO CREATE. Each edit's payload.path is a relative path that does not yet exist; payload.content is the COMPLETE new file body. Do not include baseContentHash for new files. The whole_file strategy will write each file via the ACP filesystem."
              : hasNoSnapshots
                ? "No file snapshots were attached for this turn; treat this as a greenfield task and propose new files to create."
                : "Some files are read-only context; the edit set may still include create-from-scratch files alongside modifications.",
            "For verification commands, propose ones that match the task's language/runtime (e.g., python3 -m py_compile <file>, bash -n <file.sh>, cargo check, go build ./...). Do NOT default to npm unless this is clearly a Node project.",
            ...input.editContext.renderedContract.promptFragments,
          ].join("\n\n"),
        },
        {
          role: "user",
          content: [
            `Task:\n${input.task}`,
            `Knowledge:\n${input.knowledge.slice(0, 6000)}`,
            `Repo context:\n${input.repoContext.slice(0, 12000)}`,
            `Edit task shape:\n${compactJson(input.editContext.taskShape)}`,
            `Router candidates:\n${compactJson(input.editContext.decision.candidates.slice(0, 6))}`,
            ...(input.verifierResults && input.verifierResults.length > 0
              ? [renderVerifierResultsForLlm(input.verifierResults, input.repairRound)]
              : []),
            ...(input.postApplyFailures && input.postApplyFailures.length > 0
              ? [
                  [
                    "Post-apply consistency findings:",
                    ...input.postApplyFailures.map(
                      (f) => `- ${f.path}: ${f.status}${f.reason ? ` — ${f.reason}` : ""}`,
                    ),
                  ].join("\n"),
                ]
              : []),
            "Files:",
            ...input.fileSnapshots.map(
              (file) => {
                const tag =
                  file.kind === "create" && file.content.length === 0
                    ? "(new file to create — no current content)\n"
                    : file.kind === "create"
                      ? "(new file — current draft content shown below)\n"
                      : "";
                return `## ${file.relativePath}\n${tag}Absolute: ${file.path}\nHash: sha256:${file.hash}\n\`\`\`\n${file.content.slice(0, 40000)}\n\`\`\``;
              },
            ),
          ].join("\n\n"),
        },
      ],
    });
    const parsed = parseJsonObject<{
      summary?: unknown;
      edits?: unknown;
      commands?: unknown;
      risks?: unknown;
      structuredImpossibility?: unknown;
    }>(raw, fallback);
    const rejectedEdits: string[] = [];
    const rawEditCount = Array.isArray(parsed.edits)
      ? parsed.edits.filter((edit) => edit != null && typeof edit === "object").length
      : 0;
    const structuredImpossibility = structuredImpossibilityFrom(parsed.structuredImpossibility);
    const edits = Array.isArray(parsed.edits)
      ? parsed.edits
          .filter((edit) => edit != null && typeof edit === "object")
          .flatMap((edit, index): CodingEditOperation[] => {
            const parsedEdit = parseCodingEditOperation({
              rawEdit: edit as Record<string, unknown>,
              index,
              editContext: input.editContext,
              fileSnapshots: input.fileSnapshots,
            });
            if (parsedEdit.parseFailure !== undefined) {
              rejectedEdits.push(parsedEdit.parseFailure);
              return [];
            }
            return parsedEdit.edit === undefined ? [] : [parsedEdit.edit];
          })
      : [];
    return {
      summary: String(parsed.summary ?? fallback.summary),
      editStrategy,
      generation: {
        modelAvailable: true,
        modelRole: role,
        rawEditCount,
        rawCommandCount: Array.isArray(parsed.commands) ? parsed.commands.length : 0,
        ...(structuredImpossibility === undefined ? {} : { structuredImpossibility }),
      },
      ...(structuredImpossibility === undefined ? {} : { structuredImpossibility }),
      edits,
      commands: Array.isArray(parsed.commands)
        ? parsed.commands
            .filter((command) => command != null && typeof command === "object")
            .map((command) => ({
              command: String((command as { command?: unknown }).command ?? ""),
              args: Array.isArray((command as { args?: unknown }).args)
                ? ((command as { args: unknown[] }).args).map(String)
                : [],
              reason: String((command as { reason?: unknown }).reason ?? "Model-proposed verification."),
            }))
            .filter((command) => command.command.length > 0)
        : fallback.commands,
      risks: [
        ...(Array.isArray(parsed.risks) ? parsed.risks.map(String) : []),
        ...rejectedEdits.map((reason) => `Rejected malformed edit payload: ${reason}`),
      ],
      parseFailures: rejectedEdits,
    };
  } catch (error) {
    return {
      ...fallback,
      generation: {
        modelAvailable: true,
        modelRole: role,
        modelError: error instanceof Error ? error.message : String(error),
        rawEditCount: 0,
        rawCommandCount: 0,
      },
      risks: [error instanceof Error ? error.message : String(error), ...fallback.risks],
    };
  }
};

const structuredImpossibilityFrom = (value: unknown): CodingStructuredImpossibility | undefined => {
  const parsed = CodingStructuredImpossibilitySchema.safeParse(value);
  return parsed.success ? parsed.data : undefined;
};
