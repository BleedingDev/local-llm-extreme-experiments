import { createHash } from "node:crypto";
import { spawn } from "node:child_process";
import { chmod, mkdir, mkdtemp, readdir, readFile, rm, writeFile } from "node:fs/promises";
import { realpathSync } from "node:fs";
import { basename, dirname, isAbsolute, join, relative, resolve, sep } from "node:path";
import { tmpdir } from "node:os";
import {
  EvalCaseSchema,
  EvalRunResultSchema,
  FixtureWorkspaceSchema,
  type EvalAssertion,
  type EvalAssertionResult,
  type EvalCase,
  type EvalComparableContext,
  type EvalRunResult,
  type EvalRunStatus,
  type FixtureWorkspace,
} from "./types";
import { JsonValueSchema, type JsonValue } from "../optimizer/types";

export type MaterializedFixtureWorkspace = {
  fixtureWorkspaceId: string;
  workspacePath: string;
  cleanup: () => Promise<void>;
};

export type WorkspaceSnapshotFile = {
  path: string;
  hash: string;
  sizeBytes: number;
  content: string;
};

export type WorkspaceSnapshot = {
  workspacePath: string;
  files: WorkspaceSnapshotFile[];
};

export type ChangedWorkspaceFile = {
  path: string;
  changeKind: "added" | "modified" | "deleted";
  beforeHash?: string;
  afterHash?: string;
};

export type EvalCommandResult = {
  commandId: string;
  command: string[];
  cwd: string;
  exitCode: number | null;
  signal: NodeJS.Signals | null;
  stdout: string;
  stderr: string;
  timedOut: boolean;
  durationMs: number;
  startedAt: string;
  completedAt: string;
};

export type EvalExecutorContext = {
  signal: AbortSignal;
  timeoutMs: number;
};

export type EvalRunExecutorOutput = {
  telemetry?: JsonValue;
  commandResults?: EvalCommandResult[];
};

export type EvalRunExecutor = (
  workspacePath: string,
  evalCase: EvalCase,
  context: EvalExecutorContext,
) => Promise<void | EvalRunExecutorOutput>;

export type EvalRunExecution = {
  result: EvalRunResult;
  workspacePath: string;
  initialSnapshot: WorkspaceSnapshot;
  finalSnapshot: WorkspaceSnapshot;
  changedFiles: ChangedWorkspaceFile[];
  protectedPathChanges: ChangedWorkspaceFile[];
  setupCommandResults: EvalCommandResult[];
  verificationCommandResults: EvalCommandResult[];
  cleanup: () => Promise<void>;
};

export type RunEvalCaseInput = {
  evalCase: EvalCase;
  runRole: "baseline" | "candidate";
  context: EvalComparableContext;
  comparisonRunId: string;
  executor?: EvalRunExecutor;
  candidatePatchId?: string;
  timeoutMs?: number;
  baseDir?: string;
  signal?: AbortSignal;
};

export type EvalComparisonExecution = {
  baseline: EvalRunExecution;
  candidate: EvalRunExecution;
  cleanup: () => Promise<void>;
};

export type RunEvalComparisonInput = {
  evalCase: EvalCase;
  context: EvalComparableContext;
  baselineComparisonRunId: string;
  candidateComparisonRunId: string;
  baselineExecutor?: EvalRunExecutor;
  candidateExecutor: EvalRunExecutor;
  candidatePatchId?: string;
  timeoutMs?: number;
  baseDir?: string;
  signal?: AbortSignal;
};

type CommandPhase = "setup" | "verify";

type ExecutorOutcome =
  | { status: "completed"; output?: EvalRunExecutorOutput }
  | { status: "error"; error: string }
  | { status: "timeout" };

const MAX_COMMAND_OUTPUT_BYTES = 64 * 1024;
const TELEMETRY_DIR = ".bag/evals";

export const materializeFixtureWorkspace = async (
  workspaceInput: FixtureWorkspace,
  options: { baseDir?: string } = {},
): Promise<MaterializedFixtureWorkspace> => {
  const workspace = FixtureWorkspaceSchema.parse(workspaceInput);
  const baseDir = resolve(options.baseDir ?? tmpdir());
  await mkdir(baseDir, { recursive: true });
  assertPathInsideTempDirectory(baseDir);
  const materializedRoot = await mkdtemp(join(baseDir, "bleeding-agent-eval-"));
  const workspacePath = join(materializedRoot, "workspace");
  await mkdir(workspacePath, { recursive: true });
  assertPathInsideTempDirectory(workspacePath);

  for (const file of workspace.files) {
    const filePath = resolveWorkspacePath(workspacePath, file.path);
    await mkdir(dirname(filePath), { recursive: true });
    await writeFile(filePath, file.content, "utf8");
    if (file.executable) {
      await chmod(filePath, 0o755);
    }
  }

  return {
    fixtureWorkspaceId: workspace.fixtureWorkspaceId,
    workspacePath,
    cleanup: async () => {
      await rm(materializedRoot, { recursive: true, force: true });
    },
  };
};

export const snapshotWorkspace = async (workspacePath: string): Promise<WorkspaceSnapshot> => {
  assertPathInsideTempDirectory(workspacePath);
  const root = resolve(workspacePath);
  const files: WorkspaceSnapshotFile[] = [];

  const visit = async (directory: string): Promise<void> => {
    const entries = await readdir(directory, { withFileTypes: true });
    for (const entry of entries.sort((left, right) => left.name.localeCompare(right.name))) {
      const absolutePath = join(directory, entry.name);
      if (entry.isDirectory()) {
        await visit(absolutePath);
        continue;
      }
      if (!entry.isFile()) {
        continue;
      }
      const contentBuffer = await readFile(absolutePath);
      const relativePath = toPosixRelativePath(root, absolutePath);
      files.push({
        path: relativePath,
        hash: sha256(contentBuffer),
        sizeBytes: contentBuffer.byteLength,
        content: contentBuffer.toString("utf8"),
      });
    }
  };

  await visit(root);
  return {
    workspacePath: root,
    files: files.sort((left, right) => left.path.localeCompare(right.path)),
  };
};

export const detectChangedFiles = (
  before: WorkspaceSnapshot,
  after: WorkspaceSnapshot,
): ChangedWorkspaceFile[] => {
  const beforeByPath = snapshotMap(before);
  const afterByPath = snapshotMap(after);
  const paths = [...new Set([...before.files.map((file) => file.path), ...after.files.map((file) => file.path)])]
    .sort((left, right) => left.localeCompare(right));

  return paths.flatMap((path): ChangedWorkspaceFile[] => {
    const beforeFile = beforeByPath.get(path);
    const afterFile = afterByPath.get(path);
    if (beforeFile === undefined && afterFile !== undefined) {
      return [{ path, changeKind: "added", afterHash: afterFile.hash }];
    }
    if (beforeFile !== undefined && afterFile === undefined) {
      return [{ path, changeKind: "deleted", beforeHash: beforeFile.hash }];
    }
    if (beforeFile !== undefined && afterFile !== undefined && beforeFile.hash !== afterFile.hash) {
      return [{ path, changeKind: "modified", beforeHash: beforeFile.hash, afterHash: afterFile.hash }];
    }
    return [];
  });
};

export const detectProtectedPathChanges = (
  changedFiles: readonly ChangedWorkspaceFile[],
  protectedPaths: readonly string[],
): ChangedWorkspaceFile[] => {
  const normalizedProtectedPaths = protectedPaths.map(assertSafeRelativePath);
  return changedFiles.filter((changedFile) =>
    normalizedProtectedPaths.some(
      (protectedPath) =>
        changedFile.path === protectedPath || changedFile.path.startsWith(`${protectedPath}/`),
    ),
  );
};

export const runCommand = async (input: {
  commandId: string;
  command: readonly string[];
  cwd: string;
  timeoutMs: number;
  signal?: AbortSignal;
}): Promise<EvalCommandResult> => {
  assertPathInsideTempDirectory(input.cwd);
  if (input.command.length === 0) {
    throw new Error("eval command must include an executable");
  }

  const [executable, ...args] = input.command;
  if (executable === undefined) {
    throw new Error("eval command must include an executable");
  }

  const startedAt = now();
  const startedMs = performance.now();
  let stdout = "";
  let stderr = "";
  let exitCode: number | null = null;
  let exitSignal: NodeJS.Signals | null = null;
  let timedOut = false;
  let spawnError = "";

  const child = spawn(executable, args, {
    cwd: input.cwd,
    env: process.env,
    stdio: ["ignore", "pipe", "pipe"],
  });

  const killForCancellation = (): void => {
    timedOut = input.signal?.aborted !== true;
    if (child.exitCode === null && child.signalCode === null) {
      child.kill("SIGTERM");
    }
  };

  const timeout = setTimeout(killForCancellation, input.timeoutMs);
  input.signal?.addEventListener("abort", killForCancellation, { once: true });
  if (input.signal?.aborted === true) {
    killForCancellation();
  }

  child.stdout?.on("data", (chunk: Buffer) => {
    stdout = appendBounded(stdout, chunk.toString("utf8"));
  });
  child.stderr?.on("data", (chunk: Buffer) => {
    stderr = appendBounded(stderr, chunk.toString("utf8"));
  });
  child.on("error", (error) => {
    spawnError = error.message;
  });

  await new Promise<void>((resolvePromise) => {
    child.on("close", (code, signal) => {
      exitCode = code;
      exitSignal = signal;
      resolvePromise();
    });
  });

  clearTimeout(timeout);
  input.signal?.removeEventListener("abort", killForCancellation);

  if (spawnError.length > 0) {
    stderr = appendBounded(stderr, spawnError);
  }

  return {
    commandId: input.commandId,
    command: [...input.command],
    cwd: input.cwd,
    exitCode,
    signal: exitSignal,
    stdout,
    stderr,
    timedOut,
    durationMs: Math.round(performance.now() - startedMs),
    startedAt,
    completedAt: now(),
  };
};

export const runEvalCase = async (input: RunEvalCaseInput): Promise<EvalRunExecution> => {
  const evalCase = EvalCaseSchema.parse(input.evalCase);
  const runTimeoutMs = input.timeoutMs ?? evalCase.timeoutMs;
  const startedAt = now();
  const materialized = await materializeFixtureWorkspace(evalCase.fixtureWorkspace, {
    ...(input.baseDir === undefined ? {} : { baseDir: input.baseDir }),
  });
  const allCommandResults: EvalCommandResult[] = [];
  let initialSnapshot = await snapshotWorkspace(materialized.workspacePath);
  let finalSnapshot = initialSnapshot;
  let executorOutcome: ExecutorOutcome = { status: "completed" };

  const setupCommandResults = await runFixtureCommands({
    commands: evalCase.fixtureWorkspace.setupCommands,
    phase: "setup",
    workspacePath: materialized.workspacePath,
    timeoutMs: runTimeoutMs,
    ...(input.signal === undefined ? {} : { signal: input.signal }),
  });
  allCommandResults.push(...setupCommandResults);
  initialSnapshot = await snapshotWorkspace(materialized.workspacePath);

  if (input.executor !== undefined) {
    executorOutcome = await runExecutorWithTimeout({
      executor: input.executor,
      workspacePath: materialized.workspacePath,
      evalCase,
      timeoutMs: runTimeoutMs,
      ...(input.signal === undefined ? {} : { signal: input.signal }),
    });
    if (executorOutcome.status === "completed" && executorOutcome.output?.commandResults !== undefined) {
      allCommandResults.push(...executorOutcome.output.commandResults);
    }
  }

  const verificationCommandResults = await runFixtureCommands({
    commands: evalCase.fixtureWorkspace.verificationCommands,
    phase: "verify",
    workspacePath: materialized.workspacePath,
    timeoutMs: runTimeoutMs,
    ...(input.signal === undefined ? {} : { signal: input.signal }),
  });
  allCommandResults.push(...verificationCommandResults);
  finalSnapshot = await snapshotWorkspace(materialized.workspacePath);

  const changedFiles = detectChangedFiles(initialSnapshot, finalSnapshot);
  const protectedPathChanges = detectProtectedPathChanges(
    changedFiles,
    evalCase.fixtureWorkspace.protectedPaths,
  );
  const runResultId = runResultIdFor({
    comparisonRunId: input.comparisonRunId,
    evalCaseId: evalCase.evalCaseId,
    runRole: input.runRole,
  });
  const executorTelemetry = executorOutcome.status === "completed"
    ? executorOutcome.output?.telemetry
    : undefined;
  const telemetryArtifact = mergeTelemetryArtifact({
    schemaVersion: "eval-runner-telemetry.v1",
    runResultId,
    comparisonRunId: input.comparisonRunId,
    runRole: input.runRole,
    evalCaseId: evalCase.evalCaseId,
    fixtureWorkspaceId: evalCase.fixtureWorkspace.fixtureWorkspaceId,
    executorOutcome: summarizeExecutorOutcome(executorOutcome),
    setupCommandResults,
    verificationCommandResults,
    executorCommandResults: executorOutcome.status === "completed"
      ? executorOutcome.output?.commandResults ?? []
      : [],
    changedFiles,
    protectedPathChanges,
  }, executorTelemetry);
  const commandResultsById = commandResultMap(allCommandResults);
  const assertionResults = evalCase.assertions.map((assertion) =>
    evaluateAssertion({
      assertion,
      finalSnapshot,
      changedFiles,
      commandResultsById,
      telemetryArtifact,
    }),
  );

  const completedAt = now();
  const hardFailure = assertionResults.some(
    (result) => !result.passed && (result.severity === "failure" || result.severity === "critical"),
  );
  const timedOut = executorOutcome.status === "timeout" || allCommandResults.some((result) => result.timedOut);
  const errored =
    executorOutcome.status === "error" ||
    setupCommandResults.some((result) => result.exitCode === null && !result.timedOut);
  const status = determineStatus({ timedOut, errored, hardFailure });
  const score = assertionResults.length === 0
    ? 0
    : assertionResults.filter((result) => result.passed).length / assertionResults.length;
  const telemetryArtifactPath = await writeTelemetryArtifact(
    materialized.workspacePath,
    runResultId,
    telemetryArtifact,
  );

  const result = EvalRunResultSchema.parse({
    runResultId,
    comparisonRunId: input.comparisonRunId,
    runRole: input.runRole,
    evalCaseId: evalCase.evalCaseId,
    split: evalCase.split,
    context: input.context,
    ...(input.candidatePatchId === undefined ? {} : { candidatePatchId: input.candidatePatchId }),
    status,
    score,
    assertionResults,
    objectiveMetrics: [
      {
        metricId: "changed-file-count",
        name: "Changed file count",
        value: changedFiles.length,
        unit: "count",
        higherIsBetter: false,
      },
      {
        metricId: "protected-path-change-count",
        name: "Protected path change count",
        value: protectedPathChanges.length,
        unit: "count",
        higherIsBetter: false,
      },
    ],
    changedFiles: changedFiles.map((changedFile) => changedFile.path),
    telemetryArtifactPath,
    startedAt,
    completedAt,
  });

  return {
    result,
    workspacePath: materialized.workspacePath,
    initialSnapshot,
    finalSnapshot,
    changedFiles,
    protectedPathChanges,
    setupCommandResults,
    verificationCommandResults,
    cleanup: materialized.cleanup,
  };
};

export const runEvalComparison = async (
  input: RunEvalComparisonInput,
): Promise<EvalComparisonExecution> => {
  const baseline = await runEvalCase({
    evalCase: input.evalCase,
    runRole: "baseline",
    context: input.context,
    comparisonRunId: input.baselineComparisonRunId,
    ...(input.baselineExecutor === undefined ? {} : { executor: input.baselineExecutor }),
    ...(input.timeoutMs === undefined ? {} : { timeoutMs: input.timeoutMs }),
    ...(input.baseDir === undefined ? {} : { baseDir: input.baseDir }),
    ...(input.signal === undefined ? {} : { signal: input.signal }),
  });
  const candidate = await runEvalCase({
    evalCase: input.evalCase,
    runRole: "candidate",
    context: input.context,
    comparisonRunId: input.candidateComparisonRunId,
    executor: input.candidateExecutor,
    ...(input.candidatePatchId === undefined ? {} : { candidatePatchId: input.candidatePatchId }),
    ...(input.timeoutMs === undefined ? {} : { timeoutMs: input.timeoutMs }),
    ...(input.baseDir === undefined ? {} : { baseDir: input.baseDir }),
    ...(input.signal === undefined ? {} : { signal: input.signal }),
  });

  return {
    baseline,
    candidate,
    cleanup: async () => {
      await Promise.all([baseline.cleanup(), candidate.cleanup()]);
    },
  };
};

const runFixtureCommands = async (input: {
  commands: readonly (readonly string[])[];
  phase: CommandPhase;
  workspacePath: string;
  timeoutMs: number;
  signal?: AbortSignal;
}): Promise<EvalCommandResult[]> => {
  const results: EvalCommandResult[] = [];
  for (const [index, command] of input.commands.entries()) {
    results.push(await runCommand({
      commandId: commandIdFor(input.phase, index, command),
      command,
      cwd: input.workspacePath,
      timeoutMs: input.timeoutMs,
      ...(input.signal === undefined ? {} : { signal: input.signal }),
    }));
  }
  return results;
};

const runExecutorWithTimeout = async (input: {
  executor: EvalRunExecutor;
  workspacePath: string;
  evalCase: EvalCase;
  timeoutMs: number;
  signal?: AbortSignal;
}): Promise<ExecutorOutcome> => {
  const controller = new AbortController();
  const abort = (): void => controller.abort();
  input.signal?.addEventListener("abort", abort, { once: true });
  if (input.signal?.aborted === true) {
    controller.abort();
  }

  let timeout: ReturnType<typeof setTimeout> | undefined;
  const executorPromise = Promise.resolve()
    .then(() => input.executor(input.workspacePath, input.evalCase, {
      signal: controller.signal,
      timeoutMs: input.timeoutMs,
    }))
    .then(
      (output): ExecutorOutcome => ({
        status: "completed",
        ...normalizeExecutorOutput(output),
      }),
      (error: unknown): ExecutorOutcome => ({ status: "error", error: errorMessage(error) }),
    );
  const timeoutPromise = new Promise<ExecutorOutcome>((resolvePromise) => {
    timeout = setTimeout(() => {
      controller.abort();
      resolvePromise({ status: "timeout" });
    }, input.timeoutMs);
  });

  const outcome = await Promise.race([executorPromise, timeoutPromise]);
  if (timeout !== undefined) {
    clearTimeout(timeout);
  }
  input.signal?.removeEventListener("abort", abort);
  return outcome;
};

const evaluateAssertion = (input: {
  assertion: EvalAssertion;
  finalSnapshot: WorkspaceSnapshot;
  changedFiles: ChangedWorkspaceFile[];
  commandResultsById: Map<string, EvalCommandResult>;
  telemetryArtifact: JsonValue;
}): EvalAssertionResult => {
  const assertion = input.assertion;
  switch (assertion.assertionKind) {
    case "file_contains": {
      const file = snapshotMap(input.finalSnapshot).get(assertion.path);
      const passed = file?.content.includes(assertion.text) ?? false;
      return assertionResult(assertion, passed, {
        expected: assertion.text,
        actual: file?.content ?? null,
        message: passed
          ? `${assertion.path} contains expected text.`
          : `${assertion.path} does not contain expected text.`,
      });
    }
    case "file_not_contains": {
      const file = snapshotMap(input.finalSnapshot).get(assertion.path);
      const passed = file === undefined || !file.content.includes(assertion.text);
      return assertionResult(assertion, passed, {
        expected: `not ${assertion.text}`,
        actual: file?.content ?? null,
        message: passed
          ? `${assertion.path} does not contain forbidden text.`
          : `${assertion.path} contains forbidden text.`,
      });
    }
    case "command_exit_code": {
      const commandResult = input.commandResultsById.get(assertion.commandId);
      const actual = commandResult?.exitCode ?? null;
      const passed = actual === assertion.expectedExitCode;
      return assertionResult(assertion, passed, {
        expected: assertion.expectedExitCode,
        actual,
        message: commandResult === undefined
          ? `Command ${assertion.commandId} was not run.`
          : `Command ${assertion.commandId} exited with ${String(actual)}.`,
      });
    }
    case "no_forbidden_path_changed": {
      const protectedChanges = detectProtectedPathChanges(input.changedFiles, assertion.paths);
      const passed = protectedChanges.length === 0;
      return assertionResult(assertion, passed, {
        expected: [],
        actual: protectedChanges.map((changedFile) => changedFile.path),
        message: passed
          ? "No forbidden paths changed."
          : `Forbidden paths changed: ${protectedChanges.map((changedFile) => changedFile.path).join(", ")}`,
      });
    }
    case "json_pointer_equals": {
      if (assertion.artifact !== "telemetry") {
        return assertionResult(assertion, false, {
          expected: assertion.expected,
          actual: null,
          message: `JSON pointer assertions for ${assertion.artifact} are not available in the runner.`,
        });
      }
      const actual = jsonPointerGet(input.telemetryArtifact, assertion.pointer);
      const passed = jsonEquals(actual, assertion.expected);
      return assertionResult(assertion, passed, {
        expected: assertion.expected,
        actual: actual ?? null,
        message: passed
          ? `${assertion.pointer} matched expected telemetry.`
          : `${assertion.pointer} did not match expected telemetry.`,
      });
    }
    case "llm_judge_min_score":
      return assertionResult(assertion, false, {
        expected: assertion.minimumScore,
        actual: null,
        message: "LLM judge assertions are reserved for the scorer lane.",
      });
  }
};

const assertionResult = (
  assertion: EvalAssertion,
  passed: boolean,
  detail: { message: string; expected?: JsonValue; actual?: JsonValue },
): EvalAssertionResult => ({
  assertionId: assertion.assertionId,
  assertionKind: assertion.assertionKind,
  passed,
  severity: assertion.severity,
  message: detail.message,
  ...(detail.expected === undefined ? {} : { expected: detail.expected }),
  ...(detail.actual === undefined ? {} : { actual: detail.actual }),
});

const determineStatus = (input: {
  timedOut: boolean;
  errored: boolean;
  hardFailure: boolean;
}): EvalRunStatus => {
  if (input.timedOut) {
    return "timeout";
  }
  if (input.errored) {
    return "error";
  }
  if (input.hardFailure) {
    return "failed";
  }
  return "passed";
};

const normalizeExecutorOutput = (
  output: void | EvalRunExecutorOutput,
): { output?: EvalRunExecutorOutput } => {
  if (output == null) {
    return {};
  }
  const normalized: EvalRunExecutorOutput = {};
  if (output.telemetry !== undefined) {
    normalized.telemetry = JsonValueSchema.parse(output.telemetry);
  }
  if (output.commandResults !== undefined) {
    normalized.commandResults = output.commandResults;
  }
  return { output: normalized };
};

const summarizeExecutorOutcome = (outcome: ExecutorOutcome): JsonValue => {
  switch (outcome.status) {
    case "completed":
      return {
        status: outcome.status,
        hasTelemetry: outcome.output?.telemetry !== undefined,
        commandResultCount: outcome.output?.commandResults?.length ?? 0,
      };
    case "error":
      return {
        status: outcome.status,
        error: outcome.error,
      };
    case "timeout":
      return {
        status: outcome.status,
      };
  }
};

const mergeTelemetryArtifact = (
  base: Record<string, JsonValue>,
  executorTelemetry: JsonValue | undefined,
): JsonValue => {
  if (executorTelemetry == null) {
    return base;
  }
  if (isJsonObject(executorTelemetry)) {
    return {
      ...base,
      ...executorTelemetry,
      executorTelemetry,
    };
  }
  return {
    ...base,
    executorTelemetry,
  };
};

const jsonPointerGet = (value: JsonValue, pointer: string): JsonValue | undefined => {
  if (pointer === "") {
    return value;
  }
  const segments = pointer.split("/").slice(1).map((segment) =>
    segment.replaceAll("~1", "/").replaceAll("~0", "~"),
  );
  let current: JsonValue | undefined = value;
  for (const segment of segments) {
    if (Array.isArray(current)) {
      if (!/^(0|[1-9]\d*)$/.test(segment)) {
        return undefined;
      }
      current = current[Number(segment)];
      continue;
    }
    if (isJsonObject(current)) {
      current = current[segment];
      continue;
    }
    return undefined;
  }
  return current;
};

const jsonEquals = (left: JsonValue | undefined, right: JsonValue): boolean =>
  left !== undefined && JSON.stringify(left) === JSON.stringify(right);

const isJsonObject = (value: JsonValue | undefined): value is Record<string, JsonValue> =>
  value != null && typeof value === "object" && !Array.isArray(value);

const commandResultMap = (results: readonly EvalCommandResult[]): Map<string, EvalCommandResult> => {
  const map = new Map<string, EvalCommandResult>();
  for (const result of results) {
    map.set(result.commandId, result);
  }
  return map;
};

const commandIdFor = (phase: CommandPhase, index: number, command: readonly string[]): string => {
  const candidatePath = command[1] ?? command[0] ?? `${phase}-${index + 1}`;
  const candidateName = basename(candidatePath).replace(/\.[cm]?js$/, "").replace(/\.[^.]+$/, "");
  const slug = candidateName
    .replace(/[^A-Za-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "")
    .toLowerCase();
  return `${phase}.${slug.length === 0 ? String(index + 1) : slug}`;
};

const writeTelemetryArtifact = async (
  workspacePath: string,
  runResultId: string,
  artifact: JsonValue,
): Promise<string> => {
  const relativePath = assertSafeRelativePath(`${TELEMETRY_DIR}/${runResultId}.json`);
  const artifactPath = resolveWorkspacePath(workspacePath, relativePath);
  await mkdir(dirname(artifactPath), { recursive: true });
  await writeFile(artifactPath, `${JSON.stringify(artifact, null, 2)}\n`, "utf8");
  return relativePath;
};

const runResultIdFor = (input: {
  comparisonRunId: string;
  evalCaseId: string;
  runRole: "baseline" | "candidate";
}): string => `run.${input.runRole}.${input.evalCaseId}.${hashText(input.comparisonRunId).slice(0, 12)}`;

const snapshotMap = (snapshot: WorkspaceSnapshot): Map<string, WorkspaceSnapshotFile> =>
  new Map(snapshot.files.map((file) => [file.path, file]));

const resolveWorkspacePath = (workspacePath: string, relativePath: string): string => {
  const safeRelativePath = assertSafeRelativePath(relativePath);
  const root = resolve(workspacePath);
  const absolutePath = resolve(root, ...safeRelativePath.split("/"));
  if (!isPathInside(root, absolutePath)) {
    throw new Error(`eval path escapes workspace: ${relativePath}`);
  }
  return absolutePath;
};

const assertSafeRelativePath = (path: string): string => {
  if (
    path.length === 0 ||
    isAbsolute(path) ||
    path.includes("\\") ||
    path.split("/").some((segment) => segment.length === 0 || segment === "." || segment === "..")
  ) {
    throw new Error(`unsafe eval relative path: ${path}`);
  }
  return path;
};

const assertPathInsideTempDirectory = (path: string): void => {
  const realTemp = realpathSync(tmpdir());
  const realPath = realpathSync(resolve(path));
  if (!isPathInside(realTemp, realPath)) {
    throw new Error(`eval workspace must be inside the OS temp directory: ${path}`);
  }
};

const isPathInside = (parent: string, child: string): boolean => {
  const pathBetween = relative(parent, child);
  return pathBetween === "" || (!pathBetween.startsWith("..") && !isAbsolute(pathBetween));
};

const toPosixRelativePath = (root: string, absolutePath: string): string => {
  const path = relative(root, absolutePath).split(sep).join("/");
  return assertSafeRelativePath(path);
};

const appendBounded = (current: string, next: string): string => {
  const combined = current + next;
  if (Buffer.byteLength(combined) <= MAX_COMMAND_OUTPUT_BYTES) {
    return combined;
  }
  return combined.slice(0, MAX_COMMAND_OUTPUT_BYTES);
};

const sha256 = (content: Buffer): string => `sha256:${createHash("sha256").update(content).digest("hex")}`;

const hashText = (content: string): string => createHash("sha256").update(content).digest("hex");

const now = (): string => new Date().toISOString();

const errorMessage = (error: unknown): string => error instanceof Error ? error.message : String(error);
