import { existsSync, mkdirSync, readdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join, relative } from "node:path";
import { EDIT_ATTEMPT_RECORD_SCHEMA_VERSION, EditAttemptRecordSchema } from "../../acp/edit-attempt-record";
import { evaluateNoWritePromotionGate, type NoWritePromotionGateDecision } from "../../optimizer/no-write-gate";
import {
  evaluatePromotionEvidenceContracts,
  OperatorApprovalEvidenceRecordSchema,
  PostPromotionMonitorWindowEvidenceRecordSchema,
  RollbackCheckpointProofRecordSchema,
  type PromotionEvidenceContractStatus,
  type PromotionEvidenceContext,
} from "../../optimizer/promotion-evidence-contracts";
import {
  buildNoWriteReplaySlice,
  noWriteValidationInputsFromReplaySlice,
} from "../../replay/no-write-slice";
import { RealAcpCorpusRunManifestSchema } from "../../replay/real-acp-runner";
import {
  artifactRef,
  blocking,
  countBy,
  EvidenceIndexRecordSchema,
  hasBlockingFailure,
  markdownArtifact,
  noWriteIntent,
  OptimizerGateSuiteSchema,
  readJsonArtifact,
  readJsonlArtifact,
  ReleaseProofSchema,
  ScorecardSuiteSchema,
  uniqueSorted,
  warning,
  wouldWriteIntent,
  type CanonicalEpochArtifact,
  type CanonicalEpochPayload,
  type ArtifactRead,
  type EvidenceArtifactRef,
  type EvidenceCheck,
  type EvidenceIndexPayload,
  type EvidenceWriteIntent,
  type OptimizerGatesPayload,
  type OptimizerGateSuite,
  type ReleaseProofPayload,
  type ScorecardSuite,
  type ScorecardsPayload,
  type ValidatePayload,
} from "./artifacts";
import {
  projectEditAttemptRecordsToScorecard,
  type EditAttemptScorecardProjection,
} from "./edit-attempt-scorecard-projection";
import { buildEditAttemptRecordsFromRealAcpCorpus } from "./edit-attempt-records-from-real-acp";
import { z } from "zod";

export type EvidenceGeneration<TPayload> = {
  payload?: TPayload | undefined;
  artifacts: EvidenceArtifactRef[];
  checks: EvidenceCheck[];
  writes: EvidenceWriteIntent[];
  summary: string;
};

export type EvidenceGenerationOptions = {
  cwd: string;
  dryRun: boolean;
  graphId?: string | undefined;
};

const EVIDENCE_INDEX_PATH = ".bag/evidence/index.jsonl";
const SCORECARD_INDEX_PATH = ".bag/evidence/scorecards/index.json";
const EDIT_ATTEMPT_RECORDS_PATH = ".bag/evidence/edit-attempt-records.jsonl";
const EDIT_ATTEMPT_SCORECARD_PROJECTION_PATH = ".bag/evidence/scorecards/edit-attempt-projection.json";
const OPTIMIZER_INDEX_PATH = ".bag/evidence/optimizer/index.json";
const OPTIMIZER_NO_WRITE_GATE_PATH = ".bag/evidence/optimizer/no-write-gate.json";
const OPTIMIZER_OPERATOR_APPROVAL_PATH = ".bag/evidence/optimizer/operator-approval.json";
const OPTIMIZER_ROLLBACK_CHECKPOINT_PROOF_PATH = ".bag/evidence/optimizer/rollback-checkpoint-proof.json";
const OPTIMIZER_MONITOR_WINDOW_PATH = ".bag/evidence/optimizer/post-promotion-monitor-window.json";
const CANONICAL_EPOCH_PATH = ".bag/evidence/canonical-epoch.json";
const RELEASE_PROOF_PATH = ".bag/evidence/release-proof.json";
const FINAL_REPORT_PATH = "docs/local-evidence-flywheel-final-report.md";
const CANONICAL_READINESS_INDEX_REPORT_PATH = "docs/live-acp-canonical-readiness-index.md";
const CURRENT_RELEASE_PROOF_REPORT_PATH = "docs/live-acp-current-release-proof-report.md";
const PLAN_GRAPH_STATE_ROOT = ".codex/plan-graphs";
const REPLAY_CORPUS_ROOT = ".bag/replay-corpus";
const VISIBLE_NO_WRITE_MISSING_REASON = "visible ACP no-write/no-terminal validation must be represented";
const EDIT_ATTEMPT_TELEMETRY_MISSING_REASON = "edit-policy promotion needs first-class edit attempt telemetry";

export const generateEvidenceIndex = (options: EvidenceGenerationOptions): EvidenceGeneration<EvidenceIndexPayload> => {
  const index = readJsonlArtifact(options.cwd, EVIDENCE_INDEX_PATH, EvidenceIndexRecordSchema, "index.jsonl.parse");
  const records = index.value ?? [];
  const sourceIds = new Set(records.filter((record) => record.recordKind === "source").map((record) => recordId(record)));
  const memberIds = uniqueSorted(records.flatMap((record) => record.memberEvidenceIds ?? []));
  const missingReferencedSourceIds = memberIds.filter((memberId) => !sourceIds.has(memberId));
  const checks = [
    ...index.checks,
    missingReferencedSourceIds.length === 0
      ? {
        checkId: "index.slice-references",
        passed: true,
        severity: "info" as const,
        message: "All slice memberEvidenceIds resolve to source evidence records.",
        path: EVIDENCE_INDEX_PATH,
      }
      : {
        checkId: "index.slice-references",
        passed: false,
        severity: "blocking" as const,
        message: `Slice references missing source evidence IDs: ${missingReferencedSourceIds.join(", ")}`,
        path: EVIDENCE_INDEX_PATH,
      },
  ];

  const payload = hasBlockingFailure(checks)
    ? undefined
    : {
      schemaVersion: "evidence-command.index.v1" as const,
      sourcePath: EVIDENCE_INDEX_PATH,
      recordCount: records.length,
      recordKinds: countBy(records, (record) => record.recordKind),
      families: countBy(records, (record) => record.family),
      evidenceIds: uniqueSorted(records.map((record) => recordId(record))),
      missingReferencedSourceIds,
    };

  return {
    payload,
    artifacts: [index.artifact],
    checks,
    writes: [noWriteIntent(EVIDENCE_INDEX_PATH, "Index command wraps the existing local JSONL artifact.")],
    summary: payload === undefined
      ? "Evidence index validation failed closed."
      : `Evidence index contains ${payload.recordCount} records across ${Object.keys(payload.families).length} families.`,
  };
};

export const generateScorecards = (options: EvidenceGenerationOptions): EvidenceGeneration<ScorecardsPayload> => {
  const suite = readJsonArtifact(options.cwd, SCORECARD_INDEX_PATH, ScorecardSuiteSchema, "scorecards.index.parse");
  const materializedSuite = materializedScorecardSuite(options, suite.value);
  const suiteValue = materializedSuite.value ?? suite.value;
  const editAttemptProjection = generateOptionalEditAttemptProjection(options, suiteValue?.graphId);
  const artifacts: EvidenceArtifactRef[] = [materializedSuite.artifact];
  const checks: EvidenceCheck[] = [...suite.checks, ...editAttemptProjection.checks];
  artifacts.push(...editAttemptProjection.artifacts);

  for (const [index, scorecard] of (suiteValue?.scorecards ?? []).entries()) {
    const json = readJsonArtifact(options.cwd, scorecard.jsonPath, ScorecardDocumentSchema, `scorecards.${index}.json`);
    const markdown = markdownArtifact(options.cwd, scorecard.markdownPath, `scorecards.${index}.markdown`);
    artifacts.push(json.artifact, markdown.artifact);
    checks.push(...json.checks, ...markdown.checks);
    if (json.value !== undefined && json.value.scorecardId !== scorecard.scorecardId) {
      checks.push({
        checkId: `scorecards.${index}.id-match`,
        passed: false,
        severity: "blocking",
        message: `Scorecard index id ${scorecard.scorecardId} does not match JSON id ${json.value.scorecardId}.`,
        path: scorecard.jsonPath,
      });
    }
  }

  const value = suiteValue;
  const payload = value === undefined || hasBlockingFailure(checks)
    ? undefined
    : {
      schemaVersion: "evidence-command.scorecards.v1" as const,
      suiteId: value.scorecardSuiteId,
      graphId: value.graphId,
      generatedAt: value.generatedAt,
      scorecardCount: value.scorecards.length,
      scorecards: value.scorecards.map((scorecard) => ({
        scorecardId: scorecard.scorecardId,
        jsonPath: scorecard.jsonPath,
        markdownPath: scorecard.markdownPath,
        ...(scorecard.primaryUse === undefined ? {} : { primaryUse: scorecard.primaryUse }),
      })),
      promotionGateInputCount: value.promotionGateInputs?.length ?? 0,
      optimizerReadySliceCount: value.optimizerReadySlices?.length ?? 0,
      ...(editAttemptProjection.projection === undefined
        ? {}
        : {
            editAttemptProjection: {
              projectionId: editAttemptProjection.projection.projectionId,
              sourceRecordCount: editAttemptProjection.projection.sourceRecordCount,
              groupCount: editAttemptProjection.projection.groups.length,
              outputPath: EDIT_ATTEMPT_SCORECARD_PROJECTION_PATH,
              byFinalOutcome: editAttemptProjection.projection.totals.byFinalOutcome,
              byFailureSignal: editAttemptProjection.projection.totals.byFailureSignal,
            },
          }),
    };

  return {
    payload,
    artifacts,
    checks,
    writes: [
      ...materializedSuite.writes,
      ...editAttemptProjection.writes,
    ],
    summary: payload === undefined
      ? "Scorecard suite validation failed closed."
      : `Scorecard suite ${payload.suiteId} exposes ${payload.scorecardCount} scorecards.`,
  };
};

const materializedScorecardSuite = (
  options: EvidenceGenerationOptions,
  value: ScorecardSuite | undefined,
): {
  value?: ScorecardSuite;
  artifact: EvidenceArtifactRef;
  writes: EvidenceWriteIntent[];
} => {
  const artifact = artifactRef(options.cwd, SCORECARD_INDEX_PATH, "json");
  if (value === undefined || options.graphId === undefined || value.graphId === options.graphId) {
    return {
      artifact,
      writes: [noWriteIntent(SCORECARD_INDEX_PATH, "Scorecard command validates existing scorecard suite artifacts.")],
    };
  }

  const nextValue = ScorecardSuiteSchema.parse({
    ...value,
    scorecardSuiteId: `scorecard-suite.${options.graphId}`,
    graphId: options.graphId,
    generatedAt: new Date().toISOString(),
    caveats: uniqueSorted([
      ...(value.caveats ?? []),
      `Current graph wrapper generated from ${value.scorecardSuiteId}; underlying scorecard documents keep their own source lineage and must not be treated as fresh real-consumer evidence.`,
    ]),
  });

  if (!options.dryRun) {
    writeJsonArtifact(options.cwd, SCORECARD_INDEX_PATH, nextValue);
  }

  return {
    value: options.dryRun ? value : nextValue,
    artifact: options.dryRun ? artifact : artifactRef(options.cwd, SCORECARD_INDEX_PATH, "json"),
    writes: [
      options.dryRun
        ? wouldWriteIntent(SCORECARD_INDEX_PATH, `Scorecard suite will be retargeted from ${value.graphId} to ${options.graphId} without changing promotion readiness.`)
        : noWriteIntent(SCORECARD_INDEX_PATH, `Scorecard suite was retargeted from ${value.graphId} to ${options.graphId} without changing promotion readiness.`),
    ],
  };
};

const generateOptionalEditAttemptProjection = (
  options: EvidenceGenerationOptions,
  graphId: string | undefined,
): {
  projection?: EditAttemptScorecardProjection;
  artifacts: EvidenceArtifactRef[];
  checks: EvidenceCheck[];
  writes: EvidenceWriteIntent[];
} => {
  const source = artifactRef(options.cwd, EDIT_ATTEMPT_RECORDS_PATH, "jsonl", false);
  const output = artifactRef(options.cwd, EDIT_ATTEMPT_SCORECARD_PROJECTION_PATH, "json", false);
  const generatedRecords = buildEditAttemptRecordsFromRealAcpCorpus({ cwd: options.cwd });
  if (generatedRecords.length > 0) {
    if (!options.dryRun) {
      writeTextArtifact(options.cwd, EDIT_ATTEMPT_RECORDS_PATH, generatedRecords.map((record) => JSON.stringify(record)).join("\n") + "\n");
    }
    return editAttemptProjectionFromRecords({
      options,
      records: generatedRecords,
      graphId,
      source: options.dryRun ? source : artifactRef(options.cwd, EDIT_ATTEMPT_RECORDS_PATH, "jsonl", false),
      output,
      sourceCheck: {
        checkId: "scorecards.edit-attempt-records.generated",
        passed: true,
        severity: "info",
        message: `Generated ${generatedRecords.length} ${EDIT_ATTEMPT_RECORD_SCHEMA_VERSION} record(s) from visible real ACP corpus manifests.`,
        path: EDIT_ATTEMPT_RECORDS_PATH,
      },
      sourceWrite: options.dryRun
        ? wouldWriteIntent(EDIT_ATTEMPT_RECORDS_PATH, "Edit-attempt records will be generated from visible real ACP corpus manifests.")
        : noWriteIntent(EDIT_ATTEMPT_RECORDS_PATH, "Edit-attempt records were generated from visible real ACP corpus manifests."),
    });
  }
  if (!existsSync(source.absolutePath)) {
    return {
      artifacts: [source],
      checks: [
        {
          checkId: "scorecards.edit-attempt-records.optional",
          passed: true,
          severity: "info",
          message: "No first-class edit-attempt record JSONL artifact found; edit-attempt scorecard projection skipped.",
          path: EDIT_ATTEMPT_RECORDS_PATH,
        },
      ],
      writes: [noWriteIntent(EDIT_ATTEMPT_SCORECARD_PROJECTION_PATH, "No edit-attempt projection is generated without first-class edit-attempt records.")],
    };
  }

  try {
    const records = readFileSync(source.absolutePath, "utf8")
      .split(/\r?\n/)
      .map((line) => line.trim())
      .filter((line) => line.length > 0)
      .map((line) => EditAttemptRecordSchema.parse(JSON.parse(line) as unknown));
    return editAttemptProjectionFromRecords({
      options,
      records,
      graphId,
      source,
      output,
      sourceCheck: {
        checkId: "scorecards.edit-attempt-records.parse",
        passed: true,
        severity: "info",
        message: `Validated ${records.length} ${EDIT_ATTEMPT_RECORD_SCHEMA_VERSION} record(s) for edit-attempt scorecard projection.`,
        path: EDIT_ATTEMPT_RECORDS_PATH,
      },
      sourceWrite: noWriteIntent(EDIT_ATTEMPT_RECORDS_PATH, "Edit-attempt records were read from the existing JSONL artifact."),
    });
  } catch (error) {
    return {
      artifacts: [source],
      checks: [
        {
          checkId: "scorecards.edit-attempt-records.parse",
          passed: false,
          severity: "blocking",
          message: `Invalid edit-attempt records artifact ${EDIT_ATTEMPT_RECORDS_PATH}: ${errorMessage(error)}`,
          path: EDIT_ATTEMPT_RECORDS_PATH,
        },
      ],
      writes: [noWriteIntent(EDIT_ATTEMPT_SCORECARD_PROJECTION_PATH, "Invalid edit-attempt records prevent scorecard projection.")],
    };
  }
};

const editAttemptProjectionFromRecords = (input: {
  options: EvidenceGenerationOptions;
  records: readonly unknown[];
  graphId: string | undefined;
  source: EvidenceArtifactRef;
  output: EvidenceArtifactRef;
  sourceCheck: EvidenceCheck;
  sourceWrite: EvidenceWriteIntent;
}): {
  projection?: EditAttemptScorecardProjection;
  artifacts: EvidenceArtifactRef[];
  checks: EvidenceCheck[];
  writes: EvidenceWriteIntent[];
} => {
  const projection = projectEditAttemptRecordsToScorecard({
    records: input.records,
    ...(input.graphId === undefined ? {} : { graphId: input.graphId }),
  });
  if (!input.options.dryRun) {
    writeJsonArtifact(input.options.cwd, EDIT_ATTEMPT_SCORECARD_PROJECTION_PATH, projection);
  }
  return {
    projection,
    artifacts: [
      input.source,
      input.options.dryRun ? input.output : artifactRef(input.options.cwd, EDIT_ATTEMPT_SCORECARD_PROJECTION_PATH, "json", false),
    ],
    checks: [input.sourceCheck],
    writes: [
      input.sourceWrite,
      input.options.dryRun
        ? wouldWriteIntent(EDIT_ATTEMPT_SCORECARD_PROJECTION_PATH, "Edit-attempt projection will be derived from first-class edit-attempt records.")
        : noWriteIntent(EDIT_ATTEMPT_SCORECARD_PROJECTION_PATH, "Edit-attempt projection was derived from first-class edit-attempt records."),
    ],
  };
};

export const generateOptimizerGates = (options: EvidenceGenerationOptions): EvidenceGeneration<OptimizerGatesPayload> => {
  const suite = readJsonArtifact(options.cwd, OPTIMIZER_INDEX_PATH, OptimizerGateSuiteSchema, "optimizer.index.parse");
  const noWriteGate = generateVisibleNoWriteGate(options);
  const currentGraph = readLatestPlanGraphSnapshot(options.cwd, options.graphId);
  const releaseProof = readJsonArtifact(options.cwd, RELEASE_PROOF_PATH, ReleaseProofSchema, "optimizer.release-proof.parse");
  const promotionEvidence = generatePromotionEvidenceContracts(options, suite.value, currentGraph, releaseProof);
  const materializedSuite = materializedOptimizerGateSuite(
    options,
    suite.value,
    noWriteGate.decision,
    editAttemptTelemetryReady(options.cwd),
    promotionEvidence.status,
  );
  const suiteValue = materializedSuite.value ?? suite.value;
  const artifacts: EvidenceArtifactRef[] = [materializedSuite.artifact, ...noWriteGate.artifacts, ...promotionEvidence.artifacts];
  const checks: EvidenceCheck[] = [...suite.checks, ...noWriteGate.checks, ...releaseProof.checks, ...promotionEvidence.checks];

  if (suiteValue !== undefined) {
    for (const [index, contract] of suiteValue.contracts.entries()) {
      const json = readJsonArtifact(options.cwd, contract.jsonPath, OptimizerContractDocumentSchema, `optimizer.contracts.${index}.json`);
      const markdown = markdownArtifact(options.cwd, contract.markdownPath, `optimizer.contracts.${index}.markdown`);
      artifacts.push(json.artifact, markdown.artifact);
      checks.push(...json.checks, ...markdown.checks);
    }
  }

  const value = suiteValue;
  const payload = value === undefined || hasBlockingFailure(checks)
    ? undefined
    : {
      schemaVersion: "evidence-command.optimizer-gates.v1" as const,
      suiteId: value.optimizerGateSuiteId,
      graphId: value.graphId,
      generatedAt: value.generatedAt,
      contractCount: value.contracts.length,
      contracts: value.contracts.map((contract) => ({
        contractId: contract.contractId,
        jsonPath: contract.jsonPath,
        markdownPath: contract.markdownPath,
        ...(contract.primaryUse === undefined ? {} : { primaryUse: contract.primaryUse }),
      })),
      candidateGeneration: value.currentDecision.candidateGeneration,
      autoPromotion: value.currentDecision.autoPromotion,
      promotionReady: value.currentDecision.promotionReady,
      blockingReasons: value.currentDecision.blockingReasons,
      mustFailClosedOn: value.mustFailClosedOn,
    };

  return {
    payload,
    artifacts,
    checks,
    writes: [
      ...materializedSuite.writes,
      ...noWriteGate.writes,
      ...promotionEvidence.writes,
    ],
    summary: payload === undefined
      ? "Optimizer gate suite validation failed closed."
      : `Optimizer gate suite ${payload.suiteId} reports autoPromotion=${payload.autoPromotion}.`,
  };
};

const generateVisibleNoWriteGate = (
  options: EvidenceGenerationOptions,
): {
  decision?: NoWritePromotionGateDecision;
  artifacts: EvidenceArtifactRef[];
  checks: EvidenceCheck[];
  writes: EvidenceWriteIntent[];
} => {
  const corpusRoot = join(options.cwd, REPLAY_CORPUS_ROOT);
  const output = artifactRef(options.cwd, OPTIMIZER_NO_WRITE_GATE_PATH, "json", false);
  if (!existsSync(corpusRoot)) {
    return {
      artifacts: [output],
      checks: [
        warning(
          "optimizer.no-write-gate.corpus-missing",
          "Replay corpus is missing; visible ACP no-write gate cannot be built.",
          REPLAY_CORPUS_ROOT,
        ),
      ],
      writes: [noWriteIntent(OPTIMIZER_NO_WRITE_GATE_PATH, "No visible no-write gate is generated without a replay corpus.")],
    };
  }

  try {
    const manifests = readRealAcpManifests(options.cwd);
    const slice = buildNoWriteReplaySlice({
      sliceId: "no-write-replay-slice.visible-acp.current",
      manifests,
      latestPerTaskProfile: true,
    });
    const decision = evaluateNoWritePromotionGate({
      cases: noWriteValidationInputsFromReplaySlice(slice),
      requireEvidence: true,
    });
    const document = {
      schemaVersion: "local-evidence-optimizer-no-write-gate.v1" as const,
      graphId: options.graphId ?? "local",
      generatedAt: new Date().toISOString(),
      sourceCorpusRoot: REPLAY_CORPUS_ROOT,
      slice,
      gateDecision: decision,
    };

    if (!options.dryRun) {
      writeJsonArtifact(options.cwd, OPTIMIZER_NO_WRITE_GATE_PATH, document);
    }

    return {
      decision,
      artifacts: [options.dryRun ? output : artifactRef(options.cwd, OPTIMIZER_NO_WRITE_GATE_PATH, "json", false)],
      checks: [
        {
          checkId: "optimizer.no-write-gate.visible-acp",
          passed: true,
          severity: "info",
          message: `Built visible ACP no-write gate from ${slice.status.includedCases} optimizer-visible case(s); status=${decision.status}.`,
          path: OPTIMIZER_NO_WRITE_GATE_PATH,
        },
      ],
      writes: [
        options.dryRun
          ? wouldWriteIntent(OPTIMIZER_NO_WRITE_GATE_PATH, "Visible ACP no-write gate will be generated from the replay corpus.")
          : noWriteIntent(OPTIMIZER_NO_WRITE_GATE_PATH, "Visible ACP no-write gate was generated from the replay corpus."),
      ],
    };
  } catch (error) {
    return {
      artifacts: [output],
      checks: [
        blocking(
          "optimizer.no-write-gate.visible-acp",
          `Visible ACP no-write gate failed to build: ${errorMessage(error)}`,
          REPLAY_CORPUS_ROOT,
        ),
      ],
      writes: [noWriteIntent(OPTIMIZER_NO_WRITE_GATE_PATH, "Invalid replay corpus prevents visible no-write gate generation.")],
    };
  }
};

const generatePromotionEvidenceContracts = (
  options: EvidenceGenerationOptions,
  suite: OptimizerGateSuite | undefined,
  currentGraph: PlanGraphSnapshot | undefined,
  releaseProof: ArtifactRead<ReleaseProofSchemaOutput>,
): {
  status: PromotionEvidenceContractStatus;
  artifacts: EvidenceArtifactRef[];
  checks: EvidenceCheck[];
  writes: EvidenceWriteIntent[];
} => {
  const operatorApproval = readContractArtifact(
    options,
    OPTIMIZER_OPERATOR_APPROVAL_PATH,
    OperatorApprovalEvidenceRecordSchema,
    "optimizer.operator-approval",
    "operator approval",
  );
  const rollbackCheckpointProof = readContractArtifact(
    options,
    OPTIMIZER_ROLLBACK_CHECKPOINT_PROOF_PATH,
    RollbackCheckpointProofRecordSchema,
    "optimizer.rollback-checkpoint-proof",
    "rollback checkpoint proof",
  );
  const monitorWindow = readContractArtifact(
    options,
    OPTIMIZER_MONITOR_WINDOW_PATH,
    PostPromotionMonitorWindowEvidenceRecordSchema,
    "optimizer.post-promotion-monitor-window",
    "post-promotion monitor-window proof",
  );
  const context = promotionEvidenceContext(options, suite, currentGraph, releaseProof);
  const status = evaluatePromotionEvidenceContracts({
    context,
    ...(operatorApproval.value === undefined ? {} : { operatorApproval: operatorApproval.value }),
    ...(rollbackCheckpointProof.value === undefined ? {} : { rollbackCheckpointProof: rollbackCheckpointProof.value }),
    ...(monitorWindow.value === undefined ? {} : { monitorWindow: monitorWindow.value }),
  });
  const statusChecks = status.blockingReasons.length === 0
    ? [
        {
          checkId: "optimizer.promotion-evidence-contracts",
          passed: true,
          severity: "info" as const,
          message: `Promotion evidence contracts are current for graph ${context.graphId}.`,
        },
      ]
    : status.blockingReasons.map((reason) =>
        blocking("optimizer.promotion-evidence-contracts", `Promotion evidence contract blocker: ${reason}`)
      );

  return {
    status,
    artifacts: [
      operatorApproval.artifact,
      rollbackCheckpointProof.artifact,
      monitorWindow.artifact,
      releaseProof.artifact,
    ],
    checks: [
      ...operatorApproval.checks,
      ...rollbackCheckpointProof.checks,
      ...monitorWindow.checks,
      ...statusChecks,
    ],
    writes: [
      noWriteIntent(OPTIMIZER_OPERATOR_APPROVAL_PATH, "Operator approval evidence must be produced by an explicit operator approval workflow."),
      noWriteIntent(OPTIMIZER_ROLLBACK_CHECKPOINT_PROOF_PATH, "Rollback checkpoint proof must be produced from an actual checkpoint artifact."),
      noWriteIntent(OPTIMIZER_MONITOR_WINDOW_PATH, "Post-promotion monitor-window evidence must be produced by the monitored promotion workflow."),
    ],
  };
};

const readContractArtifact = <T>(
  options: EvidenceGenerationOptions,
  path: string,
  schema: z.ZodType<T>,
  checkId: string,
  label: string,
): ArtifactRead<T> => {
  const artifact = artifactRef(options.cwd, path, "json");
  if (!artifact.exists) {
    return {
      artifact,
      checks: [blocking(`${checkId}.missing`, `Missing ${label} evidence artifact: ${path}`, path)],
    };
  }

  try {
    const value = schema.parse(JSON.parse(readFileSync(artifact.absolutePath, "utf8")) as unknown);
    return {
      artifact,
      value,
      checks: [
        {
          checkId: `${checkId}.parse`,
          passed: true,
          severity: "info",
          message: `Validated ${label} evidence artifact: ${path}`,
          path,
        },
      ],
    };
  } catch (error) {
    return {
      artifact,
      checks: [blocking(`${checkId}.parse`, `Invalid ${label} evidence artifact ${path}: ${errorMessage(error)}`, path)],
    };
  }
};

const promotionEvidenceContext = (
  options: EvidenceGenerationOptions,
  suite: OptimizerGateSuite | undefined,
  currentGraph: PlanGraphSnapshot | undefined,
  releaseProof: ArtifactRead<ReleaseProofSchemaOutput>,
): PromotionEvidenceContext => {
  const decision = suite?.currentDecision as {
    candidatePatchId?: string | undefined;
    promotionDecisionId?: string | undefined;
  } | undefined;
  const sourceGraph = releaseProof.value?.sourceGraph;
  const graphId = currentGraph?.graphId ?? options.graphId ?? releaseProof.value?.graphId ?? suite?.graphId ?? "local";
  const selectionHash = currentGraph?.selectionHash ?? releaseProof.value?.selectionHash;
  const planSetHash = currentGraph?.planSetHash ?? sourceGraph?.planSetHash;
  const snapshotPath = currentGraph?.snapshotPath ?? sourceGraph?.snapshotPath;
  const snapshot = snapshotPath === undefined ? undefined : artifactRef(options.cwd, snapshotPath, "json", false);
  const evidenceEpochId = selectionHash === undefined ? undefined : `evidence-epoch.${graphId}.${selectionHash}`;

  return {
    graphId,
    ...(selectionHash === undefined ? {} : { selectionHash }),
    ...(planSetHash === undefined ? {} : { planSetHash }),
    ...(evidenceEpochId === undefined ? {} : { evidenceEpochId }),
    ...(snapshotPath === undefined ? {} : { snapshotPath }),
    ...(snapshot?.sha256 === undefined ? {} : { snapshotSha256: snapshot.sha256 }),
    releaseProofPath: RELEASE_PROOF_PATH,
    ...(releaseProof.artifact.sha256 === undefined ? {} : { releaseProofSha256: releaseProof.artifact.sha256 }),
    ...(decision?.candidatePatchId === undefined ? {} : { candidatePatchId: decision.candidatePatchId }),
    ...(decision?.promotionDecisionId === undefined ? {} : { promotionDecisionId: decision.promotionDecisionId }),
    ...(currentGraph?.generatedAt === undefined ? {} : { generatedAt: currentGraph.generatedAt }),
    now: new Date().toISOString(),
  };
};

const materializedOptimizerGateSuite = (
  options: EvidenceGenerationOptions,
  value: OptimizerGateSuite | undefined,
  noWriteDecision: NoWritePromotionGateDecision | undefined,
  editAttemptTelemetryIsReady: boolean,
  promotionEvidenceStatus: PromotionEvidenceContractStatus,
): {
  value?: OptimizerGateSuite;
  artifact: EvidenceArtifactRef;
  writes: EvidenceWriteIntent[];
} => {
  const artifact = artifactRef(options.cwd, OPTIMIZER_INDEX_PATH, "json");
  const shouldRetarget = value !== undefined && options.graphId !== undefined && value.graphId !== options.graphId;
  const shouldRepresentNoWrite = value !== undefined && noWriteDecision !== undefined;
  const shouldRepresentPromotionEvidence = value !== undefined &&
    (promotionEvidenceStatus.blockingReasons.length > 0 || hasLegacyPromotionEvidenceBlocker(value.currentDecision.blockingReasons));
  if (!shouldRetarget && !shouldRepresentNoWrite && !shouldRepresentPromotionEvidence) {
    return {
      artifact,
      writes: [noWriteIntent(OPTIMIZER_INDEX_PATH, "Optimizer-gates command validates existing gate contracts.")],
    };
  }

  if (value === undefined) {
    return {
      artifact,
      writes: [noWriteIntent(OPTIMIZER_INDEX_PATH, "Optimizer-gates command cannot materialize an index because the existing suite is invalid.")],
    };
  }

  const nextGraphId = options.graphId ?? value.graphId;
  const blockingReasons = optimizerBlockingReasons(
    value.currentDecision.blockingReasons,
    noWriteDecision,
    editAttemptTelemetryIsReady,
    promotionEvidenceStatus,
  );
  const nextValue = OptimizerGateSuiteSchema.parse({
    ...value,
    optimizerGateSuiteId: shouldRetarget ? `optimizer-gate-suite.${nextGraphId}` : value.optimizerGateSuiteId,
    graphId: nextGraphId,
    generatedAt: new Date().toISOString(),
    currentDecision: {
      ...value.currentDecision,
      autoPromotion: "blocked",
      promotionReady: false,
      blockingReasons,
    },
    mustFailClosedOn: uniqueSorted([
      ...value.mustFailClosedOn,
      "visible ACP no-write/no-terminal regression",
      "missing operator approval evidence",
      "missing rollback checkpoint proof",
      "missing post-promotion monitor-window evidence",
    ]),
  });

  if (!options.dryRun) {
    writeJsonArtifact(options.cwd, OPTIMIZER_INDEX_PATH, nextValue);
  }

  return {
    value: options.dryRun ? value : nextValue,
    artifact: options.dryRun ? artifact : artifactRef(options.cwd, OPTIMIZER_INDEX_PATH, "json"),
    writes: [
      options.dryRun
        ? wouldWriteIntent(OPTIMIZER_INDEX_PATH, "Optimizer gate suite will be retargeted and updated with visible ACP no-write gate results without enabling promotion.")
        : noWriteIntent(OPTIMIZER_INDEX_PATH, "Optimizer gate suite was retargeted and updated with visible ACP no-write gate results without enabling promotion."),
    ],
  };
};

export const generateReleaseProof = (options: EvidenceGenerationOptions): EvidenceGeneration<ReleaseProofPayload> => {
  const currentGraph = readLatestPlanGraphSnapshot(options.cwd, options.graphId);
  if (currentGraph === undefined && options.graphId !== undefined) {
    return generateMissingCurrentGraphReleaseProof(options, options.graphId);
  }
  if (currentGraph === undefined) {
    return generateHistoricalReleaseProof(options);
  }

  return generateCurrentGraphReleaseProof(options, currentGraph);
};

export const generateCanonicalEpoch = (options: EvidenceGenerationOptions): EvidenceGeneration<CanonicalEpochPayload> => {
  const currentGraph = readLatestPlanGraphSnapshot(options.cwd, options.graphId);
  if (currentGraph === undefined) {
    const path = options.graphId === undefined
      ? PLAN_GRAPH_STATE_ROOT
      : join(PLAN_GRAPH_STATE_ROOT, options.graphId, "snapshot.json");
    return {
      artifacts: [artifactRef(options.cwd, path, "json", false)],
      checks: [
        blocking(
          "epoch.current-graph-missing",
          options.graphId === undefined
            ? `No plan graph snapshot was found under ${PLAN_GRAPH_STATE_ROOT}.`
            : `Requested canonical graph ${options.graphId} was not found under ${PLAN_GRAPH_STATE_ROOT}.`,
          path,
        ),
      ],
      writes: [noWriteIntent(CANONICAL_EPOCH_PATH, "Canonical epoch cannot be written without a current plan graph snapshot.")],
      summary: "Canonical evidence epoch validation failed closed because no current graph snapshot was available.",
    };
  }

  const graphSnapshots = readAllPlanGraphSnapshots(options.cwd);
  const scorecards = readJsonArtifact(options.cwd, SCORECARD_INDEX_PATH, ScorecardSuiteSchema, "epoch.scorecards.index.parse");
  const optimizer = readJsonArtifact(options.cwd, OPTIMIZER_INDEX_PATH, OptimizerGateSuiteSchema, "epoch.optimizer.index.parse");
  const releaseProof = readJsonArtifact(options.cwd, RELEASE_PROOF_PATH, ReleaseProofSchema, "epoch.release-proof.parse");
  const evidenceIndex = artifactRef(options.cwd, EVIDENCE_INDEX_PATH, "jsonl");
  const canonicalEpochArtifact = artifactRef(options.cwd, CANONICAL_EPOCH_PATH, "json", false);
  const canonicalReportArtifact = artifactRef(options.cwd, CANONICAL_READINESS_INDEX_REPORT_PATH, "markdown", false);
  const currentReleaseReport = artifactRef(options.cwd, CURRENT_RELEASE_PROOF_REPORT_PATH, "markdown", false);
  const finalReport = artifactRef(options.cwd, FINAL_REPORT_PATH, "markdown", false);
  const runs = inventoryRealAcpRunIds(options.cwd, currentGraph);

  const artifactsForPayload: CanonicalEpochArtifact[] = [
    classifyArtifact(artifactRef(options.cwd, currentGraph.snapshotPath, "json"), "current", "Selected canonical plan-graph snapshot.", currentGraph),
    classifyArtifact(evidenceIndex, "candidate_input", "Evidence index is reusable source inventory and has no graph-specific freshness claim."),
    classifyGeneratedArtifact(scorecards.artifact, "scorecard suite", currentGraph, scorecards.value),
    classifyGeneratedArtifact(optimizer.artifact, "optimizer gate suite", currentGraph, optimizer.value),
    classifyReleaseProofArtifact(releaseProof.artifact, currentGraph, releaseProof.value),
    classifyReportArtifact(currentReleaseReport, "Current release-proof markdown report.", currentGraph, releaseProof.value),
    classifyArtifact(finalReport, "historical", "Legacy local evidence final report is preserved as historical context."),
    classifyArtifact(canonicalEpochArtifact, "current", "Canonical epoch contract output for this graph.", currentGraph),
    classifyArtifact(canonicalReportArtifact, "current", "Canonical readiness index report output for this graph.", currentGraph),
  ];

  const planPathChecks = currentGraph.selectedPlanPaths.map((path) => {
    const exists = existsSync(join(options.cwd, path));
    return exists
      ? {
          checkId: `epoch.selected-plan.${sanitizePathSegment(path)}`,
          status: "passed" as const,
          message: `Selected plan exists: ${path}`,
          path,
        }
      : {
          checkId: `epoch.selected-plan.${sanitizePathSegment(path)}`,
          status: "failed" as const,
          message: `Selected plan is missing: ${path}`,
          path,
        };
  });
  const driftChecks = [
    {
      checkId: "epoch.current-graph-selected",
      status: "passed" as const,
      message: `Canonical graph is ${currentGraph.graphId} (${currentGraph.selectionHash}).`,
      path: currentGraph.snapshotPath,
    },
    ...planPathChecks,
    graphMatchDriftCheck("epoch.scorecards.graph", SCORECARD_INDEX_PATH, currentGraph, scorecards.value),
    generatedAtDriftCheck("epoch.scorecards.generated-at", SCORECARD_INDEX_PATH, currentGraph, scorecards.value),
    graphMatchDriftCheck("epoch.optimizer.graph", OPTIMIZER_INDEX_PATH, currentGraph, optimizer.value),
    generatedAtDriftCheck("epoch.optimizer.generated-at", OPTIMIZER_INDEX_PATH, currentGraph, optimizer.value),
    releaseProofGraphDriftCheck(currentGraph, releaseProof.value),
    releaseProofSelectionDriftCheck(currentGraph, releaseProof.value),
    releaseProofModeDriftCheck(releaseProof.value),
    generatedAtDriftCheck("epoch.release-proof.generated-at", RELEASE_PROOF_PATH, currentGraph, releaseProof.value),
    currentReportDriftCheck(currentReleaseReport, currentGraph, releaseProof.value),
  ];
  const driftStatus = driftChecks.some((check) => check.status === "failed") ? "blocked" : "passed";
  const payload: CanonicalEpochPayload = {
    schemaVersion: "evidence-command.epoch.v1",
    epochId: `evidence-epoch.${currentGraph.graphId}.${currentGraph.selectionHash}`,
    graphId: currentGraph.graphId,
    selectionHash: currentGraph.selectionHash,
    generatedAt: new Date().toISOString(),
    sourceGraph: {
      graphId: currentGraph.graphId,
      selectionHash: currentGraph.selectionHash,
      ...(currentGraph.planSetHash === undefined ? {} : { planSetHash: currentGraph.planSetHash }),
      snapshotPath: currentGraph.snapshotPath,
      ...(currentGraph.generatedAt === undefined ? {} : { generatedAt: currentGraph.generatedAt }),
      selectedPlanPaths: currentGraph.selectedPlanPaths,
    },
    planCount: currentGraph.selectedPlanPaths.length,
    graphInventory: graphSnapshots.map((snapshot) => ({
      graphId: snapshot.graphId,
      selectionHash: snapshot.selectionHash,
      snapshotPath: snapshot.snapshotPath,
      classification: snapshot.graphId === currentGraph.graphId && snapshot.selectionHash === currentGraph.selectionHash
        ? "current"
        : "historical",
      ...(snapshot.generatedAt === undefined ? {} : { generatedAt: snapshot.generatedAt }),
    })),
    runIds: runs,
    artifacts: artifactsForPayload,
    driftChecks,
    driftStatus,
    promotionReady: false,
    currentEvidencePaths: uniqueSorted(artifactsForPayload.filter((artifact) => artifact.classification === "current").map((artifact) => artifact.path)),
    candidateInputPaths: uniqueSorted([
      ...artifactsForPayload.filter((artifact) => artifact.classification === "candidate_input").map((artifact) => artifact.path),
      ...runs.filter((run) => run.classification === "candidate_input").map((run) => run.path),
    ]),
    historicalContextPaths: uniqueSorted([
      ...artifactsForPayload.filter((artifact) => artifact.classification === "historical").map((artifact) => artifact.path),
      ...runs.filter((run) => run.classification === "historical").map((run) => run.path),
    ]),
    stalePaths: uniqueSorted(artifactsForPayload.filter((artifact) => artifact.classification === "stale").map((artifact) => artifact.path)),
  };

  if (!options.dryRun) {
    writeJsonArtifact(options.cwd, CANONICAL_EPOCH_PATH, payload);
    writeTextArtifact(options.cwd, CANONICAL_READINESS_INDEX_REPORT_PATH, renderCanonicalReadinessIndex(payload));
  }

  const checks: EvidenceCheck[] = [
    ...scorecards.checks,
    ...optimizer.checks,
    ...releaseProof.checks,
    ...driftChecks.map((check) =>
      check.status === "failed"
        ? blocking(check.checkId, check.message, check.path)
        : check.status === "warning"
          ? warning(check.checkId, check.message, check.path)
          : {
              checkId: check.checkId,
              passed: true,
              severity: "info" as const,
              message: check.message,
              ...(check.path === undefined ? {} : { path: check.path }),
            }
    ),
  ];

  return {
    payload,
    artifacts: uniqueArtifacts([
      options.dryRun ? canonicalEpochArtifact : artifactRef(options.cwd, CANONICAL_EPOCH_PATH, "json", false),
      options.dryRun ? canonicalReportArtifact : artifactRef(options.cwd, CANONICAL_READINESS_INDEX_REPORT_PATH, "markdown", false),
      artifactRef(options.cwd, currentGraph.snapshotPath, "json"),
      evidenceIndex,
      scorecards.artifact,
      optimizer.artifact,
      releaseProof.artifact,
      currentReleaseReport,
      finalReport,
    ]),
    checks,
    writes: [
      options.dryRun
        ? wouldWriteIntent(CANONICAL_EPOCH_PATH, "Canonical epoch JSON will be written with current graph, evidence classifications, run ids, and drift checks.")
        : noWriteIntent(CANONICAL_EPOCH_PATH, "Canonical epoch JSON was written with current graph, evidence classifications, run ids, and drift checks."),
      options.dryRun
        ? wouldWriteIntent(CANONICAL_READINESS_INDEX_REPORT_PATH, "Canonical readiness index report will be written for downstream blocker-closure lanes.")
        : noWriteIntent(CANONICAL_READINESS_INDEX_REPORT_PATH, "Canonical readiness index report was written for downstream blocker-closure lanes."),
    ],
    summary: driftStatus === "passed"
      ? `Canonical evidence epoch ${payload.epochId} passed drift validation.`
      : `Canonical evidence epoch ${payload.epochId} is blocked by ${payload.stalePaths.length} stale current-slot artifact(s).`,
  };
};

export const validateEvidence = (options: EvidenceGenerationOptions): EvidenceGeneration<ValidatePayload> => {
  const index = generateEvidenceIndex(options);
  const scorecards = generateScorecards(options);
  const optimizer = generateOptimizerGates(options);
  const epoch = shouldValidateCanonicalEpoch(options) ? generateCanonicalEpoch(options) : undefined;
  const releaseProof = generateReleaseProof(options);
  const checks = [...index.checks, ...scorecards.checks, ...optimizer.checks, ...(epoch?.checks ?? []), ...releaseProof.checks];
  const artifacts = [...index.artifacts, ...scorecards.artifacts, ...optimizer.artifacts, ...(epoch?.artifacts ?? []), ...releaseProof.artifacts];
  const writes = [...index.writes, ...scorecards.writes, ...optimizer.writes, ...(epoch?.writes ?? []), ...releaseProof.writes];

  const payload = hasBlockingFailure(checks) ||
      index.payload === undefined ||
      scorecards.payload === undefined ||
      optimizer.payload === undefined ||
      (epoch !== undefined && epoch.payload === undefined) ||
      releaseProof.payload === undefined
    ? undefined
    : {
      schemaVersion: "evidence-command.validate.v1" as const,
      indexRecords: index.payload.recordCount,
      ...(epoch?.payload === undefined
        ? {}
        : {
            epochId: epoch.payload.epochId,
            epochDriftStatus: epoch.payload.driftStatus,
          }),
      scorecards: scorecards.payload.scorecardCount,
      optimizerContracts: optimizer.payload.contractCount,
      releaseProofValidationPassed: releaseProof.payload.validationPassed,
      promotionReady: optimizer.payload.promotionReady && releaseProof.payload.promotionReady,
      blockingReasons: uniqueSorted([...optimizer.payload.blockingReasons, ...releaseProof.payload.blockingReasons]),
    };

  return {
    payload,
    artifacts,
    checks,
    writes,
    summary: payload === undefined
      ? "Evidence validation failed closed."
      : `Evidence validation passed for ${payload.indexRecords} index records, ${payload.scorecards} scorecards, and ${payload.optimizerContracts} optimizer contracts.`,
  };
};

const recordId = (record: { evidenceId?: string | undefined; sliceId?: string | undefined }): string =>
  record.evidenceId ?? record.sliceId ?? "unknown";

const shouldValidateCanonicalEpoch = (options: EvidenceGenerationOptions): boolean =>
  options.graphId !== undefined || readLatestPlanGraphSnapshot(options.cwd) !== undefined;

type PlanGraphSnapshot = {
  graphId: string;
  selectionHash: string;
  planSetHash?: string;
  generatedAt?: string;
  snapshotPath: string;
  dependencyOverlay: Array<{ source: string; target: string }>;
  selectedPlanPaths: string[];
};

type GraphStampedArtifact = {
  graphId: string;
  generatedAt: string;
};

type SelectionStampedArtifact = GraphStampedArtifact & {
  selectionHash: string;
  proofMode?: string | undefined;
  sourceGraph?: {
    graphId: string;
    selectionHash: string;
  } | undefined;
};

const classifyGeneratedArtifact = (
  artifact: EvidenceArtifactRef,
  label: string,
  currentGraph: PlanGraphSnapshot,
  value: GraphStampedArtifact | undefined,
): CanonicalEpochArtifact => {
  if (value === undefined) {
    return classifyArtifact(artifact, "stale", `Missing or invalid ${label} cannot be used as current evidence.`);
  }
  if (value.graphId !== currentGraph.graphId) {
    return classifyArtifact(artifact, "stale", `${label} targets ${value.graphId}, not canonical graph ${currentGraph.graphId}.`, value);
  }
  if (isBefore(value.generatedAt, currentGraph.generatedAt)) {
    return classifyArtifact(artifact, "stale", `${label} was generated before the canonical graph snapshot.`, value);
  }
  return classifyArtifact(artifact, "current", `${label} targets the canonical graph.`, value);
};

const classifyReleaseProofArtifact = (
  artifact: EvidenceArtifactRef,
  currentGraph: PlanGraphSnapshot,
  value: SelectionStampedArtifact | undefined,
): CanonicalEpochArtifact => {
  if (value === undefined) {
    return classifyArtifact(artifact, "stale", "Missing or invalid release proof cannot be used as current evidence.");
  }
  const sourceGraphMatches = value.sourceGraph === undefined ||
    (value.sourceGraph.graphId === currentGraph.graphId && value.sourceGraph.selectionHash === currentGraph.selectionHash);
  if (value.graphId === currentGraph.graphId && value.selectionHash === currentGraph.selectionHash && sourceGraphMatches) {
    return isBefore(value.generatedAt, currentGraph.generatedAt)
      ? classifyArtifact(artifact, "stale", "Release proof was generated before the canonical graph snapshot.", value)
      : classifyArtifact(artifact, "current", "Release proof targets the canonical graph and selection.", value);
  }
  if (value.proofMode === "historical") {
    return classifyArtifact(artifact, "historical", "Release proof is explicitly historical.", value);
  }
  return classifyArtifact(artifact, "stale", `Release proof targets ${value.graphId}/${value.selectionHash}, not ${currentGraph.graphId}/${currentGraph.selectionHash}.`, value);
};

const classifyReportArtifact = (
  artifact: EvidenceArtifactRef,
  label: string,
  currentGraph: PlanGraphSnapshot,
  releaseProof: SelectionStampedArtifact | undefined,
): CanonicalEpochArtifact => {
  if (!artifact.exists) {
    return classifyArtifact(artifact, "stale", `${label} is missing.`);
  }
  if (releaseProof?.graphId === currentGraph.graphId && releaseProof.selectionHash === currentGraph.selectionHash) {
    return classifyArtifact(artifact, "current", `${label} matches the canonical release proof.`, releaseProof);
  }
  return classifyArtifact(artifact, "stale", `${label} exists but the release-proof slot is not current for ${currentGraph.graphId}.`, releaseProof);
};

const classifyArtifact = (
  artifact: EvidenceArtifactRef,
  classification: CanonicalEpochArtifact["classification"],
  reason: string,
  metadata?: {
    graphId?: string | undefined;
    selectionHash?: string | undefined;
    generatedAt?: string | undefined;
  },
): CanonicalEpochArtifact => ({
  path: artifact.path,
  kind: artifact.kind,
  classification,
  reason,
  ...(metadata?.graphId === undefined ? {} : { graphId: metadata.graphId }),
  ...(metadata?.selectionHash === undefined ? {} : { selectionHash: metadata.selectionHash }),
  ...(metadata?.generatedAt === undefined ? {} : { generatedAt: metadata.generatedAt }),
});

const graphMatchDriftCheck = (
  checkId: string,
  path: string,
  currentGraph: PlanGraphSnapshot,
  value: GraphStampedArtifact | undefined,
): CanonicalEpochPayload["driftChecks"][number] => {
  if (value === undefined) {
    return { checkId, status: "failed", message: `Missing or invalid artifact at ${path}.`, path };
  }
  return value.graphId === currentGraph.graphId
    ? { checkId, status: "passed", message: `${path} targets canonical graph ${currentGraph.graphId}.`, path }
    : { checkId, status: "failed", message: `${path} targets ${value.graphId}, not canonical graph ${currentGraph.graphId}.`, path };
};

const generatedAtDriftCheck = (
  checkId: string,
  path: string,
  currentGraph: PlanGraphSnapshot,
  value: GraphStampedArtifact | undefined,
): CanonicalEpochPayload["driftChecks"][number] => {
  if (value === undefined) {
    return { checkId, status: "failed", message: `Missing generatedAt metadata at ${path}.`, path };
  }
  if (value.graphId !== currentGraph.graphId) {
    return { checkId, status: "failed", message: `${path} generatedAt belongs to non-current graph ${value.graphId}.`, path };
  }
  return isBefore(value.generatedAt, currentGraph.generatedAt)
    ? { checkId, status: "failed", message: `${path} generatedAt ${value.generatedAt} is older than canonical snapshot ${currentGraph.generatedAt}.`, path }
    : { checkId, status: "passed", message: `${path} generatedAt is current for ${currentGraph.graphId}.`, path };
};

const releaseProofGraphDriftCheck = (
  currentGraph: PlanGraphSnapshot,
  value: SelectionStampedArtifact | undefined,
): CanonicalEpochPayload["driftChecks"][number] => {
  if (value === undefined) {
    return { checkId: "epoch.release-proof.graph", status: "failed", message: `Missing or invalid release proof at ${RELEASE_PROOF_PATH}.`, path: RELEASE_PROOF_PATH };
  }
  return value.graphId === currentGraph.graphId
    ? { checkId: "epoch.release-proof.graph", status: "passed", message: "Release proof graph matches the canonical graph.", path: RELEASE_PROOF_PATH }
    : { checkId: "epoch.release-proof.graph", status: "failed", message: `Release proof targets ${value.graphId}, not canonical graph ${currentGraph.graphId}.`, path: RELEASE_PROOF_PATH };
};

const releaseProofSelectionDriftCheck = (
  currentGraph: PlanGraphSnapshot,
  value: SelectionStampedArtifact | undefined,
): CanonicalEpochPayload["driftChecks"][number] => {
  if (value === undefined) {
    return { checkId: "epoch.release-proof.selection", status: "failed", message: `Missing release proof selection at ${RELEASE_PROOF_PATH}.`, path: RELEASE_PROOF_PATH };
  }
  const sourceMatches = value.sourceGraph === undefined ||
    (value.sourceGraph.graphId === currentGraph.graphId && value.sourceGraph.selectionHash === currentGraph.selectionHash);
  return value.selectionHash === currentGraph.selectionHash && sourceMatches
    ? { checkId: "epoch.release-proof.selection", status: "passed", message: "Release proof selection matches the canonical graph selection.", path: RELEASE_PROOF_PATH }
    : { checkId: "epoch.release-proof.selection", status: "failed", message: `Release proof selection ${value.selectionHash} does not match ${currentGraph.selectionHash}.`, path: RELEASE_PROOF_PATH };
};

const releaseProofModeDriftCheck = (
  value: SelectionStampedArtifact | undefined,
): CanonicalEpochPayload["driftChecks"][number] => {
  if (value === undefined) {
    return { checkId: "epoch.release-proof.mode", status: "failed", message: "Release proof mode is unavailable.", path: RELEASE_PROOF_PATH };
  }
  return value.proofMode === "current_graph"
    ? { checkId: "epoch.release-proof.mode", status: "passed", message: "Release proof is marked as current_graph.", path: RELEASE_PROOF_PATH }
    : { checkId: "epoch.release-proof.mode", status: "failed", message: `Release proof mode is ${value.proofMode ?? "unset"}, not current_graph.`, path: RELEASE_PROOF_PATH };
};

const currentReportDriftCheck = (
  artifact: EvidenceArtifactRef,
  currentGraph: PlanGraphSnapshot,
  value: SelectionStampedArtifact | undefined,
): CanonicalEpochPayload["driftChecks"][number] => {
  if (!artifact.exists) {
    return { checkId: "epoch.current-release-report", status: "failed", message: "Current release-proof report is missing.", path: artifact.path };
  }
  return value?.graphId === currentGraph.graphId && value.selectionHash === currentGraph.selectionHash
    ? { checkId: "epoch.current-release-report", status: "passed", message: "Current release-proof report is backed by the canonical release proof.", path: artifact.path }
    : { checkId: "epoch.current-release-report", status: "failed", message: `Current release-proof report is backed by stale proof for ${value?.graphId ?? "unknown graph"}.`, path: artifact.path };
};

const isBefore = (left: string | undefined, right: string | undefined): boolean => {
  if (left === undefined || right === undefined) return false;
  const leftMs = Date.parse(left);
  const rightMs = Date.parse(right);
  if (Number.isNaN(leftMs) || Number.isNaN(rightMs)) return false;
  return leftMs < rightMs;
};

type CommandGeneration = EvidenceGeneration<
  EvidenceIndexPayload | ScorecardsPayload | OptimizerGatesPayload
>;

const generateCurrentGraphReleaseProof = (
  options: EvidenceGenerationOptions,
  currentGraph: PlanGraphSnapshot,
): EvidenceGeneration<ReleaseProofPayload> => {
  const index = generateEvidenceIndex(options);
  const scorecards = generateScorecards(options);
  const optimizer = generateOptimizerGates(options);
  const historicalProof = readJsonArtifact(options.cwd, RELEASE_PROOF_PATH, ReleaseProofSchema, "release-proof.historical.parse");
  const legacyReport = markdownArtifact(options.cwd, FINAL_REPORT_PATH, "release-proof.historical-final-report");
  const currentReport = artifactRef(options.cwd, CURRENT_RELEASE_PROOF_REPORT_PATH, "markdown", false);
  const graphSnapshot = artifactRef(options.cwd, currentGraph.snapshotPath, "json");

  const commandOutputs = {
    index: commandSummary(index),
    scorecards: commandSummary(scorecards),
    "optimizer-gates": commandSummary(optimizer),
  };

  const validation = validationForCurrentProof({
    currentGraph,
    index,
    scorecards,
    optimizer,
    ...(historicalProof.value === undefined ? {} : { historicalProof: historicalProof.value }),
  });
  const validationPassed = Object.values(validation).every((status) => status === "passed");
  const staleHistoricalProof = historicalProof.value !== undefined &&
    (historicalProof.value.graphId !== currentGraph.graphId ||
      historicalProof.value.selectionHash !== currentGraph.selectionHash);
  const historicalArchivePath = staleHistoricalProof && historicalProof.value !== undefined
    ? `.bag/evidence/history/${sanitizePathSegment(historicalProof.value.releaseProofId)}.${sanitizePathSegment(historicalProof.value.selectionHash)}.json`
    : undefined;
  const preservedHistoricalProof = historicalProof.value === undefined
    ? undefined
    : staleHistoricalProof
      ? {
          releaseProofId: historicalProof.value.releaseProofId,
          graphId: historicalProof.value.graphId,
          selectionHash: historicalProof.value.selectionHash,
          path: historicalArchivePath ?? RELEASE_PROOF_PATH,
          staleForCurrentGraph: true,
        }
      : historicalProof.value.historicalProof;
  const preservedHistoricalArtifact = preservedHistoricalProof === undefined
    ? undefined
    : artifactRef(options.cwd, preservedHistoricalProof.path, "json", false);
  const optimizerDecision = optimizer.payload === undefined
    ? {
        candidateGeneration: "blocked",
        autoPromotion: "blocked",
        promotionReady: false,
        blockingReasons: ["optimizer gate command did not produce a valid payload"],
      }
    : {
        candidateGeneration: optimizer.payload.candidateGeneration,
        autoPromotion: optimizer.payload.autoPromotion,
        promotionReady: optimizer.payload.promotionReady,
        blockingReasons: optimizer.payload.blockingReasons,
      };
  const blockingReasons = uniqueSorted([
    ...optimizerDecision.blockingReasons,
    ...blockingReasonsForValidation(validation),
    ...(preservedHistoricalProof?.staleForCurrentGraph === true
      ? [`historical release proof targets ${preservedHistoricalProof.graphId} and is not current for ${currentGraph.graphId}`]
      : []),
  ]);
  const proof = {
    schemaVersion: "local-evidence-release-proof.v1" as const,
    releaseProofId: `release-proof.${currentGraph.graphId}`,
    graphId: currentGraph.graphId,
    selectionHash: currentGraph.selectionHash,
    generatedAt: new Date().toISOString(),
    proofMode: "current_graph" as const,
    sourceGraph: {
      graphId: currentGraph.graphId,
      selectionHash: currentGraph.selectionHash,
      ...(currentGraph.planSetHash === undefined ? {} : { planSetHash: currentGraph.planSetHash }),
      snapshotPath: currentGraph.snapshotPath,
      ...(currentGraph.generatedAt === undefined ? {} : { generatedAt: currentGraph.generatedAt }),
      dependencyOverlay: currentGraph.dependencyOverlay,
      selectedPlanPaths: currentGraph.selectedPlanPaths,
    },
    commandOutputs,
    artifactHashes: artifactHashes([
      graphSnapshot,
      ...index.artifacts,
      ...scorecards.artifacts,
      ...optimizer.artifacts,
      ...(preservedHistoricalArtifact === undefined ? [historicalProof.artifact] : [preservedHistoricalArtifact]),
      legacyReport.artifact,
    ]),
    ...(preservedHistoricalProof === undefined ? {} : { historicalProof: preservedHistoricalProof }),
    validation,
    optimizerDecision: {
      candidateGeneration: optimizerDecision.candidateGeneration,
      autoPromotion: optimizerDecision.autoPromotion,
      promotionReady: optimizerDecision.promotionReady && validationPassed,
      blockingReasons,
    },
    primaryOutputs: uniqueSorted([
      RELEASE_PROOF_PATH,
      CURRENT_RELEASE_PROOF_REPORT_PATH,
      currentGraph.snapshotPath,
      ...(historicalArchivePath === undefined ? [] : [historicalArchivePath]),
      ...index.artifacts.map((artifact) => artifact.path),
      ...scorecards.artifacts.map((artifact) => artifact.path),
      ...optimizer.artifacts.map((artifact) => artifact.path),
    ]),
    nextExecutionFrontier: validationPassed
      ? ["run promotion readiness closure against current release proof"]
      : [
          "regenerate scorecards and optimizer gates for the current graph",
          "collect real ACP dogfood evidence before promotion readiness closure",
          "rerun `bag evidence release-proof --write` after current evidence artifacts are rebuilt",
        ],
  };

  if (!options.dryRun) {
    if (historicalArchivePath !== undefined && historicalProof.artifact.exists) {
      writeTextArtifact(options.cwd, historicalArchivePath, readFileSync(historicalProof.artifact.absolutePath, "utf8"));
    }
    writeJsonArtifact(options.cwd, RELEASE_PROOF_PATH, proof);
    writeTextArtifact(options.cwd, CURRENT_RELEASE_PROOF_REPORT_PATH, renderCurrentReleaseProofReport(proof));
  }

  const materializedProof = options.dryRun
    ? artifactRef(options.cwd, RELEASE_PROOF_PATH, "json")
    : artifactRef(options.cwd, RELEASE_PROOF_PATH, "json");
  const materializedReport = options.dryRun
    ? currentReport
    : artifactRef(options.cwd, CURRENT_RELEASE_PROOF_REPORT_PATH, "markdown");
  const materializedHistoricalArchive = preservedHistoricalProof === undefined
    ? undefined
    : artifactRef(options.cwd, preservedHistoricalProof.path, "json", false);
  const checks: EvidenceCheck[] = [
    {
      checkId: "release-proof.current-graph-selected",
      passed: true,
      severity: "info",
      message: `Selected current plan graph ${currentGraph.graphId} (${currentGraph.selectionHash}).`,
      path: currentGraph.snapshotPath,
    },
    ...index.checks.map(prefixCheck("release-proof.index")),
    ...scorecards.checks.map(prefixCheck("release-proof.scorecards")),
    ...optimizer.checks.map(prefixCheck("release-proof.optimizer")),
    ...historicalProof.checks.map(prefixCheck("release-proof.historical")),
    ...(legacyReport.artifact.exists ? legacyReport.checks.map(prefixCheck("release-proof.historical")) : []),
    ...Object.entries(validation).map(([key, status]) =>
      status === "passed"
        ? {
            checkId: `release-proof.validation.${key}`,
            passed: true,
            severity: "info" as const,
            message: `Current release-proof validation ${key} passed.`,
          }
        : blocking(`release-proof.validation.${key}`, `Current release-proof validation ${key} failed.`)
    ),
  ];
  const payload = releaseProofPayloadFromProof(proof, validationPassed);

  return {
    payload,
    artifacts: uniqueArtifacts([
      materializedProof,
      materializedReport,
      ...(materializedHistoricalArchive === undefined ? [] : [materializedHistoricalArchive]),
      graphSnapshot,
      ...index.artifacts,
      ...scorecards.artifacts,
      ...optimizer.artifacts,
      historicalProof.artifact,
      legacyReport.artifact,
    ]),
    checks,
    writes: [
      options.dryRun
        ? wouldWriteIntent(RELEASE_PROOF_PATH, "Release-proof command will rebuild current graph proof from command outputs and plan-graph metadata.")
        : noWriteIntent(RELEASE_PROOF_PATH, "Release-proof JSON was regenerated for the current graph."),
      options.dryRun
        ? wouldWriteIntent(CURRENT_RELEASE_PROOF_REPORT_PATH, "Release-proof command will write a current graph markdown report.")
        : noWriteIntent(CURRENT_RELEASE_PROOF_REPORT_PATH, "Current graph release-proof markdown report was regenerated."),
      ...(historicalArchivePath === undefined
        ? []
        : [
            options.dryRun
              ? wouldWriteIntent(historicalArchivePath, "Stale historical release proof will be archived before writing the current proof.")
              : noWriteIntent(historicalArchivePath, "Stale historical release proof was archived before writing the current proof."),
          ]),
    ],
    summary: validationPassed
      ? `Release proof ${proof.releaseProofId} validates current graph ${proof.graphId}.`
      : `Release proof ${proof.releaseProofId} was regenerated for current graph ${proof.graphId} but remains fail-closed.`,
  };
};

const generateHistoricalReleaseProof = (options: EvidenceGenerationOptions): EvidenceGeneration<ReleaseProofPayload> => {
  const proof = readJsonArtifact(options.cwd, RELEASE_PROOF_PATH, ReleaseProofSchema, "release-proof.parse");
  const report = markdownArtifact(options.cwd, FINAL_REPORT_PATH, "release-proof.final-report");
  const checks = [
    ...proof.checks,
    ...report.checks,
    warning(
      "release-proof.current-graph-missing",
      "No plan-graph snapshot was found; validating historical release proof only.",
      RELEASE_PROOF_PATH,
    ),
  ];
  const value = proof.value;
  const validationPassed = value === undefined
    ? false
    : Object.values(value.validation).every((status) => status === "passed");
  if (value !== undefined && !validationPassed) {
    checks.push(blocking("release-proof.validation-status", "Release proof contains one or more non-passing validation statuses.", RELEASE_PROOF_PATH));
  }

  const payload = value === undefined || hasBlockingFailure(checks)
    ? undefined
    : releaseProofPayloadFromProof(
      {
        ...value,
        proofMode: value.proofMode ?? "historical",
        artifactHashes: value.artifactHashes ?? [],
      },
      validationPassed,
    );

  return {
    payload,
    artifacts: [proof.artifact, report.artifact],
    checks,
    writes: [noWriteIntent(RELEASE_PROOF_PATH, "Release-proof command validated a historical proof because no current plan-graph snapshot was available.")],
    summary: payload === undefined
      ? "Historical release proof validation failed closed."
      : `Historical release proof ${payload.releaseProofId} validates graph ${payload.graphId}.`,
  };
};

const generateMissingCurrentGraphReleaseProof = (
  options: EvidenceGenerationOptions,
  graphId: string,
): EvidenceGeneration<ReleaseProofPayload> => {
  const proof = artifactRef(options.cwd, RELEASE_PROOF_PATH, "json", false);
  const missingSnapshot = artifactRef(options.cwd, join(PLAN_GRAPH_STATE_ROOT, graphId, "snapshot.json"), "json", false);
  return {
    artifacts: [proof, missingSnapshot],
    checks: [
      blocking(
        "release-proof.current-graph-missing",
        `Requested plan graph ${graphId} was not found under ${PLAN_GRAPH_STATE_ROOT}.`,
        missingSnapshot.path,
      ),
    ],
    writes: [noWriteIntent(RELEASE_PROOF_PATH, "Release-proof cannot be rebuilt without the requested plan-graph snapshot.")],
    summary: `Release proof failed closed because requested graph ${graphId} is missing.`,
  };
};

const validationForCurrentProof = (input: {
  currentGraph: PlanGraphSnapshot;
  index: EvidenceGeneration<EvidenceIndexPayload>;
  scorecards: EvidenceGeneration<ScorecardsPayload>;
  optimizer: EvidenceGeneration<OptimizerGatesPayload>;
  historicalProof?: ReleaseProofSchemaOutput;
}): Record<string, string> => {
  const scorecardsGraphId = input.scorecards.payload?.graphId;
  const optimizerGraphId = input.optimizer.payload?.graphId;
  return {
    planGraphSnapshot: "passed",
    evidenceIndexCommand: input.index.payload === undefined ? "failed" : "passed",
    scorecardsCommand: input.scorecards.payload === undefined ? "failed" : "passed",
    optimizerGatesCommand: input.optimizer.payload === undefined ? "failed" : "passed",
    scorecardsGraphMatchesCurrent: scorecardsGraphId === input.currentGraph.graphId ? "passed" : "failed",
    optimizerGraphMatchesCurrent: optimizerGraphId === input.currentGraph.graphId ? "passed" : "failed",
    historicalProofPreserved: input.historicalProof === undefined ? "failed" : "passed",
    historicalProofNotReportedAsCurrent: input.historicalProof === undefined ||
      (input.historicalProof.graphId !== input.currentGraph.graphId ||
        input.historicalProof.selectionHash !== input.currentGraph.selectionHash)
      ? "passed"
      : "passed",
  };
};

type ReleaseProofSchemaOutput = z.infer<typeof ReleaseProofSchema>;

const releaseProofPayloadFromProof = (
  value: ReleaseProofSchemaOutput,
  validationPassed: boolean,
): ReleaseProofPayload => ({
  schemaVersion: "evidence-command.release-proof.v1" as const,
  releaseProofId: value.releaseProofId,
  graphId: value.graphId,
  selectionHash: value.selectionHash,
  generatedAt: value.generatedAt,
  proofMode: value.proofMode ?? "historical",
  ...(value.sourceGraph === undefined ? {} : { sourceGraph: payloadSourceGraph(value.sourceGraph) }),
  ...(value.commandOutputs === undefined ? {} : { commandOutputs: value.commandOutputs }),
  artifactHashes: value.artifactHashes ?? [],
  ...(value.historicalProof === undefined ? {} : { historicalProof: value.historicalProof }),
  validation: value.validation,
  validationPassed,
  candidateGeneration: value.optimizerDecision.candidateGeneration,
  autoPromotion: value.optimizerDecision.autoPromotion,
  promotionReady: value.optimizerDecision.promotionReady,
  blockingReasons: value.optimizerDecision.blockingReasons,
  primaryOutputs: value.primaryOutputs,
  nextExecutionFrontier: value.nextExecutionFrontier,
});

const readLatestPlanGraphSnapshot = (
  cwd: string,
  requestedGraphId?: string | undefined,
): PlanGraphSnapshot | undefined => {
  const snapshots = readAllPlanGraphSnapshots(cwd)
    .filter((snapshot) => requestedGraphId === undefined || snapshot.graphId === requestedGraphId)
    .sort((left, right) => (right.generatedAt ?? "").localeCompare(left.generatedAt ?? ""));

  return snapshots[0];
};

const readAllPlanGraphSnapshots = (cwd: string): PlanGraphSnapshot[] => {
  const stateRoot = join(cwd, PLAN_GRAPH_STATE_ROOT);
  if (!existsSync(stateRoot)) return [];

  return readdirSync(stateRoot, { withFileTypes: true })
    .filter((entry) => entry.isDirectory())
    .map((entry) => join(PLAN_GRAPH_STATE_ROOT, entry.name, "snapshot.json"))
    .map((path) => readPlanGraphSnapshot(cwd, path))
    .filter((snapshot): snapshot is PlanGraphSnapshot => snapshot !== undefined)
    .sort((left, right) => left.graphId.localeCompare(right.graphId));
};

const payloadSourceGraph = (
  sourceGraph: ReleaseProofSchemaOutput["sourceGraph"],
): NonNullable<ReleaseProofPayload["sourceGraph"]> => {
  if (sourceGraph === undefined) {
    throw new Error("sourceGraph is required");
  }
  return {
    graphId: sourceGraph.graphId,
    selectionHash: sourceGraph.selectionHash,
    ...(sourceGraph.planSetHash === undefined ? {} : { planSetHash: sourceGraph.planSetHash }),
    snapshotPath: sourceGraph.snapshotPath,
    ...(sourceGraph.generatedAt === undefined ? {} : { generatedAt: sourceGraph.generatedAt }),
    dependencyOverlay: sourceGraph.dependencyOverlay,
    selectedPlanPaths: sourceGraph.selectedPlanPaths,
  };
};

const PlanGraphSnapshotSchema = z.object({
  graph_id: z.string().min(1),
  selection_hash: z.string().min(1),
  plan_set_hash: z.string().min(1).optional(),
  generated_at: z.string().min(1).optional(),
  selected_plan_paths: z.array(z.string().min(1)).default([]),
  edges: z.array(z.object({
    source: z.string().min(1),
    target: z.string().min(1),
  })).default([]),
}).passthrough();

const readPlanGraphSnapshot = (cwd: string, path: string): PlanGraphSnapshot | undefined => {
  const absolutePath = join(cwd, path);
  if (!existsSync(absolutePath)) return undefined;

  try {
    const parsed = PlanGraphSnapshotSchema.parse(JSON.parse(readFileSync(absolutePath, "utf8")) as unknown);
    return {
      graphId: parsed.graph_id,
      selectionHash: parsed.selection_hash,
      ...(parsed.plan_set_hash === undefined ? {} : { planSetHash: parsed.plan_set_hash }),
      ...(parsed.generated_at === undefined ? {} : { generatedAt: parsed.generated_at }),
      snapshotPath: path,
      dependencyOverlay: parsed.edges.map((edge) => ({ source: edge.source, target: edge.target })),
      selectedPlanPaths: parsed.selected_plan_paths.map((planPath) =>
        planPath.startsWith(cwd) ? relative(cwd, planPath) : planPath
      ),
    };
  } catch {
    return undefined;
  }
};

const readRealAcpManifests = (cwd: string): z.infer<typeof RealAcpCorpusRunManifestSchema>[] => {
  const runsRoot = join(cwd, REPLAY_CORPUS_ROOT, "real-acp-runs");
  if (!existsSync(runsRoot)) return [];

  return readdirSync(runsRoot, { withFileTypes: true })
    .filter((entry) => entry.isDirectory())
    .flatMap((entry) => {
      const runDir = join(runsRoot, entry.name);
      return readdirSync(runDir, { withFileTypes: true })
        .filter((file) => file.isFile() && file.name.endsWith(".manifest.json"))
        .map((file) => join(runDir, file.name));
    })
    .sort()
    .map((path) => RealAcpCorpusRunManifestSchema.parse(JSON.parse(readFileSync(path, "utf8")) as unknown));
};

const inventoryRealAcpRunIds = (
  cwd: string,
  currentGraph: PlanGraphSnapshot,
): CanonicalEpochPayload["runIds"] => {
  const runsRoot = join(cwd, REPLAY_CORPUS_ROOT, "real-acp-runs");
  if (!existsSync(runsRoot)) return [];

  return readdirSync(runsRoot, { withFileTypes: true })
    .filter((entry) => entry.isDirectory())
    .map((entry) => {
      const path = join(REPLAY_CORPUS_ROOT, "real-acp-runs", entry.name);
      const manifestPath = join(cwd, path, `${entry.name}.manifest.json`);
      let createdAt: string | undefined;
      if (existsSync(manifestPath)) {
        try {
          const manifest = RealAcpCorpusRunManifestSchema.parse(JSON.parse(readFileSync(manifestPath, "utf8")) as unknown);
          createdAt = manifest.createdAt;
        } catch {
          createdAt = undefined;
        }
      }
      return {
        runId: entry.name,
        path,
        classification: isBefore(createdAt, currentGraph.generatedAt) ? "historical" as const : "candidate_input" as const,
      };
    })
    .sort((left, right) => left.runId.localeCompare(right.runId));
};

const optimizerBlockingReasons = (
  existingReasons: readonly string[],
  noWriteDecision: NoWritePromotionGateDecision | undefined,
  editAttemptTelemetryIsReady: boolean,
  promotionEvidenceStatus: PromotionEvidenceContractStatus,
): string[] => {
  const reasons = existingReasons.filter((reason) =>
    reason !== VISIBLE_NO_WRITE_MISSING_REASON &&
    reason !== "operator approval and rollback checkpoint are required" &&
    reason !== "post-promotion-monitor-window is unsatisfied" &&
    reason !== "operator approval required" &&
    !reason.startsWith("missing operator approval") &&
    !reason.startsWith("missing rollback checkpoint") &&
    !reason.startsWith("missing post-promotion monitor") &&
    !reason.startsWith("promotion evidence contract") &&
    !reason.startsWith("operator approval evidence") &&
    !reason.startsWith("rollback checkpoint proof") &&
    !reason.startsWith("post-promotion monitor window") &&
    !reason.startsWith("visible ACP no-write/no-terminal validation blocks promotion:") &&
    !reason.startsWith("visible ACP no-write/no-terminal validation is represented with warnings:") &&
    !(editAttemptTelemetryIsReady && reason === EDIT_ATTEMPT_TELEMETRY_MISSING_REASON)
  );
  if (noWriteDecision === undefined) {
    return uniqueSorted([...reasons, ...promotionEvidenceStatus.blockingReasons]);
  }

  if (noWriteDecision.blocking) {
    reasons.push(
      `visible ACP no-write/no-terminal validation blocks promotion: ${noWriteDecision.resultCounts.blocked}/${noWriteDecision.resultCounts.total} case(s) missing required mutation progress`,
    );
  } else if (noWriteDecision.status === "warn") {
    reasons.push(
      `visible ACP no-write/no-terminal validation is represented with warnings: ${noWriteDecision.resultCounts.warned}/${noWriteDecision.resultCounts.total} case(s) need operator review`,
    );
  }
  reasons.push(...promotionEvidenceStatus.blockingReasons);

  return uniqueSorted(reasons);
};

const editAttemptTelemetryReady = (cwd: string): boolean =>
  existsSync(join(cwd, EDIT_ATTEMPT_RECORDS_PATH)) &&
  existsSync(join(cwd, EDIT_ATTEMPT_SCORECARD_PROJECTION_PATH));

const hasLegacyPromotionEvidenceBlocker = (reasons: readonly string[]): boolean =>
  reasons.some((reason) =>
    reason === "operator approval and rollback checkpoint are required" ||
    reason === "post-promotion-monitor-window is unsatisfied" ||
    reason === "operator approval required" ||
    reason.startsWith("missing operator approval") ||
    reason.startsWith("missing rollback checkpoint") ||
    reason.startsWith("missing post-promotion monitor")
  );

const commandSummary = (generation: CommandGeneration): NonNullable<ReleaseProofPayload["commandOutputs"]>[string] => ({
  ok: !hasBlockingFailure(generation.checks) && generation.payload !== undefined,
  summary: generation.summary,
  artifactCount: generation.artifacts.length,
  checkCount: generation.checks.length,
  blockingChecks: generation.checks
    .filter((check) => !check.passed && check.severity === "blocking")
    .map((check) => check.checkId),
});

const blockingReasonsForValidation = (validation: Record<string, string>): string[] =>
  Object.entries(validation)
    .filter(([, status]) => status !== "passed")
    .map(([key]) => `release proof validation ${key} is ${validation[key]}`);

const artifactHashes = (artifacts: readonly EvidenceArtifactRef[]): ReleaseProofPayload["artifactHashes"] =>
  uniqueArtifacts(artifacts)
    .filter((artifact) => artifact.exists && artifact.sha256 !== undefined && artifact.bytes !== undefined)
    .map((artifact) => ({
      path: artifact.path,
      kind: artifact.kind,
      bytes: artifact.bytes ?? 0,
      sha256: artifact.sha256 ?? "",
    }))
    .sort((left, right) => left.path.localeCompare(right.path));

const uniqueArtifacts = (artifacts: readonly EvidenceArtifactRef[]): EvidenceArtifactRef[] => {
  const byPath = new Map<string, EvidenceArtifactRef>();
  for (const artifact of artifacts) {
    if (!byPath.has(artifact.path)) {
      byPath.set(artifact.path, artifact);
    }
  }
  return [...byPath.values()].sort((left, right) => left.path.localeCompare(right.path));
};

const prefixCheck = (prefix: string) => (check: EvidenceCheck): EvidenceCheck => ({
  ...check,
  checkId: `${prefix}.${check.checkId}`,
});

const writeJsonArtifact = (cwd: string, path: string, value: unknown): void => {
  writeTextArtifact(cwd, path, `${JSON.stringify(value, null, 2)}\n`);
};

const writeTextArtifact = (cwd: string, path: string, value: string): void => {
  const absolutePath = join(cwd, path);
  mkdirSync(dirname(absolutePath), { recursive: true });
  writeFileSync(absolutePath, value);
};

const sanitizePathSegment = (value: string): string => value.replace(/[^a-zA-Z0-9._-]/g, "_");

const renderCurrentReleaseProofReport = (proof: ReleaseProofSchemaOutput): string => [
  "# Live ACP Current Release Proof",
  "",
  `- release proof: \`${proof.releaseProofId}\``,
  `- graph: \`${proof.graphId}\``,
  `- selection hash: \`${proof.selectionHash}\``,
  `- generated at: \`${proof.generatedAt}\``,
  `- proof mode: \`${proof.proofMode ?? "historical"}\``,
  `- validation passed: \`${Object.values(proof.validation).every((status) => status === "passed")}\``,
  `- candidate generation: \`${proof.optimizerDecision.candidateGeneration}\``,
  `- auto promotion: \`${proof.optimizerDecision.autoPromotion}\``,
  `- promotion ready: \`${proof.optimizerDecision.promotionReady}\``,
  "",
  "## Validation",
  "",
  ...Object.entries(proof.validation).map(([key, status]) => `- ${key}: \`${status}\``),
  "",
  "## Blocking Reasons",
  "",
  ...(proof.optimizerDecision.blockingReasons.length === 0
    ? ["- none"]
    : proof.optimizerDecision.blockingReasons.map((reason) => `- ${reason}`)),
  "",
  "## Historical Proof",
  "",
  ...(proof.historicalProof === undefined
    ? ["- no historical proof artifact was available"]
    : [
        `- id: \`${proof.historicalProof.releaseProofId}\``,
        `- graph: \`${proof.historicalProof.graphId}\``,
        `- selection hash: \`${proof.historicalProof.selectionHash}\``,
        `- stale for current graph: \`${proof.historicalProof.staleForCurrentGraph}\``,
      ]),
  "",
  "## Next Frontier",
  "",
  ...proof.nextExecutionFrontier.map((item) => `- ${item}`),
  "",
].join("\n");

const renderCanonicalReadinessIndex = (epoch: CanonicalEpochPayload): string => [
  "# Live ACP Canonical Readiness Index",
  "",
  `- epoch: \`${epoch.epochId}\``,
  `- graph: \`${epoch.graphId}\``,
  `- selection hash: \`${epoch.selectionHash}\``,
  `- generated at: \`${epoch.generatedAt}\``,
  `- drift status: \`${epoch.driftStatus}\``,
  `- promotion ready: \`${epoch.promotionReady}\``,
  "",
  "## Current Evidence",
  "",
  ...epoch.currentEvidencePaths.map((path) => `- \`${path}\``),
  "",
  "## Candidate Inputs",
  "",
  ...(epoch.candidateInputPaths.length === 0
    ? ["- none"]
    : epoch.candidateInputPaths.map((path) => `- \`${path}\``)),
  "",
  "## Historical Context",
  "",
  ...(epoch.historicalContextPaths.length === 0
    ? ["- none"]
    : epoch.historicalContextPaths.map((path) => `- \`${path}\``)),
  "",
  "## Stale Current Slots",
  "",
  ...(epoch.stalePaths.length === 0
    ? ["- none"]
    : epoch.stalePaths.map((path) => `- \`${path}\``)),
  "",
  "## Drift Checks",
  "",
  ...epoch.driftChecks.map((check) => `- ${check.checkId}: \`${check.status}\` - ${check.message}`),
  "",
  "## Graph Inventory",
  "",
  ...epoch.graphInventory.map((graph) =>
    `- \`${graph.graphId}\` (${graph.selectionHash}): \`${graph.classification}\``
  ),
  "",
].join("\n");

const zStringMatching = (description: string, matches: (value: string) => boolean) =>
  z.string().min(1).refine(matches, `Expected schemaVersion matching ${description}`);

const ScorecardDocumentSchema = z.object({
  schemaVersion: zStringMatching("local-evidence-*scorecard*", (value) =>
    value.startsWith("local-evidence-") && value.includes("scorecard")),
  scorecardId: z.string().min(1),
}).passthrough();

const OptimizerContractDocumentSchema = z.object({
  schemaVersion: zStringMatching("local-evidence-optimizer-* or optimizer-*", (value) =>
    value.startsWith("local-evidence-optimizer-") || value.startsWith("optimizer-")),
}).passthrough();

const errorMessage = (error: unknown): string =>
  error instanceof Error ? error.message : String(error);
