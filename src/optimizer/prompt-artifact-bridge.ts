import { existsSync, lstatSync, mkdirSync, renameSync, rmSync, symlinkSync, unlinkSync, writeFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import type { CandidatePatch } from "./types";
import type { CandidatePromotionResult } from "./promotion";

export type MaterializePromotedPromptArtifactInput = {
  promotion: CandidatePromotionResult;
  candidate: CandidatePatch;
  resolvedPromptText: string;
  cwd: string;
  /**
   * Optional override for the artifact run directory id. Defaults to the
   * promotion decision id, which is stable per candidate patch promotion.
   */
  runId?: string;
  /**
   * Optional ISO timestamp. Defaults to now.
   */
  promotedAt?: string;
};

export type MaterializePromotedPromptArtifactResult = {
  artifactPath: string;
  latestSymlink: string;
  runId: string;
};

const ARTIFACT_REL_DIR = "artifacts/optimized-prompts";

const stableRunId = (input: MaterializePromotedPromptArtifactInput): string => {
  if (input.runId != null && input.runId.length > 0) return input.runId;
  return input.promotion.decision.promotionDecisionId;
};

/**
 * Write a promoted prompt candidate to
 * `<cwd>/artifacts/optimized-prompts/<runId>/best_candidate.json` and rotate the
 * `latest` symlink atomically (write `.tmp-link`, then rename). The path layout
 * matches `loadOptimizedExecutorPrompt` in `src/optimized-prompt-loader.ts`.
 */
export const materializePromotedPromptArtifact = (
  input: MaterializePromotedPromptArtifactInput,
): MaterializePromotedPromptArtifactResult => {
  const root = resolve(input.cwd);
  const baseDir = join(root, ARTIFACT_REL_DIR);
  const runId = stableRunId(input);
  const runDir = join(baseDir, runId);
  const artifactPath = join(runDir, "best_candidate.json");
  const latestSymlink = join(baseDir, "latest");
  const promotedAt = input.promotedAt ?? new Date().toISOString();

  mkdirSync(runDir, { recursive: true });

  const payload = {
    schemaVersion: "optimized-prompt.v1",
    system: input.resolvedPromptText,
    runId,
    candidatePatchId: input.candidate.candidatePatchId,
    promotedAt,
  };
  writeFileSync(artifactPath, `${JSON.stringify(payload, null, 2)}\n`, "utf8");

  rotateLatestSymlink({ baseDir, latestSymlink, runDir });

  return { artifactPath, latestSymlink, runId };
};

const rotateLatestSymlink = (input: { baseDir: string; latestSymlink: string; runDir: string }): void => {
  // Best-effort atomic rotate: create a temp symlink alongside the target, then
  // rename it over `latest`. Fall back to remove-then-create if rename across a
  // pre-existing symlink is rejected by the platform.
  const tmpLink = `${input.latestSymlink}.tmp-link`;
  // Clean any stale tmp link.
  try {
    if (existsSync(tmpLink) || isSymlink(tmpLink)) unlinkSync(tmpLink);
  } catch {
    // ignore
  }
  mkdirSync(dirname(input.latestSymlink), { recursive: true });
  symlinkSync(input.runDir, tmpLink, "dir");
  try {
    renameSync(tmpLink, input.latestSymlink);
  } catch {
    // Some platforms refuse to rename onto an existing symlink/dir — fall back.
    if (isSymlink(input.latestSymlink) || existsSync(input.latestSymlink)) {
      try {
        const stat = lstatSync(input.latestSymlink);
        if (stat.isSymbolicLink() || stat.isFile()) {
          unlinkSync(input.latestSymlink);
        } else if (stat.isDirectory()) {
          rmSync(input.latestSymlink, { recursive: true, force: true });
        }
      } catch {
        // ignore
      }
    }
    renameSync(tmpLink, input.latestSymlink);
  }
};

const isSymlink = (path: string): boolean => {
  try {
    return lstatSync(path).isSymbolicLink();
  } catch {
    return false;
  }
};
