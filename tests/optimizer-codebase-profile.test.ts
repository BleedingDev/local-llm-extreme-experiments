import { describe, expect, test } from "bun:test";
import { mkdtempSync, mkdirSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import {
  detectCodebaseProfileDrift,
  evaluateCodebaseProfilePin,
  generateCodebaseProfile,
  refreshCodebaseProfileForReview,
} from "../src/optimizer/codebase-profile";

const withTempCwd = (fn: (cwd: string) => void): void => {
  const cwd = mkdtempSync(join(tmpdir(), "bag-codebase-profile-"));
  try {
    fn(cwd);
  } finally {
    rmSync(cwd, { recursive: true, force: true });
  }
};

const writeProject = (cwd: string, scripts: Record<string, string>): void => {
  mkdirSync(join(cwd, "src"), { recursive: true });
  mkdirSync(join(cwd, "tests"), { recursive: true });
  mkdirSync(join(cwd, "scripts"), { recursive: true });
  mkdirSync(join(cwd, "dist"), { recursive: true });
  writeFileSync(join(cwd, "src", "index.ts"), "export const value: number = 1;\n");
  writeFileSync(join(cwd, "tests", "index.test.ts"), "import { value } from '../src/index';\n");
  writeFileSync(join(cwd, "scripts", "smoke.sh"), "#!/usr/bin/env bash\nexit 0\n");
  writeFileSync(join(cwd, "dist", "index.js"), "export const value = 1;\n");
  writeFileSync(join(cwd, ".gitignore"), "node_modules/\ndist/\ncoverage/\n");
  writeFileSync(join(cwd, "README.md"), "# Fixture\n");
  writeFileSync(join(cwd, "tsconfig.json"), JSON.stringify({ compilerOptions: { strict: true } }));
  writeFileSync(join(cwd, "rspack.config.ts"), "export default {};\n");
  writeFileSync(join(cwd, "package-lock.json"), JSON.stringify({ lockfileVersion: 3 }));
  writeFileSync(join(cwd, "package.json"), JSON.stringify({
    name: "fixture-agent",
    scripts,
  }, null, 2));
};

describe("optimizer codebase profile generation", () => {
  test("builds a profile from structured repo, package, command, and verifier evidence", () => {
    withTempCwd((cwd) => {
      writeProject(cwd, {
        test: "bun test tests",
        typecheck: "tsc -p tsconfig.json --noEmit",
        lint: "eslint src tests",
      });

      const generated = generateCodebaseProfile({
        cwd,
        codebaseProfileId: "codebase.fixture",
        protectedPathDefaults: [".bag"],
        observedVerifierBehavior: [
          {
            kind: "lint",
            commandId: "lint.observed",
            command: ["npm", "run", "lint"],
            lastExitCode: 0,
          },
        ],
      });

      expect(generated.profile.codebaseProfileId).toBe("codebase.fixture");
      expect(generated.profile.displayName).toBe("fixture-agent");
      expect(generated.profile.rootFingerprint).toStartWith("sha256:");
      expect(generated.profile.languages).toEqual(["typescript", "javascript", "shell", "markdown"]);
      expect(generated.profile.packageManagers).toEqual(["npm"]);
      expect(generated.profile.primaryPackageManager).toBe("npm");
      expect(generated.profile.sourceRoots).toEqual(["scripts", "src", "tests"]);
      expect(generated.profile.generatedDirs).toEqual(["dist"]);
      expect(generated.profile.ignoredDirs).toContain(".bag");
      expect(generated.profile.ignoredDirs).toContain("dist");
      expect(generated.profile.ignoredDirs).toContain("node_modules");
      expect(generated.profile.testCommands).toEqual([
        { commandId: "test", command: ["npm", "test"], required: true },
      ]);
      expect(generated.profile.typecheckCommands).toEqual([
        { commandId: "typecheck", command: ["npm", "run", "typecheck"], required: true },
      ]);
      expect(generated.profile.lintCommands).toEqual([
        { commandId: "lint", command: ["npm", "run", "lint"], required: true },
      ]);
      expect(generated.profile.protectedPaths).toContain(".bag");
      expect(generated.profile.protectedPaths).toContain("node_modules");
      expect(generated.profile.testRiskTiers.map((tier) => tier.tierId)).toEqual([
        "risk.lint",
        "risk.protected-paths",
        "risk.test",
        "risk.typecheck",
      ]);
      expect(generated.profile.conventions).toContain("config.package.json");
      expect(generated.profile.conventions).toContain("generated-dir.dist");
      expect(generated.profile.conventions).toContain("ignored-dir.dist");
      expect(generated.profile.conventions).toContain("package-script.typecheck");
      expect(generated.profile.conventions).toContain("verifier.lint.passing");
      expect(generated.profile.acpClientQuirks.map((quirk) => quirk.quirkId)).toContain("acp.client.terminal-create.optional");
      expect(generated.evidence.packageJson?.scripts.typecheck).toBe("tsc -p tsconfig.json --noEmit");
    });
  });

  test("detects meaningful drift without changing the active profile", () => {
    withTempCwd((cwd) => {
      writeProject(cwd, {
        test: "bun test tests",
        typecheck: "tsc -p tsconfig.json --noEmit",
      });
      const active = generateCodebaseProfile({ cwd, codebaseProfileId: "codebase.fixture" }).profile;

      writeProject(cwd, {
        test: "bun test tests",
        typecheck: "tsc -p tsconfig.json --noEmit",
        lint: "eslint src tests",
      });
      const proposed = generateCodebaseProfile({ cwd, codebaseProfileId: "codebase.fixture" }).profile;
      const drift = detectCodebaseProfileDrift(active, proposed);

      expect(drift.decision).toBe("blocked");
      expect(drift.activeProfile).toEqual(active);
      expect(drift.proposedProfile).toEqual(proposed);
      expect(drift.diagnostics.map((diagnostic) => diagnostic.field)).toContain("rootFingerprint");
      expect(drift.diagnostics.map((diagnostic) => diagnostic.field)).toContain("lintCommands");
      expect(drift.diagnostics.map((diagnostic) => diagnostic.field)).toContain("conventions");
      expect(drift.diagnostics.some((diagnostic) => diagnostic.severity === "blocked")).toBe(true);
      expect(active.lintCommands).toEqual([]);
      expect(proposed.lintCommands).toEqual([
        { commandId: "lint", command: ["npm", "run", "lint"], required: true },
      ]);
    });
  });

  test("refresh review reports no drift for an unchanged profile", () => {
    withTempCwd((cwd) => {
      writeProject(cwd, {
        test: "bun test tests",
        typecheck: "tsc -p tsconfig.json --noEmit",
      });
      const active = generateCodebaseProfile({ cwd, codebaseProfileId: "codebase.fixture" }).profile;
      const review = refreshCodebaseProfileForReview(active, { cwd });

      expect(review.decision).toBe("no_change");
      expect(review.diagnostics).toEqual([]);
      expect(review.activeProfile).toEqual(active);
      expect(review.proposedProfile).toEqual(active);
    });
  });

  test("records verifier failures and blocks mismatched profile pins", () => {
    withTempCwd((cwd) => {
      writeProject(cwd, {
        test: "bun test tests",
      });
      const generated = generateCodebaseProfile({
        cwd,
        codebaseProfileId: "codebase.fixture",
        observedVerifierBehavior: [
          {
            kind: "test",
            commandId: "test.integration",
            command: ["npm", "test"],
            lastExitCode: 1,
            required: true,
          },
        ],
      });

      expect(generated.profile.knownFailures).toEqual([
        expect.objectContaining({
          failureId: "known-failure.test.integration",
          commandId: "test.integration",
          lastExitCode: 1,
        }),
      ]);
      expect(evaluateCodebaseProfilePin(
        {
          codebaseProfileId: "codebase.fixture",
          codebaseRootFingerprint: generated.profile.rootFingerprint,
        },
        {
          codebaseProfileId: "codebase.fixture",
          codebaseRootFingerprint: "sha256:other",
        },
      )).toMatchObject({
        decision: "blocked",
        reason: expect.stringContaining("fingerprint mismatch"),
      });
    });
  });
});
