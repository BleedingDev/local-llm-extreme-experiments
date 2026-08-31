import { createHash } from "node:crypto";
import {
  existsSync,
  readdirSync,
  readFileSync,
  statSync,
} from "node:fs";
import type { Dirent } from "node:fs";
import { basename, join, relative, resolve, sep } from "node:path";
import {
  CodebaseProfileSchema,
  type CodebaseProfile,
  type CommandSpec,
} from "./types";

type CodebaseLanguage = CodebaseProfile["languages"][number];
type PackageManager = CodebaseProfile["packageManagers"][number];

export type VerifierCommandKind = "test" | "typecheck" | "lint";

export interface ObservedVerifierBehavior {
  kind: VerifierCommandKind;
  commandId?: string;
  command: readonly string[];
  cwd?: string;
  required?: boolean;
  lastExitCode?: number;
}

export interface GenerateCodebaseProfileInput {
  cwd?: string;
  codebaseProfileId?: string;
  displayName?: string;
  verificationPolicyVersion?: string;
  protectedPathDefaults?: readonly string[];
  observedVerifierBehavior?: readonly ObservedVerifierBehavior[];
}

export interface CodebaseProfileEvidence {
  repoRoot: string;
  repoEntries: string[];
  packageJson?: {
    path: string;
    name?: string;
    scripts: Record<string, string>;
  };
  lockfiles: string[];
  configFiles: string[];
  sourceRoots: string[];
  languageFiles: Partial<Record<CodebaseLanguage, string[]>>;
  observedVerifierBehavior: Array<{
    kind: VerifierCommandKind;
    commandId: string;
    command: string[];
    cwd?: string;
    required: boolean;
    lastExitCode?: number;
  }>;
}

export interface GeneratedCodebaseProfile {
  profile: CodebaseProfile;
  evidence: CodebaseProfileEvidence;
}

export type ProfileDriftField =
  | "rootFingerprint"
  | "languages"
  | "packageManagers"
  | "primaryPackageManager"
  | "sourceRoots"
  | "generatedDirs"
  | "ignoredDirs"
  | "testCommands"
  | "typecheckCommands"
  | "lintCommands"
  | "testRiskTiers"
  | "protectedPaths"
  | "conventions"
  | "knownFailures"
  | "acpClientQuirks"
  | "verificationPolicyVersion";

export type ProfileDriftKind = "added" | "removed" | "changed";

export interface CodebaseProfileDriftDiagnostic {
  field: ProfileDriftField;
  kind: ProfileDriftKind;
  previous: unknown;
  current: unknown;
  severity: "review" | "blocked";
  message: string;
}

export interface CodebaseProfileDriftDecision {
  decision: "no_change" | "review_required" | "blocked";
  activeProfile: CodebaseProfile;
  proposedProfile: CodebaseProfile;
  diagnostics: CodebaseProfileDriftDiagnostic[];
}

export interface CodebaseProfilePin {
  codebaseProfileId: string;
  codebaseRootFingerprint?: string | undefined;
}

export interface CodebaseProfileGateDecision {
  decision: "pass" | "blocked";
  expected: CodebaseProfilePin;
  actual: CodebaseProfilePin;
  reason?: string | undefined;
}

const DEFAULT_VERIFICATION_POLICY_VERSION = "verification.v1";
const DEFAULT_PROTECTED_PATHS = [".bag", ".codex", ".cursor", ".git", "node_modules", "dist", "build", "coverage"];
const DEFAULT_GENERATED_DIRS = ["dist", "build", "coverage", ".next", "out", "target"];
const DEFAULT_IGNORED_DIRS = [".bag", ".codex", ".cursor", ".git", "node_modules", "dist", "build", "coverage", "target", ".venv", "venv", "__pycache__"];
const MAX_SCANNED_FILES = 2_000;
const MAX_LANGUAGE_EVIDENCE_FILES = 12;

const SKIPPED_DIRS = new Set([
  ".bag",
  ".codex",
  ".cursor",
  ".git",
  ".hg",
  ".svn",
  "coverage",
  "dist",
  "build",
  "node_modules",
  "target",
  ".venv",
  "venv",
  "__pycache__",
]);

const SOURCE_ROOT_CANDIDATES = [
  "src",
  "lib",
  "app",
  "tests",
  "test",
  "scripts",
  "packages",
  "crates",
  "cmd",
  "internal",
];

const CONFIG_FILE_CANDIDATES = [
  "package.json",
  "tsconfig.json",
  "rspack.config.ts",
  "webpack.config.js",
  "vite.config.ts",
  "bun.lock",
  "bun.lockb",
  "package-lock.json",
  "pnpm-lock.yaml",
  "yarn.lock",
  "requirements.txt",
  "pyproject.toml",
  "Cargo.toml",
  "Cargo.lock",
  "go.mod",
  "go.sum",
  "README.md",
];

const LOCKFILE_TO_PACKAGE_MANAGER: Array<[string, PackageManager]> = [
  ["bun.lock", "bun"],
  ["bun.lockb", "bun"],
  ["package-lock.json", "npm"],
  ["pnpm-lock.yaml", "pnpm"],
  ["yarn.lock", "yarn"],
  ["requirements.txt", "pip"],
  ["pyproject.toml", "pip"],
  ["Cargo.lock", "cargo"],
  ["Cargo.toml", "cargo"],
  ["go.mod", "go"],
];

export const generateCodebaseProfile = (
  input: GenerateCodebaseProfileInput = {},
): GeneratedCodebaseProfile => {
  const cwd = resolve(input.cwd ?? process.cwd());
  const files = scanRepoFiles(cwd);
  const packageJson = readPackageJson(cwd);
  const repoEntries = listRepoEntries(cwd);
  const lockfiles = CONFIG_FILE_CANDIDATES.filter((path) => existsSync(join(cwd, path)) && lockfilePackageManager(path) !== undefined);
  const configFiles = CONFIG_FILE_CANDIDATES.filter((path) => existsSync(join(cwd, path)));
  const packageManagers = detectPackageManagers(cwd, packageJson, lockfiles);
  const primaryPackageManager = preferredPackageManager(packageManagers);
  const sourceRoots = detectSourceRoots(cwd, files);
  const generatedDirs = detectGeneratedDirs(cwd);
  const ignoredDirs = detectIgnoredDirs(cwd);
  const languageFiles = collectLanguageEvidence(files);
  const languages = detectLanguages(cwd, packageJson, languageFiles);
  const protectedPaths = uniqueSorted([
    ...DEFAULT_PROTECTED_PATHS,
    ...(input.protectedPathDefaults ?? []),
  ].map(normalizeRelativePath).filter((path) => path.length > 0));
  const observedVerifierBehavior = normalizeObservedVerifierBehavior(input.observedVerifierBehavior ?? []);
  const testCommands = mergeObservedCommands(
    scriptCommands("test", packageJson?.scripts, primaryPackageManager),
    observedVerifierBehavior.filter((entry) => entry.kind === "test"),
  );
  const typecheckCommands = mergeObservedCommands(
    scriptCommands("typecheck", packageJson?.scripts, primaryPackageManager),
    observedVerifierBehavior.filter((entry) => entry.kind === "typecheck"),
  );
  const lintCommands = mergeObservedCommands(
    scriptCommands("lint", packageJson?.scripts, primaryPackageManager),
    observedVerifierBehavior.filter((entry) => entry.kind === "lint"),
  );
  const conventions = detectConventions({
    configFiles,
    generatedDirs,
    ignoredDirs,
    packageManagers,
    primaryPackageManager,
    packageScripts: packageJson?.scripts ?? {},
    sourceRoots,
    languages,
    observedVerifierBehavior,
  });
  const testRiskTiers = buildTestRiskTiers({
    testCommands,
    typecheckCommands,
    lintCommands,
    protectedPaths,
  });
  const knownFailures = knownFailuresFromVerifierBehavior(observedVerifierBehavior);
  const acpClientQuirks = defaultAcpClientQuirks();

  const evidence: CodebaseProfileEvidence = {
    repoRoot: cwd,
    repoEntries,
    ...(packageJson === undefined ? {} : { packageJson }),
    lockfiles,
    configFiles,
    sourceRoots,
    languageFiles,
    observedVerifierBehavior,
  };
  const rootFingerprint = fingerprintCodebaseProfileEvidence({
    repoRoot: cwd,
    repoEntries,
    packageJson,
    lockfiles,
    configFiles: configFiles.map((path) => ({ path, hash: hashFile(join(cwd, path)) })),
    sourceRoots,
    languages,
    packageManagers,
    primaryPackageManager,
    generatedDirs,
    ignoredDirs,
    testCommands,
    typecheckCommands,
    lintCommands,
    testRiskTiers,
    protectedPaths,
    conventions,
    knownFailures,
    acpClientQuirks,
    observedVerifierBehavior,
  });

  const profile = CodebaseProfileSchema.parse({
    codebaseProfileId: input.codebaseProfileId ?? codebaseProfileIdFor(cwd),
    displayName: input.displayName ?? packageJson?.name ?? basename(cwd),
    rootFingerprint,
    languages,
    packageManagers,
    primaryPackageManager,
    sourceRoots,
    generatedDirs,
    ignoredDirs,
    testCommands,
    typecheckCommands,
    lintCommands,
    testRiskTiers,
    protectedPaths,
    conventions,
    knownFailures,
    acpClientQuirks,
    verificationPolicyVersion: input.verificationPolicyVersion ?? DEFAULT_VERIFICATION_POLICY_VERSION,
  });

  return { profile, evidence };
};

export const detectCodebaseProfileDrift = (
  activeProfile: CodebaseProfile,
  proposedProfile: CodebaseProfile,
): CodebaseProfileDriftDecision => {
  const active = CodebaseProfileSchema.parse(activeProfile);
  const proposed = CodebaseProfileSchema.parse(proposedProfile);
  const diagnostics = driftFields.flatMap((field) => driftDiagnostic(field, active[field], proposed[field]));
  return {
    decision: diagnostics.length === 0
      ? "no_change"
      : diagnostics.some((diagnostic) => diagnostic.severity === "blocked") ? "blocked" : "review_required",
    activeProfile: active,
    proposedProfile: proposed,
    diagnostics,
  };
};

export const evaluateCodebaseProfilePin = (
  expected: CodebaseProfilePin,
  actual: CodebaseProfilePin,
): CodebaseProfileGateDecision => {
  if (expected.codebaseProfileId !== actual.codebaseProfileId) {
    return {
      decision: "blocked",
      expected,
      actual,
      reason: `codebase profile id mismatch: expected ${expected.codebaseProfileId}, got ${actual.codebaseProfileId}`,
    };
  }
  if (
    expected.codebaseRootFingerprint !== undefined &&
    actual.codebaseRootFingerprint !== undefined &&
    expected.codebaseRootFingerprint !== actual.codebaseRootFingerprint
  ) {
    return {
      decision: "blocked",
      expected,
      actual,
      reason: `codebase profile fingerprint mismatch for ${expected.codebaseProfileId}`,
    };
  }
  return { decision: "pass", expected, actual };
};

export const refreshCodebaseProfileForReview = (
  activeProfile: CodebaseProfile,
  input: GenerateCodebaseProfileInput = {},
): CodebaseProfileDriftDecision & { evidence: CodebaseProfileEvidence } => {
  const generated = generateCodebaseProfile({
    ...input,
    codebaseProfileId: input.codebaseProfileId ?? activeProfile.codebaseProfileId,
    displayName: input.displayName ?? activeProfile.displayName,
    verificationPolicyVersion: input.verificationPolicyVersion ?? activeProfile.verificationPolicyVersion,
  });
  return {
    ...detectCodebaseProfileDrift(activeProfile, generated.profile),
    evidence: generated.evidence,
  };
};

const driftFields: ProfileDriftField[] = [
  "rootFingerprint",
  "languages",
  "packageManagers",
  "primaryPackageManager",
  "sourceRoots",
  "generatedDirs",
  "ignoredDirs",
  "testCommands",
  "typecheckCommands",
  "lintCommands",
  "testRiskTiers",
  "protectedPaths",
  "conventions",
  "knownFailures",
  "acpClientQuirks",
  "verificationPolicyVersion",
];

const driftDiagnostic = (
  field: ProfileDriftField,
  previous: unknown,
  current: unknown,
): CodebaseProfileDriftDiagnostic[] => {
  const normalizedPrevious = stableValue(previous);
  const normalizedCurrent = stableValue(current);
  if (JSON.stringify(normalizedPrevious) === JSON.stringify(normalizedCurrent)) {
    return [];
  }
  const severity = blockedDriftFields.has(field) ? "blocked" : "review";
  return [{
    field,
    kind: driftKind(normalizedPrevious, normalizedCurrent),
    previous: normalizedPrevious,
    current: normalizedCurrent,
    severity,
    message: severity === "blocked"
      ? `codebase profile ${field} changed and blocks promotion until reviewed`
      : `codebase profile ${field} changed and requires review before promotion`,
  }];
};

const blockedDriftFields = new Set<ProfileDriftField>([
  "rootFingerprint",
  "sourceRoots",
  "generatedDirs",
  "ignoredDirs",
  "testCommands",
  "typecheckCommands",
  "lintCommands",
  "testRiskTiers",
  "protectedPaths",
  "knownFailures",
  "acpClientQuirks",
  "verificationPolicyVersion",
]);

const driftKind = (previous: unknown, current: unknown): ProfileDriftKind => {
  if (Array.isArray(previous) && Array.isArray(current)) {
    if (previous.length < current.length && previous.every((value) => current.includes(value))) {
      return "added";
    }
    if (previous.length > current.length && current.every((value) => previous.includes(value))) {
      return "removed";
    }
  }
  return "changed";
};

const readPackageJson = (cwd: string): CodebaseProfileEvidence["packageJson"] | undefined => {
  const path = join(cwd, "package.json");
  if (!existsSync(path)) {
    return undefined;
  }
  try {
    const parsed = JSON.parse(readFileSync(path, "utf8")) as unknown;
    if (parsed === null || typeof parsed !== "object") {
      return undefined;
    }
    const record = parsed as Record<string, unknown>;
    const scripts = record.scripts !== null && typeof record.scripts === "object"
      ? Object.fromEntries(
        Object.entries(record.scripts as Record<string, unknown>)
          .filter(([, value]) => typeof value === "string")
          .sort(([left], [right]) => left.localeCompare(right)),
      ) as Record<string, string>
      : {};
    return {
      path: "package.json",
      ...(typeof record.name === "string" && record.name.length > 0 ? { name: record.name } : {}),
      scripts,
    };
  } catch {
    return undefined;
  }
};

const listRepoEntries = (cwd: string): string[] => {
  try {
    return readdirSync(cwd, { withFileTypes: true })
      .filter((entry) => !SKIPPED_DIRS.has(entry.name))
      .map((entry) => entry.isDirectory() ? `${entry.name}/` : entry.name)
      .sort((left, right) => left.localeCompare(right));
  } catch {
    return [];
  }
};

const scanRepoFiles = (cwd: string): string[] => {
  const files: string[] = [];
  const visit = (dir: string): void => {
    if (files.length >= MAX_SCANNED_FILES) {
      return;
    }
    let entries: Dirent[];
    try {
      entries = readdirSync(dir, { withFileTypes: true });
    } catch {
      return;
    }
    for (const entry of entries.sort((left, right) => left.name.localeCompare(right.name))) {
      if (files.length >= MAX_SCANNED_FILES) {
        return;
      }
      const absolutePath = join(dir, entry.name);
      const relativePath = normalizeRelativePath(relative(cwd, absolutePath));
      if (entry.isDirectory()) {
        if (!SKIPPED_DIRS.has(entry.name)) {
          visit(absolutePath);
        }
        continue;
      }
      if (entry.isFile()) {
        files.push(relativePath);
      }
    }
  };
  visit(cwd);
  return files.sort((left, right) => left.localeCompare(right));
};

const detectPackageManagers = (
  cwd: string,
  packageJson: CodebaseProfileEvidence["packageJson"] | undefined,
  lockfiles: readonly string[],
): PackageManager[] => {
  const managers = new Set<PackageManager>();
  for (const lockfile of lockfiles) {
    const manager = lockfilePackageManager(lockfile);
    if (manager !== undefined) {
      managers.add(manager);
    }
  }
  if (packageJson !== undefined) {
    managers.add("npm");
  }
  if (existsSync(join(cwd, "go.mod"))) {
    managers.add("go");
  }
  if (existsSync(join(cwd, "Cargo.toml"))) {
    managers.add("cargo");
  }
  if (managers.size === 0) {
    managers.add("none");
  }
  return sortPackageManagers([...managers]);
};

const lockfilePackageManager = (path: string): PackageManager | undefined =>
  LOCKFILE_TO_PACKAGE_MANAGER.find(([lockfile]) => lockfile === path)?.[1];

const sortPackageManagers = (managers: PackageManager[]): PackageManager[] => {
  const rank: Record<PackageManager, number> = {
    bun: 0,
    npm: 1,
    pnpm: 2,
    yarn: 3,
    pip: 4,
    cargo: 5,
    go: 6,
    none: 7,
  };
  return managers.sort((left, right) => rank[left] - rank[right] || left.localeCompare(right));
};

const preferredPackageManager = (managers: readonly PackageManager[]): PackageManager =>
  managers.find((manager) => manager !== "none" && ["bun", "npm", "pnpm", "yarn"].includes(manager)) ?? managers[0] ?? "none";

const detectSourceRoots = (cwd: string, files: readonly string[]): string[] => {
  const roots = new Set<string>();
  for (const candidate of SOURCE_ROOT_CANDIDATES) {
    if (existsSync(join(cwd, candidate))) {
      try {
        if (statSync(join(cwd, candidate)).isDirectory()) {
          roots.add(candidate);
        }
      } catch {
        // Ignore transient filesystem errors while profiling.
      }
    }
  }
  for (const file of files) {
    if (!file.includes("/")) {
      continue;
    }
    const [first] = file.split("/");
    if (first !== undefined && first.length > 0 && hasSourceExtension(file) && !SKIPPED_DIRS.has(first)) {
      roots.add(first);
    }
  }
  return uniqueSorted([...roots]);
};

const detectGeneratedDirs = (cwd: string): string[] =>
  DEFAULT_GENERATED_DIRS
    .filter((path) => {
      try {
        return existsSync(join(cwd, path)) && statSync(join(cwd, path)).isDirectory();
      } catch {
        return false;
      }
    })
    .map(normalizeRelativePath)
    .sort((left, right) => left.localeCompare(right));

const detectIgnoredDirs = (cwd: string): string[] => {
  const ignored = new Set(DEFAULT_IGNORED_DIRS.map(normalizeRelativePath));
  for (const entry of readGitignoreDirEntries(cwd)) {
    ignored.add(entry);
  }
  return uniqueSorted([...ignored]);
};

const readGitignoreDirEntries = (cwd: string): string[] => {
  const path = join(cwd, ".gitignore");
  if (!existsSync(path)) {
    return [];
  }
  try {
    return readFileSync(path, "utf8")
      .split(/\r?\n/)
      .map((line) => line.trim())
      .filter((line) => line.length > 0 && !line.startsWith("#") && !line.startsWith("!"))
      .map((line) => normalizeRelativePath(line.replace(/^\//, "").replace(/\*+$/g, "")))
      .filter((line) => line.length > 0 && !line.includes("*"))
      .map((line) => line.endsWith("/") ? line.slice(0, -1) : line)
      .filter((line) => line.length > 0);
  } catch {
    return [];
  }
};

const collectLanguageEvidence = (files: readonly string[]): Partial<Record<CodebaseLanguage, string[]>> => {
  const evidence: Partial<Record<CodebaseLanguage, string[]>> = {};
  for (const file of files) {
    const language = languageForFile(file);
    if (language === undefined) {
      continue;
    }
    const current = evidence[language] ?? [];
    if (current.length < MAX_LANGUAGE_EVIDENCE_FILES) {
      current.push(file);
      evidence[language] = current;
    }
  }
  return stableValue(evidence) as Partial<Record<CodebaseLanguage, string[]>>;
};

const detectLanguages = (
  cwd: string,
  packageJson: CodebaseProfileEvidence["packageJson"] | undefined,
  languageFiles: Partial<Record<CodebaseLanguage, string[]>>,
): CodebaseLanguage[] => {
  const languages = new Set<CodebaseLanguage>();
  Object.entries(languageFiles).forEach(([language, files]) => {
    if ((files?.length ?? 0) > 0) {
      languages.add(language as CodebaseLanguage);
    }
  });
  if (existsSync(join(cwd, "tsconfig.json"))) {
    languages.add("typescript");
  }
  if (packageJson !== undefined) {
    languages.add("javascript");
  }
  if (existsSync(join(cwd, "requirements.txt")) || existsSync(join(cwd, "pyproject.toml"))) {
    languages.add("python");
  }
  if (existsSync(join(cwd, "Cargo.toml"))) {
    languages.add("rust");
  }
  if (existsSync(join(cwd, "go.mod"))) {
    languages.add("go");
  }
  return sortLanguages([...languages]);
};

const sortLanguages = (languages: CodebaseLanguage[]): CodebaseLanguage[] => {
  const rank: Record<CodebaseLanguage, number> = {
    typescript: 0,
    javascript: 1,
    python: 2,
    rust: 3,
    go: 4,
    shell: 5,
    markdown: 6,
    other: 7,
  };
  return languages.sort((left, right) => rank[left] - rank[right] || left.localeCompare(right));
};

const languageForFile = (path: string): CodebaseLanguage | undefined => {
  if (/\.(ts|tsx|mts|cts)$/.test(path)) {
    return "typescript";
  }
  if (/\.(js|jsx|mjs|cjs)$/.test(path)) {
    return "javascript";
  }
  if (/\.py$/.test(path)) {
    return "python";
  }
  if (/\.rs$/.test(path)) {
    return "rust";
  }
  if (/\.go$/.test(path)) {
    return "go";
  }
  if (/\.(sh|bash|zsh)$/.test(path)) {
    return "shell";
  }
  if (/\.(md|mdx)$/.test(path)) {
    return "markdown";
  }
  return undefined;
};

const hasSourceExtension = (path: string): boolean => languageForFile(path) !== undefined;

const scriptCommands = (
  kind: VerifierCommandKind,
  scripts: Record<string, string> | undefined,
  packageManager: PackageManager,
): CommandSpec[] => {
  if (scripts === undefined) {
    return [];
  }
  const names = scriptNamesFor(kind).filter((name) => {
    const script = scripts[name];
    return script !== undefined && !isPlaceholderTestScript(script);
  });
  return names.map((name) => ({
    commandId: name === kind ? kind : `${kind}.${sanitizeIdPart(name)}`,
    command: packageScriptCommand(packageManager, name),
    required: true,
  }));
};

const scriptNamesFor = (kind: VerifierCommandKind): string[] => {
  switch (kind) {
    case "test":
      return ["test", "test:unit", "test:ci"];
    case "typecheck":
      return ["typecheck", "check:types", "tsc"];
    case "lint":
      return ["lint", "lint:ci"];
  }
};

const isPlaceholderTestScript = (script: string): boolean =>
  /no test specified/i.test(script) || /exit 1/.test(script) && /error/i.test(script);

const packageScriptCommand = (packageManager: PackageManager, scriptName: string): string[] => {
  switch (packageManager) {
    case "bun":
      return ["bun", "run", scriptName];
    case "pnpm":
      return ["pnpm", "run", scriptName];
    case "yarn":
      return ["yarn", scriptName];
    case "npm":
    case "pip":
    case "cargo":
    case "go":
    case "none":
      return scriptName === "test" ? ["npm", "test"] : ["npm", "run", scriptName];
  }
};

const normalizeObservedVerifierBehavior = (
  observed: readonly ObservedVerifierBehavior[],
): CodebaseProfileEvidence["observedVerifierBehavior"] =>
  observed.map((entry, index) => ({
    kind: entry.kind,
    commandId: entry.commandId ?? `${entry.kind}.observed.${index + 1}`,
    command: [...entry.command],
    ...(entry.cwd === undefined ? {} : { cwd: normalizeRelativePath(entry.cwd) }),
    required: entry.required ?? entry.lastExitCode === 0,
    ...(entry.lastExitCode === undefined ? {} : { lastExitCode: entry.lastExitCode }),
  })).sort((left, right) => left.commandId.localeCompare(right.commandId));

const mergeObservedCommands = (
  detected: readonly CommandSpec[],
  observed: readonly CodebaseProfileEvidence["observedVerifierBehavior"][number][],
): CommandSpec[] => {
  const commands = new Map<string, CommandSpec>();
  for (const command of detected) {
    commands.set(commandKey(command), command);
  }
  for (const entry of observed) {
    const command: CommandSpec = {
      commandId: sanitizeCommandId(entry.commandId),
      command: entry.command,
      ...(entry.cwd === undefined ? {} : { cwd: entry.cwd }),
      required: entry.required,
    };
    if (!commands.has(commandKey(command))) {
      commands.set(commandKey(command), command);
    }
  }
  return [...commands.values()].sort((left, right) => left.commandId.localeCompare(right.commandId));
};

const commandKey = (command: CommandSpec): string =>
  JSON.stringify([command.command, command.cwd ?? "", command.required]);

const detectConventions = (input: {
  configFiles: readonly string[];
  generatedDirs: readonly string[];
  ignoredDirs: readonly string[];
  packageManagers: readonly PackageManager[];
  primaryPackageManager: PackageManager;
  packageScripts: Record<string, string>;
  sourceRoots: readonly string[];
  languages: readonly CodebaseLanguage[];
  observedVerifierBehavior: readonly CodebaseProfileEvidence["observedVerifierBehavior"][number][];
}): string[] => {
  const conventions = new Set<string>();
  for (const language of input.languages) {
    conventions.add(`language.${language}`);
  }
  for (const manager of input.packageManagers) {
    conventions.add(`package-manager.${manager}`);
  }
  conventions.add(`primary-package-manager.${input.primaryPackageManager}`);
  for (const sourceRoot of input.sourceRoots) {
    conventions.add(`source-root.${sanitizeIdPart(sourceRoot)}`);
  }
  for (const generatedDir of input.generatedDirs) {
    conventions.add(`generated-dir.${sanitizeIdPart(generatedDir)}`);
  }
  for (const ignoredDir of input.ignoredDirs) {
    conventions.add(`ignored-dir.${sanitizeIdPart(ignoredDir)}`);
  }
  for (const configFile of input.configFiles) {
    conventions.add(`config.${sanitizeIdPart(configFile)}`);
  }
  for (const scriptName of Object.keys(input.packageScripts).sort((left, right) => left.localeCompare(right))) {
    if (scriptNamesFor("test").includes(scriptName) || scriptNamesFor("typecheck").includes(scriptName) || scriptNamesFor("lint").includes(scriptName)) {
      conventions.add(`package-script.${sanitizeIdPart(scriptName)}`);
    }
  }
  for (const observation of input.observedVerifierBehavior) {
    const outcome = observation.lastExitCode === undefined
      ? "observed"
      : observation.lastExitCode === 0 ? "passing" : "failing";
    conventions.add(`verifier.${observation.kind}.${outcome}`);
  }
  return [...conventions].sort((left, right) => left.localeCompare(right));
};

const buildTestRiskTiers = (input: {
  testCommands: readonly CommandSpec[];
  typecheckCommands: readonly CommandSpec[];
  lintCommands: readonly CommandSpec[];
  protectedPaths: readonly string[];
}): CodebaseProfile["testRiskTiers"] => {
  const tiers: CodebaseProfile["testRiskTiers"] = [];
  if (input.typecheckCommands.length > 0) {
    tiers.push({
      tierId: "risk.typecheck",
      description: "Type-level regression gate for source changes.",
      commandIds: input.typecheckCommands.map((command) => command.commandId),
      protectedPaths: [],
      required: input.typecheckCommands.some((command) => command.required),
    });
  }
  if (input.testCommands.length > 0) {
    tiers.push({
      tierId: "risk.test",
      description: "Behavioral regression gate for runnable tests.",
      commandIds: input.testCommands.map((command) => command.commandId),
      protectedPaths: [],
      required: input.testCommands.some((command) => command.required),
    });
  }
  if (input.lintCommands.length > 0) {
    tiers.push({
      tierId: "risk.lint",
      description: "Static hygiene gate for style and low-risk defects.",
      commandIds: input.lintCommands.map((command) => command.commandId),
      protectedPaths: [],
      required: input.lintCommands.some((command) => command.required),
    });
  }
  if (input.protectedPaths.length > 0) {
    tiers.push({
      tierId: "risk.protected-paths",
      description: "Protected path gate for metadata, dependencies, and generated outputs.",
      commandIds: [],
      protectedPaths: [...input.protectedPaths],
      required: true,
    });
  }
  return tiers.sort((left, right) => left.tierId.localeCompare(right.tierId));
};

const knownFailuresFromVerifierBehavior = (
  observed: readonly CodebaseProfileEvidence["observedVerifierBehavior"][number][],
): CodebaseProfile["knownFailures"] =>
  observed
    .filter((entry) => entry.lastExitCode !== undefined && entry.lastExitCode !== 0)
    .map((entry) => ({
      failureId: `known-failure.${sanitizeIdPart(entry.commandId)}`,
      source: "verifier" as const,
      commandId: sanitizeCommandId(entry.commandId),
      severity: "failure" as const,
      summary: `${entry.kind} verifier ${entry.commandId} last exited ${entry.lastExitCode}`,
      lastExitCode: entry.lastExitCode,
    }))
    .sort((left, right) => left.failureId.localeCompare(right.failureId));

const defaultAcpClientQuirks = (): CodebaseProfile["acpClientQuirks"] => [
  {
    quirkId: "acp.client.fs-write-text-file.optional",
    affectedCapability: "fs/write_text_file",
    behavior: "Some ACP consumers do not advertise direct text-file writes.",
    mitigation: "Route edits through the configured edit strategy and record write failures as optimizer evidence.",
  },
  {
    quirkId: "acp.client.terminal-create.optional",
    affectedCapability: "terminal/create",
    behavior: "Some ACP consumers do not provide an interactive terminal capability.",
    mitigation: "Treat verifier execution as capability-gated and keep terminal failures distinct from verifier failures.",
  },
  {
    quirkId: "acp.client.permissions.variable",
    affectedCapability: "permissions",
    behavior: "ACP consumers can run in permissioned, auto, or YOLO modes.",
    mitigation: "Pin replay and promotion evidence to the active profile before applying optimizer policy changes.",
  },
];

const codebaseProfileIdFor = (cwd: string): string =>
  `codebase.${sha256(resolve(cwd)).slice(0, 12)}`;

const fingerprintCodebaseProfileEvidence = (value: unknown): string =>
  `sha256:${sha256(JSON.stringify(stableValue(value)))}`;

const hashFile = (path: string): string => {
  try {
    return `sha256:${sha256(readFileSync(path))}`;
  } catch {
    return "sha256:unreadable";
  }
};

const sha256 = (content: string | Buffer): string =>
  createHash("sha256").update(content).digest("hex");

const normalizeRelativePath = (path: string): string =>
  path.split(sep).join("/").replace(/^\.\/+/, "").replace(/\/+$/, "");

const sanitizeCommandId = (value: string): string => {
  const sanitized = sanitizeIdPart(value);
  return /^[A-Za-z0-9]/.test(sanitized) ? sanitized : `command.${sanitized}`;
};

const sanitizeIdPart = (value: string): string =>
  value
    .trim()
    .replace(/[^A-Za-z0-9._:-]+/g, ".")
    .replace(/^[._:-]+|[._:-]+$/g, "")
    .replace(/[._:-]{2,}/g, ".")
    .toLowerCase() || "default";

const uniqueSorted = (values: readonly string[]): string[] =>
  [...new Set(values)].sort((left, right) => left.localeCompare(right));

const stableValue = (value: unknown): unknown => {
  if (Array.isArray(value)) {
    return value.map((entry) => stableValue(entry));
  }
  if (value !== null && typeof value === "object") {
    return Object.fromEntries(
      Object.entries(value as Record<string, unknown>)
        .filter(([, entry]) => entry !== undefined)
        .sort(([left], [right]) => left.localeCompare(right))
        .map(([key, entry]) => [key, stableValue(entry)]),
    );
  }
  return value;
};
