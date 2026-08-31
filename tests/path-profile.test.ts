import { describe, expect, test } from "bun:test";
import {
  BagConfigSchema,
  DEFAULT_PATH_PROFILE,
  PathProfileSchema,
} from "../src/types";
import { auditScratchHygiene } from "../src/scratch-hygiene";
import {
  buildSnapshotCommand,
  renderFindMetadataExcludes,
} from "../src/instruction-verifier";
import { buildSelfCheckSystemPrompt } from "../src/pre-submit-self-check";
import { buildExecutorSystemPrompt } from "../src/autonomous-coding-turn";

const trace = (
  entries: Array<{ command: string; output?: string; exitCode?: number | null }>,
) =>
  entries.map((e) => ({
    command: e.command,
    output: e.output ?? "",
    exitCode: e.exitCode === undefined ? 0 : e.exitCode,
  }));

describe("PathProfile — schema defaults", () => {
  test("default profile returns the historical Linux conventions", () => {
    const parsed = PathProfileSchema.parse(undefined);
    expect(parsed).toEqual({
      metadataDirs: [".bag", ".git"],
      scratchDirs: ["/tmp", "/var/tmp"],
      systemPathDirs: ["/usr/local/bin", "/usr/bin", "/bin"],
    });
    // Module-level constant matches the schema default exactly.
    expect(DEFAULT_PATH_PROFILE).toEqual(parsed);
  });

  test("BagConfigSchema parses a valid pathProfile and falls back to default when missing", () => {
    // 1. Missing pathProfile → schema fills in the Linux defaults so older
    //    bag.config.json files (e.g. the one written by bench/bag_agent/agent.py)
    //    keep working with no edits.
    const fallback = BagConfigSchema.parse({});
    expect(fallback.pathProfile).toEqual(DEFAULT_PATH_PROFILE);
    // Same shape as bench/bag_agent/agent.py emits — must round-trip cleanly.
    const benchConfig = BagConfigSchema.parse({
      artifactDir: ".bag",
      master: {
        provider: "openai",
        model: "claude-opus-4-7",
        baseUrl: "https://api.anthropic.com/v1",
        apiKeyEnv: "ANTHROPIC_AUTH_TOKEN",
        maxTokens: 4096,
        temperature: 0.2,
      },
      local: {
        provider: "openai-compatible",
        model: "claude-opus-4-7",
        baseUrl: "https://api.anthropic.com/v1",
        apiKey: "unused",
        apiKeyEnv: "ANTHROPIC_AUTH_TOKEN",
        maxTokens: 4096,
        temperature: 0.2,
      },
    });
    expect(benchConfig.pathProfile).toEqual(DEFAULT_PATH_PROFILE);

    // 2. Explicit override is preserved.
    const explicit = BagConfigSchema.parse({
      pathProfile: {
        metadataDirs: [".bag"],
        scratchDirs: ["/scratch"],
        systemPathDirs: ["/usr/local/bin"],
      },
    });
    expect(explicit.pathProfile).toEqual({
      metadataDirs: [".bag"],
      scratchDirs: ["/scratch"],
      systemPathDirs: ["/usr/local/bin"],
    });
  });

  test("empty array overrides are rejected", () => {
    expect(() => PathProfileSchema.parse({ metadataDirs: [] })).toThrow();
    expect(() => PathProfileSchema.parse({ scratchDirs: [] })).toThrow();
    expect(() => PathProfileSchema.parse({ systemPathDirs: [] })).toThrow();
    // Empty-string entries are rejected (would render an empty `find -not
    // -path` glob and silently match every file under cwd).
    expect(() =>
      PathProfileSchema.parse({ metadataDirs: [""] }),
    ).toThrow();
  });
});

describe("PathProfile — flow into scratch-hygiene", () => {
  test("override scratchDirs flows into scratch-hygiene audit", () => {
    const profile = {
      ...DEFAULT_PATH_PROFILE,
      scratchDirs: ["/scratch", "/cache"],
    };
    // /scratch and /cache writes should now be flagged; /tmp should NOT (the
    // override REPLACES the default — explicit-is-better-than-implicit).
    const result = auditScratchHygiene(
      trace([
        { command: "echo data > /scratch/foo.txt" },
        { command: "echo other > /cache/bar.txt" },
        { command: "echo nope > /tmp/notflagged.txt" },
      ]),
      profile,
    );
    const paths = result.tmpWrites.map((w) => w.path).sort();
    expect(paths).toEqual(["/cache/bar.txt", "/scratch/foo.txt"]);
  });

  test("override scratchDirs picks up sweeping cleanup using the override path", () => {
    const profile = { ...DEFAULT_PATH_PROFILE, scratchDirs: ["/scratch"] };
    const result = auditScratchHygiene(
      trace([
        { command: "echo data > /scratch/foo.txt" },
        { command: "rm -rf /scratch/*" },
      ]),
      profile,
    );
    expect(result.tmpWrites).toEqual([]);
  });
});

describe("PathProfile — flow into instruction-verifier snapshot exclusion", () => {
  test("override metadataDirs flows into snapshot exclusion command", () => {
    // Default profile reproduces the historical exclusion globs verbatim.
    const defaults = renderFindMetadataExcludes(
      DEFAULT_PATH_PROFILE.metadataDirs,
    );
    expect(defaults).toBe("-not -path '*/.bag/*' -not -path '*/.git/*'");

    // Override flows through.
    const overridden = renderFindMetadataExcludes([
      ".bag",
      ".git",
      ".cache",
      "node_modules",
    ]);
    expect(overridden).toBe(
      "-not -path '*/.bag/*' -not -path '*/.git/*' -not -path '*/.cache/*' -not -path '*/node_modules/*'",
    );

    // The full snapshot command picks up the override too.
    const snapshotCmd = buildSnapshotCommand({
      cwd: "/work",
      pathProfile: {
        ...DEFAULT_PATH_PROFILE,
        metadataDirs: [".bag", ".git", ".cache"],
      },
      outputPath: "/tmp/snap.txt",
    });
    expect(snapshotCmd).toContain("-not -path '*/.cache/*'");
    expect(snapshotCmd).toContain('"/work"'); // cwd is JSON-encoded
  });
});

describe("PathProfile — flow into pre-submit-self-check prompt", () => {
  test("override systemPathDirs flows into self-check prompt content", () => {
    const linuxPrompt = buildSelfCheckSystemPrompt(DEFAULT_PATH_PROFILE);
    // Default cites the historical /usr/local/bin:/usr/bin:/bin colon-PATH
    // verbatim — preserves byte-equivalent behaviour for the default config.
    expect(linuxPrompt).toContain("/usr/local/bin:/usr/bin:/bin");

    const altPrompt = buildSelfCheckSystemPrompt({
      ...DEFAULT_PATH_PROFILE,
      systemPathDirs: ["/nix/var/nix/profiles/default/bin", "/usr/bin"],
    });
    expect(altPrompt).toContain(
      "/nix/var/nix/profiles/default/bin:/usr/bin",
    );
    // The original colon-form should be GONE — the override replaced it.
    expect(altPrompt).not.toContain("/usr/local/bin:/usr/bin:/bin");
  });

  test("override flows into the executor (autonomous-coding-turn) prompt", () => {
    const linuxPrompt = buildExecutorSystemPrompt(DEFAULT_PATH_PROFILE);
    // Default-profile output retains the historical /tmp / PATH wording.
    expect(linuxPrompt).toContain("/tmp");
    expect(linuxPrompt).toContain("/usr/local/bin:/usr/bin:/bin");

    const dockerPrompt = buildExecutorSystemPrompt({
      ...DEFAULT_PATH_PROFILE,
      scratchDirs: ["/scratch"],
      systemPathDirs: ["/opt/bin", "/usr/bin"],
    });
    expect(dockerPrompt).toContain("/scratch");
    expect(dockerPrompt).toContain("/opt/bin:/usr/bin");
    // /tmp examples should be gone; persistTarget should reflect override.
    expect(dockerPrompt).not.toContain("Test in /tmp");
    expect(dockerPrompt).toContain("cp X /opt/bin/");
  });
});
