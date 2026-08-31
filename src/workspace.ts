import { execFileSync } from "node:child_process";
import { existsSync, readFileSync, statSync } from "node:fs";
import { join, relative, resolve } from "node:path";
import { parseJsonObject, type LlmRouter } from "./llm";
import type { BagConfig, ContextScoutFinding } from "./types";

const TEXT_FILE_EXTENSIONS = new Set([
  ".cjs",
  ".css",
  ".js",
  ".json",
  ".jsx",
  ".md",
  ".mjs",
  ".py",
  ".rs",
  ".sh",
  ".sql",
  ".tsx",
  ".ts",
  ".txt",
  ".yaml",
  ".yml",
]);

const extname = (path: string): string => {
  const index = path.lastIndexOf(".");
  return index < 0 ? "" : path.slice(index);
};

export type ProjectKind = "node" | "python" | "shell" | "rust" | "go" | "unknown";

export const detectProjectKind = (cwd: string): ProjectKind => {
  if (existsSync(join(cwd, "package.json"))) {
    return "node";
  }
  if (existsSync(join(cwd, "Cargo.toml"))) {
    return "rust";
  }
  if (existsSync(join(cwd, "go.mod"))) {
    return "go";
  }
  const pythonMarkers = ["pyproject.toml", "setup.py", "setup.cfg", "requirements.txt"];
  if (pythonMarkers.some((marker) => existsSync(join(cwd, marker)))) {
    return "python";
  }

  let entries: string[];
  try {
    entries = listWorkspaceFiles(cwd).slice(0, 50);
  } catch {
    return "unknown";
  }

  let pyCount = 0;
  let shCount = 0;
  for (const entry of entries) {
    const ext = extname(entry);
    if (ext === ".py") {
      pyCount += 1;
    } else if (ext === ".sh") {
      shCount += 1;
    }
  }

  if (pyCount > shCount && pyCount >= 1) {
    return "python";
  }
  if (shCount >= 1) {
    return "shell";
  }
  return "unknown";
};

export const listWorkspaceFiles = (cwd = process.cwd()): string[] => {
  const parseLines = (text: string): string[] =>
    [...new Set(text.split("\n").map((line) => line.trim()).filter(Boolean))];

  try {
    const gitFiles = parseLines(
      execFileSync("git", ["ls-files", "--cached", "--others", "--exclude-standard"], {
        cwd,
        encoding: "utf8",
        stdio: ["ignore", "pipe", "ignore"],
      }),
    );
    if (gitFiles.length > 0) {
      return gitFiles;
    }
  } catch {
    // Fall through to rg below. Isolated ACP fixture workspaces can be inside
    // an ignored parent path, where git intentionally reports no files.
  }
  try {
    return parseLines(execFileSync("rg", ["--files"], { cwd, encoding: "utf8", stdio: ["ignore", "pipe", "ignore"] }));
  } catch {
    return [];
  }
};

export const readRelevantFileSamples = (input: {
  cwd?: string;
  config: BagConfig;
  task: string;
}): Array<{ path: string; content: string }> => {
  const cwd = input.cwd ?? process.cwd();
  const taskWords = new Set(
    input.task
      .toLowerCase()
      .split(/[^a-z0-9_-]+/)
      .filter((word) => word.length >= 3),
  );

  return listWorkspaceFiles(cwd)
    .filter((file) => TEXT_FILE_EXTENSIONS.has(extname(file)))
    .map((file) => {
      const keywordScore = [...taskWords].filter((word) => file.toLowerCase().includes(word)).length;
      const priority =
        file === "README.md" || file === "package.json" || file.endsWith("AGENTS.md") ? 3 : 0;
      return { file, score: keywordScore + priority };
    })
    .sort((left, right) => right.score - left.score || left.file.localeCompare(right.file))
    .slice(0, input.config.policy.contextFiles)
    .flatMap(({ file }) => {
      const path = resolve(cwd, file);
      if (!existsSync(path) || statSync(path).size > 512_000) {
        return [];
      }
      const content = readFileSync(path, "utf8").slice(0, input.config.policy.contextCharsPerFile);
      return [{ path: file, content }];
    });
};

export const buildRepoContext = (input: {
  cwd?: string;
  config: BagConfig;
  task: string;
  findings?: ContextScoutFinding[];
}): string => {
  const cwd = input.cwd ?? process.cwd();
  const files = readRelevantFileSamples({ cwd, config: input.config, task: input.task });
  const gitRoot = (() => {
    try {
      return execFileSync("git", ["rev-parse", "--show-toplevel"], { cwd, encoding: "utf8" }).trim();
    } catch {
      return cwd;
    }
  })();

  const fileList = files.map((file) => `- ${file.path}`).join("\n");
  const findingText =
    input.findings == null || input.findings.length === 0
      ? "No scout findings yet."
      : input.findings
          .map((finding) => `- ${finding.file}: ${finding.reason} (${finding.confidence})`)
          .join("\n");
  const samples = files
    .slice(0, 24)
    .map((file) => `### ${file.path}\n\`\`\`\n${file.content}\n\`\`\``)
    .join("\n\n");

  return [
    `Repo root: ${gitRoot}`,
    `Relative cwd: ${relative(gitRoot, cwd) || "."}`,
    "",
    "Candidate files:",
    fileList,
    "",
    "Scout findings:",
    findingText,
    "",
    "File samples:",
    samples,
  ].join("\n");
};

export const runLocalContextScouts = async (input: {
  router: LlmRouter;
  config: BagConfig;
  task: string;
  cwd?: string;
}): Promise<ContextScoutFinding[]> => {
  const localReady = await input.router.localAvailable();
  const files = readRelevantFileSamples({
    config: input.config,
    task: input.task,
    ...(input.cwd != null ? { cwd: input.cwd } : {}),
  }).slice(0, input.config.policy.executorConcurrency);

  if (!localReady || files.length === 0) {
    return files.slice(0, 12).map((file, index) => ({
      file: file.path,
      reason: index === 0 ? "High-priority repo metadata or task keyword match." : "Relevant sample selected by filename ranking.",
      signals: ["deterministic-fallback"],
      confidence: 0.35,
    }));
  }

  const limit = Math.min(input.config.policy.executorConcurrency, input.config.policy.maxExecutorConcurrency);
  const queue = [...files];
  const findings: ContextScoutFinding[] = [];

  const worker = async () => {
    for (;;) {
      const file = queue.shift();
      if (file == null) {
        return;
      }
      const raw = await input.router.chatText({
        role: "local",
        json: true,
        maxTokens: 512,
        purpose: "context-scout",
        messages: [
          {
            role: "system",
            content:
              "You are a repo context scout. Return compact JSON only: {\"file\",\"reason\",\"signals\",\"confidence\"}.",
          },
          {
            role: "user",
            content: `Task:\n${input.task}\n\nFile: ${file.path}\n\nContent sample:\n${file.content.slice(0, 5000)}`,
          },
        ],
      });
      const parsed = parseJsonObject(raw, {
        file: file.path,
        reason: "Local scout did not return parseable JSON.",
        signals: ["parse-fallback"],
        confidence: 0.25,
      });
      const parsedObject = parsed as { file?: unknown; reason?: unknown; signals?: unknown; confidence?: unknown };
      findings.push({
        file: String(parsedObject.file ?? file.path),
        reason: String(parsedObject.reason ?? "No reason returned."),
        signals: Array.isArray(parsedObject.signals) ? parsedObject.signals.map(String) : [],
        confidence:
          typeof parsedObject.confidence === "number"
            ? Math.max(0, Math.min(1, parsedObject.confidence))
            : 0.5,
      });
    }
  };

  await Promise.all(Array.from({ length: limit }, () => worker()));
  return findings.sort((left, right) => right.confidence - left.confidence);
};

export const loadKnowledge = (cwd = process.cwd()): string => {
  const candidates = [
    join(cwd, ".bag", "knowledge.md"),
    join(cwd, ".bag", "tool-guidance.md"),
    join(cwd, "AGENTS.md"),
    join(cwd, "README.md"),
  ];
  return candidates
    .filter((path) => existsSync(path))
    .map((path) => `# ${path}\n${readFileSync(path, "utf8").slice(0, 8000)}`)
    .join("\n\n");
};
