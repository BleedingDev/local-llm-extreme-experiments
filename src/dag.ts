import { parseJsonObject, type LlmRouter } from "./llm";
import { DagPlanSchema, type DagPlan, type PrdArtifact } from "./types";

const idFromTitle = (prefix: string, title: string, index: number): string =>
  `${prefix}-${index + 1}-${title
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-|-$/g, "")
    .slice(0, 36)}`;

export const fallbackDag = (prd: PrdArtifact): DagPlan => {
  const epics = [
    "ACP server contract and session behavior",
    "Coding and planning user workflows",
    "Trace replay and evaluation harness",
    "Self-optimization and promotion controls",
  ];
  const tasks = [
    "Define the consumer-agnostic ACP product contract",
    "Implement ACP session modes and slash command behavior",
    "Implement mixed GPT master and local MLX executor routing",
    "Generate planning artifacts for read-only plan/report turns",
    "Persist run manifests metrics traces and telemetry",
    "Convert failures and repairs into replayable eval evidence",
    "Gate self-optimization candidates with promotion and rollback controls",
  ];

  const issues = [
    ...epics.map((title, index) => ({
      issueId: idFromTitle("epic", title, index),
      issueType: "epic" as const,
      title,
      body: `Epic derived from ${prd.documentTitle}.`,
      status: "open" as const,
      sortIndex: index,
      plannerMetadata: {
        suggestedOwner: "master" as const,
        expectedFiles: [],
        verificationCommands: ["npm run typecheck"],
        risk: "medium" as const,
      },
    })),
    ...tasks.map((title, index) => ({
      issueId: idFromTitle("task", title, index),
      issueType: "task" as const,
      title,
      body: "Implement and verify this slice with explicit artifacts and metrics.",
      status: "open" as const,
      sortIndex: epics.length + index,
      plannerMetadata: {
        suggestedOwner: index < 2 ? ("master" as const) : ("executor" as const),
        expectedFiles: ["src/**/*.ts", "package.json", "docs/bleeding-agent.md"],
        verificationCommands: ["npm run typecheck"],
        risk: index === 1 ? ("high" as const) : ("medium" as const),
      },
    })),
  ];

  return {
    summary: {
      planId: `bag-plan-${Date.now()}`,
      title: prd.documentTitle,
      status: "draft",
      issueCount: issues.length,
      dependencyCount: tasks.length - 1,
      chosenTier: "medium",
    },
    issues,
    dependencies: tasks.slice(1).map((title, index) => ({
      fromIssueId: idFromTitle("task", title, index + 1),
      toIssueId: idFromTitle("task", tasks[index] ?? title, index),
      relation: "depends_on" as const,
    })),
  };
};

export const renderDagMarkdown = (plan: DagPlan): string => {
  const issues = plan.issues
    .map(
      (issue) =>
        `- [ ] ${issue.issueId} (${issue.issueType}, ${issue.plannerMetadata.risk}): ${issue.title}\n  ${issue.body}`,
    )
    .join("\n");
  const deps = plan.dependencies
    .map((dep) => `- ${dep.fromIssueId} ${dep.relation} ${dep.toIssueId}`)
    .join("\n");
  return `# ${plan.summary.title}\n\n## Issues\n\n${issues}\n\n## Dependencies\n\n${deps}\n`;
};

export const generateDag = async (input: {
  router: LlmRouter;
  prd: PrdArtifact;
  repoContext: string;
}): Promise<DagPlan> => {
  const fallback = fallbackDag(input.prd);
  if (!input.router.masterAvailable) {
    return fallback;
  }

  const raw = await input.router.chatText({
    role: "master",
    json: true,
    maxTokens: 5000,
    purpose: "prd-to-dag",
    messages: [
      {
        role: "system",
        content: [
          "Turn the PRD into a dependency-aware coding DAG. Return JSON ONLY (no prose, no code fences) matching this shape exactly:",
          "{",
          '  "summary": { "planId": string, "title": string, "status": "draft"|"ready"|"running"|"completed"|"failed", "issueCount": int, "dependencyCount": int, "chosenTier": "small"|"medium"|"large" },',
          '  "issues": [ { "issueId": string, "issueType": "epic"|"task", "title": string, "body": string, "status": "open"|"in_progress"|"blocked"|"closed", "sortIndex": int, "plannerMetadata": { "suggestedOwner": "master"|"executor"|"critic", "expectedFiles": string[], "verificationCommands": string[], "risk": "low"|"medium"|"high" } } ],',
          '  "dependencies": [ { "fromIssueId": string, "toIssueId": string, "relation": "depends_on"|"blocks"|"relates_to" } ]',
          "}",
          "Enum fields MUST use the exact lowercase strings shown — no synonyms, no capitalization variants.",
        ].join("\n"),
      },
      {
        role: "user",
        content: [
          `PRD:\n${JSON.stringify(input.prd, null, 2)}`,
          `Repo context:\n${input.repoContext.slice(0, 12000)}`,
        ].join("\n\n"),
      },
    ],
  });

  const candidate = parseJsonObject(raw, fallback);
  const safeParsed = DagPlanSchema.safeParse(candidate);
  const parsed = safeParsed.success ? safeParsed.data : fallback;
  return {
    ...parsed,
    summary: {
      ...parsed.summary,
      issueCount: parsed.issues.length,
      dependencyCount: parsed.dependencies.length,
    },
  };
};
