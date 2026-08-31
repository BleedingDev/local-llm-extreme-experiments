import { describe, expect, test } from "bun:test";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import {
  classifyMcpToolPolicy,
  connectMcpStdioRuntimeServer,
  createMcpRuntimeToolBridge,
  mcpRuntimeToolResultToOptimizerFeedback,
  normalizeMcpServerTools,
  prepareMcpRenderedToolContracts,
  type McpServerMetadata,
} from "../src/mcp/runtime-tools";
import type { ResolvedOptimizerPolicy } from "../src/optimizer/policy-resolver";
import { renderToolContracts } from "../src/optimizer/tool-renderer";
import type { ModelProfile } from "../src/optimizer/types";

const nativeModelProfile: ModelProfile = {
  modelProfileId: "model.native",
  displayName: "Native Tool Model",
  provider: "openai",
  model: "native-tool-model",
  endpointKind: "responses",
  contextWindowTokens: 128000,
  maxOutputTokens: 4096,
  defaultTemperature: 0.1,
  toolCallingMode: "native",
  structuredOutputMode: "json_schema",
  supportsParallelToolCalls: true,
  promptStyle: "system_user",
  resultStyleVersion: "result-style.model",
  verificationPolicyVersion: "verification.model",
};

const resolvedPolicy = (modelProfile: ModelProfile): ResolvedOptimizerPolicy => ({
  source: "seed",
  modelProfile,
  codebaseProfile: {
    codebaseProfileId: "codebase.test",
    displayName: "Test Codebase",
    rootFingerprint: "sha256:test",
    languages: ["typescript"],
    packageManagers: ["npm"],
    sourceRoots: ["src"],
    testCommands: [],
    typecheckCommands: [],
    lintCommands: [],
    protectedPaths: [],
    conventions: [],
    verificationPolicyVersion: "verification.codebase",
  },
  policy: {
    policyId: "policy.test",
    modelProfileId: modelProfile.modelProfileId,
    codebaseProfileId: "codebase.test",
    status: "promoted",
    canonicalToolVersion: "canonical-tools.policy",
    renderedToolVersion: "rendered-tools.policy",
    resultStyleVersion: "result-style.policy",
    verificationPolicyVersion: "verification.policy",
    editStrategyVersion: "edit-strategy.v1",
    renderedEditContractVersion: "rendered-edit-contract.v1",
    editFallbackPolicyVersion: "edit-fallback.v1",
    editRepairPolicyVersion: "edit-repair.v1",
    editVerifierPolicyVersion: "edit-verifier.v1",
    editObjectiveSetId: "edit-objectives.default.v1",
    candidateScopes: [],
    verificationGates: [],
    maxConcurrentEvaluations: 1,
    riskTolerance: "low",
  },
  modelProfileId: modelProfile.modelProfileId,
  codebaseProfileId: "codebase.test",
  policyId: "policy.test",
  canonicalToolVersion: "canonical-tools.policy",
  renderedToolVersion: "rendered-tools.policy",
  resultStyleVersion: "result-style.policy",
  verificationPolicyVersion: "verification.policy",
  editStrategyVersion: "edit-strategy.v1",
  renderedEditContractVersion: "rendered-edit-contract.v1",
  editFallbackPolicyVersion: "edit-fallback.v1",
  editRepairPolicyVersion: "edit-repair.v1",
  editVerifierPolicyVersion: "edit-verifier.v1",
  editObjectiveSetId: "edit-objectives.default.v1",
  recordIds: {
    modelProfileRecordId: `registry.${modelProfile.modelProfileId}`,
    codebaseProfileRecordId: "registry.codebase.test",
    policyRecordId: "registry.policy.test",
  },
});

const fakeServer: McpServerMetadata = {
  serverId: "github-enterprise",
  name: "GitHub Enterprise",
  tools: [
    {
      name: "repos/read_file",
      title: "Read File",
      description: "Read one file from a repository.",
      annotations: { readOnlyHint: true },
      inputSchema: {
        type: "object",
        properties: {
          owner: { type: "string" },
          repo: { type: "string" },
          path: { type: "string" },
        },
        required: ["owner", "repo", "path"],
      },
      outputSchema: {
        type: "object",
        properties: {
          content: { type: "string" },
        },
      },
      examples: [
        {
          name: "read package",
          input: { owner: "acme", repo: "app", path: "package.json" },
          output: { content: "{...}" },
        },
      ],
    },
    {
      name: "issues/update",
      description: "Update an issue title or labels.",
      annotations: { destructiveHint: true },
      inputSchema: {
        type: "object",
        properties: {
          issueNumber: { type: "number" },
          title: { type: "string" },
        },
        required: ["issueNumber"],
      },
    },
  ],
};

const fakeRiskServer: McpServerMetadata = {
  serverId: "runtime-risk",
  name: "Runtime Risk",
  tools: [
    {
      name: "workspace/write_file",
      title: "Write File",
      description: "Write content to a workspace file.",
      annotations: { destructiveHint: true },
      inputSchema: {
        type: "object",
        properties: {
          path: { type: "string" },
          content: { type: "string" },
        },
        required: ["path", "content"],
      },
    },
    {
      name: "web/fetch_url",
      title: "Fetch URL",
      description: "Fetch a URL over HTTP.",
      annotations: { openWorldHint: true },
      inputSchema: {
        type: "object",
        properties: {
          url: { type: "string" },
        },
        required: ["url"],
      },
    },
    {
      name: "terminal/exec",
      title: "Run Command",
      description: "Run a shell command in a subprocess.",
      inputSchema: {
        type: "object",
        properties: {
          command: { type: "string" },
        },
        required: ["command"],
      },
    },
  ],
};

describe("MCP runtime tool normalization", () => {
  test("converts fake MCP tool metadata into canonical optimizer tool specs", () => {
    const normalized = normalizeMcpServerTools(fakeServer, {
      canonicalToolVersion: "canonical-tools.test",
    });

    expect(normalized).toHaveLength(2);
    expect(normalized[0]!.serverId).toBe("github-enterprise");
    expect(normalized[0]!.canonicalSpec).toMatchObject({
      canonicalToolVersion: "canonical-tools.test",
      namespace: "mcp.github-enterprise",
      name: "repos_read_file",
      title: "Read File",
      description: "Read one file from a repository.",
      resultStyle: "json",
      sideEffectLevel: "read",
      requiresConfirmation: false,
    });
    expect(normalized[0]!.canonicalSpec.canonicalToolId).toMatch(/^tool\.mcp\.github-enterprise\.repos_read_file\.[a-f0-9]{12}$/);
    expect(normalized[0]!.canonicalSpec.inputSchema.properties).toHaveProperty("owner");
    expect(normalized[0]!.canonicalSpec.outputSchema?.properties).toHaveProperty("content");
    expect(normalized[0]!.canonicalSpec.examples[0]).toMatchObject({
      name: "read package",
      input: { owner: "acme", path: "package.json", repo: "app" },
      output: { content: "{...}" },
    });

    expect(normalized[1]!.canonicalSpec).toMatchObject({
      namespace: "mcp.github-enterprise",
      name: "issues_update",
      sideEffectLevel: "write",
      requiresConfirmation: true,
    });
  });

  test("classifies read, write, network, and process policy explicitly", () => {
    const read = classifyMcpToolPolicy({
      name: "repo/read",
      description: "Read repository metadata.",
      annotations: { readOnlyHint: true },
    });
    expect(read).toMatchObject({
      sideEffectLevel: "read",
      requiresConfirmation: false,
      safeAction: "allow",
      yoloAction: "allow",
      risks: {
        writesWorkspace: false,
        usesNetwork: false,
        runsProcess: false,
      },
    });

    const write = classifyMcpToolPolicy({
      name: "repo/delete_file",
      description: "Delete a file in the workspace.",
      annotations: { destructiveHint: true },
    });
    expect(write).toMatchObject({
      sideEffectLevel: "write",
      requiresConfirmation: true,
      safeAction: "confirm",
      yoloAction: "allow",
      risks: { writesWorkspace: true },
    });

    const network = classifyMcpToolPolicy({
      name: "web/fetch_url",
      description: "Fetch a URL over HTTP.",
      annotations: { openWorldHint: true },
    });
    expect(network).toMatchObject({
      sideEffectLevel: "network",
      requiresConfirmation: true,
      safeAction: "confirm",
      yoloAction: "allow",
      risks: { usesNetwork: true },
    });

    const process = classifyMcpToolPolicy({
      name: "terminal/exec",
      description: "Run a shell command.",
    });
    expect(process).toMatchObject({
      sideEffectLevel: "process",
      requiresConfirmation: true,
      safeAction: "confirm",
      yoloAction: "confirm",
      risks: { runsProcess: true },
    });
    expect(process.resultMaxBytes).toBeLessThan(read.resultMaxBytes);
  });

  test("prepares rendered contracts through the optimizer tool renderer", () => {
    const normalized = normalizeMcpServerTools(fakeServer);
    const policy = resolvedPolicy(nativeModelProfile);
    const prepared = prepareMcpRenderedToolContracts({
      normalizedTools: normalized,
      resolvedPolicy: policy,
    });
    const directRender = renderToolContracts({
      canonicalToolSpecs: normalized.map((tool) => tool.canonicalSpec),
      resolvedPolicy: policy,
    });

    expect(prepared.renderedContracts).toEqual(directRender);
    expect(prepared.renderedContracts.map((contract) => contract.name)).toEqual([
      "mcp.github-enterprise_issues_update",
      "mcp.github-enterprise_repos_read_file",
    ]);
    expect(prepared.policiesByCanonicalToolId[normalized[1]!.canonicalSpec.canonicalToolId]).toMatchObject({
      sideEffectLevel: "write",
      requiresConfirmation: true,
    });
    expect(prepared.resultBudgetsByCanonicalToolId[normalized[1]!.canonicalSpec.canonicalToolId]).toMatchObject({
      errorResultStyle: "structured_error",
      resultMaxBytes: 32768,
    });
    expect(prepared.resultBudgetsByCanonicalToolId[normalized[1]!.canonicalSpec.canonicalToolId]!.truncationMessage)
      .toContain("32768 bytes");

    const readRendered = prepared.renderedContracts.find((contract) => contract.name === "mcp.github-enterprise_repos_read_file")!;
    const readFacing = prepared.modelFacingContracts.find((contract) => contract.toolName === "repos/read_file")!;
    const readFacingId = readFacing.modelFacingToolId;
    const readFacingName = readFacing.modelFacingToolName;
    expect(readFacing).toMatchObject({
      kind: "mcp.model_facing_tool_contract",
      modelFacingToolId: expect.stringMatching(/^mcp\.model_facing_tool\.[a-f0-9]{12}$/),
      modelFacingToolName: expect.stringMatching(/^mcp_github-enterprise_repos_read_file_[a-f0-9]{12}$/),
      resultStyle: "json",
      resultContract: {
        resultStyle: "json",
        resultStyleVersion: "result-style.policy",
        resultMaxBytes: 65536,
        errorResultStyle: "structured_error",
      },
      lineage: {
        canonicalToolId: normalized[0]!.canonicalSpec.canonicalToolId,
        renderedToolId: readRendered.renderedToolId,
        renderedToolName: "mcp.github-enterprise_repos_read_file",
        modelFacingToolId: readFacingId,
        modelFacingToolName: readFacingName,
        renderedToolVersion: "rendered-tools.policy",
        modelProfileId: "model.native",
        policyId: "policy.test",
      },
      policy: {
        sideEffectLevel: "read",
        safeAction: "allow",
        yoloAction: "allow",
      },
    });
    expect(readFacingName.includes(".")).toBe(false);
    expect(readFacingName.length).toBeLessThanOrEqual(64);
    expect(readFacing.description).toContain("mcpResultTruncated=true");
    expect(readFacing.description).toContain("structured_error");
    expect(readFacing.promptFragments).toEqual(expect.arrayContaining([
      expect.stringContaining("Result contract for mcp.github-enterprise_repos_read_file"),
      expect.stringContaining("Structured error contract for mcp.github-enterprise_repos_read_file"),
    ]));
    expect(readFacing.resultContract.resultExamples).toEqual(expect.arrayContaining([
      expect.objectContaining({
        name: "truncated_result",
        value: expect.objectContaining({ mcpResultTruncated: true }),
      }),
      expect.objectContaining({
        name: "structured_error",
        resultStyle: "structured_error",
        value: expect.objectContaining({
          error: expect.objectContaining({ class: "invalid_arguments" }),
        }),
      }),
    ]));
    expect(prepared.modelFacingContractsByName[readFacingName]).toBe(readFacing);
    expect(prepared.modelFacingContractsById[readFacingId]).toBe(readFacing);
  });

  test("executes an approved read-only MCP tool through the runtime bridge", async () => {
    const normalized = normalizeMcpServerTools(fakeServer);
    const prepared = prepareMcpRenderedToolContracts({
      normalizedTools: normalized,
      resolvedPolicy: resolvedPolicy(nativeModelProfile),
    });
    const calls: unknown[] = [];
    const bridge = createMcpRuntimeToolBridge({
      normalizedTools: normalized,
      renderedContracts: prepared.renderedContracts,
      createCallId: () => "call.read.1",
      executor: (request) => {
        calls.push(request);
        return { content: `read:${request.arguments.path}` };
      },
    });

    const renderedRead = prepared.renderedContracts.find((contract) => contract.name === "mcp.github-enterprise_repos_read_file");
    const facingRead = prepared.modelFacingContracts.find((contract) => contract.toolName === "repos/read_file");
    expect(renderedRead).toBeDefined();
    expect(facingRead).toBeDefined();
    expect(bridge.callableTools[0]).toMatchObject({
      name: facingRead!.modelFacingToolName,
      modelFacingToolId: facingRead!.modelFacingToolId,
      renderedToolName: "mcp.github-enterprise_repos_read_file",
      resultContract: {
        resultMaxBytes: 65536,
        errorResultStyle: "structured_error",
      },
    });
    expect(bridge.callableTools[0]!.description).toContain("mcpResultTruncated=true");

    const result = await bridge.executeToolCall({
      modelFacingToolName: facingRead!.modelFacingToolName,
      arguments: { owner: "acme", repo: "app", path: "README.md" },
      retryCount: 2,
    });

    expect(result).toMatchObject({
      ok: true,
      status: "success",
      callId: "call.read.1",
      result: { content: "read:README.md" },
      policyDecision: {
        action: "allow",
        permissionStatus: "not_required",
        sideEffectLevel: "read",
      },
      trace: {
        event: "mcp.tool_call",
        spanName: "mcp.tool_call",
        status: "success",
        modelFacingToolId: facingRead!.modelFacingToolId,
        modelFacingToolName: facingRead!.modelFacingToolName,
        renderedToolId: renderedRead!.renderedToolId,
        renderedToolName: "mcp.github-enterprise_repos_read_file",
        renderedToolVersion: "rendered-tools.policy",
        modelProfileId: "model.native",
        policyId: "policy.test",
        argumentShapeHash: expect.stringMatching(/^sha256:[a-f0-9]{12}$/),
        redactionStatus: "hash_only",
        durationMs: expect.any(Number),
        retryCount: 2,
        followUpBehavior: "none",
      },
    });
    expect(result.call).toMatchObject({
      modelFacingToolId: facingRead!.modelFacingToolId,
      modelFacingToolName: facingRead!.modelFacingToolName,
      renderedToolId: renderedRead!.renderedToolId,
      renderedToolVersion: "rendered-tools.policy",
      modelProfileId: "model.native",
      policyId: "policy.test",
      resultStyleVersion: "result-style.policy",
    });
    expect(calls).toHaveLength(1);
    expect(calls[0]).toMatchObject({
      retryCount: 2,
      modelFacingToolId: facingRead!.modelFacingToolId,
      modelFacingToolName: facingRead!.modelFacingToolName,
      renderedToolId: renderedRead!.renderedToolId,
      renderedToolName: "mcp.github-enterprise_repos_read_file",
    });
    expect(result.metrics.argumentBytes).toBeGreaterThan(0);
    expect(result.metrics.truncated).toBe(false);
    const feedback = mcpRuntimeToolResultToOptimizerFeedback(result, {
      includeSuccessful: true,
      traceId: "trace.read.1",
      spanId: "span.read.1",
    });
    expect(feedback).toMatchObject({
      source: "mcp_runtime_tool_call",
      severity: "info",
      status: "success",
      retryCount: 2,
      argumentShapeHash: result.argumentShapeHash,
      redactionStatus: "hash_only",
      durationMs: result.metrics.durationMs,
      traceIds: ["trace.read.1"],
      spanIds: ["span.read.1"],
      canonicalToolId: normalized[0]!.canonicalSpec.canonicalToolId,
      modelFacingToolId: facingRead!.modelFacingToolId,
      modelFacingToolName: facingRead!.modelFacingToolName,
      renderedToolId: renderedRead!.renderedToolId,
      lineage: {
        canonicalToolIds: [normalized[0]!.canonicalSpec.canonicalToolId],
        modelFacingToolIds: [facingRead!.modelFacingToolId],
        modelFacingToolNames: [facingRead!.modelFacingToolName],
        renderedToolContractIds: [renderedRead!.renderedToolId],
        renderedToolVersions: ["rendered-tools.policy"],
        resultStyleVersions: ["result-style.policy"],
        modelProfileIds: ["model.native"],
        policyIds: ["policy.test"],
      },
      followUpBehavior: "none",
    });
  });

  test("rejects malformed and schema-invalid MCP arguments before execution", async () => {
    const normalized = normalizeMcpServerTools(fakeServer);
    let executed = false;
    const bridge = createMcpRuntimeToolBridge({
      normalizedTools: normalized,
      createCallId: () => "call.bad-args.1",
      executor: () => {
        executed = true;
        return {};
      },
    });

    const result = await bridge.executeToolCall({
      canonicalToolId: normalized[0]!.canonicalSpec.canonicalToolId,
      arguments: { owner: "acme", repo: "app" },
    });
    const resultArgumentShapeHash = result.argumentShapeHash;

    expect(executed).toBe(false);
    expect(result).toMatchObject({
      ok: false,
      status: "invalid_arguments",
      error: {
        class: "invalid_arguments",
        code: "schema_mismatch",
        details: {
          reason: "missing_required",
          fields: ["path"],
        },
      },
      failureCode: "schema_mismatch",
      argumentShapeHash: expect.stringMatching(/^sha256:[a-f0-9]{12}$/),
      redactionStatus: "hash_only",
      followUpBehavior: "repair_arguments",
      trace: {
        status: "invalid_arguments",
        policyAction: "allow",
        permissionStatus: "not_required",
        failureCode: "schema_mismatch",
        errorClass: "invalid_arguments",
        followUpBehavior: "repair_arguments",
      },
    });

    const malformed = await bridge.executeToolCall({
      canonicalToolId: normalized[0]!.canonicalSpec.canonicalToolId,
      arguments: "not-json-object",
    });

    expect(executed).toBe(false);
    expect(malformed).toMatchObject({
      ok: false,
      status: "invalid_arguments",
      error: {
        class: "invalid_arguments",
        code: "malformed_arguments",
        details: {
          reason: "not_object",
        },
      },
      failureCode: "malformed_arguments",
      followUpBehavior: "repair_arguments",
      trace: {
        failureCode: "malformed_arguments",
        redactionStatus: "hash_only",
      },
    });

    const feedback = mcpRuntimeToolResultToOptimizerFeedback(result, { maxFeedbackChars: 1_000 });
    expect(feedback).toMatchObject({
      severity: "warning",
      status: "invalid_arguments",
      failureCode: "schema_mismatch",
      errorClass: "invalid_arguments",
      policyAction: "allow",
      permissionStatus: "not_required",
      argumentShapeHash: resultArgumentShapeHash,
      redactionStatus: "hash_only",
      followUpBehavior: "repair_arguments",
    });
    expect(feedback?.feedback).toContain("missing_required");
    expect(feedback?.feedback).toContain("failureCode=schema_mismatch");
    expect(feedback?.feedback.length).toBeLessThanOrEqual(1_000);
  });

  test("separates missing MCP servers from unknown tools", async () => {
    const normalized = normalizeMcpServerTools(fakeServer);
    const bridge = createMcpRuntimeToolBridge({
      normalizedTools: normalized,
      createCallId: () => "call.lookup.1",
      executor: () => {
        throw new Error("unknown calls must not execute");
      },
    });

    const missingServer = await bridge.executeToolCall({
      serverId: "missing-server",
      toolName: "repos/read_file",
      arguments: { owner: "acme", repo: "app", path: "README.md" },
    });
    expect(missingServer).toMatchObject({
      ok: false,
      status: "missing_server",
      error: {
        class: "missing_server",
        code: "missing_server",
      },
      failureCode: "missing_server",
      followUpBehavior: "refresh_tool_inventory",
      trace: {
        status: "missing_server",
        failureCode: "missing_server",
        followUpBehavior: "refresh_tool_inventory",
      },
    });

    const unknownTool = await bridge.executeToolCall({
      serverId: normalized[0]!.serverId,
      toolName: "repos/missing_tool",
      arguments: { owner: "acme", repo: "app", path: "README.md" },
    });
    expect(unknownTool).toMatchObject({
      ok: false,
      status: "unknown_tool",
      error: {
        class: "unknown_tool",
        code: "unknown_tool",
      },
      failureCode: "unknown_tool",
      trace: {
        argumentShapeHash: expect.stringMatching(/^sha256:[a-f0-9]{12}$/),
        redactionStatus: "hash_only",
      },
    });
  });

  test("bounds oversized MCP execution results", async () => {
    const normalized = normalizeMcpServerTools(fakeServer);
    const bridge = createMcpRuntimeToolBridge({
      normalizedTools: normalized,
      createCallId: () => "call.large.1",
      executor: () => ({ content: "x".repeat(80 * 1024) }),
    });

    const result = await bridge.executeToolCall({
      canonicalToolId: normalized[0]!.canonicalSpec.canonicalToolId,
      arguments: { owner: "acme", repo: "app", path: "large.txt" },
    });

    expect(result.status).toBe("success");
    expect(result.metrics.truncated).toBe(true);
    expect(result.metrics.resultBytes).toBeLessThanOrEqual(normalized[0]!.policy.resultMaxBytes);
    expect(result.metrics.resultBytesBeforeBounding).toBeGreaterThan(normalized[0]!.policy.resultMaxBytes);
    expect(result.result).toMatchObject({
      mcpResultTruncated: true,
    });
    expect(result).toMatchObject({
      failureCode: "oversized_output",
      followUpBehavior: "narrow_request_or_paginate",
      trace: {
        resultTruncated: true,
        failureCode: "oversized_output",
        followUpBehavior: "narrow_request_or_paginate",
      },
    });
    const feedback = mcpRuntimeToolResultToOptimizerFeedback(result, { maxFeedbackChars: 800 });
    expect(feedback).toMatchObject({
      severity: "warning",
      status: "success",
      failureCode: "oversized_output",
      resultTruncated: true,
      omittedResultBytes: result.metrics.omittedResultBytes,
      followUpBehavior: "narrow_request_or_paginate",
    });
    expect(feedback?.feedback).toContain("truncated=true");
    expect(feedback?.feedback).toContain("failureCode=oversized_output");
    expect(feedback?.feedback.length).toBeLessThanOrEqual(800);
  });

  test("denies side-effecting MCP tools in safe mode without permission", async () => {
    const normalized = normalizeMcpServerTools(fakeServer);
    let executed = false;
    const bridge = createMcpRuntimeToolBridge({
      normalizedTools: normalized,
      createCallId: () => "call.denied.1",
      executor: () => {
        executed = true;
        return {};
      },
    });

    const result = await bridge.executeToolCall({
      canonicalToolId: normalized[1]!.canonicalSpec.canonicalToolId,
      arguments: { issueNumber: 7, title: "new title" },
    });

    expect(executed).toBe(false);
    expect(result).toMatchObject({
      ok: false,
      status: "denied",
      policyDecision: {
        mode: "safe",
        action: "confirm",
        permissionStatus: "denied",
        sideEffectLevel: "write",
      },
      error: {
        class: "permission_denied",
        code: "permission_denied",
      },
      failureCode: "permission_denied",
      followUpBehavior: "request_permission_or_choose_lower_risk_tool",
      trace: {
        policyAction: "confirm",
        permissionStatus: "denied",
        failureCode: "permission_denied",
        followUpBehavior: "request_permission_or_choose_lower_risk_tool",
      },
    });
    const feedback = mcpRuntimeToolResultToOptimizerFeedback(result);
    expect(feedback).toMatchObject({
      severity: "warning",
      status: "denied",
      failureCode: "permission_denied",
      errorClass: "permission_denied",
      policyAction: "confirm",
      permissionStatus: "denied",
      sideEffectLevel: "write",
      followUpBehavior: "request_permission_or_choose_lower_risk_tool",
    });
  });

  test("executes an approved write MCP tool through the permission gate", async () => {
    const normalized = normalizeMcpServerTools(fakeRiskServer);
    const writeTool = normalized.find((tool) => tool.toolName === "workspace/write_file")!;
    const permissionRequests: unknown[] = [];
    const executionRequests: unknown[] = [];
    const bridge = createMcpRuntimeToolBridge({
      normalizedTools: normalized,
      createCallId: () => "call.write.1",
      permissionHandler: (request) => {
        permissionRequests.push(request);
        return "allow";
      },
      executor: (request) => {
        executionRequests.push(request);
        return { written: request.arguments.path };
      },
    });

    const result = await bridge.executeToolCall({
      canonicalToolId: writeTool.canonicalSpec.canonicalToolId,
      arguments: { path: "CHANGELOG.md", content: "updated" },
    });

    expect(permissionRequests).toHaveLength(1);
    expect(permissionRequests[0]).toMatchObject({
      callId: "call.write.1",
      toolName: "workspace/write_file",
      mode: "safe",
      policy: {
        sideEffectLevel: "write",
        requiresConfirmation: true,
      },
      arguments: { path: "CHANGELOG.md", content: "updated" },
    });
    expect(executionRequests).toHaveLength(1);
    expect(executionRequests[0]).toMatchObject({
      callId: "call.write.1",
      toolName: "workspace/write_file",
      arguments: { path: "CHANGELOG.md", content: "updated" },
    });
    expect(result).toMatchObject({
      ok: true,
      status: "success",
      result: { written: "CHANGELOG.md" },
      policyDecision: {
        action: "confirm",
        permissionStatus: "granted",
        sideEffectLevel: "write",
      },
      trace: {
        policyAction: "confirm",
        permissionStatus: "granted",
        sideEffectLevel: "write",
      },
    });
  });

  test("denies network MCP tools in safe mode and allows them in yolo mode", async () => {
    const normalized = normalizeMcpServerTools(fakeRiskServer);
    const networkTool = normalized.find((tool) => tool.toolName === "web/fetch_url")!;
    let executions = 0;
    const safeBridge = createMcpRuntimeToolBridge({
      normalizedTools: normalized,
      createCallId: () => "call.network.safe.1",
      executor: () => {
        executions += 1;
        return {};
      },
    });

    const denied = await safeBridge.executeToolCall({
      canonicalToolId: networkTool.canonicalSpec.canonicalToolId,
      arguments: { url: "https://example.com/status.json" },
    });

    expect(executions).toBe(0);
    expect(denied).toMatchObject({
      ok: false,
      status: "denied",
      policyDecision: {
        mode: "safe",
        action: "confirm",
        permissionStatus: "denied",
        sideEffectLevel: "network",
      },
      error: { class: "permission_denied" },
      trace: {
        policyAction: "confirm",
        permissionStatus: "denied",
        sideEffectLevel: "network",
        failureCode: "permission_denied",
      },
    });

    const yoloBridge = createMcpRuntimeToolBridge({
      normalizedTools: normalized,
      mode: "yolo",
      createCallId: () => "call.network.yolo.1",
      executor: (request) => {
        executions += 1;
        return { fetched: request.arguments.url };
      },
    });

    const allowed = await yoloBridge.executeToolCall({
      canonicalToolId: networkTool.canonicalSpec.canonicalToolId,
      arguments: { url: "https://example.com/status.json" },
    });

    expect(executions).toBe(1);
    expect(allowed).toMatchObject({
      ok: true,
      status: "success",
      result: { fetched: "https://example.com/status.json" },
      policyDecision: {
        mode: "yolo",
        action: "allow",
        permissionStatus: "not_required",
        sideEffectLevel: "network",
      },
      trace: {
        policyAction: "allow",
        permissionStatus: "not_required",
        sideEffectLevel: "network",
      },
    });
  });

  test("keeps process MCP tools confirmation-bound in yolo mode", async () => {
    const normalized = normalizeMcpServerTools(fakeRiskServer);
    const processTool = normalized.find((tool) => tool.toolName === "terminal/exec")!;
    const deniedBridge = createMcpRuntimeToolBridge({
      normalizedTools: normalized,
      mode: "yolo",
      createCallId: () => "call.process.denied.1",
      executor: () => {
        throw new Error("process executor should not run without permission");
      },
    });

    const denied = await deniedBridge.executeToolCall({
      canonicalToolId: processTool.canonicalSpec.canonicalToolId,
      arguments: { command: "npm test" },
    });

    expect(denied).toMatchObject({
      ok: false,
      status: "denied",
      policyDecision: {
        mode: "yolo",
        action: "confirm",
        permissionStatus: "denied",
        sideEffectLevel: "process",
      },
      error: { class: "permission_denied", code: "permission_denied" },
    });

    const permissionRequests: unknown[] = [];
    const executionRequests: unknown[] = [];
    const approvedBridge = createMcpRuntimeToolBridge({
      normalizedTools: normalized,
      mode: "yolo",
      createCallId: () => "call.process.approved.1",
      permissionHandler: (request) => {
        permissionRequests.push(request);
        return "allow";
      },
      executor: (request) => {
        executionRequests.push(request);
        return { stdout: "ok" };
      },
    });

    const approved = await approvedBridge.executeToolCall({
      canonicalToolId: processTool.canonicalSpec.canonicalToolId,
      arguments: { command: "npm test" },
      retryCount: 3,
    });

    expect(permissionRequests).toHaveLength(1);
    expect(permissionRequests[0]).toMatchObject({
      callId: "call.process.approved.1",
      toolName: "terminal/exec",
      mode: "yolo",
      policy: {
        sideEffectLevel: "process",
        yoloAction: "confirm",
      },
    });
    expect(executionRequests).toHaveLength(1);
    expect(executionRequests[0]).toMatchObject({
      retryCount: 3,
      arguments: { command: "npm test" },
    });
    expect(approved).toMatchObject({
      ok: true,
      status: "success",
      result: { stdout: "ok" },
      policyDecision: {
        mode: "yolo",
        action: "confirm",
        permissionStatus: "granted",
        sideEffectLevel: "process",
      },
      metrics: {
        retryCount: 3,
      },
      trace: {
        retryCount: 3,
        policyAction: "confirm",
        permissionStatus: "granted",
        sideEffectLevel: "process",
      },
    });
  });

  test("preserves retry count on failed MCP executions and feedback", async () => {
    const normalized = normalizeMcpServerTools(fakeServer);
    const executionRequests: unknown[] = [];
    const bridge = createMcpRuntimeToolBridge({
      normalizedTools: normalized,
      createCallId: () => "call.retry.1",
      executor: (request) => {
        executionRequests.push(request);
        throw new Error("upstream MCP timeout after retry");
      },
    });

    const result = await bridge.executeToolCall({
      canonicalToolId: normalized[0]!.canonicalSpec.canonicalToolId,
      arguments: { owner: "acme", repo: "app", path: "README.md" },
      retryCount: 4,
    });

    expect(executionRequests).toHaveLength(1);
    expect(executionRequests[0]).toMatchObject({ retryCount: 4 });
    expect(result).toMatchObject({
      ok: false,
      status: "error",
      error: {
        class: "execution_error",
        code: "runtime_exception",
        message: "upstream MCP timeout after retry",
      },
      failureCode: "runtime_exception",
      metrics: {
        retryCount: 4,
      },
      trace: {
        status: "error",
        retryCount: 4,
        failureCode: "runtime_exception",
        errorClass: "execution_error",
      },
    });

    const feedback = mcpRuntimeToolResultToOptimizerFeedback(result);
    expect(feedback).toMatchObject({
      severity: "failure",
      status: "error",
      retryCount: 4,
      failureCode: "runtime_exception",
      errorClass: "execution_error",
      followUpBehavior: "inspect_executor_or_choose_alternate_tool",
    });
    expect(feedback?.feedback).toContain("retries=4");
  });

  test("returns retry exhaustion when execution fails on the final allowed attempt", async () => {
    const normalized = normalizeMcpServerTools(fakeServer);
    const executionRequests: unknown[] = [];
    const bridge = createMcpRuntimeToolBridge({
      normalizedTools: normalized,
      maxRetryCount: 2,
      createCallId: () => "call.retry-exhausted.1",
      executor: (request) => {
        executionRequests.push(request);
        throw new Error("upstream MCP failed again");
      },
    });

    const result = await bridge.executeToolCall({
      canonicalToolId: normalized[0]!.canonicalSpec.canonicalToolId,
      arguments: { owner: "acme", repo: "app", path: "README.md" },
      retryCount: 2,
    });

    expect(executionRequests).toHaveLength(1);
    expect(result).toMatchObject({
      ok: false,
      status: "retry_exhausted",
      failureCode: "retry_exhausted",
      error: {
        class: "retry_exhausted",
        code: "retry_exhausted",
      },
      metrics: {
        retryCount: 2,
      },
      trace: {
        status: "retry_exhausted",
        failureCode: "retry_exhausted",
        followUpBehavior: "retry_with_narrower_scope_or_abort",
      },
    });

    const feedback = mcpRuntimeToolResultToOptimizerFeedback(result);
    expect(feedback).toMatchObject({
      severity: "failure",
      status: "retry_exhausted",
      failureCode: "retry_exhausted",
      followUpBehavior: "retry_with_narrower_scope_or_abort",
    });
  });

  test("returns timeout when an MCP execution exceeds its runtime budget", async () => {
    const normalized = normalizeMcpServerTools(fakeServer);
    const executionRequests: unknown[] = [];
    const bridge = createMcpRuntimeToolBridge({
      normalizedTools: normalized,
      timeoutMs: 5,
      createCallId: () => "call.timeout.1",
      executor: (request) => {
        executionRequests.push(request);
        return new Promise((resolve) => {
          setTimeout(() => resolve({ content: "late" }), 30);
        });
      },
    });

    const result = await bridge.executeToolCall({
      canonicalToolId: normalized[0]!.canonicalSpec.canonicalToolId,
      arguments: { owner: "acme", repo: "app", path: "README.md" },
    });

    expect(executionRequests).toHaveLength(1);
    expect(executionRequests[0]).toMatchObject({
      timeoutMs: 5,
      signal: expect.any(AbortSignal),
    });
    expect(result).toMatchObject({
      ok: false,
      status: "timeout",
      failureCode: "timeout",
      error: {
        class: "timeout",
        code: "timeout",
      },
      trace: {
        status: "timeout",
        failureCode: "timeout",
        followUpBehavior: "retry_with_narrower_scope_or_abort",
      },
    });
  });

  test("returns cancelled when an MCP execution signal aborts", async () => {
    const normalized = normalizeMcpServerTools(fakeServer);
    const controller = new AbortController();
    const bridge = createMcpRuntimeToolBridge({
      normalizedTools: normalized,
      createCallId: () => "call.cancel.1",
      executor: (request) => new Promise((resolve, reject) => {
        request.signal?.addEventListener("abort", () => {
          reject(Object.assign(new Error("aborted by test"), { name: "AbortError" }));
        });
        setTimeout(() => resolve({ content: "late" }), 100);
      }),
    });

    const pending = bridge.executeToolCall({
      canonicalToolId: normalized[0]!.canonicalSpec.canonicalToolId,
      arguments: { owner: "acme", repo: "app", path: "README.md" },
      signal: controller.signal,
    });
    controller.abort();
    const result = await pending;

    expect(result).toMatchObject({
      ok: false,
      status: "cancelled",
      failureCode: "cancelled",
      error: {
        class: "cancelled",
        code: "cancelled",
      },
      trace: {
        status: "cancelled",
        failureCode: "cancelled",
        errorClass: "cancelled",
        followUpBehavior: "retry_with_narrower_scope_or_abort",
      },
    });
    const feedback = mcpRuntimeToolResultToOptimizerFeedback(result);
    expect(feedback).toMatchObject({
      severity: "warning",
      status: "cancelled",
      failureCode: "cancelled",
      errorClass: "cancelled",
      followUpBehavior: "retry_with_narrower_scope_or_abort",
    });
    expect(feedback?.feedback).toContain("cancellation");
  });
});

const controlledMcpFixturePath = (): string =>
  join(dirname(fileURLToPath(import.meta.url)), "fixtures", "controlled-mcp-server.mjs");

describe("MCP stdio transport proof", () => {
  test("discovers a controlled stdio MCP server and runs a bounded read-only tool with lineage and feedback", async () => {
    const transport = await connectMcpStdioRuntimeServer({
      serverId: "controlled-fixture",
      name: "controlled-fixture",
      displayName: "Controlled MCP fixture",
      command: "node",
      args: [controlledMcpFixturePath()],
      startupTimeoutMs: 2_000,
      requestTimeoutMs: 2_000,
    });
    try {
      expect({
        serverId: transport.server.serverId,
        name: transport.server.name,
      }).toEqual({
        serverId: "controlled-fixture",
        name: "controlled-fixture",
      });
      expect(transport.server.tools.find((tool) => tool.name === "read_note")).toMatchObject({
        name: "read_note",
        annotations: { readOnlyHint: true },
      });

      const normalized = normalizeMcpServerTools(transport.server);
      const prepared = prepareMcpRenderedToolContracts({
        normalizedTools: normalized,
        resolvedPolicy: resolvedPolicy(nativeModelProfile),
      });
      const readTool = normalized.find((tool) => tool.toolName === "read_note")!;
      const readContract = prepared.modelFacingContracts.find((contract) => contract.toolName === "read_note")!;
      const bridge = createMcpRuntimeToolBridge({
        normalizedTools: normalized,
        renderedContracts: prepared.renderedContracts,
        executor: transport.executor,
        timeoutMs: 500,
        maxRetryCount: 1,
      });

      const result = await bridge.executeToolCall({
        callId: "call.stdio.read.1",
        modelFacingToolName: readContract.modelFacingToolName,
        arguments: { id: "alpha" },
        retryCount: 1,
      });

      expect(result).toMatchObject({
        ok: true,
        status: "success",
        callId: "call.stdio.read.1",
        policyDecision: {
          action: "allow",
          permissionStatus: "not_required",
          sideEffectLevel: "read",
        },
        trace: {
          event: "mcp.tool_call",
          serverId: "controlled-fixture",
          serverName: "controlled-fixture",
          toolName: "read_note",
          canonicalToolId: readTool.canonicalSpec.canonicalToolId,
          modelFacingToolId: readContract.modelFacingToolId,
          modelFacingToolName: readContract.modelFacingToolName,
          renderedToolVersion: "rendered-tools.policy",
          modelProfileId: "model.native",
          policyId: "policy.test",
          retryCount: 1,
          resultTruncated: false,
          followUpBehavior: "none",
        },
      });
      expect(result.result).toMatchObject({
        structuredContent: {
          id: "alpha",
          note: "controlled fixture note",
        },
      });
      expect(result.metrics.durationMs).toBeGreaterThanOrEqual(0);
      expect(result.metrics.resultBytes).toBeGreaterThan(0);
      expect(result.metrics.resultBytes).toBeLessThanOrEqual(readTool.policy.resultMaxBytes);

      const feedback = mcpRuntimeToolResultToOptimizerFeedback(result, {
        includeSuccessful: true,
        traceId: "trace.stdio.read",
        spanId: "span.stdio.read",
      });
      expect(feedback).toMatchObject({
        severity: "info",
        status: "success",
        traceIds: ["trace.stdio.read"],
        spanIds: ["span.stdio.read"],
        canonicalToolId: readTool.canonicalSpec.canonicalToolId,
        modelFacingToolId: readContract.modelFacingToolId,
        modelFacingToolName: readContract.modelFacingToolName,
        serverId: "controlled-fixture",
        serverName: "controlled-fixture",
        toolName: "read_note",
        policyAction: "allow",
        permissionStatus: "not_required",
        sideEffectLevel: "read",
        lineage: {
          canonicalToolIds: [readTool.canonicalSpec.canonicalToolId],
          modelFacingToolIds: [readContract.modelFacingToolId],
          modelFacingToolNames: [readContract.modelFacingToolName],
          modelProfileIds: ["model.native"],
          policyIds: ["policy.test"],
        },
      });
      expect(feedback?.feedback).toContain("modelProfileId=model.native");
      expect(feedback?.feedback).toContain("policyId=policy.test");
    } finally {
      await transport.close();
    }
  });

  test("proves safe denial and yolo execution for a real side-effecting stdio MCP tool", async () => {
    const transport = await connectMcpStdioRuntimeServer({
      serverId: "controlled-fixture",
      name: "controlled-fixture",
      command: "node",
      args: [controlledMcpFixturePath()],
      startupTimeoutMs: 2_000,
      requestTimeoutMs: 2_000,
    });
    try {
      const normalized = normalizeMcpServerTools(transport.server);
      const writeTool = normalized.find((tool) => tool.toolName === "write_note")!;
      const safeBridge = createMcpRuntimeToolBridge({
        normalizedTools: normalized,
        executor: transport.executor,
      });

      const denied = await safeBridge.executeToolCall({
        callId: "call.stdio.write.denied",
        canonicalToolId: writeTool.canonicalSpec.canonicalToolId,
        arguments: { id: "alpha", note: "safe should deny this" },
      });

      expect(denied).toMatchObject({
        ok: false,
        status: "denied",
        failureCode: "permission_denied",
        policyDecision: {
          mode: "safe",
          action: "confirm",
          permissionStatus: "denied",
          sideEffectLevel: "write",
        },
        followUpBehavior: "request_permission_or_choose_lower_risk_tool",
      });

      const yoloBridge = createMcpRuntimeToolBridge({
        normalizedTools: normalized,
        mode: "yolo",
        executor: transport.executor,
      });
      const allowed = await yoloBridge.executeToolCall({
        callId: "call.stdio.write.yolo",
        canonicalToolId: writeTool.canonicalSpec.canonicalToolId,
        arguments: { id: "alpha", note: "updated through yolo" },
      });

      expect(allowed).toMatchObject({
        ok: true,
        status: "success",
        result: {
          structuredContent: {
            id: "alpha",
            written: true,
          },
        },
        policyDecision: {
          mode: "yolo",
          action: "allow",
          permissionStatus: "not_required",
          sideEffectLevel: "write",
        },
      });
    } finally {
      await transport.close();
    }
  });

  test("classifies real stdio MCP timeout, cancellation, retry exhaustion, and JSON-RPC failures", async () => {
    const transport = await connectMcpStdioRuntimeServer({
      serverId: "controlled-fixture",
      name: "controlled-fixture",
      command: "node",
      args: [controlledMcpFixturePath()],
      startupTimeoutMs: 2_000,
      requestTimeoutMs: 2_000,
    });
    try {
      const normalized = normalizeMcpServerTools(transport.server);
      const slowTool = normalized.find((tool) => tool.toolName === "slow_read")!;
      const failTool = normalized.find((tool) => tool.toolName === "fail_read")!;

      const timeoutBridge = createMcpRuntimeToolBridge({
        normalizedTools: normalized,
        executor: transport.executor,
        timeoutMs: 5,
      });
      const timeout = await timeoutBridge.executeToolCall({
        callId: "call.stdio.timeout",
        canonicalToolId: slowTool.canonicalSpec.canonicalToolId,
        arguments: { id: "alpha", delayMs: 50 },
      });
      expect(timeout).toMatchObject({
        ok: false,
        status: "timeout",
        failureCode: "timeout",
        error: { class: "timeout", code: "timeout" },
        followUpBehavior: "retry_with_narrower_scope_or_abort",
      });

      const controller = new AbortController();
      const cancellationBridge = createMcpRuntimeToolBridge({
        normalizedTools: normalized,
        executor: transport.executor,
      });
      const pending = cancellationBridge.executeToolCall({
        callId: "call.stdio.cancelled",
        canonicalToolId: slowTool.canonicalSpec.canonicalToolId,
        arguments: { id: "alpha", delayMs: 50 },
        signal: controller.signal,
      });
      controller.abort();
      const cancelled = await pending;
      expect(cancelled).toMatchObject({
        ok: false,
        status: "cancelled",
        failureCode: "cancelled",
        error: { class: "cancelled", code: "cancelled" },
        followUpBehavior: "retry_with_narrower_scope_or_abort",
      });

      const retryBridge = createMcpRuntimeToolBridge({
        normalizedTools: normalized,
        executor: transport.executor,
        maxRetryCount: 0,
      });
      const retryExhausted = await retryBridge.executeToolCall({
        callId: "call.stdio.retry-exhausted",
        canonicalToolId: slowTool.canonicalSpec.canonicalToolId,
        arguments: { id: "alpha", delayMs: 1 },
        retryCount: 1,
      });
      expect(retryExhausted).toMatchObject({
        ok: false,
        status: "retry_exhausted",
        failureCode: "retry_exhausted",
        followUpBehavior: "retry_with_narrower_scope_or_abort",
      });

      const failureBridge = createMcpRuntimeToolBridge({
        normalizedTools: normalized,
        executor: transport.executor,
      });
      const failure = await failureBridge.executeToolCall({
        callId: "call.stdio.failure",
        canonicalToolId: failTool.canonicalSpec.canonicalToolId,
        arguments: { id: "alpha" },
      });
      expect(failure).toMatchObject({
        ok: false,
        status: "error",
        failureCode: "runtime_exception",
        error: {
          class: "execution_error",
          code: "runtime_exception",
          message: "controlled fixture failure",
        },
        followUpBehavior: "inspect_executor_or_choose_alternate_tool",
      });
      const failureFeedback = mcpRuntimeToolResultToOptimizerFeedback(failure, {
        traceId: "trace.stdio.failure",
        spanId: "span.stdio.failure",
      });
      expect(failureFeedback).toMatchObject({
        source: "mcp_runtime_tool_call",
        severity: "failure",
        status: "error",
        failureCode: "runtime_exception",
        traceIds: ["trace.stdio.failure"],
        spanIds: ["span.stdio.failure"],
        serverId: "controlled-fixture",
        serverName: "controlled-fixture",
        toolName: "fail_read",
        followUpBehavior: "inspect_executor_or_choose_alternate_tool",
      });
      expect(failureFeedback?.feedback).toContain("controlled fixture failure");
    } finally {
      await transport.close();
    }
  });
});
