import { mkdtemp, mkdir, rm, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { tmpdir } from "node:os";
import { describe, expect, test } from "bun:test";
import {
  acpClientCapabilityProfileFromInitialize,
  acpConsumerCompatibilityMatrix,
  availableCommands,
  defaultAcpClientCapabilityProfile,
  maintenanceCommandHelp,
  modeState,
  promptToText,
  renderUserCapabilitySurface,
  sessionConfigOptions,
} from "../src/acp/surface";
import { readAcpSettingsSnippet, readAcpZedSettingsSnippet } from "../src/acp/settings";
import { defaultConfig } from "../src/config";
import {
  createGlassAcpConsumerReadiness,
  createStdioAcpConsumerReadiness,
  createZedAcpConsumerReadiness,
} from "../src/replay";

describe("ACP surface helpers", () => {
  test("keeps advertised commands focused on coding while maintenance remains hidden", () => {
    const names = availableCommands().map((command) => command.name);

    expect(names).toEqual(["run", "plan", "chat", "auto", "yolo", "safe", "skills", "mcp", "metrics", "traces"]);
    expect(names).not.toContain("maintenance");
    expect(maintenanceCommandHelp()).toContain("/maintenance status");
    expect(renderUserCapabilitySurface()).toContain("/run <task>");
    expect(renderUserCapabilitySurface()).not.toContain("/maintenance");
  });

  test("renders stable modes and session config defaults", () => {
    const config = defaultConfig();
    const modes = modeState("plan");
    const options = sessionConfigOptions(config, { executorConcurrency: 20, yolo: true });

    expect(modes.currentModeId).toBe("plan");
    expect(modes.availableModes.map((mode) => mode.id)).toEqual(["auto", "chat", "plan", "run"]);
    expect(options.map((option) => option.id)).toEqual(["executor-concurrency", "yolo"]);
    expect(options.find((option) => option.id === "executor-concurrency")?.currentValue).toBe("20");
    expect(options.find((option) => option.id === "yolo")?.currentValue).toBe(true);
  });

  test("normalizes prompt blocks and client capabilities without consumer-name branching", () => {
    const text = promptToText([
      { type: "text", text: "hello" },
      { type: "resource", resource: { uri: "file:///tmp/report.md", mimeType: "text/markdown", text: "report" } },
      { type: "resource_link", name: "docs", uri: "file:///tmp/docs" },
      { type: "image", mimeType: "image/png", data: "" },
    ] as never);

    expect(text).toContain("hello");
    expect(text).toContain("Embedded resource file:///tmp/report.md");
    expect(text).toContain("Resource link docs");
    expect(text).toContain("Image input (image/png)");
    expect(defaultAcpClientCapabilityProfile()).toMatchObject({
      fsReadTextFile: true,
      fsWriteTextFile: true,
      terminal: true,
      source: "not-initialized",
    });
    expect(acpClientCapabilityProfileFromInitialize({ fs: { readTextFile: true }, terminal: false }, "initialize:any"))
      .toEqual({
        fsReadTextFile: true,
        fsWriteTextFile: false,
        terminal: false,
        richDiffContent: false,
        richTerminalContent: false,
        source: "initialize:any",
      });
  });

  test("keeps compatibility matrix and settings snippets stable", () => {
    const matrix = acpConsumerCompatibilityMatrix();
    const generic = JSON.parse(readAcpSettingsSnippet()) as {
      acp_server: { command: string; args: string[]; cwd: string };
      named_examples: Record<string, { agent_servers: Record<string, { command: string; args: string[] }> }>;
    };
    const zed = JSON.parse(readAcpZedSettingsSnippet()) as {
      agent_servers: Record<string, { command: string; args: string[] }>;
    };

    expect(matrix.map((entry) => entry.id)).toEqual([
      "session-start",
      "greeting",
      "plan-report",
      "edit-run",
      "terminal-verification",
      "permissions",
      "slash-commands",
      "cancellation",
      "trace-artifacts",
    ]);
    expect(matrix.every((entry) => entry.namedConsumerFixtures.map((fixture) => fixture.consumer).join(",") === "Glass,Zed"))
      .toBe(true);
    expect(generic.acp_server).toMatchObject({ command: "bag", args: ["acp"] });
    expect(Object.keys(generic.named_examples).sort()).toEqual(["glass", "zed"]);
    expect(Object.values(generic.named_examples).map((example) => Object.values(example.agent_servers)[0]))
      .toEqual([
        { command: "bag", args: ["acp"] },
        { command: "bag", args: ["acp"] },
      ]);
    expect(Object.values(zed.agent_servers)[0]).toEqual({ command: "bag", args: ["acp"] });
  });

  test("resolves named real ACP consumer metadata without making apps mandatory for core surface", async () => {
    const root = await mkdtemp(join(tmpdir(), "acp-consumer-adapter-test-"));
    try {
      const settingsPath = join(root, "zed-settings.json");
      const appPath = join(root, "Zed.app");
      await mkdir(appPath, { recursive: true });
      await writeFile(settingsPath, `{
        // JSONC is accepted because Zed settings are usually edited by hand.
        "agent_servers": {
          "bleeding-agent": {
            "command": "bag",
            "args": ["acp"],
          },
        },
      }\n`, "utf8");

      const zed = createZedAcpConsumerReadiness({
        settingsPath,
        serverKey: "bleeding-agent",
        appPath,
      });
      expect(zed).toMatchObject({
        providerId: "real-acp.consumer.zed",
        consumerName: "Zed",
        status: "ready",
        launch: {
          command: "bag",
          args: ["acp"],
        },
        app: {
          installed: true,
        },
        clientMetadata: {
          transport: "stdio",
          acpConsumerCapabilities: {
            desktopUiParity: false,
          },
        },
      });

      const glass = createGlassAcpConsumerReadiness({
        serverKey: "bleeding-agent",
        appPath: join(root, "missing-glass.app"),
      });
      expect(glass.status).toBe("blocked");
      expect(glass.blockers.join("\n")).toContain("Glass app not found");
      expect(glass.blockers.join("\n")).toContain("Glass ACP settings path is not configured");

      const stdio = createStdioAcpConsumerReadiness({
        command: "bag",
        args: ["acp"],
      });
      expect(stdio.status).toBe("ready");
      expect(stdio.clientMetadata.acpConsumerCapabilities.desktopUiParity).toBe(false);
    } finally {
      await rm(root, { recursive: true, force: true });
    }
  });
});
