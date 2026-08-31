import { existsSync, readFileSync } from "node:fs";
import { resolve } from "node:path";

export type AcpSettingsConsumer = "generic" | "zed";

const agentServersSettings = (packageName: string, command: string, args: string[]) => ({
  agent_servers: {
    [packageName]: {
      command,
      args,
    },
  },
});

export const acpServerLaunchConfig = (cwd = process.cwd()) => {
  const packageJsonPath = resolve(cwd, "package.json");
  const packageName = existsSync(packageJsonPath)
    ? (JSON.parse(readFileSync(packageJsonPath, "utf8")) as { name?: string }).name ?? "bleeding-agent"
    : "bleeding-agent";
  return {
    packageName,
    command: "bag",
    args: ["acp"],
    cwd: resolve(cwd),
  };
};

export const readAcpSettingsSnippet = (cwd = process.cwd(), consumer: AcpSettingsConsumer = "generic"): string => {
  const launch = acpServerLaunchConfig(cwd);
  if (consumer === "zed") {
    return JSON.stringify(agentServersSettings(launch.packageName, launch.command, launch.args), null, 2);
  }
  const namedAgentServers = agentServersSettings(launch.packageName, launch.command, launch.args);
  return JSON.stringify(
    {
      acp_server: {
        command: launch.command,
        args: launch.args,
        cwd: launch.cwd,
      },
      named_examples: {
        glass: namedAgentServers,
        zed: namedAgentServers,
      },
    },
    null,
    2,
  );
};

export const readAcpZedSettingsSnippet = (cwd = process.cwd()): string => readAcpSettingsSnippet(cwd, "zed");
