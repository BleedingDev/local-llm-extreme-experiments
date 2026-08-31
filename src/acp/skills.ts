import { existsSync, readFileSync, readdirSync } from "node:fs";
import { join, resolve } from "node:path";
import type { SkillSummary } from "./slash-router";

export const discoverAcpSkills = (home = process.env.HOME ?? ""): SkillSummary[] => {
  const roots = [
    resolve(home, ".codex", "skills"),
    resolve(home, ".agents", "skills"),
    resolve(home, "side", "experiments", "skills"),
  ];
  return roots.flatMap((root) => {
    if (!existsSync(root)) {
      return [];
    }
    return readdirSync(root, { withFileTypes: true }).flatMap((entry) => {
      if (!entry.isDirectory()) {
        return [];
      }
      const path = join(root, entry.name, "SKILL.md");
      if (!existsSync(path)) {
        return [];
      }
      const content = readFileSync(path, "utf8");
      const description =
        content.match(/^description:\s*(.+)$/m)?.[1]?.trim() ??
        content
          .split("\n")
          .find((line) => line.trim().length > 0 && !line.startsWith("---") && !line.startsWith("#"))
          ?.trim() ??
        "No description.";
      return [{ name: entry.name, description, path }];
    });
  });
};
