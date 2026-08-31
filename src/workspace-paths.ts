import { existsSync, realpathSync, statSync } from "node:fs";
import { dirname, isAbsolute, relative, resolve } from "node:path";

export class WorkspacePathError extends Error {
  readonly code = "workspace_path_escape";

  constructor(message: string) {
    super(message);
    this.name = "WorkspacePathError";
  }
}

export type ResolveSessionPathInput = {
  cwd: string;
  additionalDirectories?: readonly string[];
  path: string;
  kind?: "file" | "directory" | "any";
};

const isInsideOrSame = (root: string, path: string): boolean => {
  const rel = relative(root, path);
  return rel === "" || (!rel.startsWith("..") && !isAbsolute(rel));
};

const uniqueResolvedRoots = (cwd: string, additionalDirectories: readonly string[] = []): string[] => {
  const roots = [cwd, ...additionalDirectories].map((root) => resolve(root));
  return [...new Set(roots)];
};

const realpathIfExists = (path: string): string | undefined => {
  try {
    return existsSync(path) ? realpathSync.native(path) : undefined;
  } catch {
    return undefined;
  }
};

const nearestExistingPath = (path: string): string | undefined => {
  let current = path;
  for (;;) {
    if (existsSync(current)) {
      return current;
    }
    const next = dirname(current);
    if (next === current) {
      return undefined;
    }
    current = next;
  }
};

export const sessionRootPaths = (cwd: string, additionalDirectories: readonly string[] = []): string[] =>
  uniqueResolvedRoots(cwd, additionalDirectories);

export const sessionRelativePath = (
  cwd: string,
  additionalDirectories: readonly string[],
  path: string,
): string => {
  const absolutePath = resolveSessionPath({ cwd, additionalDirectories, path });
  const primaryRoot = resolve(cwd);
  const relativePath = relative(primaryRoot, absolutePath).replaceAll("\\", "/");
  return relativePath === "" || relativePath.startsWith("../") || relativePath === ".." ? absolutePath : relativePath;
};

export const resolveSessionPath = (input: ResolveSessionPathInput): string => {
  const kind = input.kind ?? "file";
  if (input.path.includes("\0")) {
    throw new WorkspacePathError("workspace path contains a null byte");
  }

  const lexicalPath = isAbsolute(input.path) ? resolve(input.path) : resolve(input.cwd, input.path);
  const lexicalRoots = sessionRootPaths(input.cwd, input.additionalDirectories);
  const lexicalRoot = lexicalRoots.find((root) => isInsideOrSame(root, lexicalPath));
  if (lexicalRoot === undefined) {
    throw new WorkspacePathError(`workspace path escapes allowed roots: ${input.path}`);
  }

  const existingTarget = realpathIfExists(lexicalPath);
  const realRoots = lexicalRoots.map((root) => realpathIfExists(root) ?? root);
  if (existingTarget !== undefined) {
    const allowed = realRoots.some((root) => isInsideOrSame(root, existingTarget));
    if (!allowed) {
      throw new WorkspacePathError(`workspace path resolves outside allowed roots: ${input.path}`);
    }
    const stats = statSync(existingTarget);
    if (kind === "file" && stats.isDirectory()) {
      throw new WorkspacePathError(`workspace path points to a directory: ${input.path}`);
    }
    if (kind === "directory" && !stats.isDirectory()) {
      throw new WorkspacePathError(`workspace path does not point to a directory: ${input.path}`);
    }
    return lexicalPath;
  }

  if (kind === "directory") {
    throw new WorkspacePathError(`workspace directory does not exist: ${input.path}`);
  }

  const nearest = nearestExistingPath(lexicalPath);
  const nearestReal = nearest == null ? undefined : realpathIfExists(nearest);
  if (nearestReal != null && !realRoots.some((root) => isInsideOrSame(root, nearestReal))) {
    throw new WorkspacePathError(`workspace path parent resolves outside allowed roots: ${input.path}`);
  }

  return lexicalPath;
};
