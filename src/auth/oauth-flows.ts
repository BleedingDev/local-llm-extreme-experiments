/**
 * OAuth credential storage and refresh for paid subscription endpoints
 * (Anthropic Pro/Max, ChatGPT Plus/Pro Codex, GitHub Copilot).
 *
 * The login *flows* live in scripts/bag_oauth_login.ts to keep this module a
 * pure runtime helper that the LlmRouter can call without dragging in
 * node:http or browser-launching side effects. Both files are derived from
 * pi-mono (https://github.com/badlogic/pi-mono, MIT, Copyright (c) 2025
 * Mario Zechner) — see scripts/bag_oauth_login.ts for license text and
 * upstream provenance.
 *
 * Token storage shape (matches pi-mono so users can drop in existing tokens):
 *
 *   {
 *     "access":   "<access token>",
 *     "refresh":  "<refresh token>",
 *     "expires":  1735689600000,    // ms-since-epoch absolute deadline
 *     // GitHub Copilot only: enterprise domain ("company.ghe.com") to use on refresh
 *     "enterpriseUrl": "...",
 *     // OpenAI Codex only: chatgpt_account_id from the JWT, required by chatgpt-backend
 *     "accountId": "..."
 *   }
 *
 * Storage location: ~/.bag/oauth/<provider>.json, chmod 600.
 */

import { chmodSync, existsSync, mkdirSync, readFileSync, statSync, writeFileSync } from "node:fs";
import { homedir } from "node:os";
import { dirname, join } from "node:path";

export type OAuthProviderId = "anthropic" | "openai" | "github-copilot";

/** All built-in providers — exported so callers can iterate (login UX, doctor, etc.). */
export const OAUTH_PROVIDER_IDS: readonly OAuthProviderId[] = [
  "anthropic",
  "openai",
  "github-copilot",
] as const;

export type OAuthCredentials = {
  /** Short-lived bearer token actually sent to the API. */
  access: string;
  /** Long-lived token used to mint a new `access` once it expires. */
  refresh: string;
  /** Absolute expiry deadline (ms since epoch). When `Date.now() >= expires`, refresh. */
  expires: number;
  /** GitHub Copilot enterprise domain ("company.ghe.com"); undefined for github.com. */
  enterpriseUrl?: string;
  /** OpenAI Codex chatgpt_account_id (extracted from JWT) — required by `chatgpt-account-id` header. */
  accountId?: string;
  /** Free-form passthrough so future providers can persist extra fields without a schema bump. */
  [key: string]: unknown;
};

export type OAuthHeaderResult = {
  /** Headers to merge into the outbound LLM request (Authorization, x-api-key, anthropic-beta, etc.). */
  headers: Record<string, string>;
  /** Optional baseUrl override — Copilot proxies vary per token, so callers should honor this. */
  baseUrlOverride?: string;
  /** Resolved credentials after any refresh. Useful for callers that want to display "expires in N min". */
  credentials: OAuthCredentials;
};

export type OAuthFileSystem = {
  exists: (path: string) => boolean;
  read: (path: string) => string;
  write: (path: string, contents: string) => void;
  mkdir: (path: string) => void;
  chmod600: (path: string) => void;
  mode: (path: string) => number;
};

const defaultFs: OAuthFileSystem = {
  exists: (path) => existsSync(path),
  read: (path) => readFileSync(path, "utf8"),
  write: (path, contents) => writeFileSync(path, contents, "utf8"),
  mkdir: (path) => mkdirSync(path, { recursive: true, mode: 0o700 }),
  chmod600: (path) => {
    try {
      chmodSync(path, 0o600);
    } catch {
      // chmod is best-effort: filesystems like exFAT, fat32, or Windows volumes
      // may not honour POSIX modes. We still wrote the file in the user's own
      // home dir (~/.bag/oauth/), so this is not catastrophic.
    }
  },
  mode: (path) => statSync(path).mode,
};

/** Root directory under the user's home. Override via `BAG_OAUTH_DIR` for tests. */
export const oauthDir = (envOverride?: string): string => {
  const fromEnv = envOverride ?? process.env.BAG_OAUTH_DIR;
  if (fromEnv != null && fromEnv !== "") return fromEnv;
  return join(homedir(), ".bag", "oauth");
};

/** Absolute path to the credentials file for a given provider. */
export const oauthFilePath = (provider: OAuthProviderId, baseDir?: string): string =>
  join(baseDir ?? oauthDir(), `${provider}.json`);

const isOAuthCredentials = (value: unknown): value is OAuthCredentials => {
  if (value == null || typeof value !== "object") return false;
  const candidate = value as Record<string, unknown>;
  return (
    typeof candidate.access === "string" &&
    typeof candidate.refresh === "string" &&
    typeof candidate.expires === "number"
  );
};

export type LoadOAuthOptions = {
  fs?: OAuthFileSystem;
  baseDir?: string;
};

export const loadOAuthCredentials = (
  provider: OAuthProviderId,
  options: LoadOAuthOptions = {},
): OAuthCredentials | undefined => {
  const fs = options.fs ?? defaultFs;
  const path = oauthFilePath(provider, options.baseDir);
  if (!fs.exists(path)) return undefined;
  let parsed: unknown;
  try {
    parsed = JSON.parse(fs.read(path));
  } catch {
    return undefined;
  }
  if (!isOAuthCredentials(parsed)) return undefined;
  return parsed;
};

export const saveOAuthCredentials = (
  provider: OAuthProviderId,
  credentials: OAuthCredentials,
  options: LoadOAuthOptions = {},
): void => {
  const fs = options.fs ?? defaultFs;
  const path = oauthFilePath(provider, options.baseDir);
  fs.mkdir(dirname(path));
  fs.write(path, `${JSON.stringify(credentials, null, 2)}\n`);
  // chmod 600 — only the owner can read the refresh token. Best-effort because
  // some filesystems do not support POSIX modes (see defaultFs.chmod600).
  fs.chmod600(path);
};

/** True if the credentials are within `marginMs` of expiry (default 60s) — i.e. should refresh now. */
export const isExpired = (credentials: OAuthCredentials, marginMs = 60_000): boolean =>
  Date.now() >= credentials.expires - marginMs;

// ----- Refresh transport (HTTP) ---------------------------------------------

export type OAuthHttp = (url: string, init: { method: string; headers: Record<string, string>; body: string | URLSearchParams }) => Promise<{
  ok: boolean;
  status: number;
  text: () => Promise<string>;
  json: () => Promise<unknown>;
}>;

const defaultHttp: OAuthHttp = async (url, init) => {
  const response = await fetch(url, {
    method: init.method,
    headers: init.headers,
    body: init.body instanceof URLSearchParams ? init.body : init.body,
  });
  return {
    ok: response.ok,
    status: response.status,
    text: () => response.text(),
    json: () => response.json() as Promise<unknown>,
  };
};

const ANTHROPIC_TOKEN_URL = "https://platform.claude.com/v1/oauth/token";
// Same client_id pi-mono uses (decoded from its base64 obfuscation upstream).
// Source: https://github.com/badlogic/pi-mono/blob/main/packages/ai/src/utils/oauth/anthropic.ts
const ANTHROPIC_CLIENT_ID = "9d1c250a-e61b-44d9-88ed-5944d1962f5e";

const OPENAI_TOKEN_URL = "https://auth.openai.com/oauth/token";
const OPENAI_CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann";
const OPENAI_JWT_CLAIM_PATH = "https://api.openai.com/auth";

// GitHub Copilot is a token-exchange (Bearer the long-lived github oauth token,
// receive a short-lived copilot token good for ~25min), not a refresh-grant.
const GITHUB_COPILOT_HEADERS: Record<string, string> = {
  "User-Agent": "GitHubCopilotChat/0.35.0",
  "Editor-Version": "vscode/1.107.0",
  "Editor-Plugin-Version": "copilot-chat/0.35.0",
  "Copilot-Integration-Id": "vscode-chat",
};

const refreshAnthropic = async (
  credentials: OAuthCredentials,
  http: OAuthHttp,
): Promise<OAuthCredentials> => {
  const response = await http(ANTHROPIC_TOKEN_URL, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Accept: "application/json",
    },
    body: JSON.stringify({
      grant_type: "refresh_token",
      client_id: ANTHROPIC_CLIENT_ID,
      refresh_token: credentials.refresh,
    }),
  });
  if (!response.ok) {
    const detail = await response.text();
    throw new Error(`Anthropic token refresh failed (${response.status}): ${detail.slice(0, 300)}`);
  }
  const payload = (await response.json()) as {
    access_token?: string;
    refresh_token?: string;
    expires_in?: number;
  };
  if (
    typeof payload.access_token !== "string" ||
    typeof payload.refresh_token !== "string" ||
    typeof payload.expires_in !== "number"
  ) {
    throw new Error("Anthropic token refresh response missing required fields");
  }
  return {
    ...credentials,
    access: payload.access_token,
    refresh: payload.refresh_token,
    // 5-minute clock-skew margin (matches pi-mono).
    expires: Date.now() + payload.expires_in * 1000 - 5 * 60 * 1000,
  };
};

const decodeJwtAccount = (token: string): string | undefined => {
  const parts = token.split(".");
  if (parts.length !== 3) return undefined;
  try {
    const payload = JSON.parse(
      Buffer.from(parts[1] ?? "", "base64").toString("utf8"),
    ) as Record<string, unknown>;
    const claim = payload[OPENAI_JWT_CLAIM_PATH];
    if (claim != null && typeof claim === "object") {
      const accountId = (claim as Record<string, unknown>).chatgpt_account_id;
      if (typeof accountId === "string" && accountId.length > 0) return accountId;
    }
  } catch {
    return undefined;
  }
  return undefined;
};

const refreshOpenAI = async (
  credentials: OAuthCredentials,
  http: OAuthHttp,
): Promise<OAuthCredentials> => {
  const body = new URLSearchParams({
    grant_type: "refresh_token",
    refresh_token: credentials.refresh,
    client_id: OPENAI_CLIENT_ID,
  });
  const response = await http(OPENAI_TOKEN_URL, {
    method: "POST",
    headers: { "Content-Type": "application/x-www-form-urlencoded" },
    body,
  });
  if (!response.ok) {
    const detail = await response.text();
    throw new Error(`OpenAI Codex token refresh failed (${response.status}): ${detail.slice(0, 300)}`);
  }
  const payload = (await response.json()) as {
    access_token?: string;
    refresh_token?: string;
    expires_in?: number;
  };
  if (
    typeof payload.access_token !== "string" ||
    typeof payload.refresh_token !== "string" ||
    typeof payload.expires_in !== "number"
  ) {
    throw new Error("OpenAI Codex token refresh response missing required fields");
  }
  const accountId = decodeJwtAccount(payload.access_token) ?? credentials.accountId;
  return {
    ...credentials,
    access: payload.access_token,
    refresh: payload.refresh_token,
    expires: Date.now() + payload.expires_in * 1000,
    ...(accountId == null ? {} : { accountId }),
  };
};

const githubCopilotDomain = (credentials: OAuthCredentials): string =>
  typeof credentials.enterpriseUrl === "string" && credentials.enterpriseUrl !== ""
    ? credentials.enterpriseUrl
    : "github.com";

const refreshGitHubCopilot = async (
  credentials: OAuthCredentials,
  http: OAuthHttp,
): Promise<OAuthCredentials> => {
  const domain = githubCopilotDomain(credentials);
  const url = `https://api.${domain}/copilot_internal/v2/token`;
  const response = await http(url, {
    method: "GET",
    headers: {
      Accept: "application/json",
      Authorization: `Bearer ${credentials.refresh}`,
      ...GITHUB_COPILOT_HEADERS,
    },
    // Even though it's GET semantically, fetch tolerates an empty body string.
    body: "",
  });
  if (!response.ok) {
    const detail = await response.text();
    throw new Error(`GitHub Copilot token refresh failed (${response.status}): ${detail.slice(0, 300)}`);
  }
  const payload = (await response.json()) as { token?: string; expires_at?: number };
  if (typeof payload.token !== "string" || typeof payload.expires_at !== "number") {
    throw new Error("GitHub Copilot token response missing required fields");
  }
  return {
    ...credentials,
    access: payload.token,
    refresh: credentials.refresh,
    // 5-minute skew margin to match pi-mono.
    expires: payload.expires_at * 1000 - 5 * 60 * 1000,
  };
};

const refreshDispatch: Record<
  OAuthProviderId,
  (creds: OAuthCredentials, http: OAuthHttp) => Promise<OAuthCredentials>
> = {
  anthropic: refreshAnthropic,
  openai: refreshOpenAI,
  "github-copilot": refreshGitHubCopilot,
};

export const refreshOAuthCredentials = async (
  provider: OAuthProviderId,
  credentials: OAuthCredentials,
  http: OAuthHttp = defaultHttp,
): Promise<OAuthCredentials> => refreshDispatch[provider](credentials, http);

// ----- Header derivation per provider ---------------------------------------

const githubCopilotApiBaseUrl = (token: string, enterpriseUrl?: string): string => {
  // Token format: tid=...;exp=...;proxy-ep=proxy.individual.githubcopilot.com;...
  // Convert proxy.<host> -> api.<host>. (Logic ported from pi-mono.)
  const match = token.match(/proxy-ep=([^;]+)/);
  if (match != null && match[1] != null) {
    return `https://${match[1].replace(/^proxy\./, "api.")}`;
  }
  if (enterpriseUrl != null && enterpriseUrl !== "") return `https://copilot-api.${enterpriseUrl}`;
  return "https://api.individual.githubcopilot.com";
};

const headersForAnthropic = (credentials: OAuthCredentials): OAuthHeaderResult => ({
  // Anthropic OAuth tokens go in `Authorization: Bearer ...` AND require the
  // `anthropic-beta: oauth-2025-04-20` header — without the beta flag the
  // Pro/Max tier will reject the bearer token. (See pi-mono provider table.)
  headers: {
    Authorization: `Bearer ${credentials.access}`,
    "anthropic-beta": "oauth-2025-04-20",
    "anthropic-version": "2023-06-01",
  },
  credentials,
});

const headersForOpenAI = (credentials: OAuthCredentials): OAuthHeaderResult => {
  const accountId = credentials.accountId ?? decodeJwtAccount(credentials.access);
  return {
    headers: {
      Authorization: `Bearer ${credentials.access}`,
      // chatgpt-backend rejects requests without the account id once you're on
      // an OAuth (subscription) token. Only API-key flows can omit it.
      ...(accountId == null ? {} : { "chatgpt-account-id": accountId, "OpenAI-Beta": "responses=experimental" }),
    },
    credentials: accountId == null ? credentials : { ...credentials, accountId },
  };
};

const headersForGitHubCopilot = (credentials: OAuthCredentials): OAuthHeaderResult => ({
  headers: {
    Authorization: `Bearer ${credentials.access}`,
    ...GITHUB_COPILOT_HEADERS,
  },
  baseUrlOverride: githubCopilotApiBaseUrl(
    credentials.access,
    typeof credentials.enterpriseUrl === "string" ? credentials.enterpriseUrl : undefined,
  ),
  credentials,
});

const headersDispatch: Record<OAuthProviderId, (creds: OAuthCredentials) => OAuthHeaderResult> = {
  anthropic: headersForAnthropic,
  openai: headersForOpenAI,
  "github-copilot": headersForGitHubCopilot,
};

export type GetOAuthHeadersOptions = {
  fs?: OAuthFileSystem;
  baseDir?: string;
  http?: OAuthHttp;
  /** Override Date.now (used in tests). */
  now?: () => number;
  /** Refresh margin in ms (default 60s). */
  refreshMarginMs?: number;
};

/**
 * The single entry point the LlmRouter calls. Resolves OAuth credentials
 * from `~/.bag/oauth/<provider>.json`, refreshes them if expired, persists the
 * refreshed token (chmod 600), and returns the headers (and optional baseUrl
 * override for Copilot's per-token proxy).
 *
 * Throws if the file is missing — callers should treat that as "user has not
 * run `bag-oauth-login <provider>` yet".
 */
export const getOAuthHeaders = async (
  provider: OAuthProviderId,
  options: GetOAuthHeadersOptions = {},
): Promise<OAuthHeaderResult> => {
  const fs = options.fs ?? defaultFs;
  const http = options.http ?? defaultHttp;
  const now = options.now ?? (() => Date.now());
  const margin = options.refreshMarginMs ?? 60_000;

  const stored = loadOAuthCredentials(provider, { fs, ...(options.baseDir == null ? {} : { baseDir: options.baseDir }) });
  if (stored == null) {
    throw new Error(
      `No OAuth credentials found for "${provider}". Run \`bag-oauth-login ${provider}\` to authenticate.`,
    );
  }

  let credentials = stored;
  if (now() >= credentials.expires - margin) {
    credentials = await refreshOAuthCredentials(provider, credentials, http);
    saveOAuthCredentials(provider, credentials, { fs, ...(options.baseDir == null ? {} : { baseDir: options.baseDir }) });
  }

  return headersDispatch[provider](credentials);
};

// ----- Test seam -----------------------------------------------------------

/** Exported for unit tests; not part of the stable public API. */
export const __testing = {
  defaultFs,
  decodeJwtAccount,
  githubCopilotApiBaseUrl,
  isExpired,
  ANTHROPIC_CLIENT_ID,
  OPENAI_CLIENT_ID,
  ANTHROPIC_TOKEN_URL,
  OPENAI_TOKEN_URL,
  OPENAI_JWT_CLAIM_PATH,
};
