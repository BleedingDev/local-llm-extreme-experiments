#!/usr/bin/env tsx
/**
 * bag-oauth-login — interactive OAuth login for paid LLM subscriptions.
 *
 * Usage:
 *   tsx scripts/bag_oauth_login.ts <provider>
 *     where <provider> is one of: anthropic | openai | github-copilot
 *
 *   tsx scripts/bag_oauth_login.ts --list
 *   tsx scripts/bag_oauth_login.ts --status
 *
 * After a successful login the credentials are written to
 * ~/.bag/oauth/<provider>.json (chmod 600). The LlmRouter will pick them up
 * automatically once `bag.config.json` has `master.authType: "oauth"`.
 *
 * ----------------------------------------------------------------------------
 *
 * Portions of this file are derived from pi-mono
 *   (https://github.com/badlogic/pi-mono, packages/ai/src/utils/oauth/*)
 *   Copyright (c) 2025 Mario Zechner. Licensed under the MIT License.
 *
 *   Permission is hereby granted, free of charge, to any person obtaining a
 *   copy of this software and associated documentation files (the "Software"),
 *   to deal in the Software without restriction, including without limitation
 *   the rights to use, copy, modify, merge, publish, distribute, sublicense,
 *   and/or sell copies of the Software, and to permit persons to whom the
 *   Software is furnished to do so, subject to the following conditions:
 *
 *   The above copyright notice and this permission notice shall be included in
 *   all copies or substantial portions of the Software.
 *
 *   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND...
 *
 * Specific provenance:
 *   - PKCE generation: pi-mono/.../oauth/pkce.ts
 *   - Anthropic flow: pi-mono/.../oauth/anthropic.ts (CLIENT_ID, AUTHORIZE_URL,
 *     TOKEN_URL, SCOPES, callback port 53692, redirect path /callback).
 *   - OpenAI Codex flow: pi-mono/.../oauth/openai-codex.ts (CLIENT_ID,
 *     AUTHORIZE_URL, TOKEN_URL, callback port 1455, JWT account-id parsing).
 *   - GitHub Copilot device flow: pi-mono/.../oauth/github-copilot.ts (CLIENT_ID,
 *     /login/device/code, /login/oauth/access_token, copilot_internal/v2/token).
 *
 * BAG additions:
 *   - Persistence layer (~/.bag/oauth/<provider>.json with chmod 600).
 *   - Single-binary CLI shape (subcommand-per-provider) instead of pi's
 *     OAuthProviderInterface registry.
 */

import { createServer, type Server } from "node:http";
import { createInterface } from "node:readline/promises";
import { stdin, stdout } from "node:process";
import {
  OAUTH_PROVIDER_IDS,
  oauthFilePath,
  saveOAuthCredentials,
  loadOAuthCredentials,
  type OAuthCredentials,
  type OAuthProviderId,
} from "../src/auth/oauth-flows";

// ----- PKCE -----------------------------------------------------------------

const base64url = (bytes: Uint8Array): string => {
  let binary = "";
  for (const byte of bytes) binary += String.fromCharCode(byte);
  return Buffer.from(binary, "binary").toString("base64").replace(/\+/g, "-").replace(/\//g, "_").replace(/=/g, "");
};

const generatePkce = async (): Promise<{ verifier: string; challenge: string }> => {
  const verifierBytes = new Uint8Array(32);
  globalThis.crypto.getRandomValues(verifierBytes);
  const verifier = base64url(verifierBytes);
  const hash = await globalThis.crypto.subtle.digest("SHA-256", new TextEncoder().encode(verifier));
  return { verifier, challenge: base64url(new Uint8Array(hash)) };
};

const randomState = (): string => {
  const bytes = new Uint8Array(16);
  globalThis.crypto.getRandomValues(bytes);
  return Array.from(bytes, (b) => b.toString(16).padStart(2, "0")).join("");
};

// ----- Browser launcher -----------------------------------------------------

const openBrowser = async (url: string): Promise<void> => {
  const platform = process.platform;
  const cmd = platform === "darwin" ? "open" : platform === "win32" ? "cmd" : "xdg-open";
  const args = platform === "win32" ? ["/c", "start", "", url] : [url];
  try {
    const { spawn } = await import("node:child_process");
    const child = spawn(cmd, args, { detached: true, stdio: "ignore" });
    child.unref();
  } catch {
    // best-effort — user can copy/paste the URL from stdout.
  }
};

// ----- HTML responses -------------------------------------------------------

const okHtml = (msg: string): string =>
  `<!doctype html><meta charset="utf-8"><title>OAuth complete</title><style>body{font:16px/1.5 system-ui;padding:3em;color:#1a1a1a}.box{max-width:32em;margin:0 auto;padding:2em;border:1px solid #ddd;border-radius:6px}</style><div class="box"><h1>Authentication complete</h1><p>${msg}</p><p>You can close this window.</p></div>`;
const errHtml = (msg: string): string =>
  `<!doctype html><meta charset="utf-8"><title>OAuth error</title><style>body{font:16px/1.5 system-ui;padding:3em;color:#a31515}.box{max-width:32em;margin:0 auto;padding:2em;border:1px solid #f5c2c0;border-radius:6px;background:#fff5f5}</style><div class="box"><h1>Authentication failed</h1><p>${msg}</p></div>`;

// ----- Anthropic ------------------------------------------------------------

const ANTHROPIC_AUTHORIZE_URL = "https://claude.ai/oauth/authorize";
const ANTHROPIC_TOKEN_URL = "https://platform.claude.com/v1/oauth/token";
const ANTHROPIC_CLIENT_ID = "9d1c250a-e61b-44d9-88ed-5944d1962f5e";
const ANTHROPIC_CALLBACK_PORT = 53692;
const ANTHROPIC_CALLBACK_PATH = "/callback";
const ANTHROPIC_SCOPES =
  "org:create_api_key user:profile user:inference user:sessions:claude_code user:mcp_servers user:file_upload";

type CallbackResult = { code: string; state?: string };

const startCallbackServer = (
  port: number,
  path: string,
  expectedState: string,
): { server: Server; wait: () => Promise<CallbackResult> } => {
  let resolveWait: ((value: CallbackResult) => void) | undefined;
  let rejectWait: ((reason: Error) => void) | undefined;
  const wait = () =>
    new Promise<CallbackResult>((resolve, reject) => {
      resolveWait = resolve;
      rejectWait = reject;
    });
  const server = createServer((req, res) => {
    try {
      const url = new URL(req.url ?? "", "http://localhost");
      if (url.pathname !== path) {
        res.writeHead(404, { "Content-Type": "text/html; charset=utf-8" });
        res.end(errHtml("Unexpected callback path."));
        return;
      }
      const error = url.searchParams.get("error");
      if (error) {
        res.writeHead(400, { "Content-Type": "text/html; charset=utf-8" });
        res.end(errHtml(`Provider returned error: ${error}`));
        rejectWait?.(new Error(`OAuth provider returned error=${error}`));
        return;
      }
      const code = url.searchParams.get("code");
      const state = url.searchParams.get("state");
      if (code == null || state == null) {
        res.writeHead(400, { "Content-Type": "text/html; charset=utf-8" });
        res.end(errHtml("Missing code or state in callback."));
        return;
      }
      if (state !== expectedState) {
        res.writeHead(400, { "Content-Type": "text/html; charset=utf-8" });
        res.end(errHtml("State mismatch — possible CSRF. Re-run the login command."));
        rejectWait?.(new Error("OAuth state mismatch"));
        return;
      }
      res.writeHead(200, { "Content-Type": "text/html; charset=utf-8" });
      res.end(okHtml("Token issued."));
      resolveWait?.({ code, state });
    } catch (err) {
      res.writeHead(500, { "Content-Type": "text/plain; charset=utf-8" });
      res.end("Internal error");
      rejectWait?.(err instanceof Error ? err : new Error(String(err)));
    }
  });
  server.listen(port, "127.0.0.1");
  return { server, wait };
};

const loginAnthropic = async (): Promise<OAuthCredentials> => {
  const { verifier, challenge } = await generatePkce();
  // pi-mono uses the verifier itself as `state` for Anthropic — preserved for
  // wire compatibility with the upstream authorize endpoint.
  const state = verifier;
  const { server, wait } = startCallbackServer(ANTHROPIC_CALLBACK_PORT, ANTHROPIC_CALLBACK_PATH, state);
  const authUrl = new URL(ANTHROPIC_AUTHORIZE_URL);
  authUrl.searchParams.set("code", "true");
  authUrl.searchParams.set("client_id", ANTHROPIC_CLIENT_ID);
  authUrl.searchParams.set("response_type", "code");
  authUrl.searchParams.set("redirect_uri", `http://localhost:${ANTHROPIC_CALLBACK_PORT}${ANTHROPIC_CALLBACK_PATH}`);
  authUrl.searchParams.set("scope", ANTHROPIC_SCOPES);
  authUrl.searchParams.set("code_challenge", challenge);
  authUrl.searchParams.set("code_challenge_method", "S256");
  authUrl.searchParams.set("state", state);
  process.stdout.write(`\nOpen this URL in your browser to authenticate:\n  ${authUrl.toString()}\n\n`);
  await openBrowser(authUrl.toString());
  try {
    const { code } = await wait();
    const tokenResponse = await fetch(ANTHROPIC_TOKEN_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json", Accept: "application/json" },
      body: JSON.stringify({
        grant_type: "authorization_code",
        client_id: ANTHROPIC_CLIENT_ID,
        code,
        state,
        redirect_uri: `http://localhost:${ANTHROPIC_CALLBACK_PORT}${ANTHROPIC_CALLBACK_PATH}`,
        code_verifier: verifier,
      }),
    });
    if (!tokenResponse.ok) {
      const detail = await tokenResponse.text();
      throw new Error(`Anthropic token exchange failed (${tokenResponse.status}): ${detail.slice(0, 300)}`);
    }
    const json = (await tokenResponse.json()) as {
      access_token: string;
      refresh_token: string;
      expires_in: number;
    };
    return {
      access: json.access_token,
      refresh: json.refresh_token,
      expires: Date.now() + json.expires_in * 1000 - 5 * 60 * 1000,
    };
  } finally {
    server.close();
  }
};

// ----- OpenAI Codex (ChatGPT Plus/Pro) --------------------------------------

const OPENAI_AUTHORIZE_URL = "https://auth.openai.com/oauth/authorize";
const OPENAI_TOKEN_URL = "https://auth.openai.com/oauth/token";
const OPENAI_CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann";
const OPENAI_CALLBACK_PORT = 1455;
const OPENAI_CALLBACK_PATH = "/auth/callback";
const OPENAI_SCOPE = "openid profile email offline_access";
const OPENAI_JWT_CLAIM_PATH = "https://api.openai.com/auth";

const decodeJwtAccount = (token: string): string | undefined => {
  const parts = token.split(".");
  if (parts.length !== 3) return undefined;
  try {
    const payload = JSON.parse(Buffer.from(parts[1] ?? "", "base64").toString("utf8")) as Record<string, unknown>;
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

const loginOpenAI = async (): Promise<OAuthCredentials> => {
  const { verifier, challenge } = await generatePkce();
  const state = randomState();
  const { server, wait } = startCallbackServer(OPENAI_CALLBACK_PORT, OPENAI_CALLBACK_PATH, state);
  const authUrl = new URL(OPENAI_AUTHORIZE_URL);
  authUrl.searchParams.set("response_type", "code");
  authUrl.searchParams.set("client_id", OPENAI_CLIENT_ID);
  authUrl.searchParams.set("redirect_uri", `http://localhost:${OPENAI_CALLBACK_PORT}${OPENAI_CALLBACK_PATH}`);
  authUrl.searchParams.set("scope", OPENAI_SCOPE);
  authUrl.searchParams.set("code_challenge", challenge);
  authUrl.searchParams.set("code_challenge_method", "S256");
  authUrl.searchParams.set("state", state);
  authUrl.searchParams.set("id_token_add_organizations", "true");
  authUrl.searchParams.set("codex_cli_simplified_flow", "true");
  authUrl.searchParams.set("originator", "bag");
  process.stdout.write(`\nOpen this URL in your browser to authenticate:\n  ${authUrl.toString()}\n\n`);
  await openBrowser(authUrl.toString());
  try {
    const { code } = await wait();
    const response = await fetch(OPENAI_TOKEN_URL, {
      method: "POST",
      headers: { "Content-Type": "application/x-www-form-urlencoded" },
      body: new URLSearchParams({
        grant_type: "authorization_code",
        client_id: OPENAI_CLIENT_ID,
        code,
        code_verifier: verifier,
        redirect_uri: `http://localhost:${OPENAI_CALLBACK_PORT}${OPENAI_CALLBACK_PATH}`,
      }),
    });
    if (!response.ok) {
      const detail = await response.text();
      throw new Error(`OpenAI token exchange failed (${response.status}): ${detail.slice(0, 300)}`);
    }
    const json = (await response.json()) as {
      access_token: string;
      refresh_token: string;
      expires_in: number;
    };
    const accountId = decodeJwtAccount(json.access_token);
    if (accountId == null) {
      throw new Error("Failed to extract chatgpt_account_id from access token JWT");
    }
    return {
      access: json.access_token,
      refresh: json.refresh_token,
      expires: Date.now() + json.expires_in * 1000,
      accountId,
    };
  } finally {
    server.close();
  }
};

// ----- GitHub Copilot (device flow) -----------------------------------------

const GITHUB_COPILOT_CLIENT_ID = "Iv1.b507a08c87ecfe98";
const COPILOT_HEADERS = {
  "User-Agent": "GitHubCopilotChat/0.35.0",
  "Editor-Version": "vscode/1.107.0",
  "Editor-Plugin-Version": "copilot-chat/0.35.0",
  "Copilot-Integration-Id": "vscode-chat",
} as const;

const sleep = (ms: number): Promise<void> => new Promise((resolve) => setTimeout(resolve, ms));

const loginGitHubCopilot = async (rl: ReturnType<typeof createInterface>): Promise<OAuthCredentials> => {
  const enterprise = (await rl.question("GitHub Enterprise domain (blank for github.com): ")).trim();
  const domain = enterprise === "" ? "github.com" : enterprise;
  const enterpriseUrl = enterprise === "" ? undefined : enterprise;

  const deviceResp = await fetch(`https://${domain}/login/device/code`, {
    method: "POST",
    headers: {
      Accept: "application/json",
      "Content-Type": "application/x-www-form-urlencoded",
      "User-Agent": COPILOT_HEADERS["User-Agent"],
    },
    body: new URLSearchParams({ client_id: GITHUB_COPILOT_CLIENT_ID, scope: "read:user" }),
  });
  if (!deviceResp.ok) throw new Error(`Device-code request failed (${deviceResp.status}): ${await deviceResp.text()}`);
  const device = (await deviceResp.json()) as {
    device_code: string;
    user_code: string;
    verification_uri: string;
    interval: number;
    expires_in: number;
  };
  process.stdout.write(`\nOpen ${device.verification_uri} and enter code: ${device.user_code}\n\n`);
  await openBrowser(device.verification_uri);

  const deadline = Date.now() + device.expires_in * 1000;
  let intervalMs = Math.max(1000, device.interval * 1000);
  let githubAccessToken: string | undefined;
  while (Date.now() < deadline) {
    await sleep(intervalMs);
    const tokenResp = await fetch(`https://${domain}/login/oauth/access_token`, {
      method: "POST",
      headers: {
        Accept: "application/json",
        "Content-Type": "application/x-www-form-urlencoded",
        "User-Agent": COPILOT_HEADERS["User-Agent"],
      },
      body: new URLSearchParams({
        client_id: GITHUB_COPILOT_CLIENT_ID,
        device_code: device.device_code,
        grant_type: "urn:ietf:params:oauth:grant-type:device_code",
      }),
    });
    const data = (await tokenResp.json()) as { access_token?: string; error?: string; interval?: number };
    if (typeof data.access_token === "string") {
      githubAccessToken = data.access_token;
      break;
    }
    if (data.error === "authorization_pending") continue;
    if (data.error === "slow_down") {
      intervalMs = Math.max(intervalMs + 5000, (data.interval ?? 0) * 1000);
      continue;
    }
    throw new Error(`Device flow failed: ${data.error ?? "unknown"}`);
  }
  if (githubAccessToken == null) throw new Error("Device flow timed out");

  // Exchange long-lived github oauth token for a short-lived copilot token.
  const copilotResp = await fetch(`https://api.${domain}/copilot_internal/v2/token`, {
    headers: { Accept: "application/json", Authorization: `Bearer ${githubAccessToken}`, ...COPILOT_HEADERS },
  });
  if (!copilotResp.ok) throw new Error(`Copilot token exchange failed (${copilotResp.status}): ${await copilotResp.text()}`);
  const copilot = (await copilotResp.json()) as { token: string; expires_at: number };
  return {
    // For Copilot we persist the long-lived github access token as `refresh`
    // (it's what we re-present every ~25min to mint a new copilot token), and
    // the short-lived copilot token as `access`.
    access: copilot.token,
    refresh: githubAccessToken,
    expires: copilot.expires_at * 1000 - 5 * 60 * 1000,
    ...(enterpriseUrl == null ? {} : { enterpriseUrl }),
  };
};

// ----- CLI dispatcher -------------------------------------------------------

const printUsage = (): void => {
  process.stdout.write(
    [
      "bag-oauth-login — interactive OAuth login for paid LLM subscriptions",
      "",
      "Usage:",
      "  tsx scripts/bag_oauth_login.ts <provider>",
      "  tsx scripts/bag_oauth_login.ts --list",
      "  tsx scripts/bag_oauth_login.ts --status",
      "",
      `Providers: ${OAUTH_PROVIDER_IDS.join(" | ")}`,
      "",
      "Tokens are stored at ~/.bag/oauth/<provider>.json (chmod 600).",
      "",
    ].join("\n"),
  );
};

const runLogin = async (provider: OAuthProviderId): Promise<void> => {
  const rl = createInterface({ input: stdin, output: stdout });
  try {
    let credentials: OAuthCredentials;
    if (provider === "anthropic") {
      credentials = await loginAnthropic();
    } else if (provider === "openai") {
      credentials = await loginOpenAI();
    } else {
      credentials = await loginGitHubCopilot(rl);
    }
    saveOAuthCredentials(provider, credentials);
    const path = oauthFilePath(provider);
    process.stdout.write(`\nSaved ${provider} credentials to ${path} (chmod 600).\n`);
    process.stdout.write(`Token expires at ${new Date(credentials.expires).toISOString()}.\n`);
  } finally {
    rl.close();
  }
};

const runStatus = (): void => {
  for (const provider of OAUTH_PROVIDER_IDS) {
    const creds = loadOAuthCredentials(provider);
    if (creds == null) {
      process.stdout.write(`${provider.padEnd(16)} (no credentials)\n`);
    } else {
      const remaining = Math.max(0, creds.expires - Date.now());
      const minutes = Math.floor(remaining / 60_000);
      process.stdout.write(`${provider.padEnd(16)} expires in ${minutes}m  (${oauthFilePath(provider)})\n`);
    }
  }
};

const main = async (): Promise<number> => {
  const argv = process.argv.slice(2);
  if (argv.length === 0 || argv.includes("-h") || argv.includes("--help")) {
    printUsage();
    return argv.length === 0 ? 1 : 0;
  }
  if (argv[0] === "--list") {
    process.stdout.write(`${OAUTH_PROVIDER_IDS.join("\n")}\n`);
    return 0;
  }
  if (argv[0] === "--status") {
    runStatus();
    return 0;
  }
  const provider = argv[0];
  if (provider == null || !(OAUTH_PROVIDER_IDS as readonly string[]).includes(provider)) {
    process.stderr.write(`Unknown provider "${String(provider)}". Use --list to see options.\n`);
    return 2;
  }
  await runLogin(provider as OAuthProviderId);
  return 0;
};

void main()
  .then((code) => {
    if (code !== 0) process.exit(code);
  })
  .catch((err: unknown) => {
    process.stderr.write(`\n${err instanceof Error ? err.message : String(err)}\n`);
    process.exit(1);
  });
