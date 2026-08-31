import { describe, expect, test } from "bun:test";
import {
  __testing,
  getOAuthHeaders,
  isExpired,
  loadOAuthCredentials,
  oauthDir,
  oauthFilePath,
  refreshOAuthCredentials,
  saveOAuthCredentials,
  type OAuthCredentials,
  type OAuthFileSystem,
  type OAuthHttp,
  type OAuthProviderId,
} from "../src/auth/oauth-flows";

// ---------------------------------------------------------------------------
// In-memory filesystem used to verify storage shape, chmod, and refresh
// persistence WITHOUT touching the user's actual ~/.bag/oauth/ directory.
// ---------------------------------------------------------------------------

const inMemoryFs = (): OAuthFileSystem & {
  files: Map<string, { contents: string; mode: number }>;
  dirs: Set<string>;
} => {
  const files = new Map<string, { contents: string; mode: number }>();
  const dirs = new Set<string>();
  return {
    files,
    dirs,
    exists: (path) => files.has(path),
    read: (path) => {
      const entry = files.get(path);
      if (entry == null) throw new Error(`ENOENT: ${path}`);
      return entry.contents;
    },
    write: (path, contents) => {
      const existing = files.get(path);
      files.set(path, { contents, mode: existing?.mode ?? 0o644 });
    },
    mkdir: (path) => {
      dirs.add(path);
    },
    chmod600: (path) => {
      const entry = files.get(path);
      if (entry != null) files.set(path, { ...entry, mode: 0o600 });
    },
    mode: (path) => files.get(path)?.mode ?? 0,
  };
};

const sampleCreds = (provider: OAuthProviderId, expires: number): OAuthCredentials => ({
  access: `${provider}-access-v1`,
  refresh: `${provider}-refresh-v1`,
  expires,
  ...(provider === "openai" ? { accountId: "acct-123" } : {}),
});

// ---------------------------------------------------------------------------

describe("oauth-flows :: storage paths", () => {
  test("oauthDir defaults to $HOME/.bag/oauth or BAG_OAUTH_DIR override", () => {
    expect(oauthDir("/tmp/test-overide")).toBe("/tmp/test-overide");
    expect(oauthDir()).toMatch(/[\\/]\.bag[\\/]oauth$/);
  });

  test("oauthFilePath joins the base dir with <provider>.json", () => {
    expect(oauthFilePath("anthropic", "/tmp/x")).toBe("/tmp/x/anthropic.json");
    expect(oauthFilePath("openai", "/tmp/x")).toBe("/tmp/x/openai.json");
    expect(oauthFilePath("github-copilot", "/tmp/x")).toBe("/tmp/x/github-copilot.json");
  });
});

describe("oauth-flows :: load + save round-trip", () => {
  test("saveOAuthCredentials writes JSON, creates the directory, and chmods 600", () => {
    const fs = inMemoryFs();
    const creds = sampleCreds("anthropic", 1_700_000_000_000);
    saveOAuthCredentials("anthropic", creds, { fs, baseDir: "/fake/.bag/oauth" });

    expect(fs.dirs.has("/fake/.bag/oauth")).toBe(true);
    const path = "/fake/.bag/oauth/anthropic.json";
    expect(fs.files.has(path)).toBe(true);
    expect(fs.mode(path)).toBe(0o600);

    const reloaded = loadOAuthCredentials("anthropic", { fs, baseDir: "/fake/.bag/oauth" });
    expect(reloaded).toEqual(creds);
  });

  test("loadOAuthCredentials returns undefined when the file is missing", () => {
    const fs = inMemoryFs();
    expect(loadOAuthCredentials("anthropic", { fs, baseDir: "/empty" })).toBeUndefined();
  });

  test("loadOAuthCredentials returns undefined when the JSON is malformed", () => {
    const fs = inMemoryFs();
    fs.write("/x/anthropic.json", "{not json");
    expect(loadOAuthCredentials("anthropic", { fs, baseDir: "/x" })).toBeUndefined();
  });

  test("loadOAuthCredentials rejects payloads missing required fields", () => {
    const fs = inMemoryFs();
    fs.write("/x/anthropic.json", JSON.stringify({ access: "a" })); // no refresh / expires
    expect(loadOAuthCredentials("anthropic", { fs, baseDir: "/x" })).toBeUndefined();
  });
});

describe("oauth-flows :: isExpired", () => {
  test("treats credentials within the margin window as expired", () => {
    const creds: OAuthCredentials = { access: "a", refresh: "r", expires: Date.now() + 30_000 };
    expect(isExpired(creds, 60_000)).toBe(true);
  });
  test("treats credentials safely past the margin as fresh", () => {
    const creds: OAuthCredentials = { access: "a", refresh: "r", expires: Date.now() + 10 * 60_000 };
    expect(isExpired(creds, 60_000)).toBe(false);
  });
});

// ---------------------------------------------------------------------------
// Mock HTTP transport — captures requests and lets each test script the
// response. NO real network IO.
// ---------------------------------------------------------------------------

const mockHttp = (
  responder: (url: string, init: { method: string; headers: Record<string, string>; body: string | URLSearchParams }) => {
    ok?: boolean;
    status?: number;
    text?: string;
    json?: unknown;
  },
): OAuthHttp & { calls: Array<{ url: string; method: string; headers: Record<string, string>; body: string }> } => {
  const calls: Array<{ url: string; method: string; headers: Record<string, string>; body: string }> = [];
  const fn = (async (url: string, init: { method: string; headers: Record<string, string>; body: string | URLSearchParams }) => {
    const bodyString = init.body instanceof URLSearchParams ? init.body.toString() : init.body;
    calls.push({ url, method: init.method, headers: init.headers, body: bodyString });
    const response = responder(url, init);
    return {
      ok: response.ok ?? true,
      status: response.status ?? 200,
      text: async () => response.text ?? JSON.stringify(response.json ?? {}),
      json: async () => response.json ?? {},
    };
  }) as OAuthHttp & { calls: typeof calls };
  fn.calls = calls;
  return fn;
};

describe("oauth-flows :: refreshOAuthCredentials (Anthropic)", () => {
  test("posts to the Anthropic token endpoint with the configured client_id and refresh_token", async () => {
    const http = mockHttp(() => ({
      json: { access_token: "new-access", refresh_token: "new-refresh", expires_in: 3600 },
    }));
    const before = Date.now();
    const result = await refreshOAuthCredentials(
      "anthropic",
      sampleCreds("anthropic", 0),
      http,
    );
    expect(http.calls.length).toBe(1);
    expect(http.calls[0]?.url).toBe(__testing.ANTHROPIC_TOKEN_URL);
    expect(http.calls[0]?.method).toBe("POST");
    const body = JSON.parse(http.calls[0]?.body ?? "{}");
    expect(body.grant_type).toBe("refresh_token");
    expect(body.client_id).toBe(__testing.ANTHROPIC_CLIENT_ID);
    expect(body.refresh_token).toBe("anthropic-refresh-v1");
    expect(result.access).toBe("new-access");
    expect(result.refresh).toBe("new-refresh");
    // 5-minute clock-skew margin should bring expires < (now + 3600s).
    expect(result.expires).toBeGreaterThanOrEqual(before + 3600 * 1000 - 5 * 60 * 1000 - 50);
    expect(result.expires).toBeLessThanOrEqual(Date.now() + 3600 * 1000 - 5 * 60 * 1000 + 50);
  });

  test("throws a useful error on non-2xx", async () => {
    const http = mockHttp(() => ({ ok: false, status: 401, text: "invalid_grant" }));
    await expect(refreshOAuthCredentials("anthropic", sampleCreds("anthropic", 0), http)).rejects.toThrow(/401/);
  });

  test("rejects responses missing required fields", async () => {
    const http = mockHttp(() => ({ json: { access_token: "only-access" } }));
    await expect(refreshOAuthCredentials("anthropic", sampleCreds("anthropic", 0), http)).rejects.toThrow(/missing required fields/);
  });
});

describe("oauth-flows :: refreshOAuthCredentials (OpenAI Codex)", () => {
  test("posts form-encoded refresh_token grant to OpenAI", async () => {
    const http = mockHttp(() => ({
      json: { access_token: "header.payload.sig", refresh_token: "rt2", expires_in: 1800 },
    }));
    const result = await refreshOAuthCredentials("openai", sampleCreds("openai", 0), http);
    expect(http.calls[0]?.url).toBe(__testing.OPENAI_TOKEN_URL);
    expect(http.calls[0]?.headers["Content-Type"]).toBe("application/x-www-form-urlencoded");
    const params = new URLSearchParams(http.calls[0]?.body ?? "");
    expect(params.get("grant_type")).toBe("refresh_token");
    expect(params.get("client_id")).toBe(__testing.OPENAI_CLIENT_ID);
    expect(params.get("refresh_token")).toBe("openai-refresh-v1");
    expect(result.access).toBe("header.payload.sig");
    expect(result.refresh).toBe("rt2");
  });

  test("preserves the previously-stored accountId when refresh JWT cannot be decoded", async () => {
    const http = mockHttp(() => ({
      json: { access_token: "not-a-jwt", refresh_token: "rt2", expires_in: 1800 },
    }));
    const seed: OAuthCredentials = {
      access: "old", refresh: "old-refresh", expires: 0, accountId: "preserved-acct",
    };
    const result = await refreshOAuthCredentials("openai", seed, http);
    expect(result.accountId).toBe("preserved-acct");
  });
});

describe("oauth-flows :: refreshOAuthCredentials (GitHub Copilot)", () => {
  test("does a token-exchange (Bearer) against api.github.com", async () => {
    const http = mockHttp(() => ({
      json: { token: "tid=foo;exp=1;proxy-ep=proxy.individual.githubcopilot.com", expires_at: 1_800_000_000 },
    }));
    const result = await refreshOAuthCredentials("github-copilot", sampleCreds("github-copilot", 0), http);
    expect(http.calls[0]?.url).toBe("https://api.github.com/copilot_internal/v2/token");
    expect(http.calls[0]?.headers.Authorization).toBe("Bearer github-copilot-refresh-v1");
    expect(result.access).toContain("proxy-ep=");
    // expires_at is in seconds; module returns ms minus 5min margin.
    expect(result.expires).toBe(1_800_000_000 * 1000 - 5 * 60 * 1000);
  });

  test("uses the enterprise domain when present", async () => {
    const http = mockHttp(() => ({
      json: { token: "tid=foo;proxy-ep=proxy.example.ghe.com", expires_at: 2_000_000_000 },
    }));
    await refreshOAuthCredentials(
      "github-copilot",
      { access: "old", refresh: "ghe-token", expires: 0, enterpriseUrl: "company.ghe.com" },
      http,
    );
    expect(http.calls[0]?.url).toBe("https://api.company.ghe.com/copilot_internal/v2/token");
  });
});

// ---------------------------------------------------------------------------

describe("oauth-flows :: getOAuthHeaders end-to-end", () => {
  test("Anthropic: returns Bearer + anthropic-beta when credentials are fresh", async () => {
    const fs = inMemoryFs();
    const future = Date.now() + 30 * 60_000;
    saveOAuthCredentials("anthropic", sampleCreds("anthropic", future), { fs, baseDir: "/td/oauth" });
    const http = mockHttp(() => ({ json: {} }));
    const result = await getOAuthHeaders("anthropic", { fs, baseDir: "/td/oauth", http });
    expect(http.calls.length).toBe(0); // no refresh
    expect(result.headers.Authorization).toBe("Bearer anthropic-access-v1");
    expect(result.headers["anthropic-beta"]).toBe("oauth-2025-04-20");
    expect(result.headers["anthropic-version"]).toBe("2023-06-01");
    expect(result.baseUrlOverride).toBeUndefined();
  });

  test("Anthropic: refreshes when expired, persists new creds with chmod 600", async () => {
    const fs = inMemoryFs();
    const stale = Date.now() - 60_000;
    saveOAuthCredentials("anthropic", sampleCreds("anthropic", stale), { fs, baseDir: "/td/oauth" });
    const http = mockHttp(() => ({
      json: { access_token: "rotated-access", refresh_token: "rotated-refresh", expires_in: 3600 },
    }));
    const result = await getOAuthHeaders("anthropic", { fs, baseDir: "/td/oauth", http });
    expect(http.calls.length).toBe(1);
    expect(result.headers.Authorization).toBe("Bearer rotated-access");

    // Persisted refreshed credentials.
    const after = loadOAuthCredentials("anthropic", { fs, baseDir: "/td/oauth" });
    expect(after?.access).toBe("rotated-access");
    expect(after?.refresh).toBe("rotated-refresh");
    expect(fs.mode("/td/oauth/anthropic.json")).toBe(0o600);
  });

  test("OpenAI: emits chatgpt-account-id when accountId is present", async () => {
    const fs = inMemoryFs();
    saveOAuthCredentials("openai", sampleCreds("openai", Date.now() + 60 * 60_000), { fs, baseDir: "/td/oauth" });
    const http = mockHttp(() => ({ json: {} }));
    const result = await getOAuthHeaders("openai", { fs, baseDir: "/td/oauth", http });
    expect(result.headers.Authorization).toBe("Bearer openai-access-v1");
    expect(result.headers["chatgpt-account-id"]).toBe("acct-123");
    expect(result.headers["OpenAI-Beta"]).toBe("responses=experimental");
  });

  test("GitHub Copilot: returns baseUrlOverride parsed from proxy-ep", async () => {
    const fs = inMemoryFs();
    const fresh: OAuthCredentials = {
      access: "tid=abc;exp=999;proxy-ep=proxy.individual.githubcopilot.com;rt=foo",
      refresh: "ghu_long_lived_token",
      expires: Date.now() + 20 * 60_000,
    };
    saveOAuthCredentials("github-copilot", fresh, { fs, baseDir: "/td/oauth" });
    const http = mockHttp(() => ({ json: {} }));
    const result = await getOAuthHeaders("github-copilot", { fs, baseDir: "/td/oauth", http });
    expect(result.baseUrlOverride).toBe("https://api.individual.githubcopilot.com");
    expect(result.headers["Copilot-Integration-Id"]).toBe("vscode-chat");
  });

  test("Throws a helpful error when no credentials file exists for the provider", async () => {
    const fs = inMemoryFs();
    await expect(
      getOAuthHeaders("anthropic", { fs, baseDir: "/td/empty", http: mockHttp(() => ({ json: {} })) }),
    ).rejects.toThrow(/bag-oauth-login anthropic/);
  });

  test("Refresh margin is configurable", async () => {
    const fs = inMemoryFs();
    // Expires 10 minutes from now; with default 60s margin would NOT refresh.
    // With a 30-minute margin, it should refresh.
    const expires = Date.now() + 10 * 60_000;
    saveOAuthCredentials("anthropic", sampleCreds("anthropic", expires), { fs, baseDir: "/td/oauth" });
    const http = mockHttp(() => ({
      json: { access_token: "refreshed", refresh_token: "r2", expires_in: 3600 },
    }));
    const result = await getOAuthHeaders("anthropic", {
      fs,
      baseDir: "/td/oauth",
      http,
      refreshMarginMs: 30 * 60_000,
    });
    expect(result.headers.Authorization).toBe("Bearer refreshed");
    expect(http.calls.length).toBe(1);
  });
});

// ---------------------------------------------------------------------------

describe("oauth-flows :: pi-mono compatibility helpers", () => {
  test("decodeJwtAccount extracts chatgpt_account_id from a Codex-shaped JWT", () => {
    const payload = {
      [__testing.OPENAI_JWT_CLAIM_PATH]: { chatgpt_account_id: "acct-xyz" },
      sub: "user_1",
    };
    const encoded = Buffer.from(JSON.stringify(payload), "utf8").toString("base64");
    const jwt = `header.${encoded}.signature`;
    expect(__testing.decodeJwtAccount(jwt)).toBe("acct-xyz");
  });

  test("decodeJwtAccount returns undefined for non-JWT input", () => {
    expect(__testing.decodeJwtAccount("not-a-jwt")).toBeUndefined();
    expect(__testing.decodeJwtAccount("a.b")).toBeUndefined();
  });

  test("githubCopilotApiBaseUrl falls back to enterprise then default", () => {
    expect(__testing.githubCopilotApiBaseUrl("no-proxy-ep-here", undefined)).toBe(
      "https://api.individual.githubcopilot.com",
    );
    expect(__testing.githubCopilotApiBaseUrl("no-proxy-ep-here", "company.ghe.com")).toBe(
      "https://copilot-api.company.ghe.com",
    );
    expect(__testing.githubCopilotApiBaseUrl("tid=x;proxy-ep=proxy.foo.bar;y=1", undefined)).toBe(
      "https://api.foo.bar",
    );
  });
});
