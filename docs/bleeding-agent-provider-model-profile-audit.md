# BleedingAgent Provider And Model Profile Audit

Date: 2026-05-01
Lane: `next-provider-current-audit`, updated by `next-provider-role-model` and Wave 2C provider discovery/profile lineage work
Scope: provider/model-profile audit plus landed role model, deterministic offline provider/server profile IDs, doctor role/profile output, direct LLM metric lineage, optimizer seed profiles, policy resolution, and focused provider/optimizer/eval tests.

## Current Status Refresh

The audit body includes historical Wave 2 findings. A later single-owner runtime follow-up completed
the provider-policy-lineage gap for current ACP/planning runs: optimizer session pins now include
model role, provider config role, provider kind, endpoint kind, model server ID, model server profile
ID, provider discovery source, context window, max output, model/codebase/profile/policy IDs,
rendered tool versions, edit strategy versions, and verification policy versions. ACP sessions use
the shared pin, HALO root spans expose provider/server attributes, maintenance status shows pinned
lineage, and `runPlanningPipeline()` creates `RunTelemetry` with the same shared optimizer pin.

Remaining boundary: local endpoint discovery is still conservative and mostly operator-configured.
Network model-list validation, chat smoke tests, TTFT/decode/concurrency measurement, and broad
benchmark backfill remain follow-on work.

## Current config

The repo has a project-local `bag.config.json` that overrides the source defaults. Both configured runtime roles point at Anthropic's OpenAI-compatible-ish endpoint with the same model name: master uses provider `openai`, model `claude-opus-4-7`, base URL `https://api.anthropic.com/v1`, and key env `ANTHROPIC_AUTH_TOKEN`; local uses provider `openai-compatible`, the same model/base URL, `apiKey: "unused"`, and the same key env (`bag.config.json:3-19`). The same file raises executor concurrency to 64, max executor concurrency to 128, max turns to 32, and context file/char budgets to 400/12000 (`bag.config.json:20-30`).

The source defaults are different: master defaults to `gpt-5.5` on `https://api.openai.com/v1`, while local defaults to `majentik/Qwen3.6-35B-A3B-RotorQuant-MLX-3bit` at `http://127.0.0.1:18082/v1`. Source config now also defines role bindings for `master`, `local`, `planner`, `executor`, `verifier`, `critic`, `summarizer`, `fast_scout`, and `local_batch_executor`; existing `master` and `local` calls keep their prior strict provider behavior, while the new role aliases carry explicit fallback roles. Provider config supports an endpoint kind plus optional operator-supplied `serverId`, `serverProfileId`, and `contextWindowTokens`; absent those fields, BAG derives deterministic offline-safe server/profile IDs and uses a conservative `max(maxTokens, 8192)` context floor. The loader reads only `bag.config.json` from cwd when present, otherwise parses the schema defaults.

Light verification run in this checkout:

```text
npm run bag -- doctor
masterAvailable=true
localEndpointReady=true
localBaseUrl=https://api.anthropic.com/v1
localModel=claude-opus-4-7
masterModel=claude-opus-4-7
executorConcurrency=64
maxExecutorConcurrency=128
axMasterConfigured=true
axLocalConfigured=true
```

`npm run bag -- metrics` reported 5 runs, 36 step metrics, 51 LLM calls, 9 tool calls, no failures, and 170274 total tokens. Existing `.bag/telemetry/metrics.json` entries do not carry `optimizerPin`; two are legacy arrays and three are objects without pin fields, so historical local traces are not profile-comparable without backfill or exclusion. No `.bag/optimizer/active.json` exists in this checkout.

## Provider and Ax behavior

The direct router resolves a model role through the config role map, then sends raw `POST /chat/completions` requests to the selected master/local provider config with `model`, `messages`, and `max_tokens`. For Anthropic base URLs it suppresses temperature and JSON response-format options. Provider behavior is still partly inferred from URL pattern, not the `provider` field. LLM call metrics now include requested role, resolved role, provider config role, fallback role, provider kind, endpoint kind, deterministic model server ID, deterministic model server profile ID, context window, and max output.

Landed role fallback behavior is conservative. Explicit `master` calls still fail on a missing master key instead of falling back, and explicit `local` calls keep using local config. New aliases fall back only when their primary provider key is absent: `planner`, `verifier`, and `critic` use master then local; `executor`, `summarizer`, and `fast_scout` use local then master; `local_batch_executor` uses local then `executor`.

Local readiness is shallow. `localAvailable()` returns false only when the local key is empty; for Anthropic base URLs it returns true without a network model probe, otherwise it only checks `GET /models` and returns `response.ok` (`src/llm.ts:155-169`). There is no validation that the configured model is listed, no chat-completion smoke prompt, no context window detection, no tool-call capability check, and no TTFT/decode/concurrency measurement.

Ax integration is separate from the direct router. `createAxServices()` always constructs Ax services with `name: "openai"` for both master and local, passes `apiURL` from config, and does not encode provider kind, endpoint kind, local server identity, model role, or measured server profile (`src/llm.ts:194-224`). `createAxBleedingAgent()` uses master when configured, otherwise local, and sets Ax runtime/concurrency from policy (`src/llm.ts:227-255`).

The CLI doctor output now includes the configured provider role map and each role's provider config role, provider kind, model, base URL, endpoint kind, fallback role, deterministic server/profile IDs, context window/source, max output, active optimizer source, model profile ID, codebase profile ID, policy ID, and active tool/edit/verification versions. It still does not perform network-dependent model-list validation, chat smoke tests, TTFT measurement, decode speed measurement, or concurrent throughput measurement.

ACP provider/model selection is explicitly unsupported through unstable ACP provider APIs on the ACP facade (`src/acp-agent.ts`). Normal `/metrics` intentionally hides optimizer internals and says tuning is pinned for the session through `src/acp/slash-router.ts`; `/maintenance status` is the detailed inspection path in `src/acp/maintenance.ts`.

## Implemented profile and lineage behavior

Optimizer schemas distinguish model profiles, codebase profiles, and model-codebase policies. Model profiles now also carry optional `modelRole`, `providerConfigRole`, `fallbackModelRole`, `baseUrl`, `endpointKind`, `modelServerId`, `modelServerProfileId`, `providerDiscoverySource`, `contextWindowSource`, and measured-throughput placeholder fields so role/profile/policy comparisons can distinguish the same model string used for different agent responsibilities and different server envelopes. Policies continue to pin model profile, codebase profile, canonical/rendered tool versions, result style, verification policy, edit strategy, edit contract, fallback/repair/verifier policies, objective set, gates, concurrency, and risk tolerance.

Seed profile IDs are deterministic but hash-derived. Registry seed generation now creates one model profile and one promoted seed policy per model role from role plus model-name hash, codebase IDs from cwd hash, and policy IDs from `{modelProfileId, codebaseProfileId}`. With the current config and cwd, the seed IDs are:

- local model profile: `model.local.1a70c897788f`
- master model profile: `model.master.1a70c897788f`
- planner model profile: `model.planner.1a70c897788f`
- executor model profile: `model.executor.1a70c897788f`
- verifier model profile: `model.verifier.1a70c897788f`
- critic model profile: `model.critic.1a70c897788f`
- summarizer model profile: `model.summarizer.1a70c897788f`
- fast scout model profile: `model.fast_scout.1a70c897788f`
- local batch executor model profile: `model.local_batch_executor.1a70c897788f`
- codebase profile: `codebase.bb787e88c3d9`
- local policy: `policy.5025b4f7bd55`
- master policy: `policy.960122f7f8d0`
- planner policy: `policy.c0366782dd72`
- executor policy: `policy.0105183b9743`
- verifier policy: `policy.de78ac37e392`
- critic policy: `policy.7bba2ee79b8b`
- summarizer policy: `policy.4d72154f3358`
- fast scout policy: `policy.8be2d9756aec`
- local batch executor policy: `policy.58aca89c5d6c`

These model profile IDs differ only by role prefix because this checkout's project config points both master and local provider configs at `claude-opus-4-7`. Seed model records are labeled with the model role, provider config role, endpoint kind, server ID, and server profile ID; they set provider/base URL from config, set endpoint kind from config, set context window from configured value or the deterministic floor, record context source, and set max output from config. Seed policies are promoted, low-risk, and gate on `typecheck` and `test` exit code 0.

Active policy resolution prefers a valid active pointer, then promoted persisted matching records, then seed records. The resolver only accepts selectable active/promoted profiles and promoted policies, rejects pointers whose model/policy/codebase do not match, and now also requires model role compatibility unless the caller explicitly pins a `modelProfileId`. This prevents a promoted `critic` policy from being reused for `planner` just because both roles share the same provider model string.

ACP session creation snapshots the optimizer policy into `session.optimizerPin` exactly once through `src/acp/session.ts`. The chosen role is `master` when `resolveMasterApiKey()` returns a key, otherwise `local`; it passes that role and model name into `resolveLoadedOptimizerPolicy()`, then stores profile IDs, policy ID, tool/edit/verification versions, registry source, error counts, renderer IDs, and record IDs into telemetry. Existing sessions are resumed from memory without recomputing the pin. Therefore, in this host's verified environment, a new ACP session should pin `modelRole=master`, `modelProfileId=model.master.1a70c897788f`, `codebaseProfileId=codebase.bb787e88c3d9`, `policyId=policy.960122f7f8d0`, and `source=seed` unless a matching active pointer or promoted registry override is added.

Telemetry can propagate a pin when supplied. `RunTelemetry` accepts optional `OptimizerSessionPinTelemetry`, attaches it to events, metrics, and HALO spans, and writes it under each run entry in the metrics store. Direct LLM call metrics now record requested role, resolved role, provider config role, fallback source, provider kind, endpoint kind, deterministic server/profile IDs, context window, and max output. Current ACP planning/routing/coding runs construct telemetry with `session.optimizerPin.telemetry`, and the later provider-policy-lineage follow-up extended that pin with provider/server/profile fields. `runPlanningPipeline()` now creates `RunTelemetry` with the same shared optimizer pin. Historical metrics created before that follow-up remain non-comparable unless backfilled or explicitly excluded from promotion evidence.

Trace indexing already extracts optimizer profile and policy dimensions from HALO span attributes, including model/profile/policy IDs and all tool/edit version IDs (`src/trace-store.ts:821-843`, `src/trace-store.ts:852-898`, `src/trace-store.ts:914-922`). Candidate evidence preserves those dimensions in lineage (`src/optimizer/evidence.ts:17-35`, `src/optimizer/evidence.ts:153-185`). Candidate generation and GEPA reject observations with missing or ambiguous model/profile/codebase/policy lineage rather than fabricating IDs (`src/optimizer/candidates.ts:317-325`, `src/optimizer/gepa-runner.ts:494-509`).

## Missing discovery and measurement

The direct router and optimizer registry now have a first-class role model and doctor prints role/profile/policy details, but ACP runtime orchestration still chooses only `master` or `local` for current live turns and Ax services are still master/local only. Extending live ACP model-role routing requires edits outside this lane's write scope.

There is no separate model-server or server-profile registry artifact yet; server/profile IDs currently live on model profiles, optimizer pins, HALO root attributes, and LLM metrics. Eval and replay schemas already know `modelServerId` and `modelServerProfileId` (`src/eval-harness/types.ts:12-22`, `src/replay/capture.ts:166-176`), and edit-strategy ablation accepts synthetic defaults for both (`src/eval-harness/edit-strategy-ablation.ts:57-73`). The remaining gap is operational quality: local endpoint discovery, benchmark backfill, and historical trace exclusion/backfill must be solved before comparing "same model on different local server/profile/throughput envelope" across old and new runs.

Local server scripts are not wired into BAG profiles. `scripts/run_supergemma_mlx_server.sh` starts `mlx_lm.server` with defaults `127.0.0.1:8080` and model `Jiunsong/supergemma4-26b-uncensored-mlx-4bit-v2` (`scripts/run_supergemma_mlx_server.sh:7-11`, `scripts/run_supergemma_mlx_server.sh:78-93`), while BAG source defaults use `127.0.0.1:18082` (`src/types.ts:20-35`) and the concurrency benchmark defaults to `127.0.0.1:18081` (`scripts/benchmark_mlx_server_concurrency.py:108-119`). The local benchmark measures aggregate completion/total tokens per second, mean/p50/max request latency, finish reasons, and errors, but not streaming TTFT, per-token decode distribution, queueing delay, server memory, context-window probing, tool-call success, or model-list validation (`scripts/benchmark_mlx_server_concurrency.py:73-105`, `scripts/benchmark_mlx_server_concurrency.py:118-139`).

Existing benchmark docs contain useful model facts but are not fully machine-readable provider profiles. The Hermes Qwen report records MLX server command, base URL, model, 4/6 first-pass and 6/6 after one repair, plus native tool-call limitations (`docs/hermes-local-coding-eval-report.md:23-55`, `docs/hermes-local-coding-eval-report.md:57-79`). The Qwen optimization comparison recommends baseline MLX for agent loops, 8K-32K default context, 65K hard local ceiling, and avoiding DFlash-family variants for agent loops (`docs/qwen36-optimization-comparison-report.md:142-171`). The schema has optional measured fields, but this wave intentionally did not backfill benchmark measurements or run network/server probes.

## Tests needed next

Landed focused tests cover deterministic offline provider profile IDs, LLM metric server/profile lineage, role-specific seed profiles and policies, same-model role separation, active-pointer role mismatch fallback, resolver-level session/replay target stability, cross-server eval comparability rejection, and mocked provider fallback behavior without network calls (`tests/provider-role-model.test.ts`, `tests/optimizer-registry.test.ts`, `tests/optimizer-policy-resolver.test.ts`, `tests/eval-harness-types.test.ts`).

Add an ACP session-pinning regression that creates session A, records its `optimizerPin`, writes/promotes a matching active pointer or promoted persisted policy, creates session B, and proves A keeps the old policy while B resolves the new one. Existing tests prove resolver pointer behavior and maintenance read-only behavior (`tests/optimizer-policy-resolver.test.ts:75-137`, `tests/bag.test.ts:2209-2232`), but not old-session/new-session divergence after promotion.

Keep a CLI comparability regression for `bag run` so future refactors do not drop optimizer pin injection from `runPlanningPipeline()`. Legacy CLI runs that predate provider-policy-lineage should remain non-comparable unless backfilled.

Keep runtime tests that require `modelServerId` and `modelServerProfileId` to appear in ACP optimizer pins, HALO trace dimensions, replay captures, and candidate evidence. Eval scorecard tests already treat server fields as comparable context (`tests/eval-harness-scorer.test.ts:12-22`, `src/eval-harness/types.ts:235-245`); future work should focus on endpoint measurement and historical backfill/exclusion rather than adding another parallel lineage surface.

Add local endpoint doctor tests for OpenAI-compatible servers that distinguish: `/models` reachable but configured model absent, chat completions failing, JSON-object mode unsupported, tool-call payload returned as plain text, and Anthropic URL trust path. Current doctor tests cover config defaults and output shape, not these server-profile capability cases (`tests/bag.test.ts:16-23`, `src/index.ts:76-103`).

Add replay comparability tests where a captured ACP/replay record with policy/model/codebase/server context becomes an eval case and scorecard without dropping those dimensions. Replay capture context allows them (`src/replay/capture.ts:166-176`), extraction preserves trace/source IDs (`src/replay/extraction.ts:141-165`), but the skeleton does not make comparable model/server context a first-class output.

## Top implementation risks

1. Role-specific profiles and provider/server pins now exist and doctor exposes them, but ACP live routing and Ax services are still mostly master/local oriented rather than fully role-aware.
2. Local endpoint readiness can be a false positive, especially on the Anthropic trust path and on `/models`-only OpenAI-compatible checks.
3. Current runtime pins can carry server-profile IDs, but historical traces and benchmark docs are not fully backfilled, so comparisons across MLX/vLLM/Ollama/llama.cpp/server settings still need measurement and exclusion discipline.
4. Historical `.bag` metrics and any pre-lineage CLI runs lack optimizer pins, so operators should either backfill/ignore them or keep them out of promotion evidence.
5. Profile IDs are hash-derived and not operator-friendly; doctor/config UX should print both stable IDs and human-readable role/provider/model/server labels.
