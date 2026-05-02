# Codex CLI System Prompt — Provenance Research

Wave-1 Agent #D — Action 5 prep, 2026-05-01.
Wave-3 Agent #U — cleanup amendment, 2026-05-01.

## 2026-05-01 cleanup amendment (Agent #U)

**`gpt-5.2-codex` was REMOVED** as a seed module candidate. It is a legacy
model with **0% representation** in the local Codex session corpus
(`~/.codex/sessions/2026/04/`, n=300 sampled).

Corpus prevalence (Agent P's measurement, `turn_context.payload.model`):

| Model                  | Share |
|------------------------|-------|
| gpt-5.4                | 86.3% |
| gpt-5.5                |  5.3% |
| gpt-5.4-mini           |  4.3% |
| gpt-5.3-codex-spark    |  4.0% |
| gpt-5.2-codex          |  0.0% |

Template-vs-base survey (`codex-rs/core/templates/model_instructions/*.md`
across all five `codex-native*` worktrees as of 2026-05-01):

| Model                  | Has own template? | Source                                                                                          |
|------------------------|-------------------|-------------------------------------------------------------------------------------------------|
| gpt-5.4                | No                | uses base `codex-rs/core/prompt.md` (`BASE_INSTRUCTIONS`)                                       |
| gpt-5.5                | No                | uses base `prompt.md`                                                                           |
| gpt-5.4-mini           | No                | uses base `prompt.md`                                                                           |
| gpt-5.3-codex-spark    | No                | uses base `prompt.md`                                                                           |
| gpt-5.2-codex (legacy) | Yes               | `codex-rs/core/templates/model_instructions/gpt-5.2-codex_instructions_template.md` (7,319 B)   |

The only `templates/model_instructions/*.md` file present in any of the five
`codex-native*` checkouts is `gpt-5.2-codex_instructions_template.md`. None of
the four prevalent models has its own template, so all four are already
covered by the existing `--seed-module codex` (which embeds `prompt.md`
verbatim). **No new seed modules were added.**

Files removed: `trace-gepa/agent_opt/seed_gpt5_codex.py`. Choice removed:
`gpt5_codex` from `optimize.py --seed-module`. Resulting choices:
`{default, bag, codex}`.

---


## TL;DR

The Codex CLI base system prompt is hardcoded in the open-source `codex-rs`
crate as the contents of `codex-rs/core/prompt.md`, embedded at compile time
via `include_str!` and exposed as the constant `BASE_INSTRUCTIONS`. The version
captured into `seed_codex.py` is the file shipped in
`codex-native-main101-sync` (also identical to the copy in the `codex-native`
checkout at this writing).

- File: `/Users/satan/side/experiments/codex-native-main101-sync/codex-rs/core/prompt.md`
- Size: 20,923 bytes on disk (20,771 chars after UTF-8 decode — the file uses
  a handful of typographic characters such as curly apostrophes and en-dashes).
- Rust binding: `codex-rs/core/src/models_manager/model_info.rs:15`
  ```rust
  pub const BASE_INSTRUCTIONS: &str = include_str!("../../prompt.md");
  ```
- Override path: `model_info.rs:46` checks
  `if let Some(base_instructions) = &config.base_instructions { model.base_instructions = base_instructions.clone(); }`.
  The user's `~/.codex/config.toml` (10,489 bytes) does **not** define a
  `base_instructions` key — only `hide_gpt5_1_migration_prompt` and
  `hide_gpt-5.1-codex-max_migration_prompt` flags. Therefore the runtime uses
  `BASE_INSTRUCTIONS` verbatim for every session.

## Why session_meta `instructions` is null

`rollout-*.jsonl` files store the request as the user/agent constructed it.
For all five sampled sessions in `~/.codex/sessions/2026/05/`, the
`payload.instructions` field on the `session_meta` event is `null`. Codex CLI
serialises the override field rather than the entire baseline; when no user
override is present the field is null and the binary substitutes
`BASE_INSTRUCTIONS` from `prompt.md` at request time. So GEPA cannot recover
the actual baseline prompt from session traces alone — it must come from the
codex-rs source.

## Candidates considered, in priority order

1. **`codex-rs/core/prompt.md`** (chosen). The canonical
   `BASE_INSTRUCTIONS` constant. 20,771 chars.
2. **`codex-rs/core/prompt_with_apply_patch_instructions.md`** (24,008 bytes,
   not chosen). This is a strict superset that prepends `apply_patch` tool
   docs — used only when `codex.rs:5505` falls back for models that do not
   register `apply_patch` as a first-class tool. The base path is more
   representative of the steady-state system prompt that a frontier
   GPT-5-class model sees.
3. **`codex-rs/core/templates/model_instructions/gpt-5.2-codex_instructions_template.md`**
   (7,319 bytes, not chosen). A templated instruction body used for the
   GPT-5.2-codex model variant. Begins `You are Codex, a coding agent based on
   GPT-5...` and contains `{{ personality }}` placeholders. Worth optimising
   separately if the GPT-5.2-codex path becomes the dominant runtime, but it
   does not match the prompt observed for default Codex CLI sessions.
4. **`~/.codex/prompts/ralph-*.md`** (not chosen). These are user slash
   commands (`/ralph-discover`, `/ralph-plan`, etc.) — they are appended on
   demand as user messages; they are not the system prompt.
5. **Per-session override in session_meta** (not present). All sampled
   sessions have `instructions: null`.

## Files in scope

- New: `trace-gepa/agent_opt/seed_codex.py` — exposes
  `SEED_PROMPT_CODEX: str` containing the embedded `prompt.md` content.
- Modified: `trace-gepa/agent_opt/optimize.py` — added `"codex"` to
  `--seed-module` choices and an `elif args.seed_module == "codex"` import
  branch. Coordinated minimally with Agent C's earlier
  `["default", "bag"]` block — only the choices list and the if/elif chain
  were touched.

## Reproducing the lookup

```bash
rg -n 'BASE_INSTRUCTIONS|include_str!\("\.\./\.\./prompt.md"\)' \
   /Users/satan/side/experiments/codex-native-main101-sync/codex-rs/
wc -c /Users/satan/side/experiments/codex-native-main101-sync/codex-rs/core/prompt.md
grep -i 'instructions\|prompt' ~/.codex/config.toml
```

## Open questions / caveats

- The three checked-out `codex-native*` repos are at slightly different
  revisions; the current `prompt.md` is byte-identical between
  `codex-native/` and `codex-native-main101-sync/` but I did not diff every
  worktree. If a future Codex CLI release alters `prompt.md`, the seed will
  drift and should be regenerated from the version that produced the trace
  dataset under optimisation.
- The `gpt-5.2-codex_instructions_template.md` path is templated
  (`{{ personality }}` placeholder is rendered at runtime by
  `models_manager`). If the trace dataset turns out to be dominated by
  GPT-5.2-codex sessions rather than the generic path, we should add a
  fourth seed module for the rendered template.
- The seed includes the `apply_patch` reference in section "Task execution"
  but not the longer `apply_patch_tool_instructions.md` block. This matches
  the default runtime behaviour for models that register `apply_patch` as a
  function tool (the common path).
