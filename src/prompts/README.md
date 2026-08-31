# BAG system prompt — modular tactics with attestation

The BAG executor system prompt is no longer a single hardcoded constant. It is
assembled at module load from this directory:

- `principles.md` — generic agent skeleton; always loaded.
- `tactics/*.md` — one file per forensic clause / gate. Each has a YAML
  frontmatter so we can audit, deprecate, and refactor each rule
  independently.
- `loader.ts` — pure-Node loader; no third-party YAML dep.

## How the loader assembles the prompt

`buildSystemPrompt({ sentinel })` does:

1. Read `principles.md` (frontmatter stripped).
2. Glob `tactics/*.md`, parse frontmatter for each.
3. Drop tactics whose `status` is anything other than `active`.
4. Sort active tactics by `order:` (ascending; missing fields sort last).
5. Concatenate their bodies with `\n` and inject at the `${TACTICS}`
   placeholder inside `principles.md`.
6. Replace every `${SUBMIT_SENTINEL}` with the sentinel passed in.
7. Append a one-line attestation footer:
   `[Tactics loaded: N — auditable in src/prompts/tactics/]`

The loader is generic. No BAG-specific tactic content is hard-coded in
`loader.ts`.

## Frontmatter schema

```yaml
---
id: <slug-matching-the-filename>
status: active            # active | deprecated
order: <integer>          # injection order; lower = earlier
incident: <human-readable forensic incident pointer>
introduced: <YYYY-MM-DD>
review_by: <YYYY-MM-DD>   # auto-flagged for retirement after this date
trigger: "<one-sentence description of when this rule applies>"
merged_into: <optional id when status=deprecated>
---
```

Only `id` and `status` are required by the loader. All other fields are
metadata that `scripts/bag_tactics_audit.ts` reads.

## Adding a new tactic

1. Reproduce the failure in a forensic run; capture the verifier output.
2. Pick a slug (`subprocess-path-gate`, `no-tmp-leak`, ...).
3. Create `src/prompts/tactics/<slug>.md` with frontmatter + body.
4. Set `order:` to the position the body should appear within the workflow
   block (between principles' marker `${TACTICS}` and the rest).
5. Add a test fragment in `tests/prompts-loader.test.ts` if the tactic
   substantively changes the prompt shape.
6. Run `bun test tests/prompts-loader.test.ts` to confirm byte-equivalence is
   preserved.

## Deprecating an old tactic

1. Set `status: deprecated` in its frontmatter.
2. Add `merged_into: <other-id-or-"principles">` if the content moved.
3. Leave the file in place — `scripts/bag_tactics_audit.ts` lists deprecated
   tactics and the next operator can decide to delete them after one
   release cycle.

## Tactic ordering inside the prompt

The current active tactics fill the workflow block between step 7's first
sentence (in principles.md) and step 9's submit-sentinel call. Ordering:

1. `cleanup-before-submit` — COMPILED-LANGUAGE GATE (sub-bullet of step 7).
2. `no-tmp-leak` — SCRATCH-DIR HYGIENE (sub-bullet of step 7).
3. `pre-submit-final-check` — step 8 header + sub-bullets (a) (b).
4. `enumerate-deliverables` — sub-bullet (c).
5. `subprocess-path-gate` — sub-bullet (d) + closing line.

Higher orders (90+) are reserved for deprecated stubs that exist for
documentation continuity.

## Audit & deprecation tooling

`scripts/bag_tactics_audit.ts` lists all tactics with their incident,
introduction date, and review date. Use it before each major release:

```sh
tsx scripts/bag_tactics_audit.ts          # human-readable
tsx scripts/bag_tactics_audit.ts --json   # for CI / dashboards
```

Tactics whose `review_by` has passed are flagged in red.

## Runtime sync

`bench/bag-runtime/` is uploaded into the task container via
`environment.upload_dir(source_dir=HOST_RUNTIME_DIR, target_dir=BAG_DIR)` in
`bench/bag_agent/agent.py`. `upload_dir` includes ALL files under the
directory — markdown is uploaded verbatim, no rsync filter is in play. Just
make sure your tactic .md files are under `bench/bag-runtime/src/prompts/`
when you sync the runtime bundle.
