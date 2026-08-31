# BAG `pathProfile` configuration

## What it is

`pathProfile` is a config block in `bag.config.json` that centralises every
filesystem-path convention BAG previously hard-coded across the codebase
(workspace-snapshot exclusion globs, scratch directories flagged by hygiene
audits, the default subprocess `PATH` cited by the pre-submit self-check
prompt and the executor system prompt).

Before this consolidation, the same Linux conventions were duplicated as
inline literals in:

- `src/instruction-verifier.ts` — `find ... -not -path '*/.bag/*' -not -path '*/.git/*'`
- `src/scratch-hygiene.ts` — `/tmp/...` regexes
- `src/pre-submit-self-check.ts` — `/usr/local/bin:/usr/bin:/bin` PATH wording
- `src/autonomous-coding-turn.ts` — `Test in /tmp` and SUBPROCESS-PATH GATE prose

A deployment running BAG inside a container with a non-standard FHS layout
(e.g. NixOS PATH, `/scratch` instead of `/tmp`, alt metadata directories)
had to fork the source. `pathProfile` replaces every fork point with one
declarative config block that defaults to the historical Linux conventions.

## Why it exists

1. **Single source of truth.** Operators tune one block instead of grepping
   for path literals across four files.
2. **Portability.** Containers, Nix-style layouts, Windows POSIX layers,
   and BSDs no longer require source patches to flag the right scratch
   dirs / metadata directories.
3. **Backward compatibility.** Each field carries a Zod `.default()` whose
   value reproduces the historical Linux conventions byte-for-byte. Older
   `bag.config.json` files (including the one written by
   `bench/bag_agent/agent.py`) keep working with no edits.
4. **Test surface.** Centralising the config makes it possible to assert,
   in unit tests, that overrides actually flow into the snapshot command,
   the hygiene audit, and the auditor system prompt.

## Default values

```json
{
  "pathProfile": {
    "metadataDirs": [".bag", ".git"],
    "scratchDirs": ["/tmp", "/var/tmp"],
    "systemPathDirs": ["/usr/local/bin", "/usr/bin", "/bin"]
  }
}
```

| Field | Default | Consumed by |
|-------|---------|-------------|
| `metadataDirs` | `[".bag", ".git"]` | `instruction-verifier.ts` snapshot/restore — passed to `find` as `-not -path '*/<dir>/*'` so probe-cleanup ignores BAG-private cache and git metadata. |
| `scratchDirs` | `["/tmp", "/var/tmp"]` | `scratch-hygiene.ts` audit — flags writes to any of these directories that are not later cleaned up. |
| `systemPathDirs` | `["/usr/local/bin", "/usr/bin", "/bin"]` | `pre-submit-self-check.ts` SUBPROCESS-PATH GATE prompt and the autonomous-coding-turn executor system prompt — interpolated as the colon-joined default subprocess PATH. The prompt's persistence-step example uses `systemPathDirs[0]` (`ln -s ... <dir>/X`, `cp X <dir>/`). |

Empty arrays are rejected at config-parse time — every field must contain at
least one entry. An empty `metadataDirs` would let probes silently mutate
`.bag/` artifacts the harbor verifier scans; an empty `scratchDirs` would
disable hygiene auditing entirely; an empty `systemPathDirs` would render
an empty colon-PATH that no shell command could resolve.

## Common override scenarios

### Docker image with `/scratch` instead of `/tmp`

```json
{
  "pathProfile": {
    "metadataDirs": [".bag", ".git"],
    "scratchDirs": ["/scratch"],
    "systemPathDirs": ["/usr/local/bin", "/usr/bin", "/bin"]
  }
}
```

The hygiene auditor will flag `> /scratch/foo`, `cat > /scratch/foo`, etc.
without cleanup. The auditor system prompt will tell the agent to keep
`/scratch/` clean. The executor system prompt's "Test in <X>" guidance now
references `/scratch`. `/tmp` writes are NOT flagged in this configuration —
the override REPLACES the default rather than extending it.

### NixOS PATH layout

```json
{
  "pathProfile": {
    "systemPathDirs": [
      "/run/current-system/sw/bin",
      "/usr/bin",
      "/bin"
    ]
  }
}
```

Other fields fall back to defaults. The pre-submit self-check now cites
`PATH=/run/current-system/sw/bin:/usr/bin:/bin` as the authoritative
subprocess PATH; the agent-side persistence step suggests
`/run/current-system/sw/bin/X` instead of `/usr/local/bin/X`.

### Extra metadata dirs to exclude from snapshots

```json
{
  "pathProfile": {
    "metadataDirs": [".bag", ".git", ".cache", "node_modules"]
  }
}
```

The snapshot command becomes
`find <cwd> -type f -not -path '*/.bag/*' -not -path '*/.git/*' -not -path '*/.cache/*' -not -path '*/node_modules/*' ...`,
so probe-cleanup will not delete files under those dirs even if a probe
mutated them.

## How modules consume it

All consumers accept `pathProfile` as an OPTIONAL parameter and fall back to
`DEFAULT_PATH_PROFILE` (the canonical Linux conventions exported from
`src/types.ts`) when none is supplied. This keeps every existing call site
byte-equivalent until it is opportunistically threaded through with the
deployment's `BagConfig.pathProfile`.

- `buildVerifierFromInstruction({ router, instruction, pathProfile? })`
  builds the snapshot/restore commands by calling
  `renderFindMetadataExcludes(pathProfile.metadataDirs)`.
- `auditScratchHygiene(trace, pathProfile?)` scans the bash tail using the
  configured `scratchDirs` for writes AND for sweeping cleanup detection.
- `runPreSubmitSelfCheck({ ..., pathProfile? })` interpolates
  `pathProfile.systemPathDirs` into the SUBPROCESS-PATH GATE prompt
  wording, and forwards the same profile into `auditScratchHygiene`.
- `runAutonomousCodingTurn({ ..., pathProfile? })` flows the profile into
  `runPreSubmitSelfCheck`. The default executor system prompt
  (`SYSTEM_PROMPT_DEFAULT`) is now produced by `buildExecutorSystemPrompt(
  DEFAULT_PATH_PROFILE)` so deployments that build the prompt with a
  custom profile via `buildExecutorSystemPrompt(profile)` get matching
  scratch / PATH wording end-to-end.

## Notes

- The descriptive comma-list enumeration of "well-known default-PATH
  locations" inside the SUBPROCESS-PATH GATE prompt
  (`/usr/local/bin, /usr/bin, /bin, /sbin`) stays literal because it is
  prose listing common Linux conventions rather than the authoritative
  subprocess PATH that BAG actually checks against. Deployments that
  need to override the descriptive list can supply a custom
  `systemPromptOverride` via the executor config.
- `DEFAULT_PATH_PROFILE` is exported from `src/types.ts` for direct
  consumption by tests and fall-back paths that cannot reach the
  `BagConfig`. Always prefer threading the live `BagConfig.pathProfile`
  through public APIs; reach for `DEFAULT_PATH_PROFILE` only when the
  config is genuinely unavailable.
