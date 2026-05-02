# Persona Profile

- sessions: 220  records: 30313
- datasets: data/dataset.jsonl, data/dataset_v2.jsonl

## Tools (top 10)
- Bash: 13257
- exec_command: 5039
- Read: 4393
- Edit: 3143
- SendMessage: 1209
- Write: 1048
- ToolSearch: 840
- write_stdin: 824
- spawn_agent: 162
- TaskCreate: 121

## Bash verbs (top 20)
- git: 3470
- zig: 2895
- grep: 2173
- ls: 1045
- cat: 463
- pnpm: 383
- find: 297
- wc: 231
- python3: 146
- tail: 112
- sleep: 112
- rg: 108
- head: 97
- sed: 90
- mkdir: 86
- time: 74
- ps: 74
- timeout: 68
- which: 67
- until: 51

## Path prefixes
- /Users/satan/side/experiments/: 18381
- ~/.claude/: 92
- /Users/satan/.claude/: 91
- ~/.codex/sessions/: 23
- /Users/satan/.codex/sessions/: 19

## Czech corrective tokens
- vůbec: 80
- lepší: 64
- nene: 50
- počkej: 48
- fakt: 26
- pozor: 14

## Skills (top 10)
- subagent-graph: 2
- plan-graph: 2
- helm: 2
- frontend-design: 2

## Subagents (top 10)
- explorer: 162

## Repos (top 5)
- ir-multivector-retrieval: 14702
- ir-expo: 2739
- supergemma-dflash-ddtree-mlx: 426
- bonsai-android-llamacpp: 202
- codex-native: 175

## Bash flag combos (top 5 per top verb)
- **git**: `cd && git log` x659, `cd && git status` x551, `cd && git add` x420, `cd && git stash` x362, `cd && git diff` x302
- **zig**: `cd && zig build` x2451, `cd && zig test` x334, `cd && zig version` x80, `cd && zig fmt` x28, `zig version 2>&1; which` x2
- **grep**: `cd && grep -n` x303, `grep -n "pub fn` x303, `grep -n | head` x218, `grep -n` x73, `grep -rn | head` x54
- **ls**: `ls 2>&1` x173, `ls` x141, `ls 2>&1 | head` x137, `ls -la` x47, `cd && ls &&` x44
- **cat**: `cat 2>&1 | head` x78, `cat` x61, `cat 2>&1 | tail` x40, `cd && cat` x30, `cat > <<'EOF' const` x27
- **pnpm**: `pnpm exec vp check` x151, `pnpm exec vp test` x96, `pnpm exec vp lint` x30, `pnpm exec tsc --noEmit` x26, `pnpm typecheck 2>&1 |` x21
- **find**: `find -path | head` x78, `find -name -path "*std*"` x30, `find -type f -name` x25, `find -maxdepth 3 -type` x23, `find -name -type f` x19
- **wc**: `cd && wc -l` x137, `wc -l` x60, `wc -l 2>&1` x12, `wc -c` x6, `wc -l && grep` x4

## Failure-recovery (top 5)
- `Bash` -> `Bash` x2000
- `exec_command` -> `exec_command` x600
- `Bash` -> `Read` x452
- `Bash` -> `Edit` x252
- `Bash` -> `SendMessage` x121

## Czech sample contexts
- `počkej`: 'Počkej, Michael už to dělá, takže jen pullni změny.'
- `počkej`: 'Počkej, Michael už to dělá, takže jen pullni změny.'
- `pozor`: 'rkovat našeho vlastního agenta BAG. Ale pozor, aktivně se tu vyvíjí, je tu další agent, který implem'
- `pozor`: 'rkovat našeho vlastního agenta BAG. Ale pozor, aktivně se tu vyvíjí, je tu další agent, který implem'
- `pozor`: 'rkovat našeho vlastního agenta BAG. Ale pozor, aktivně se tu vyvíjí, je tu další agent, který implem'
- `pozor`: 'rkovat našeho vlastního agenta BAG. Ale pozor, aktivně se tu vyvíjí, je tu další agent, který implem'
