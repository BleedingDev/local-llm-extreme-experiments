# Proposal: Intra-Tool Argument/Flag Histograms (Brainstorm R7 #BB)

## TLDR
- **Two-tier decision**: picking the *tool* (`Bash:git`) is decision A; picking the *args* (`log -50 --oneline`) is decision B. Models often nail A and botch B (e.g., bare `git log` -> SIGPIPE/timeout 141, the largest failure bucket at 243 records in v2).
- **Mine traces for per-verb flag/sub-command n-grams**: for every `Bash` invocation extract first 1-3 tokens, build top-20 histograms keyed by verb (`git`, `bun`, `rg`, `find`, `gh`, `npm`, `docker`, `psql`, ...).
- **Three downstream uses**: (i) inject "user-typical" few-shots into prompts, (ii) add a tier-1 verifier that hard-rejects predicted commands missing critical flags (timeout-prone), (iii) expose as MCP resource `bash-args://verb/{verb}` for live retrieval.
- **Distinctness**: orthogonal to R1 (cross-tool MCP recipes) and R6 (Tool-A -> Tool-B transitions). This is *within-one-Bash-call* token statistics — a different axis of the same trace.

## Hypothesis
Tool selection and tool argument selection are statistically independent learnable distributions. Mining each separately yields a richer prior than treating `Bash:git log -50 --oneline` as a single opaque event. In particular, *catastrophic flags* (those whose absence causes timeout/OOM/SIGPIPE) are a small enumerable set we can verify.

## Concrete Output
1. **Extractor**: tokenize `tool_input.command` on whitespace (respecting quotes), drop pipes/redirects, capture `(verb, sub, flag1, flag2)`.
2. **Histogram**: `data/tool_args_histograms.json`:
   ```json
   { "git": { "log -50 --oneline": 312, "log --oneline": 280, "status": 1840, "diff --stat": 95, ... },
     "bun": { "test": 410, "run build": 188, "install": 76, ... },
     "rg":  { "-n <pat>": 920, "--files -g": 144, ... } }
   ```
3. **Optional Markov chain**: `P(token_{i+1} | verb, token_i)` for autocomplete-style biasing during decoding.
4. **Critical-flag table** (hand-curated from histogram tail-failures): `git log` REQUIRES `-N` or `--oneline | head`; `find /` REQUIRES `-maxdepth`; `docker logs` REQUIRES `--tail`; `psql` REQUIRES `-c` or `-f`.

## Use Cases
- **Few-shots**: per task, top-3 histogram entries for each verb the task likely needs.
- **Verifier**: regex predicate `^git log(?!.*(-\d|--oneline.*head|-n ))` -> score 0.
- **MCP resource**: `bash-args://verb/git` returns ranked list with frequencies.

## Effort + ROI
- **Effort**: ~120 LOC tokenizer + JSON dump; verifier predicates are 10-line regex table. Half-day.
- **ROI**: directly attacks `bash_timeout_141` (243 records, ~largest single bucket). Expected lift: 30-60% reduction in that category since most timeouts are missing-flag, not genuine slow tools.

## Self-Critique
Histograms reflect *successful* past usage, so rare-but-correct flag combos get filtered out — verifier should warn-not-block, and we must refresh the histogram per repo (a `bun` monorepo's typical args differ from a Python repo's).

---
**Path**: `trace-gepa/proposals/tool_input_flags.md`
**Self-critique (1 sentence)**: The proposal optimizes for *modal* arg patterns and may bias the model away from legitimate niche flag combinations, so the verifier must be advisory (penalty) rather than hard-reject for non-timeout-causing cases.
