# Qwen3.5 DDTree-style MLX prototype

Date: 2026-04-14

## What was implemented

This repo now includes an **experimental, runnable DDTree-style MLX path**:

- `scripts/run_qwen_ddtree_mlx_prototype.py`
- `scripts/run_qwen_ddtree_mlx_prototype.sh` (wrapper)

The prototype uses:

- target model: `Qwen/Qwen3.5-4B` (default)
- draft model: `z-lab/Qwen3.5-4B-DFlash` (default)
- existing MLX + DFlash stack (`dflash.model_mlx.load`, `load_draft`, hidden-state patching, MLX sampler/cache utilities)

## Prototype algorithm (approximation)

Per decoding round:

1. Run target on current context to get:
   - first target token sample for this round
   - hidden states used as DFlash draft context feature
2. Run draft once on `[root_token] + [mask]*depth`.
3. Build a bounded candidate tree from draft logits:
   - `tree_budget` = max retained candidate paths
   - `depth` = expansion depth
4. Verify candidate paths against target:
   - match candidate prefix token-by-token
   - choose candidate with longest accepted prefix (score tie-break)
5. Commit:
   - accepted draft prefix
   - plus one fallback token from target

## Exposed knobs

- `--tree-budget`
- `--depth`
- `--max-new-tokens`
- `--max-kv-size`
- `--temperature`

Wrapper exposes the same defaults with `DDTREE_MLX_*` env overrides.

## Metrics + artifacts

Each run writes artifacts under `artifacts/ddtree-mlx-prototype/runs/<run-name>/`:

- `config.json`
- `prompt.txt`
- `generation.txt`
- `round_traces.jsonl`
- `result.json`

`result.json` includes:

- generation tokens/sec
- accepted length stats
- candidate count stats
- fallback event counts/rate and reasons
- target/draft forward call counts
- total context tokens dropped by `max_kv_size`
- peak memory (when available from MLX)

## Limitations vs full DDTree (explicit)

This is **not** paper-parity DDTree:

1. No fused one-pass tree-attention verification mask like CUDA DDTree.
2. Candidate verification is sequential per candidate path on MLX.
3. Tree expansion uses depth-wise draft logits from a single draft pass (branch-conditioned tree logits are not implemented).
4. Throughput should be treated as prototype instrumentation, not final DDTree performance.

## Run commands

Wrapper:

```bash
scripts/run_qwen_ddtree_mlx_prototype.sh \
  --prompt "Answer in one short Czech sentence." \
  --tree-budget 4 \
  --depth 2 \
  --max-new-tokens 12 \
  --max-kv-size 512 \
  --temperature 0.0
```

Direct Python:

```bash
.venv/bin/python scripts/run_qwen_ddtree_mlx_prototype.py \
  --model Qwen/Qwen3.5-4B \
  --draft-model z-lab/Qwen3.5-4B-DFlash \
  --prompt "Answer in one short Czech sentence." \
  --tree-budget 4 \
  --depth 2 \
  --max-new-tokens 12 \
  --max-kv-size 512 \
  --temperature 0.0
```

## Local best prototype invocation captured

Short local variant search (no long baselines) was run across:

- `tree_budget=4, depth=2`
- `tree_budget=2, depth=2`
- `tree_budget=4, depth=1` (**best**)

Best invocation:

```bash
scripts/run_qwen_ddtree_mlx_prototype.sh \
  --prompt "Reply with one short Czech sentence." \
  --tree-budget 4 \
  --depth 1 \
  --max-new-tokens 6 \
  --max-kv-size 512 \
  --temperature 0.0 \
  --run-name final-b41
```

Best run summary (`final-b41`):

- generation throughput: **6.45 tok/s**
- accepted length mean: **1.0**
- fallback events: **0**
- peak memory: **9.587 GB**

Artifacts:

- `artifacts/ddtree-mlx-prototype/runs/final-b41/result.json`
- `artifacts/ddtree-mlx-prototype/runs/final-b41/round_traces.jsonl`
- `artifacts/ddtree-mlx-prototype/runs/final-b41/generation.txt`

## TODO toward closer DDTree parity

- Branch-conditioned multi-step draft expansion with cache reuse.
- Batched or fused tree verification pass on MLX.
- Cache compaction/rollback semantics closer to upstream DDTree.
- Accuracy/perf parity checks against CUDA DDTree reference traces.
