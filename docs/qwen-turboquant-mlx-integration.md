# Qwen3.5 + turboquant-mlx integration

Date: 2026-04-14

## Scope

Integrated `sharpner/turboquant-mlx` into this repo as a reproducible local workflow (`.venv` + pinned `vendor/` checkout), then executed real Qwen3.5 runs locally on MLX.

Pinned upstream source:
- repo: `https://github.com/sharpner/turboquant-mlx.git`
- commit: `aca1ceb734f75d2646b3d0178f4bd11eac185445`

---

## 1) Upstream research summary (sharpner/turboquant-mlx)

### Runtime/tooling expectations

From upstream README + code:
- target platform: Apple Silicon + MLX (`mlx`, `mlx-lm`)
- Python: 3.10+
- deps (`requirements.txt`): `mlx>=0.22.0`, `mlx-lm>=0.21.0`, `numpy`, `pytest`
- integration model: monkey-patch `mlx_lm.models.base.scaled_dot_product_attention` via `turboquant.patch.apply()`

### Main APIs

- `turboquant.cache_v2.TurboQuantKVCacheV2`
  - fast path with `mx.quantized_matmul`
  - key args: `bits`, `group_size`, `use_rotation`, `use_normalization`, `use_qjl`
- `turboquant.cache_v3.TurboQuantKVCacheV3`
  - Lloyd-Max/codebook path
  - key args: `bits`, `n_outlier`, `outlier_bits`, `use_qjl`
- `turboquant.patch.apply()`
  - routes SDPA to TurboQuant attention kernels when TurboQuant cache objects are present

### Supported quant modes exposed by upstream code

Common practical modes:
- V2 fast: `tqv2_4bit_lean`, `tqv2_4bit_rot`, `tqv2_3bit_rot_qjl`
- V3 quality/compression: `tqv3_3bit`, mixed-bit (`3.5`, `3.25`, `2.75`, `2.5`)

Upstream benchmark model set (README/scripts):
- Llama 3.2 3B, Llama 3.1 8B, Mistral 7B, Gemma 3 4B

Qwen3.5 is not listed upstream, so local verification was required (done below).

---

## 2) Reproducible local workflow in this repo

### Added scripts

- `scripts/setup_turboquant_mlx.sh`
  - creates/updates `.venv`
  - fetches pinned `vendor/turboquant-mlx`
  - installs root + turboquant requirements into `.venv`
  - writes local `.pth` mapping for `vendor/turboquant-mlx`
  - validates imports/strategy registry

- `scripts/run_qwen_turboquant_mlx.sh`
  - runs Qwen3.5 with selected TurboQuant strategy
  - supports `--compare-fp16`
  - emits machine-readable output via `--output-json`

Also updated source pinning:
- `scripts/pinned_sources.sh`
- `scripts/fetch_vendor_sources.sh` (`turboquant-mlx` component)

---

## 3) Exact setup/run commands

### Setup (local-only)

```bash
scripts/setup_turboquant_mlx.sh
```

### Quick config check

```bash
scripts/run_qwen_turboquant_mlx.sh --dry-run --strategy tqv2_4bit_lean --compare-fp16
```

### Real local execution (Qwen3.5)

Throughput-oriented variant:

```bash
scripts/run_qwen_turboquant_mlx.sh \
  --strategy tqv2_4bit_lean \
  --compare-fp16 \
  --max-tokens 24 \
  --output-json artifacts/benchmarks/qwen-turboquant-mlx-<timestamp>/result.json
```

Higher KV compression variant:

```bash
scripts/run_qwen_turboquant_mlx.sh \
  --strategy tqv3_3.5bit_mixed \
  --compare-fp16 \
  --max-tokens 24 \
  --output-json artifacts/benchmarks/qwen-turboquant-mlx-<timestamp>-v3/result.json
```

---

## 4) Local execution evidence

Executed successfully on this machine:

1) `tqv2_4bit_lean` run  
Artifacts:
- `artifacts/benchmarks/qwen-turboquant-mlx-20260414T193913Z/run.log`
- `artifacts/benchmarks/qwen-turboquant-mlx-20260414T193913Z/result.json`

Key results:
- TurboQuant-applied layers: 8 (`[3,7,11,15,19,23,27,31]`)
- KV-layer compression vs fp16 (those layers): **3.56x**
- Total cache bytes vs fp16 baseline: **-13.24%**
- tok/s ratio vs fp16: **0.82x**

2) `tqv3_3.5bit_mixed` run  
Artifacts:
- `artifacts/benchmarks/qwen-turboquant-mlx-20260414T193928Z-v3/run.log`
- `artifacts/benchmarks/qwen-turboquant-mlx-20260414T193928Z-v3/result.json`

Key results:
- KV-layer compression vs fp16 (those layers): **4.27x**
- Total cache bytes vs fp16 baseline: **-11.64%**
- tok/s ratio vs fp16: **0.18x**

---

## 5) Practical recommendation for this repo

- Use `tqv2_4bit_lean` for best local speed/benefit balance on Qwen3.5.
- Use `tqv3_3.5bit_mixed` only when testing aggressive KV compression and accepting major throughput loss.

### Important limitation (not a hard blocker)

Qwen3.5 in MLX uses hybrid layers. Only layers backed by `KVCache` are replaceable by turboquant-mlx (8/32 here).  
So end-to-end memory savings are bounded even when KV-layer compression is strong.
