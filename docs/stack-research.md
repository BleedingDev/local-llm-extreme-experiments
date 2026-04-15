# Stack research: DDTree + DFlash + MLX + SuperGemma 26B A4B (MacBook Pro M5 32GB)

## Executive summary
- **SuperGemma 26B 4-bit on MLX is feasible** on Apple Silicon M5 with 32GB unified memory.
- **Full requested stack (DDTree + DFlash + MLX + SuperGemma)** is **not currently achievable as-is**:
  - DFlash (commit `c95d242`) has no Gemma4 draft models and no DDTree integration.
  - DDTree upstream is CUDA/PyTorch-oriented, not MLX/Apple-Silicon.
- Practical immediate path: run **SuperGemma via `mlx_lm`**, and treat DFlash/DDTree as separate R&D tracks.

## Sources reviewed
- X post: https://x.com/liranringel/status/2043813397972607477
- DDTree project page/paper/code:
  - https://liranringel.github.io/ddtree
  - https://github.com/liranringel/ddtree
- DFlash repo at exact commit:
  - https://github.com/z-lab/dflash/tree/c95d24215a3e63f10c9d4dbe826b80e85772c1ae
- SuperGemma MLX model card:
  - https://huggingface.co/Jiunsong/supergemma4-26b-uncensored-mlx-4bit-v2
- MLX / MLX-LM docs and metadata:
  - https://github.com/ml-explore/mlx-lm
  - https://github.com/ml-explore/mlx
  - PyPI JSON for `mlx`, `mlx-lm`

## Compatibility matrix

| Stack slice | Status on M5/32GB | Notes |
|---|---|---|
| MLX + `mlx-lm` + `Jiunsong/supergemma4-26b-uncensored-mlx-4bit-v2` | ✅ Supported | Model is tagged `library_name: mlx`; has MLX 4-bit weights and chat template. |
| DFlash `c95d242` + MLX backend + Qwen3/Qwen3.5 DFlash drafts | ✅ Supported | README says MLX backend tested on Apple M5 Pro for Qwen3/Qwen3.5. |
| DFlash `c95d242` + SuperGemma (Gemma4) on MLX | ❌ Not supported | No Gemma DFlash draft model listed; MLX code path is Qwen-oriented and expects DFlash draft config. |
| DDTree (liranringel/ddtree) + CUDA PyTorch + DFlash draft models | ✅ Supported (CUDA env) | Upstream explicitly targets CUDA-enabled PyTorch + `flash-attn`. |
| DDTree on Apple Silicon MLX | ❌ Not supported | Upstream uses `torch.cuda.*`, `flash-attn`, CUDA assumptions. |
| DDTree integrated inside z-lab/dflash `c95d242` | ❌ Not present | No DDTree method/CLI/config in that commit. |

## Required software versions (realistic baseline)

### For SuperGemma on Mac (recommended immediate path)
- **macOS**: `>= 15.0` (MLX-LM “large models” wiring path requirement)
- **Python**: `>= 3.10` (current `mlx` requires >=3.10)
- **mlx**: `0.31.1` (current stable from PyPI)
- **mlx-lm**: `0.31.2` (current stable from PyPI)
- Extra runtime deps: `huggingface_hub`, `sentencepiece`, `safetensors`

### For DFlash repo @ `c95d242`
- `requires-python >= 3.10`
- MLX extra in `pyproject.toml`: `mlx`, `mlx-lm>=0.31.2`
- Transformers extra pins `transformers==4.57.1` (CUDA path)

### For DDTree repo (separate upstream)
- CUDA-enabled PyTorch environment
- `flash-attn` (required by their benchmark path)
- Not a Mac/MLX-native stack today

## Feasibility on M5 32GB unified memory

### Model footprint facts (SuperGemma MLX repo)
- `model.safetensors.index.json` reports:
  - `metadata.total_size = 14,200,055,868 bytes` (~13.23 GiB)
  - `metadata.total_parameters = 25,233,053,440`
- 3 shard files total roughly 14.2 GB on disk/cache (plus tokenizer/templates/metadata).

### Runtime memory estimate
Given config (`num_hidden_layers=30`, `num_key_value_heads=8`, `head_dim=256`, bf16 KV cache):
- Approx KV cache per token:
  - `30 * 8 * 256 * 2 (K+V) * 2 bytes ≈ 245,760 bytes/token` (~0.234 MiB/token)

Approx KV usage by context size:
- 4k tokens: ~0.94 GiB
- 8k tokens: ~1.88 GiB
- 16k tokens: ~3.75 GiB
- 32k tokens: ~7.50 GiB

Practical total memory on 32GB machine:
- Weights (~13.2 GiB) + runtime overhead + KV cache + OS/apps
- Realistic comfortable zone: **short/medium contexts (e.g., 4k–8k)**
- Long contexts increase paging risk sharply; use `--max-kv-size` to cap.

### Performance expectation
- Model card reports ~`46.2 tok/s` average generation speed (author benchmark).
- On M5 32GB, practical expectation is typically in that broad band for short contexts, with drop-off as context and system pressure rise.

### Constraints and caveats
- SuperGemma release is text-focused despite Gemma4 family multimodal roots.
- Avoid incorrect chat-template wiring; model notes warn passing a path string where template body is expected can corrupt responses.
- MLX-LM large-model performance can depend on wired-memory behavior (macOS 15+).

## DDTree integration status in DFlash @ `c95d242`

### What exists in this DFlash commit
- vLLM path uses speculative config with `"method": "dflash"`
- SGLang path uses `--speculative-algorithm DFLASH`
- MLX local API path:
  ```python
  from dflash.model_mlx import load, load_draft, stream_generate
  ```
- Benchmark CLI choices are only: `transformers`, `sglang`, `vllm`, `mlx`

### What does *not* exist in this commit
- No `DDTREE` algorithm switch
- No `ddtree_generate` import
- No DDTree CLI/config flag

### Where DDTree is actually invoked upstream
In `liranringel/ddtree` (separate repo), `benchmark.py` imports:
```python
from ddtree import ddtree_generate
```
and drives tree mode via `--tree-budget`.
That is a **separate CUDA research harness**, not integrated into z-lab/dflash `c95d242`.

## Recommended setup strategy (no global pollution)

Use a project-local venv only:

```bash
cd /Users/satan/side/experiments/supergemma-dflash-ddtree-mlx
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install "mlx==0.31.1" "mlx-lm==0.31.2" \
  "huggingface_hub>=0.23.0" "sentencepiece>=0.2.0" "safetensors>=0.4.3"
```

Optional (if you need exact research refs locally without polluting global env):
- Keep DFlash/DDTree checkouts under `vendor/` and install editable into `.venv` only.
- Do **not** mix CUDA and MLX assumptions in one runtime plan.

## Concrete run commands (actionable now)

### 1) Quick local generation test (recommended first)
```bash
source .venv/bin/activate
mlx_lm.generate \
  --model Jiunsong/supergemma4-26b-uncensored-mlx-4bit-v2 \
  --prompt "Write a concise Python function for prime numbers up to n." \
  --max-tokens 256 \
  --max-kv-size 512
```

If you hit Metal out-of-memory errors, keep `--max-kv-size` low (start at `512`) and increase gradually.

### 2) OpenAI-compatible local server
```bash
source .venv/bin/activate
mlx_lm.server \
  --model Jiunsong/supergemma4-26b-uncensored-mlx-4bit-v2 \
  --port 8080
```

Health/request check:
```bash
curl -s http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Say hello in Czech."}],"max_tokens":64}'
```

### 3) Optional wired-memory tuning (macOS 15+)
Set only if you observe large-model warning/slowdowns:
```bash
sudo sysctl iogpu.wired_limit_mb=24576
```
(Keep below total RAM; adjust carefully.)

### 4) DFlash on MLX (separate, Qwen-only path)
Only meaningful with supported Qwen target+draft pair, e.g.:
```bash
python -m dflash.benchmark --backend mlx \
  --model Qwen/Qwen3.5-4B \
  --draft-model z-lab/Qwen3.5-4B-DFlash \
  --dataset gsm8k --max-samples 16
```

### 5) DDTree (not for Mac MLX; CUDA-only reference)
```bash
python benchmark.py \
  --model-name-or-path Qwen/Qwen3-8B \
  --draft-name-or-path z-lab/Qwen3-8B-DFlash-b16 \
  --dataset gsm8k --max-samples 16 --tree-budget 64
```

## Risks and fallback options

### Primary risks
1. **Architecture mismatch**: SuperGemma (Gemma4) cannot directly use current open DFlash drafts (Qwen/Llama focused).
2. **Backend mismatch**: DDTree upstream currently assumes CUDA stack, not MLX.
3. **Memory pressure** on 32GB with long contexts.

### Fallbacks
- **Fallback A (recommended now):** run SuperGemma directly on MLX (`mlx_lm.generate` / `mlx_lm.server`) with controlled KV size.
- **Fallback B:** if speculative decoding is required on Mac now, use MLX-LM native draft-model speculative path (non-DDTree/DFlash), if a compatible small draft model is available.
- **Fallback C:** if DDTree is mandatory, run it in a CUDA environment with Qwen + DFlash drafts.
- **Fallback D (R&D):** port DDTree algorithm to MLX and train/convert a Gemma4-specific DFlash draft model.

## Handoff checklist for implementation agent
1. Keep `.venv` local and pinned to MLX stack above.
2. Validate SuperGemma startup with a short `mlx_lm.generate` smoke test.
3. Bring up `mlx_lm.server` on `:8080` and verify OpenAI-compatible response.
4. Explicitly document that DDTree+DFlash acceleration for SuperGemma is currently unsupported in this commit set.
5. If acceleration is still required, pick one path: CUDA DDTree pipeline vs. MLX-native speculative fallback.
