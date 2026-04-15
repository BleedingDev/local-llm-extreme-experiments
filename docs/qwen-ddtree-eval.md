# Qwen3.5 DDTree evaluation (repo-local, concrete status)

Date: 2026-04-14

## 1) What `vendor/ddtree` does and how it pairs with DFlash drafts

`vendor/ddtree` benchmark entrypoint is `vendor/ddtree/benchmark.py`.

- It runs:
  - baseline (`block_size=1`) via `dflash_generate`
  - DFlash speculative decoding (`method_key=dflash`)
  - DDTree speculative decoding (`method_key=ddtree_tb*`) when **not** using `--flash-attn`.
- Draft model class is `vendor/ddtree/model/dflash.py::DFlashDraftModel` (Qwen3-based draft architecture).
- Upstream `vendor/ddtree/run_benchmark.sh` defaults to Qwen3 pairs, not Qwen3.5.

Qwen3.5 pairing check in this repo context:

- Target candidate: `Qwen/Qwen3.5-4B`
  - config `model_type=qwen3_5`
- Draft candidate: `z-lab/Qwen3.5-4B-DFlash`
  - config `model_type=qwen3`

This is the pinned pair used by the new helper script below.

## 2) Exact runnable status on **this** machine

Host checks run locally in this repo:

- `uname -m` → `arm64`
- `sw_vers` → macOS `26.3`
- `nvidia-smi` → not found
- `.venv/bin/python` package probe:
  - `torch`: missing
  - `flash_attn`: missing
  - `ddtree`: missing
  - `dflash`: installed (editable from `vendor/dflash`)
- `scripts/smoke_test.sh --venv-path .venv` passes with warning: missing optional dependency `DDTree`.
- `.venv/bin/python vendor/ddtree/benchmark.py --help` fails immediately:
  - `ModuleNotFoundError: No module named 'torch'`

**Conclusion for this host:** DDTree benchmark is blocked here (no CUDA/NVIDIA stack, no PyTorch CUDA, no flash-attn).

## 3) Practical scaffolding added in this repo

New helper:

- `scripts/run_qwen_ddtree_benchmark.sh`

Purpose:

- Runs DDTree reference benchmark with pinned Qwen3.5 + DFlash pair:
  - target: `Qwen/Qwen3.5-4B`
  - draft: `z-lab/Qwen3.5-4B-DFlash`
- Performs explicit prerequisite checks before launch:
  - CUDA-capable PyTorch present
  - `torch.cuda.is_available()` true
  - NCCL available (`torch.distributed`)
  - `flash_attn` importable
  - target/draft model configs resolvable and draft `model_type=qwen3`
- Emits clear actionable failure text when checks fail.

## 4) Exact commands for a CUDA-capable environment

Use a Linux CUDA box (not this Apple Silicon machine):

```bash
cd /path/to/supergemma-dflash-ddtree-mlx

# Prefer Python 3.10-3.12 for CUDA wheel availability
python3.11 -m venv .venv-cuda
source .venv-cuda/bin/activate
python -m pip install --upgrade pip setuptools wheel

# Repo requirements (mlx lines are Darwin-gated and will be skipped on Linux)
python -m pip install -r requirements.txt

# Install CUDA PyTorch (example index; pick version matching host CUDA driver)
python -m pip install torch --index-url https://download.pytorch.org/whl/cu128

# DDTree runtime deps
python -m pip install -r vendor/ddtree/requirements.txt

# Optional: pin vendor sources from this repo scripts
scripts/fetch_vendor_sources.sh --component ddtree

# Sanity (should report CUDA + flash_attn + NCCL ready)
NPROC_PER_NODE=1 scripts/run_qwen_ddtree_benchmark.sh \
  --venv-path .venv-cuda \
  --dataset gsm8k \
  --max-samples 1 \
  --dry-run

# Actual run
NPROC_PER_NODE=1 scripts/run_qwen_ddtree_benchmark.sh \
  --venv-path .venv-cuda \
  --dataset gsm8k \
  --max-samples 16 \
  --max-new-tokens 256 \
  --tree-budget 16,32,64,128
```

For multi-GPU:

```bash
NPROC_PER_NODE=8 MASTER_PORT=29600 scripts/run_qwen_ddtree_benchmark.sh \
  --venv-path .venv-cuda \
  --dataset gsm8k \
  --max-samples 128
```

Outputs are written under `artifacts/ddtree/{logs,runs}` by default.

## 5) What is required to integrate DDTree with current MLX flow

Current MLX path (`scripts/run_dflash_mlx_benchmark.sh`) is separate from DDTree and does not execute DDTree tree verification logic.

To integrate DDTree into MLX flow, you would need all of:

1. Port `vendor/ddtree/ddtree.py` generation algorithm to MLX (`mx`) tensors.
2. Re-implement DDTree tree-attention mask compilation + cache compaction on top of `mlx_lm` cache APIs.
3. Add a DDTree-capable MLX benchmark/inference path (today only DFlash MLX exists).
4. Validate Qwen3.5 parity (acceptance lengths, throughput, correctness) against CUDA reference outputs.

So, in the current repo state: **DDTree is a CUDA/PyTorch track; MLX remains a separate DFlash-only track.**
