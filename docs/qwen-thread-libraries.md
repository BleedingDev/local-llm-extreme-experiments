# Qwen 3.5 optimization libraries from X thread

Thread: https://x.com/mc1_e/status/2044065179374088457

## Source access + certainty

- Direct `x.com` rendering was partially inaccessible from this environment.
- Fallback used:
  1. `r.jina.ai/http://x.com/...` mirror to read tweet text.
  2. `curl -I` on each `t.co` URL to resolve real links.
- Resolved links from the thread:
  - https://github.com/WeianMao/triattention
  - https://github.com/scrya-com/rotorquant
  - https://github.com/rookiemann/multi-turboquant
  - https://github.com/czg1225/DMax
- Additional immediate referenced links used for evaluation:
  - https://github.com/liranringel/ddtree (explicitly requested)
  - https://github.com/TheTom/llama-cpp-turboquant (TurboQuant implementation reference)
  - https://github.com/ParaMind2025/isoquant (referenced by RotorQuant)

Certainty legend:
- **High**: verified directly from repo README/code/deps.
- **Medium**: repo claims are clear, but no local reproduction run in this task.
- **Low**: incomplete/inaccessible primary source.

## Environment assumed for verdicts

- Apple Silicon **M5**, **32GB unified memory**.
- Existing repo stack: **MLX + mlx-lm + DFlash Qwen path**.
- Local DFlash pin supports MLX benchmark path and has no integrated DDTree switch in this repo state.

## Libraries/ideas and compatibility verdict

| Library / idea | Primary area | What it does | Verdict | Why for this M5 + MLX + DFlash setup | Certainty |
|---|---|---|---|---|---|
| **DDTree** (`liranringel/ddtree`) | Speculative decoding | Block diffusion draft tree decoding (`ddtree_generate`) on top of draft+target models. | **not feasible in this environment** | Repo/setup is CUDA + PyTorch + `flash-attn`; code uses `torch.cuda.*`; local target stack is Apple MLX. | High |
| **TurboQuant** (WHT KV compression; `TheTom/llama-cpp-turboquant`) | KV cache quantization | Walsh–Hadamard KV-cache compression (`turbo*` cache types) in llama.cpp forks. | **feasible with major effort** | Requires moving Qwen3.5 inference path from MLX/DFlash to llama.cpp + GGUF workflow and validating Metal behavior/perf on M5. | Medium |
| **TriAttention** (`WeianMao/triattention`) | KV cache eviction/compression | Trigonometric token scoring + KV budgeted retention; vLLM plugin; experimental MLX hook documented. | **feasible with major effort** | MLX support is marked experimental; package depends on Torch/Triton/FlashAttn-oriented stack; integration with existing DFlash MLX flow is non-trivial and likely requires patching/calibration for Qwen3.5. | Medium |
| **RotorQuant** (`scrya-com/rotorquant`) | KV cache quantization + runtime kernels | IsoQuant/PlanarQuant (block rotations) as faster alternatives to TurboQuant; llama.cpp integration path shown. | **feasible with major effort** | Practical route is llama.cpp (Metal) track, not current MLX runtime. Requires separate inference stack, conversion/testing, and quality validation on Qwen3.5. | Medium |
| **Multi-TurboQuant** (`rookiemann/multi-turboquant`) | Unified quantization toolkit + planner | Single API over TurboQuant/TCQ/Iso/Planar/TriAttention, command generation for llama.cpp/vLLM, capacity planning. | **feasible with major effort** | Mac table in repo limits runtime methods to iso/planar in llama.cpp path. Useful as orchestration layer, but still needs migration away from MLX runtime for real speedups. | Medium |
| **DMax** (`czg1225/DMax`) | Alternative decoding paradigm / model family | Aggressive parallel decoding for diffusion-style dLLMs (LLaDA-derived models, custom training/inference stack). | **not feasible in this environment** | Not a drop-in Qwen3.5 optimization; targets different model family and CUDA-centric training/eval stack (sglang/vLLM paths). | High |
| **IsoQuant / PlanarQuant** (`ParaMind2025/isoquant`, via RotorQuant refs) | KV cache quantization | Blockwise rotational quantization (4D/2D) that RotorQuant and related forks use. | **feasible with major effort** | Standalone repo is prototype/CUDA-focused; practical Mac route is via llama.cpp integrations, not current MLX+DFlash stack. | Medium |

## Classification rollup

### feasible now
- **None as a direct drop-in** to the current MLX + DFlash Qwen3.5 runtime without architecture/runtime migration.

### feasible with major effort
- TurboQuant
- TriAttention
- RotorQuant
- Multi-TurboQuant
- IsoQuant / PlanarQuant

### not feasible in this environment
- DDTree
- DMax

## Recommended experiment order for this repo

1. **Keep MLX+DFlash Qwen3.5 as baseline** (already working in this repo) and lock current metrics/artifacts.
2. **Create a separate llama.cpp-Metal branch** for Qwen3.5 GGUF and test **RotorQuant/Iso/Planar** first (best Mac-fit among thread links).
3. Add **Multi-TurboQuant** only as a planner/config layer after step 2 works, using Mac-supported methods (`iso*`/`planar*`).
4. Run a **TriAttention exploratory branch** only if you are willing to patch dependencies and generate/validate Qwen3.5 calibration stats; treat as experimental.
5. Treat **TurboQuant WHT/TCQ** as optional follow-up in the llama.cpp track (verify real Metal support/perf before deep work).
6. Keep **DDTree** and **DMax** as off-device CUDA research tracks (remote Linux GPU), not local M5 objectives.

## Notes

- The thread extraction itself is high confidence (resolved via `t.co` redirects).
- Performance claims in linked READMEs were not reproduced locally in this task; they should be treated as provisional until benchmarked on this machine.

## Post-research validation update (implemented in this repo)

After this research pass, we implemented and measured additional paths locally:

1. `sharpner/turboquant-mlx` was integrated and validated on Qwen3.5 in this repo:
   - setup: `scripts/setup_turboquant_mlx.sh`
   - run: `scripts/run_qwen_turboquant_mlx.sh`
   - details: `docs/qwen-turboquant-mlx-integration.md`
2. Experimental DDTree-style MLX prototype was implemented:
   - run: `scripts/run_qwen_ddtree_mlx_prototype.sh`
   - details: `docs/qwen-ddtree-mlx-prototype.md`

Updated practical status:
- **feasible now (experimental):** turboquant-mlx (partial-layer impact on Qwen3.5), DDTree-style MLX prototype
- **feasible with major effort:** RotorQuant / Multi-TurboQuant / IsoQuant / TriAttention production-grade integration
- **not feasible locally:** CUDA DDTree reference path, DMax full stack
