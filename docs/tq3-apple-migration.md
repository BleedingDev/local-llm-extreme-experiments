# TQ3 Apple Migration Plan

Date: 2026-04-22

## Goal

Run `Nishant2414/Qwen3.6-35B-A3B-TQ3_4S` on Apple Silicon through a real `llama.cpp` backend path, without:

- changing the model
- converting to a different quant format
- CPU fallback as the primary execution path
- monkey-patching around missing backend support

The target is a first-class Metal implementation for:

- `TQ3_4S` weights
- `TQ3_1S` weights
- `TQ3_0` KV cache
- Flash Attention on Metal where the runtime already expects it

## What is true today

### 1. The fork builds on Apple already

`llama.cpp-tq3` configures and builds with `GGML_METAL=ON` on this machine.

That means the blocker is not CMake or platform bring-up. The blocker is missing `TQ3_*` coverage in the Metal backend.

### 2. TQ3 is implemented in generic and CPU code

The fork already has:

- `ggml/src/ggml-common.h`
  - `block_tq3_0`
  - `block_tq3_1s`
  - `block_tq3_4s`
- `ggml/src/ggml-quants.c`
  - `dequantize_row_tq3_0`
  - `dequantize_row_tq3_1s`
  - `dequantize_row_tq3_4s`
- `ggml/src/ggml-cpu/quants.c`
  - `ggml_vec_dot_tq3_0_q8_0`
  - `ggml_vec_dot_tq3_1s_q8_0`
  - `ggml_vec_dot_tq3_4s_q8_0`

So the format semantics are already stable and portable. Metal does not need new math research. It needs a backend port.

### 3. The high-performance TQ3 path is CUDA-only today

The CUDA fork adds TQ3-specific acceleration in:

- `ggml/src/ggml-cuda/mmq.cu`
- `ggml/src/ggml-cuda/mmq.cuh`
- `ggml/src/ggml-cuda/tq3-native.cuh`
- `ggml/src/ggml-cuda/convert.cu`
- `ggml/src/ggml-cuda/fattn-common.cuh`
- `ggml/src/ggml-cuda/getrows.cu`
- `ggml/src/ggml-cuda/set-rows.cu`

That is the code we need to functionally migrate to Metal.

### 4. Metal currently rejects or lacks TQ3 across the critical surfaces

Evidence in `ggml/src/ggml-metal`:

- `ggml-metal-device.cpp`
  - `ggml_metal_library_get_pipeline_mul_mv(...)` has no `GGML_TYPE_TQ3_0`, `GGML_TYPE_TQ3_1S`, or `GGML_TYPE_TQ3_4S`
  - `ggml_metal_library_get_pipeline_mul_mv_id(...)` has no TQ3 cases
  - pipeline names exist only for already-instantiated kernels
- `ggml-metal-device.m`
  - `GGML_OP_FLASH_ATTN_EXT` allows only `F32`, `F16`, `BF16`, `Q8_0`, `Q4_0`, `Q4_1`, `Q5_0`, `Q5_1`
  - `CPY`, `SET_ROWS`, and related type gates do not include TQ3
- `ggml-metal.metal`
  - no `kernel_mul_mv_*` instantiations for TQ3
  - no `kernel_mul_mm_*` instantiations for TQ3
  - no `kernel_mul_mm_id_*` instantiations for TQ3
  - no `kernel_get_rows_tq3_*`
  - no `kernel_cpy_*_tq3_*`
  - no `kernel_set_rows_tq3_*`
  - no `kernel_flash_attn_ext_*tq3_0*`

### 5. Why the model does not "just work" on Apple now

The loader understands `TQ3_4S`, but the Metal executor cannot keep those tensors on GPU for the needed ops.

Practical effect:

- weights may be loaded, but `MUL_MAT` lacks a Metal implementation for TQ3
- KV cache `TQ3_0` cannot be quantized/dequantized on Metal
- Flash Attention on Metal cannot consume `TQ3_0`
- the scheduler falls back away from the intended device path, which defeats the whole point for a 35B model on Apple

## Required migration

There are two valid targets:

### Target A: make the model runnable on Apple

This means:

- `TQ3_4S` and `TQ3_1S` weights execute on Metal
- KV stays on supported existing types, for example `F16` or `Q8_0`
- no primary CPU fallback

This is enough to boot and benchmark the exact model.

### Target B: reach public runtime feature parity

This adds:

- `-ctk tq3_0`
- `-ctv tq3_0`
- Metal Flash Attention support for the TQ3 KV path

This is the full CUDA-to-Metal migration.

The correct implementation order is A first, B second.

## Patch set breakdown

## Patch 1: add TQ3 to Metal capability plumbing

Files:

- `ggml/src/ggml-metal/ggml-metal-device.m`
- `ggml/src/ggml-metal/ggml-metal-device.cpp`

### Changes

1. Extend Metal op support tables to recognize TQ3 where the backend will soon have kernels.
2. Keep the gating strict until each kernel family exists.
3. Do not add optimistic support flags before kernels are present, otherwise the scheduler will select Metal and then fail pipeline lookup at runtime.

### Concrete edits

In `ggml-metal-device.m`:

- add `GGML_TYPE_TQ3_0`, `GGML_TYPE_TQ3_1S`, `GGML_TYPE_TQ3_4S` to:
  - `GGML_OP_MUL_MAT`
  - `GGML_OP_MUL_MAT_ID`
  - `GGML_OP_GET_ROWS`
  - `GGML_OP_CPY`
  - weight-only types to any copy/dequant path that needs q->f
- add `GGML_TYPE_TQ3_0` to:
  - `GGML_OP_SET_ROWS`
  - `GGML_OP_FLASH_ATTN_EXT`
  - only after Metal quantize and FA kernels exist

In `ggml-metal-device.cpp`:

- add TQ3 cases in:
  - `ggml_metal_library_get_pipeline_mul_mv(...)`
  - `ggml_metal_library_get_pipeline_mul_mv_id(...)`
- define `nsg`, `nr0`, `smem` for TQ3 families

## Patch 2: add TQ3 dequantize and quantize helpers in MSL

File:

- `ggml/src/ggml-metal/ggml-metal.metal`

### Why this is the foundation

All of these kernel families reuse the same two primitives:

- decode packed 3-bit indices
- reconstruct values from TQ3 scales and centroids

For `TQ3_0` and `TQ3_1S`, that also includes the inverse RHT/WHT-style transform.

### New helper functions to add

- `inline float tq3_centroid(uint8_t idx)`
- `inline uint8_t tq3_unpack_idx_8(const device uint8_t * qp, int r)`
- `inline float tq3_sign(int i)`
- `inline float tq3_4s_decode_scale(uint8_t byte)`
- `inline void tq3_rht_forward_32(thread float * x)`
- `inline void tq3_rht_inverse_32(thread float * x)`

Then build typed dequantizers:

- `dequantize_tq3_0(...)`
- `dequantize_tq3_0_t4(...)`
- `dequantize_tq3_1s(...)`
- `dequantize_tq3_1s_t4(...)`
- `dequantize_tq3_4s(...)`
- `dequantize_tq3_4s_t4(...)`

And quantizers needed for KV writeback:

- `quantize_tq3_0(...)`
- optionally later `quantize_tq3_1s(...)`
- optionally later `quantize_tq3_4s(...)`

### Design rule

Do not invent a new TQ3 math path in Metal. Port the existing reference/CUDA semantics:

- centroids must match the CUDA/reference path
- packing and unpacking must be bit-identical
- `TQ3_4S` E3M5 scale decode must match `ggml-quants.c`
- `TQ3_0` inverse transform must match the CPU reference ordering

## Patch 3: weight matvec support on Metal

Files:

- `ggml/src/ggml-metal/ggml-metal.metal`
- `ggml/src/ggml-metal/ggml-metal-device.cpp`

### Required kernels

For `MUL_MAT` fallback matvec:

- `kernel_mul_mv_tq3_0_f32`
- `kernel_mul_mv_tq3_1s_f32`
- `kernel_mul_mv_tq3_4s_f32`

For `MUL_MAT_ID` matvec:

- `kernel_mul_mv_id_tq3_0_f32`
- `kernel_mul_mv_id_tq3_1s_f32`
- `kernel_mul_mv_id_tq3_4s_f32`

Also F16 destination variants if the code path requests them.

### Implementation notes

`TQ3_4S` has the cleanest Metal port:

- no inverse WHT
- four per-8 scales
- direct packed-index decode

For `TQ3_0` and `TQ3_1S`:

- decode 8 groups
- reconstruct rotated-domain values
- apply inverse transform
- dot with source activation

This is slower than the CUDA native warp trick, but correct and portable.

### Recommendation

Get `TQ3_4S` working first. That is the model the user asked for.

Then add `TQ3_1S`.

Then add `TQ3_0` weight support if desired for completeness.

## Patch 4: weight matmul support on Metal

Files:

- `ggml/src/ggml-metal/ggml-metal.metal`
- `ggml/src/ggml-metal/ggml-metal-device.cpp`
- `ggml/src/ggml-metal/ggml-metal-ops.cpp`

### Why this matters

Prompt prefill does not stay on the simple matvec path. For real throughput on Apple, `MUL_MAT` needs the `mul_mm` kernels too.

Without this patch, long prompt prefill will be the wrong shape or will fall back to a very poor path.

### Required kernel families

- `kernel_mul_mm_tq3_0_f32`
- `kernel_mul_mm_tq3_1s_f32`
- `kernel_mul_mm_tq3_4s_f32`
- F16 destination variants

And MoE/id variants:

- `kernel_mul_mm_id_tq3_0_f32`
- `kernel_mul_mm_id_tq3_1s_f32`
- `kernel_mul_mm_id_tq3_4s_f32`
- F16 destination variants

### Correct porting strategy

Do not try to transliterate CUDA warp intrinsics one-to-one.

Instead, map the CUDA idea to the existing Metal tiling pattern already used by quantized `mul_mm`:

1. Load one TQ3 tile from `src0`
2. Requantize or dequantize into the layout expected by the Metal tile kernel
3. Reuse the existing simdgroup matrix math path

### Best approach per type

For `TQ3_4S`:

- follow the CUDA `load_tiles_tq3_4s(...)` idea from `ggml/src/ggml-cuda/mmq.cuh`
- bake each subgroup into a block-level `q8` tile in shared memory
- reuse the existing `q8`-style MMA accumulation pattern

This is the most performance-faithful port and avoids materializing full FP32 tiles.

For `TQ3_0` and `TQ3_1S`:

- initial correct implementation can dequantize tile fragments to `half` or `float`
- later optimize into a q8-baked path if needed

## Patch 5: row ops and copy ops

Files:

- `ggml/src/ggml-metal/ggml-metal.metal`
- `ggml/src/ggml-metal/ggml-metal-device.m`

### Required additions

For readback/dequant:

- `kernel_get_rows_tq3_0`
- `kernel_get_rows_tq3_1s`
- `kernel_get_rows_tq3_4s`
- `kernel_cpy_tq3_0_f32`
- `kernel_cpy_tq3_1s_f32`
- `kernel_cpy_tq3_4s_f32`
- `kernel_cpy_tq3_0_f16`
- `kernel_cpy_tq3_1s_f16`
- `kernel_cpy_tq3_4s_f16`

For KV writes:

- `kernel_set_rows_tq3_0_i32`
- `kernel_set_rows_tq3_0_i64`
- optionally matching `cpy_f32_tq3_0`

### Why `TQ3_0` quantize is mandatory

The KV cache path writes activations into quantized storage. On CUDA this exists in `ggml/src/ggml-cuda/set-rows.cu`.

If Metal cannot quantize into `TQ3_0`, then `-ctk tq3_0 -ctv tq3_0` can never be native on Apple.

## Patch 6: Flash Attention for TQ3 KV on Metal

Files:

- `ggml/src/ggml-metal/ggml-metal.metal`
- `ggml/src/ggml-metal/ggml-metal-device.m`
- `ggml/src/ggml-metal/ggml-metal-ops.cpp`
- `src/llama-context.cpp`

### Existing intent in the runtime

`src/llama-context.cpp` already has a fast-path exception for:

- K type `TQ3_0`
- V type `F16`

So the runtime already expects a backend to support this pairing.

### What must be ported

From `ggml/src/ggml-cuda/fattn-common.cuh`:

- `vec_dot_fattn_vec_KQ_tq3_0(...)`
- `dequantize_V_tq3_0(...)`

Into Metal equivalents:

- `vec_dot_fattn_vec_KQ_tq3_0` in MSL
- `dequantize_V_tq3_0` in MSL

Then instantiate:

- `kernel_flash_attn_ext_tq3_0_*`
- `kernel_flash_attn_ext_vec_tq3_0_*`

for the head sizes Qwen 3.6 needs.

### Correct rollout

1. First support `K = TQ3_0`, `V = F16`
2. Then support `K = TQ3_0`, `V = TQ3_0`

That matches the existing runtime assumption and reduces first-pass complexity.

## Patch 7: Qwen3.6 validation on Apple

Once the backend port exists, validate in this order.

### A. Synthetic correctness

Add or run tests that compare CPU vs Metal for:

- `dequantize_tq3_0`
- `dequantize_tq3_1s`
- `dequantize_tq3_4s`
- `mul_mat` outputs for small tensors
- `get_rows`
- `set_rows` for `TQ3_0`
- flash-attn outputs for `TQ3_0` K against CPU fallback within tolerance

### B. Runtime smoke

Run:

```bash
./build-mac/bin/llama-cli \
  -m /path/to/Qwen3.6-35B-A3B-TQ3_4S.gguf \
  -ngl 99 \
  -c 2048 \
  -fa on \
  -n 32 \
  -p "Reply in one short sentence: Metal TQ3 is alive."
```

Then:

```bash
./build-mac/bin/llama-server \
  -m /path/to/Qwen3.6-35B-A3B-TQ3_4S.gguf \
  -ngl 99 \
  -fa on \
  -c 4096 \
  --jinja \
  --reasoning off \
  --port 8090
```

### C. KV path validation

Only after `set_rows` and FA kernels land:

```bash
./build-mac/bin/llama-server \
  -m /path/to/Qwen3.6-35B-A3B-TQ3_4S.gguf \
  -ngl 99 \
  -fa on \
  -c 8192 \
  -ctk tq3_0 \
  -ctv tq3_0 \
  --jinja \
  --reasoning off \
  --port 8090
```

## Performance expectations on Apple

There are two separate performance questions.

### 1. "Can it work?"

Yes, if the Metal backend gets native TQ3 kernels.

Nothing in the format itself prevents a Metal implementation.

### 2. "Will it match CUDA?"

Not automatically.

CUDA currently wins because:

- warp-native shuffle patterns map very naturally to the TQ3 block shape
- the fork already has q8-baked tile loaders for TQ3 in CUDA
- FA vector kernels already have TQ3-specific specializations

To get close on Apple, the important optimization is:

- do not fully dequantize TQ3 weights to FP32 in global memory
- tile locally
- bake into q8 or half fragments inside threadgroup memory
- reuse existing simdgroup MM kernels

That is the only path that has a realistic chance of delivering good prefill on Apple.

## Minimal execution order

If the goal is "make the exact model usable on Apple as fast as possible", implement in this order:

1. `TQ3_4S` dequant helpers in MSL
2. `TQ3_4S` `mul_mv`
3. `TQ3_4S` `mul_mm`
4. `TQ3_4S` `mul_mv_id` and `mul_mm_id`
5. `TQ3_4S` `get_rows` and q->f `cpy`
6. smoke-run the model with non-TQ3 KV
7. add `TQ3_0` quantize/dequantize for KV
8. add `set_rows_tq3_0`
9. add `flash_attn_ext` TQ3_0 support
10. benchmark `prefill`, `decode`, max input/output, tool calling

If the goal is "full public runtime parity", do not stop before step 9.

## What not to do

Do not:

- convert the model to `Q4_K`, `Q6_K`, or MLX
- keep weights on CPU and call that Apple support
- special-case Qwen only in high-level code while backend ops remain unsupported
- add runtime flags that silently force fallback to non-TQ3 paths

That would make the benchmark invalid for the exact model the user requested.

## Deliverable definition

The Apple port is done only when all of these are true:

1. `Qwen3.6-35B-A3B-TQ3_4S.gguf` runs on Apple Metal with the original weights.
2. `MUL_MAT` for TQ3 weights stays on Metal.
3. `-ctk tq3_0 -ctv tq3_0` works on Metal.
4. Flash Attention stays enabled on the intended TQ3 KV path.
5. We can measure:
   - prefill tok/s
   - decode tok/s
   - max stable prompt tokens
   - max stable output tokens
   - tool-calling stability

Until then, the migration is incomplete.
