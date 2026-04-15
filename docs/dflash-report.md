# DFlash report (aktuální stav v tomto repu)

## TL;DR

- DFlash je lokálně připravený a funkční v **MLX/Qwen** cestě.
- Provedené reálné benchmarky ukázaly zrychlení **1.29x až 1.62x** proti baseline.
- DFlash není v tomto setupu zapojený do SuperGemma inference (ta běží přímo přes `mlx_lm` wrappery).

## Co přesně teď používáme

1. **Pinned DFlash checkout**
   - commit: `c95d24215a3e63f10c9d4dbe826b80e85772c1ae`
   - pin je definovaný v `scripts/pinned_sources.sh`
2. **Wrapper pro DFlash benchmark**
   - `scripts/run_dflash_mlx_benchmark.sh`
   - defaultně spouští MLX backend:
     - target `Qwen/Qwen3.5-4B`
     - draft `z-lab/Qwen3.5-4B-DFlash`
     - dataset `gsm8k`
3. **SuperGemma cesta je oddělená**
   - `scripts/run_supergemma_mlx_generate.sh`
   - `scripts/run_supergemma_mlx_server.sh`

## Co audit potvrdil

Viz detailně `docs/dflash-audit.md`.

- DFlash podporuje backendy: `transformers`, `sglang`, `vllm`, `mlx`.
- MLX implementace je prakticky orientovaná na Qwen/Qwen3.5 draft páry.
- V tomto pinu není integrovaná DDTree větev v DFlash CLI/algoritmech.
- Kombinace **DDTree + DFlash + SuperGemma** v jednom MLX runtime není upstream podpořená.

## Reálné benchmark výsledky (lokální MLX)

Viz detailně `docs/dflash-benchmark.md` a logy:
- `artifacts/benchmarks/dflash-mlx-20260414-173127/benchmark.log`
- `artifacts/benchmarks/dflash-mlx-20260414-173612/benchmark.log`

| Run | max-samples | max-new-tokens | Baseline tok/s | DFlash tok/s | Speedup | Avg acceptance |
|---|---:|---:|---:|---:|---:|---:|
| dflash-mlx-20260414-173127 | 1 | 32 | 15.18 | 19.63 | 1.29x | 2.75 |
| dflash-mlx-20260414-173612 | 2 | 32 | 15.18 | 24.65 | 1.62x | 3.44 |

Poznámky:
- první běh obsahoval one-time download modelu/datasetu;
- objevila se HF warning hláška bez tokenu (`HF_TOKEN`), benchmark ale doběhl korektně.

## Praktický závěr

1. DFlash je připravený pro **separátní akcelerační track** (Qwen + draft).
2. Pro SuperGemmu dál platí přímý MLX běh mimo DFlash.
3. Dává smysl pokračovat dalším krokem na DDTree audit/integraci a vyhodnotit, jestli chceme:
   - CUDA DDTree experiment mimo MLX,
   - nebo čistě MLX cestu bez DDTree.
