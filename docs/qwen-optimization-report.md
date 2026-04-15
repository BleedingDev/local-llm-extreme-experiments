# Qwen3.5 optimization report (exhaustive local pass)

Date: 2026-04-14

> This document is a historical phase summary. For the final consolidated results across DFlash + turboquant-mlx + DDTree-style MLX prototype, see `docs/qwen-tuning-report.md`.

## Executive summary

Na tomto stroji (M5, 32 GB) jsme vyčerpali prakticky dostupné lokální možnosti v aktuální architektuře:

1. **DFlash + Qwen3.5 (MLX)** je dnes nejvýkonnější reálně provozovatelná cesta v tomto repu.
2. **DDTree** upstream CUDA/PyTorch cesta je validní, ale není lokálně spustitelná na Apple Silicon; v repu je navíc experimentální DDTree-style MLX prototyp.
3. **TurboQuant** pro MLX je nyní lokálně ověřený přes `turboquant-mlx` integraci (s menší cache a proměnlivým throughput přínosem).
4. KV tuning v čistém MLX path má měřitelný přínos (lepší `max-kv-size` nastavení), ale největší přínos přinesl DFlash sweep.

## Co bylo pokryto

| Oblast | Stav | Výstup |
|---|---|---|
| Thread knihovny + kompatibilita | hotovo | `docs/qwen-thread-libraries.md` |
| DFlash sweep (reálné běhy) | hotovo | `docs/qwen-dflash-sweep.md` + `artifacts/benchmarks/qwen-dflash-sweep-20260414-192908/` |
| DDTree applicability + scaffolding | hotovo | `docs/qwen-ddtree-eval.md` + `scripts/run_qwen_ddtree_benchmark.sh` |
| TurboQuant/KV fallback eval | hotovo | `docs/qwen-turboquant-eval.md` + Qwen KV artifacts |

## Hlavní naměřené výsledky

### 1) DFlash sweep (Qwen3.5 MLX)

- Baseline: **12.94–15.16 tok/s**
- DFlash: **22.49–33.30 tok/s**
- Speedup: **1.48x–2.39x**
- Avg acceptance length: **2.75–4.26**

Best observed command:

```bash
scripts/run_dflash_mlx_benchmark.sh \
  --model Qwen/Qwen3.5-4B \
  --draft-model z-lab/Qwen3.5-4B-DFlash \
  --max-samples 3 \
  -- --max-new-tokens 64
```

### 2) Direct MLX KV sweep (Qwen3.5)

Artifacts:
- `artifacts/benchmarks/20260414T185523Z`
- `artifacts/benchmarks/20260414T185732Z`

Průměrná generation rychlost podle `max-kv-size`:
- 512: **5.616 tok/s**
- 1024: **7.524 tok/s**
- 2048: **7.763 tok/s**
- 4096: **8.342 tok/s**

Probe:
- bez failu do `max-kv-size=32768` (krátký prompt/workload),
- `probe_best_kv=32768`.

## Co je “exhausted” vs co ne

### Exhausted v aktuálním lokálním stacku

1. DFlash benchmark tuning (`max-samples`, `max-new-tokens`, `block-size`, `temperature`) – prakticky pokryto.
2. KV tuning v MLX path (`max-kv-size` sweep + probe) – pokryto.
3. Kompatibilitní audit thread knihoven vs M5/MLX – pokryto.

### Mimo aktuální lokální stack (další fáze)

1. **DDTree full run**: vyžaduje CUDA host.
2. **TurboQuant / RotorQuant / IsoQuant**: vyžaduje llama.cpp/GGUF branch a separátní benchmark metodiku.

## Doporučený next step order

1. Stabilizovat produkční profil na DFlash best config (`max-samples=3`, `max-new-tokens=64`).
2. Pro DDTree otevřít CUDA pipeline a spustit:
   - `scripts/run_qwen_ddtree_benchmark.sh` na Linux+NVIDIA.
3. Paralelně založit samostatnou llama.cpp větev pro TurboQuant-class KV kompresi a porovnat vůči tomuto baseline reportu.
