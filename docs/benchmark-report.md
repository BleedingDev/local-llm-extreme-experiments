# Benchmark report — SuperGemma MLX (Apple M5, 32 GB)

Datum reportu: 2026-04-14 (z běhů `20260414T115232Z` a `20260414T115402Z`).

## 1) Vstupy a artefakty (reprodukovatelnost)

Použité vstupy:
- `docs/benchmark-system-baseline.md`
- `docs/benchmarking.md`

Použité benchmark artefakty:
- `artifacts/benchmarks/20260414T115232Z`
  - `results.csv`, `summary.txt`, `config.env`, `prompt.txt`, `raw/*`
- `artifacts/benchmarks/20260414T115402Z`
  - `results.csv`, `summary.txt`, `config.env`, `prompt.txt`, `raw/*`

## 2) Hardware/software baseline (snapshot `2026-04-14T11:41:40Z`)

| Oblast | Hodnota |
|---|---|
| Zařízení | MacBook Pro (`Mac17,2`) |
| Čip | Apple M5 |
| RAM (unified) | 32 GB (32.0 GiB) |
| OS | macOS 26.3 (`25D5087f`) |
| Python (.venv) | 3.14.4 |
| MLX stack | mlx 0.31.1, mlx-lm 0.31.2, dflash 0.1.0 |
| Volné místo při baseline | ~58 GiB |

## 3) Metodika a parametry testu

Harness:
- `scripts/benchmark_supergemma_mlx.sh`
- Inferenční binárka: `.venv/bin/mlx_lm.generate`
- Model: `Jiunsong/supergemma4-26b-uncensored-mlx-4bit-v2`
- Metriky (`prompt_tps`, `generation_tps`, `peak_memory_gb`) se parsují z `raw/*.combined.log` do `results.csv`.

### Běh A — rychlostní matice (`20260414T115232Z`)
- `mode=benchmark`
- `repeats=3`
- `max_tokens=64`
- `max_kv_size={512, 768, 1024, 1280, 1536}`
- Prompt (`prompt.txt`): „Napiš krátký odstavec česky o praktickém přínosu lokální AI inference.“
- Výsledek: 15/15 pokusů `status=ok`, `exit_code=0`, `benchmark_failures=0`.

### Běh B — probe kontextu (`20260414T115402Z`)
- `mode=probe`
- `probe_start_kv=512`, `probe_max_kv=4096`, `probe_max_tokens=32`
- Prompt (`prompt.txt`): „Napiš jednu krátkou českou větu.“
- Výsledek: 4/4 pokusů `status=ok`, `exit_code=0`.

## 4) Výsledky rychlosti podle KV (`20260414T115232Z/results.csv`)

| max_kv_size | opakování | prompt tok/s (avg) | prompt tok/s (min–max) | generation tok/s (avg) | generation tok/s (min–max) | peak memory (GB) |
|---:|---:|---:|---:|---:|---:|---:|
| 512  | 3 | 165.826 | 164.741–167.274 | 52.625 | 52.506–52.830 | 14.282 |
| 768  | 3 | 128.907 | 57.368–164.698  | 52.385 | 52.248–52.606 | 14.282 |
| 1024 | 3 | 165.743 | 165.199–166.666 | 52.130 | 52.023–52.299 | 14.282 |
| 1280 | 3 | 130.064 | 57.870–166.757  | 52.022 | 51.816–52.140 | 14.282 |
| 1536 | 3 | 165.219 | 164.785–166.076 | 51.909 | 51.620–52.117 | 14.282 |

Poznámka k variabilitě:
- Dva běhy mají výrazně nižší `prompt_tps` (57.368 a 57.870):
  - `artifacts/benchmarks/20260414T115232Z/raw/0006_benchmark_kv768_tok64_rep3.combined.log`
  - `artifacts/benchmarks/20260414T115232Z/raw/0011_benchmark_kv1280_tok64_rep2.combined.log`

## 5) Max ověřený kontext/KV a hranice selhání (`20260414T115402Z`)

| run_id | phase | kv | status | prompt tok/s | generation tok/s | peak memory (GB) |
|---:|---|---:|---|---:|---:|---:|
| 0001 | probe-scan | 512  | ok | 44.230  | 53.267 | 14.279 |
| 0002 | probe-scan | 1024 | ok | 136.536 | 53.405 | 14.279 |
| 0003 | probe-scan | 2048 | ok | 135.864 | 53.318 | 14.279 |
| 0004 | probe-scan | 4096 | ok | 138.527 | 53.636 | 14.279 |

Souhrn dle `summary.txt`:
- `probe_status=no-failure-observed`
- `probe_best_kv=4096`
- `probe_first_failure_kv=none`

Interpretace:
- **Maximální ověřená hodnota `max_kv_size` je 4096** (v rámci tohoto měření).
- **Hranice selhání nebyla dosažena** v testovaném rozsahu 512–4096; skutečný fail-point je >4096 nebo závisí na delším promptu/generaci.

## 6) Praktická doporučení pro tento stroj (M5, 32 GB)

1. Pro běžné použití zvažte `max_kv_size` **1024–1536**: stabilní throughput, peak paměť ~14.3 GB.
2. Pokud potřebujete větší kontext, `max_kv_size=4096` je zde **ověřený bez pádu** (ale jen pro krátký prompt + `max_tokens=32`).
3. Pro konzervativní latenci držte `max_tokens` kolem **64**; generation výkon je v datech kolem ~52 tok/s.
4. Pokud chcete skutečnou limitní hranici pro produkční workload, spusťte nový probe nad 4096 (např. 6144/8192+) se stejným modelem a realistickými prompty.

## 7) Caveats / omezení měření

- Měření je na **jednom stroji**, v jednom časovém okně.
- Testován je **jeden model a jedna kvantizace** (`supergemma4-26b ... mlx-4bit-v2`).
- Prompty jsou krátké (27–38 tokenů) a generace krátké (32/64 tokenů), takže nejde o worst-case dlouhého kontextu.
- `duration_sec` je měřeno v celých sekundách (skriptové měření), ne high-resolution profiling.
- Variabilita `prompt_tps` ukazuje, že jednotlivé běhy mohou mít odchylky; proto je vhodné používat více opakování a sledovat min/max.
