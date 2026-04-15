# Audit schopností DFlash integrace (pin `c95d24215a3e63f10c9d4dbe826b80e85772c1ae`)

## Rozsah a ověření stavu
- Audit pokrývá lokální checkout `vendor/dflash` + integrační skripty v tomto repu.
- Pin je definován v `scripts/pinned_sources.sh` (`DFLASH_COMMIT=...c95d242...`). [scripts/pinned_sources.sh:4-6]
- Lokálně ověřeno příkazem `git -C vendor/dflash rev-parse HEAD` => `c95d24215a3e63f10c9d4dbe826b80e85772c1ae` (shoda s pinem).

## 1) Přesně podporované backendy a jak se spouští

### Co deklaruje DFlash upstream
- Instalace backend extras:
  - `transformers`: `uv pip install -e ".[transformers]"`
  - `sglang`: `uv pip install -e ".[sglang]"`
  - `vllm`: `uv pip install -e ".[vllm]"` + nightly `vllm`
  - `mlx`: `pip install -e ".[mlx]"`  
  [vendor/dflash/README.md:36-47]
- CLI benchmark akceptuje přesně: `transformers`, `sglang`, `vllm`, `mlx`. [vendor/dflash/dflash/benchmark.py:482]
- V `pyproject.toml` jsou pro backendy odpovídající optional deps (`transformers`, `sglang`, `vllm`, `mlx`). [vendor/dflash/pyproject.toml:19-38]

### Jak jsou backendy volány (přesné invocation)
- **vLLM**: `vllm serve ... --speculative-config '{"method":"dflash",...}'` [vendor/dflash/README.md:51-58]
- **SGLang**: `python -m sglang.launch_server ... --speculative-algorithm DFLASH ...` [vendor/dflash/README.md:60-81]
- **Transformers**: Python API přes `DFlashDraftModel`/`spec_generate` (Qwen3/LLaMA 3.1). [vendor/dflash/README.md:83-99]
- **MLX**: `from dflash.model_mlx import load, load_draft, stream_generate` a benchmark `python -m dflash.benchmark --backend mlx ...`. [vendor/dflash/README.md:101-118,145-149]

### Jak je to napojené v tomto repu
- Wrapper pro DFlash MLX benchmark: `scripts/run_dflash_mlx_benchmark.sh` (volá `python -m dflash.benchmark --backend mlx ...`). [scripts/run_dflash_mlx_benchmark.sh:17-21,86-94]
- SuperGemma wrappery jsou oddělené (přímo `mlx_lm.generate` / `mlx_lm.server`, bez DFlash parametrů). [scripts/run_supergemma_mlx_generate.sh:87-92], [scripts/run_supergemma_mlx_server.sh:78-83]

## 2) Přesně podporované model/draft páry (docs + kód)

### 2.1 Explicitně uvedené páry v DFlash README
| Target model | DFlash draft | Stav |
|---|---|---|
| Kimi-K2.5 (Preview) | `z-lab/Kimi-K2.5-DFlash` | dostupné |
| Qwen3.5-4B | `z-lab/Qwen3.5-4B-DFlash` | dostupné |
| Qwen3.5-9B | `z-lab/Qwen3.5-9B-DFlash` | dostupné |
| Qwen3.5-27B | `z-lab/Qwen3.5-27B-DFlash` | dostupné |
| Qwen3.5-35B-A3B | `z-lab/Qwen3.5-35B-A3B-DFlash` | dostupné |
| Qwen3-Coder-Next | `z-lab/Qwen3-Coder-Next-DFlash` | dostupné |
| Qwen3-Coder-30B-A3B | `z-lab/Qwen3-Coder-30B-A3B-DFlash` | dostupné |
| gpt-oss-20b | `z-lab/gpt-oss-20b-DFlash` | dostupné |
| gpt-oss-120b | `z-lab/gpt-oss-120b-DFlash` | dostupné |
| Qwen3-4B (non-thinking) | `z-lab/Qwen3-4B-DFlash-b16` | dostupné |
| Qwen3-8B (non-thinking) | `z-lab/Qwen3-8B-DFlash-b16` | dostupné |
| Llama-3.1-8B-Instruct | `z-lab/LLaMA3.1-8B-Instruct-DFlash-UltraChat` | dostupné |
| Qwen3.5-122B-A10B | Coming soon | zatím nedostupné |
| Qwen3.5-397B-A17B | Coming soon | zatím nedostupné |
| GLM-5.1 | Coming soon | zatím nedostupné |

Zdroj tabulky: [vendor/dflash/README.md:10-29]

### 2.2 Kódová omezení (důležité pro přesnost)
- **Transformers backend** explicitně omezuje model name regexem na Qwen3 (bez Qwen3.5) a LLaMA‑3.1‑8B‑Instruct. [vendor/dflash/dflash/benchmark.py:173-182]
- Pro `transformers` a `mlx` je `--draft-model` povinný. [vendor/dflash/dflash/benchmark.py:507-514]
- `--enable-thinking` je blokováno pro `qwen3-4b`/`qwen3-8b` (assert). [vendor/dflash/dflash/benchmark.py:502-505]

### 2.3 Speciálně MLX cesta
- README říká, že MLX implementace je testovaná na Apple M5 Pro s **Qwen3/Qwen3.5**. [vendor/dflash/README.md:103]
- Ukázkový MLX pár je `Qwen/Qwen3.5-4B` + `z-lab/Qwen3.5-4B-DFlash`. [vendor/dflash/README.md:108-109,147-149]
- Kód MLX je Qwen-orientovaný (`mlx_lm.models.qwen3`, patch pro `qwen3_5`). [vendor/dflash/dflash/model_mlx.py:13,204-205]
- Draft loader očekává DFlash-specifický config (`dflash_config.target_layer_ids`, `mask_token_id`). [vendor/dflash/dflash/model_mlx.py:147-150]

### 2.4 Co je v tomto repu defaultně předpřipravené
- `run_dflash_mlx_benchmark.sh` default:  
  `--model Qwen/Qwen3.5-4B` + `--draft-model z-lab/Qwen3.5-4B-DFlash`. [scripts/run_dflash_mlx_benchmark.sh:8-9,24-26,89-92]

## 3) Algoritmické volby: co je dostupné a co chybí
- DFlash upstream invocation používá **DFLASH**:
  - vLLM: `"method": "dflash"` [vendor/dflash/README.md:55]
  - SGLang: `--speculative-algorithm DFLASH` [vendor/dflash/README.md:72]
- Lokální benchmark CLI řeší výběr backendu, nikoli alternativní spekulativní algoritmus (`--backend` only). [vendor/dflash/dflash/benchmark.py:482]
- V `vendor/dflash` nebyly nalezeny odkazy na `DDTREE`, `EAGLE`, `Medusa`, `Lookahead` (`rg` audit: 0 výsledků).
- DDTree je samostatný checkout a je cílený na CUDA PyTorch (`This codebase is intended for a CUDA-enabled PyTorch environment.`). [vendor/ddtree/README.md:16], [vendor/ddtree/requirements.txt:1-4]

## 4) Kde přesně naráží limity pro SuperGemma/Gemma4
- V oficiální tabulce podporovaných modelů DFlash není Gemma/SuperGemma. [vendor/dflash/README.md:10-29]
- MLX část DFlash je dokumentačně i kódově orientovaná na Qwen3/Qwen3.5. [vendor/dflash/README.md:103,108-109], [vendor/dflash/dflash/model_mlx.py:13,204-205]
- Lokální `combo_reality_check` explicitně hlásí, že „DDTree + DFlash + SuperGemma 26B A4B as one integrated MLX runtime stack“ není upstream podporováno. [scripts/combo_reality_check.sh:50-53]
- SuperGemma wrappery běží mimo DFlash (direct MLX runtime), takže nejde o integrovanou DFlash akceleraci SuperGemma. [scripts/run_supergemma_mlx_generate.sh:87-92], [scripts/run_supergemma_mlx_server.sh:78-83]

## 5) Konkrétní command recipes, které mají fungovat v aktuálním setupu

### 5.1 Lightweight sanity (ověřeno v tomto auditu)
```bash
cd .
git --no-pager -C vendor/dflash rev-parse HEAD
scripts/combo_reality_check.sh
scripts/run_dflash_mlx_benchmark.sh --dry-run
.venv/bin/python -m dflash.benchmark --help
```

### 5.2 Doporučené provozní příkazy (bez těžkého benchmarku)
```bash
# A) SuperGemma inference (bez DFlash, ale funkční lokální cesta)
scripts/run_supergemma_mlx_generate.sh \
  --prompt "Napiš jednu krátkou českou větu." \
  --max-tokens 64 --max-kv-size 512

# B) SuperGemma OpenAI-compatible server (bez DFlash)
scripts/run_supergemma_mlx_server.sh --host 127.0.0.1 --port 8080
curl -s http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Řekni ahoj česky."}],"max_tokens":32}'

# C) DFlash MLX path (Qwen track; lehký běh)
scripts/run_dflash_mlx_benchmark.sh \
  --model Qwen/Qwen3.5-4B \
  --draft-model z-lab/Qwen3.5-4B-DFlash \
  --dataset gsm8k \
  --max-samples 1 \
  -- --max-new-tokens 64 --temperature 0.0
```

Pozn.: Cesta C je oddělená od SuperGemma runtime a může při prvním spuštění stahovat dataset/model.

## 6) Rizika a gotchas
- `run_dflash_mlx_benchmark.sh` vyžaduje `.venv/bin/python`; bez `.venv` failne hned. [scripts/run_dflash_mlx_benchmark.sh:80-84]
- Pokud `dflash` není nainstalovaný v `.venv`, wrapper končí s návodem na instalaci z `vendor/dflash`. [scripts/run_dflash_mlx_benchmark.sh:104-109]
- Benchmark stahuje datasety do `cache/` při prvním běhu (síť + čas). [vendor/dflash/dflash/benchmark.py:26,58-81]
- vLLM backend podle README vyžaduje nightly build, ne čistě stable. [vendor/dflash/README.md:43-47]
- SGLang má experimentální volby překryvu/schedule (`SPEC_V2`), označené jako nemusí být stabilní. [vendor/dflash/README.md:65-69]
- Transformers backend má tvrdá modelová omezení a fallback bez `flash_attn` je pomalejší. [vendor/dflash/dflash/benchmark.py:173-195]
- Pro některé MLX modely je nutná gated-delta rollback podpora; jinak runtime chyba. [vendor/dflash/dflash/model_mlx.py:334-338]
- Integrace DDTree+DFlash+SuperGemma v jednom MLX stacku není v tomto pinu dostupná. [scripts/combo_reality_check.sh:50-53], [vendor/ddtree/README.md:16]

## Verdikt auditu
- **DFlash pin `c95d242` je v repu správně připnutý a lokálně přítomný.**
- **Podporovaná a ověřitelná cesta na tomto Mac setupu je:**
  1) SuperGemma přes MLX wrappery (bez DFlash),  
  2) DFlash MLX benchmark zvlášť na Qwen target+draft párech.
- **Neexistuje zde integrovaná, upstream podložená cesta DFlash+DDTree+SuperGemma(Gemma4) v jednom runtime.**
