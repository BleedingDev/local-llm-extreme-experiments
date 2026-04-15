# 128k scaling report (praktický provoz)

## Co je hotové

Pro práci s dokumenty velikosti **128k+ tokenů** je připravená pipeline:

- `scripts/long_context_map_reduce.py`
- `scripts/run_long_context_map_reduce.sh`
- návod: `docs/long-context-pipeline.md`

Pipeline dělá:
1. tokenizaci vstupu
2. chunking s overlapem
3. map sumarizace po chunkech
4. víceúrovňový reduce
5. finální syntézu odpovědi

Výstupy ukládá do `artifacts/long-context/<timestamp>/`.

## Validace

### A) Velký vstup (dry-run, 128k-scale+)

Příkaz:

```bash
scripts/run_long_context_map_reduce.sh \
  --input-file artifacts/long-context/inputs/input_128k_words.txt \
  --question "Shrň hlavní témata a rizika." \
  --dry-run \
  --chunk-size 4096 \
  --chunk-overlap 384 \
  --max-kv-size 4096 \
  --map-max-tokens 128 \
  --reduce-max-tokens 160 \
  --final-max-tokens 256
```

Artefakt:
- `/Users/satan/side/experiments/supergemma-dflash-ddtree-mlx/artifacts/long-context/20260414T152350Z`

Naměřené:
- `input_token_count = 784896`
- `chunk_count = 212`
- `reduce_call_count = 32`
- pipeline doběhla korektně, zapsala `final_answer.txt` a metadata.

### B) Reálný mini běh (bez dry-run)

Artefakt:
- `/Users/satan/side/experiments/supergemma-dflash-ddtree-mlx/artifacts/long-context/20260414T152420Z`

Naměřené:
- `input_token_count = 52`
- `chunk_count = 1`
- `dry_run = false`
- generace proběhla a vznikl `final_answer.txt`.

## Jak to použít pro ~128k tokenů

Doporučený start:

```bash
scripts/run_long_context_map_reduce.sh \
  --input-file path/to/your_128k_text.txt \
  --question "Co přesně z dokumentu potřebuju vytěžit?" \
  --chunk-size 4096 \
  --chunk-overlap 384 \
  --max-kv-size 4096 \
  --map-max-tokens 256 \
  --reduce-max-tokens 320 \
  --final-max-tokens 512
```

Hrubý počet volání modelu pro vstup kolem 128k tokenů (stejné parametry):
- map: ~35 chunků
- reduce: ~6 volání (fan-in 8)
- final: 1
- celkem ~42 inferencí

## Proč se objevila hláška o loginu

Během stahování z Hugging Face se objevila warning hláška o neautentizovaných requestech.  
**Login není povinný** pro běh, ale bez tokenu jsou nižší limity a pomalejší/stabilnější download.

Doporučení:
- pokud chceš spolehlivější download velkých modelů, nastav `HF_TOKEN` (nebo se přihlas přes HF CLI),
- inference samotná pak běží lokálně stejně jako doteď.

## Trade-offy

1. Single-window 128k je na 32GB stroji mimo reálný paměťový budget.
2. Map-reduce je škálovatelný workaround: vyšší latence, ale zvládne velmi dlouhé vstupy.
3. Kvalita výsledku závisí na chunk-size, overlapu a reduce promptu (lze ladit per use-case).
