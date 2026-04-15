# Long-context map-reduce pipeline

This pipeline processes documents larger than a single KV cache window by chunking on tokenizer tokens, running per-chunk summaries, and recursively reducing those summaries to one final answer.

## Prerequisites

- Bootstrap local env (no global installs):

```bash
scripts/setup_env.sh
```

- Input must be a plain text file (`.txt` recommended).

## Quick start (wrapper with safe defaults)

```bash
scripts/run_long_context_map_reduce.sh \
  --input-file path/to/large_document.txt \
  --question "What are the top risks, supporting evidence, and unresolved questions?"
```

Defaults are conservative for local execution:
- chunk size: `3072`
- overlap: `256`
- max KV per call: `4096`
- reduce fan-in: `8`

Artifacts are written to:

`artifacts/long-context/<timestamp>/`

## 128k-scale example

If your input is around 128k tokens, use larger chunking to reduce map call count:

```bash
scripts/run_long_context_map_reduce.sh \
  --input-file artifacts/inputs/doc_128k.txt \
  --question "Extract all architecture decisions and why they were made." \
  --chunk-size 4096 \
  --chunk-overlap 384 \
  --max-kv-size 6144 \
  --map-max-tokens 256 \
  --reduce-max-tokens 320 \
  --final-max-tokens 512
```

Or call Python directly:

```bash
.venv/bin/python scripts/long_context_map_reduce.py \
  --input-file artifacts/inputs/doc_128k.txt \
  --question "Summarize obligations, deadlines, and owners." \
  --model Jiunsong/supergemma4-26b-uncensored-mlx-4bit-v2
```

## Output layout

Each run includes structured artifacts such as:

- `config.json` / `token_stats.json`
- `chunk_summaries.jsonl`
- `map/chunk_*.{prompt,command,stdout,stderr,output}.txt`
- `reduce/level_*/group_*.{prompt,command,stdout,stderr,output}.txt`
- `final/final_{prompt,command,stdout,stderr,output}.txt`
- `final_answer.txt` and `final_answer.json`

## Lightweight sanity checks

```bash
.venv/bin/python scripts/long_context_map_reduce.py --help
scripts/run_long_context_map_reduce.sh --help
```

Tiny dry-run (no model generation; validates token chunking + artifact writes):

```bash
printf 'alpha beta gamma delta epsilon\n' > artifacts/long-context/tiny_input.txt
scripts/run_long_context_map_reduce.sh \
  --input-file artifacts/long-context/tiny_input.txt \
  --question "What tokens are present?" \
  --dry-run \
  --chunk-size 8 \
  --chunk-overlap 2
```
