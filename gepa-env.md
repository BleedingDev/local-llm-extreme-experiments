# GEPA / DSPy environment cheatsheet

This repo's MLX/benchmark `.venv` (Python 3.14, mostly empty) is left untouched.
GEPA and DSPy work happens inside an isolated venv: **`.venv-gepa/`** (Python
3.12.11, created via `uv venv --python 3.12 .venv-gepa`).

## Activate

```bash
cd /Users/satan/side/experiments/supergemma-dflash-ddtree-mlx
source .venv-gepa/bin/activate
# or, without activating:
./.venv-gepa/bin/python scripts/gepa_smoke.py
```

The venv was created with `uv` and therefore has **no `pip` shim**. Use:

```bash
VIRTUAL_ENV=$PWD/.venv-gepa uv pip install <package>
VIRTUAL_ENV=$PWD/.venv-gepa uv pip list
```

## Env var surprise — `ANTHROPIC_AUTH_TOKEN`

The repo's `.env` exposes the Anthropic key as **`ANTHROPIC_AUTH_TOKEN`**, not
the SDK-default `ANTHROPIC_API_KEY`. DSPy / LiteLLM / the `anthropic` SDK all
look for `ANTHROPIC_API_KEY` by default, so you must either:

1. Pass `api_key=` explicitly when constructing `dspy.LM(...)` (preferred,
   what `scripts/gepa_smoke.py` does):

   ```python
   from dotenv import load_dotenv; load_dotenv(".env")
   import os, dspy
   lm = dspy.LM("anthropic/claude-haiku-4-5",
                api_key=os.environ["ANTHROPIC_AUTH_TOKEN"])
   dspy.configure(lm=lm)
   ```

2. Or shim it once per shell: `export ANTHROPIC_API_KEY="$ANTHROPIC_AUTH_TOKEN"`.

Never echo or log the key — only its length / presence.

## Installed versions (Python 3.12.11, `.venv-gepa`)

| Package        | Version    |
| -------------- | ---------- |
| dspy           | 3.2.0      |
| dspy-ai        | 3.2.0      |
| gepa           | 0.0.27     |
| anthropic      | 0.97.0     |
| litellm        | 1.82.6     |
| pyarrow        | 24.0.0     |
| pandas         | 3.0.2      |
| datasets       | 4.8.5      |
| orjson         | 3.11.8     |
| tqdm           | 4.67.3     |
| rich           | 15.0.0     |
| pytest         | 9.0.3      |
| pydantic       | 2.13.3     |
| python-dotenv  | 1.2.2      |

`gepa` came from PyPI cleanly — no need to fall back to the local
`/Users/satan/side/experiments/gepa-fresh` editable install.

## Smoke test

`scripts/gepa_smoke.py` loads `.env`, builds a `dspy.LM` against
`anthropic/claude-haiku-4-5` with the explicit `api_key=`, configures DSPy,
runs `dspy.Predict("question -> answer")` on `"What is 2+2?"`, and finally
imports `gepa`. Last run: **PASS** — model returned `'2 + 2 = 4'`, `gepa`
imported cleanly (module name `gepa`), and the script exited 0. Re-run any
time with `./.venv-gepa/bin/python scripts/gepa_smoke.py` to re-validate the
environment before kicking off real GEPA training.
