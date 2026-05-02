# Trace-GEPA Bench Leaderboard
_Generated: 2026-05-02T00:56:55Z_
_Total runs ingested: 6 (ok: 6, incomplete: 0)_

## Overall ranking
| rank | harness | model | system_prompt_id | mean_score | count | wallclock_s | lm_calls | in_tok | out_tok | status |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | codex | gpt-5.5 | seed | 1.0000 | 20 | 86.36 | n/a | n/a | n/a | ok |
| 2 | anthropic | claude-opus-4-7 | seed | 1.0000 | 20 | 7.94 | 20 | 7093 | 489 | ok |
| 3 | codex | gpt-5.5 | seed | 1.0000 | 5 | 24.12 | n/a | n/a | n/a | ok |
| 4 | codex | gpt-5.5 | seed | 0.8500 | 20 | 108.34 | n/a | n/a | n/a | ok |
| 5 | mlx | mlx-community/Llama-3.2-1B-Instruct-4bit | seed | 0.4000 | 10 | 10.42 | n/a | n/a | 972 | ok |
| 6 | anthropic | claude-opus-4-7 | seed | 0.1000 | 20 | 8.50 | 20 | 7093 | 491 | ok |
## Per-category rankings
### command_synthesis

| rank | harness | model | system_prompt_id | mean_score | count |
|---|---|---|---|---|---|
| 1 | codex | gpt-5.5 | seed | n/a | 3 |

### debugging

| rank | harness | model | system_prompt_id | mean_score | count |
|---|---|---|---|---|---|
| 1 | codex | gpt-5.5 | seed | 1.0000 | 3 |

### edit_safety

| rank | harness | model | system_prompt_id | mean_score | count |
|---|---|---|---|---|---|
| 1 | codex | gpt-5.5 | seed | 1.0000 | 3 |

### path_grounding

| rank | harness | model | system_prompt_id | mean_score | count |
|---|---|---|---|---|---|
| 1 | codex | gpt-5.5 | seed | 1.0000 | 3 |

### planning

| rank | harness | model | system_prompt_id | mean_score | count |
|---|---|---|---|---|---|
| 1 | codex | gpt-5.5 | seed | 1.0000 | 2 |

### recovery

| rank | harness | model | system_prompt_id | mean_score | count |
|---|---|---|---|---|---|
| 1 | codex | gpt-5.5 | seed | 1.0000 | 3 |

### tool_routing

| rank | harness | model | system_prompt_id | mean_score | count |
|---|---|---|---|---|---|
| 1 | codex | gpt-5.5 | seed | 1.0000 | 20 |
| 2 | codex | gpt-5.5 | seed | 1.0000 | 3 |
| 3 | anthropic | claude-opus-4-7 | seed | 1.0000 | 20 |
| 4 | codex | gpt-5.5 | seed | 1.0000 | 5 |
| 5 | mlx | mlx-community/Llama-3.2-1B-Instruct-4bit | seed | 0.4000 | 10 |
| 6 | anthropic | claude-opus-4-7 | seed | 0.1000 | 20 |

## Per-difficulty rankings
### easy

| rank | harness | model | system_prompt_id | mean_score | count |
|---|---|---|---|---|---|
| 1 | codex | gpt-5.5 | seed | 1.0000 | 2 |
| 2 | anthropic | claude-opus-4-7 | seed | 1.0000 | 2 |
| 3 | codex | gpt-5.5 | seed | 1.0000 | 2 |
| 4 | codex | gpt-5.5 | seed | 0.4000 | 5 |
| 5 | anthropic | claude-opus-4-7 | seed | n/a | 2 |

### hard

| rank | harness | model | system_prompt_id | mean_score | count |
|---|---|---|---|---|---|
| 1 | codex | gpt-5.5 | seed | 1.0000 | 5 |
| 2 | codex | gpt-5.5 | seed | 1.0000 | 6 |
| 3 | anthropic | claude-opus-4-7 | seed | 1.0000 | 5 |
| 4 | anthropic | claude-opus-4-7 | seed | n/a | 5 |

### medium

| rank | harness | model | system_prompt_id | mean_score | count |
|---|---|---|---|---|---|
| 1 | codex | gpt-5.5 | seed | 1.0000 | 13 |
| 2 | codex | gpt-5.5 | seed | 1.0000 | 9 |
| 3 | anthropic | claude-opus-4-7 | seed | 1.0000 | 13 |
| 4 | codex | gpt-5.5 | seed | 1.0000 | 3 |
| 5 | anthropic | claude-opus-4-7 | seed | 0.1538 | 13 |


## Cost-effectiveness (mean_score / mean_wallclock_per_task)
| rank | harness | model | system_prompt_id | score_per_sec_per_task | mean_score | wallclock_s | count |
|---|---|---|---|---|---|---|---|
| 1 | anthropic | claude-opus-4-7 | seed | 2.518892 | 1.0000 | 7.94 | 20 |
| 2 | mlx | mlx-community/Llama-3.2-1B-Instruct-4bit | seed | 0.383877 | 0.4000 | 10.42 | 10 |
| 3 | anthropic | claude-opus-4-7 | seed | 0.235294 | 0.1000 | 8.50 | 20 |
| 4 | codex | gpt-5.5 | seed | 0.231595 | 1.0000 | 86.36 | 20 |
| 5 | codex | gpt-5.5 | seed | 0.207310 | 1.0000 | 24.12 | 5 |
| 6 | codex | gpt-5.5 | seed | 0.156913 | 0.8500 | 108.34 | 20 |

## Footer
- All known harnesses represented.
- No truncated runs.
