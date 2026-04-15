#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

DEFAULT_MODEL = os.environ.get(
    "SUPERGEMMA_MODEL",
    "Jiunsong/supergemma4-26b-uncensored-mlx-4bit-v2",
)


def positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Expected integer, got: {value}") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"Expected positive integer, got: {value}")
    return parsed


def non_negative_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Expected integer, got: {value}") from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError(f"Expected non-negative integer, got: {value}")
    return parsed


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def jsonl_dump(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def make_run_dir(base_dir: Path, timestamp_override: str | None) -> Path:
    stamp = timestamp_override or dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    candidate = base_dir / stamp
    suffix = 1
    while candidate.exists():
        candidate = base_dir / f"{stamp}-{suffix:02d}"
        suffix += 1
    candidate.mkdir(parents=True, exist_ok=False)
    return candidate


def chunk_spans(total_tokens: int, chunk_size: int, chunk_overlap: int) -> list[tuple[int, int]]:
    step = chunk_size - chunk_overlap
    spans: list[tuple[int, int]] = []
    start = 0
    while start < total_tokens:
        end = min(start + chunk_size, total_tokens)
        spans.append((start, end))
        if end >= total_tokens:
            break
        start += step
    return spans


def shorten(text: str, limit: int = 200) -> str:
    squashed = " ".join(text.split())
    if len(squashed) <= limit:
        return squashed
    return squashed[: limit - 1] + "…"


def build_map_prompt(
    *,
    objective: str,
    chunk_text: str,
    chunk_idx: int,
    chunk_total: int,
    token_start: int,
    token_end: int,
) -> str:
    return f"""You are helping analyze a very long document.

Objective / question:
{objective}

You are given chunk {chunk_idx}/{chunk_total} (token range {token_start}:{token_end}).
Summarize only facts relevant to the objective.
If this chunk is irrelevant, write: "No relevant evidence in this chunk."
Prefer concise bullet points.

Chunk text:
{chunk_text}
"""


def build_reduce_prompt(
    *,
    objective: str,
    level: int,
    group_idx: int,
    group_total: int,
    summaries: list[dict[str, Any]],
) -> str:
    rendered = []
    for index, summary in enumerate(summaries, start=1):
        rendered.append(f"Summary {index} (source={summary['id']}):\n{summary['text']}")

    combined = "\n\n".join(rendered)
    return f"""You are reducing map summaries from a long-document pipeline.

Objective / question:
{objective}

Current reduce stage: level {level}, group {group_idx}/{group_total}.
Merge the summaries into one concise summary, preserving critical details and disagreements.
If the objective is unanswered, explicitly state what is still missing.

Summaries:
{combined}
"""


def build_final_prompt(objective: str, summary_nodes: list[dict[str, Any]]) -> str:
    rendered = []
    for index, summary in enumerate(summary_nodes, start=1):
        rendered.append(f"Candidate summary {index} (source={summary['id']}):\n{summary['text']}")

    combined = "\n\n".join(rendered)
    return f"""You are producing the final answer from reduced summaries.

Objective / question:
{objective}

Using only the information below, produce:
1) Final answer
2) Key evidence bullets
3) Remaining uncertainty (if any)

Reduced summaries:
{combined}
"""


def build_mlx_command(
    *,
    mlx_generate_bin: Path,
    model: str,
    max_tokens: int,
    max_kv_size: int,
    trust_remote_code: bool,
    ignore_chat_template: bool,
    temperature: float | None,
    top_p: float | None,
    seed: int | None,
) -> list[str]:
    command = [
        str(mlx_generate_bin),
        "--model",
        model,
        "--prompt",
        "-",
        "--max-tokens",
        str(max_tokens),
        "--max-kv-size",
        str(max_kv_size),
        "--verbose",
        "False",
    ]
    if trust_remote_code:
        command.append("--trust-remote-code")
    if ignore_chat_template:
        command.append("--ignore-chat-template")
    if temperature is not None:
        command.extend(["--temp", str(temperature)])
    if top_p is not None:
        command.extend(["--top-p", str(top_p)])
    if seed is not None:
        command.extend(["--seed", str(seed)])
    return command


def run_generation(
    *,
    stage_label: str,
    stage_dir: Path,
    artifact_prefix: str,
    prompt: str,
    dry_run_output: str,
    dry_run: bool,
    command: list[str],
) -> str:
    stage_dir.mkdir(parents=True, exist_ok=True)
    prompt_path = stage_dir / f"{artifact_prefix}_prompt.txt"
    command_path = stage_dir / f"{artifact_prefix}_command.txt"
    stdout_path = stage_dir / f"{artifact_prefix}_stdout.txt"
    stderr_path = stage_dir / f"{artifact_prefix}_stderr.txt"
    output_path = stage_dir / f"{artifact_prefix}_output.txt"

    prompt_path.write_text(prompt, encoding="utf-8")
    command_path.write_text(shlex.join(command) + "\n", encoding="utf-8")

    if dry_run:
        stdout_path.write_text("[dry-run] generation skipped\n", encoding="utf-8")
        stderr_path.write_text("", encoding="utf-8")
        output = dry_run_output.strip() or "[dry-run] no output"
        output_path.write_text(output + "\n", encoding="utf-8")
        return output

    process = subprocess.run(command, text=True, input=prompt, capture_output=True, check=False)
    stdout_path.write_text(process.stdout, encoding="utf-8")
    stderr_path.write_text(process.stderr, encoding="utf-8")

    if process.returncode != 0:
        raise RuntimeError(
            f"{stage_label} failed (exit {process.returncode}). "
            f"See {stderr_path} and {stdout_path}."
        )

    output = process.stdout.strip()
    if not output:
        raise RuntimeError(f"{stage_label} returned empty output. See {stdout_path}.")

    output_path.write_text(output + "\n", encoding="utf-8")
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Chunk large text by tokenizer tokens, run map summaries with mlx_lm.generate, "
            "recursively reduce summaries, and write run artifacts."
        )
    )
    parser.add_argument("--input-file", required=True, help="Path to source text file.")
    parser.add_argument(
        "--question",
        "--objective",
        dest="question",
        required=True,
        help="Question/objective for map-reduce summarization.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Model id/path (default: {DEFAULT_MODEL}).")
    parser.add_argument(
        "--tokenizer",
        default=None,
        help="Tokenizer id/path for AutoTokenizer (default: --model).",
    )
    parser.add_argument(
        "--mlx-generate-bin",
        default=str(Path(__file__).resolve().parents[1] / ".venv" / "bin" / "mlx_lm.generate"),
        help="Path to mlx_lm.generate binary.",
    )
    parser.add_argument(
        "--artifacts-root",
        default=str(Path(__file__).resolve().parents[1] / "artifacts" / "long-context"),
        help="Base output directory for timestamped artifacts.",
    )
    parser.add_argument("--timestamp", default=None, help="Override artifact timestamp folder name.")
    parser.add_argument("--encoding", default="utf-8", help="Input file encoding (default: utf-8).")
    parser.add_argument("--chunk-size", type=positive_int, default=3072, help="Token chunk size.")
    parser.add_argument("--chunk-overlap", type=non_negative_int, default=256, help="Token overlap.")
    parser.add_argument("--map-max-tokens", type=positive_int, default=192, help="Max tokens for map outputs.")
    parser.add_argument(
        "--reduce-max-tokens",
        type=positive_int,
        default=256,
        help="Max tokens for each reduce output.",
    )
    parser.add_argument(
        "--final-max-tokens",
        type=positive_int,
        default=384,
        help="Max tokens for final synthesis output.",
    )
    parser.add_argument("--reduce-fan-in", type=positive_int, default=8, help="Summaries per reduce call.")
    parser.add_argument("--max-kv-size", type=positive_int, default=4096, help="KV cache cap per generation.")
    parser.add_argument("--temperature", type=float, default=None, help="Optional sampling temperature.")
    parser.add_argument("--top-p", type=float, default=None, help="Optional top-p.")
    parser.add_argument("--seed", type=int, default=None, help="Optional random seed.")
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass --trust-remote-code to tokenizer and generation.",
    )
    parser.add_argument(
        "--ignore-chat-template",
        action="store_true",
        help="Pass --ignore-chat-template to mlx_lm.generate.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Skip model execution; still chunk and emit artifacts.")
    parser.add_argument(
        "--write-chunk-text",
        action="store_true",
        help="Write decoded chunk text files under artifacts/map/chunks.",
    )

    args = parser.parse_args()

    if args.chunk_overlap >= args.chunk_size:
        parser.error("--chunk-overlap must be smaller than --chunk-size.")
    if args.reduce_fan_in < 2:
        parser.error("--reduce-fan-in must be at least 2.")
    if args.temperature is not None and args.temperature < 0:
        parser.error("--temperature must be >= 0.")
    if args.top_p is not None and not (0 < args.top_p <= 1):
        parser.error("--top-p must be in (0, 1].")
    return args


def main() -> int:
    args = parse_args()

    input_path = Path(args.input_file).expanduser().resolve()
    if not input_path.is_file():
        print(f"ERROR: Input file not found: {input_path}", file=sys.stderr)
        return 1

    try:
        text = input_path.read_text(encoding=args.encoding)
    except Exception as exc:
        print(f"ERROR: Failed reading input file {input_path}: {exc}", file=sys.stderr)
        return 1

    if not text.strip():
        print(f"ERROR: Input file is empty: {input_path}", file=sys.stderr)
        return 1

    mlx_generate_bin = Path(args.mlx_generate_bin).expanduser().resolve()
    if not mlx_generate_bin.exists():
        print(
            f"ERROR: mlx_lm.generate not found at {mlx_generate_bin}. "
            "Run scripts/setup_env.sh first.",
            file=sys.stderr,
        )
        return 1
    if not os.access(mlx_generate_bin, os.X_OK):
        print(f"ERROR: mlx_lm.generate is not executable: {mlx_generate_bin}", file=sys.stderr)
        return 1

    try:
        from transformers import AutoTokenizer
    except Exception as exc:
        print(
            "ERROR: transformers is required for token chunking. "
            "Install dependencies with scripts/setup_env.sh or pip install -r requirements.txt.\n"
            f"Import error: {exc}",
            file=sys.stderr,
        )
        return 1

    artifacts_root = Path(args.artifacts_root).expanduser().resolve()
    run_dir = make_run_dir(artifacts_root, args.timestamp)
    map_dir = run_dir / "map"
    map_chunks_dir = map_dir / "chunks"
    reduce_dir = run_dir / "reduce"
    final_dir = run_dir / "final"
    run_error_path = run_dir / "error.txt"

    tokenizer_name = args.tokenizer or args.model
    try:
        print(f"[info] Loading tokenizer: {tokenizer_name}", file=sys.stderr, flush=True)
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            trust_remote_code=args.trust_remote_code,
        )
    except Exception as exc:
        run_error_path.write_text(
            f"Failed to load tokenizer {tokenizer_name}: {exc}\n", encoding="utf-8"
        )
        print(
            f"ERROR: Failed to load tokenizer '{tokenizer_name}'. {exc}",
            file=sys.stderr,
        )
        return 1

    try:
        token_ids = tokenizer.encode(text, add_special_tokens=False)
    except Exception as exc:
        run_error_path.write_text(f"Failed to tokenize input: {exc}\n", encoding="utf-8")
        print(f"ERROR: Tokenization failed: {exc}", file=sys.stderr)
        return 1

    if not token_ids:
        run_error_path.write_text("Tokenizer produced zero tokens.\n", encoding="utf-8")
        print("ERROR: Tokenizer produced zero tokens for input.", file=sys.stderr)
        return 1

    spans = chunk_spans(
        total_tokens=len(token_ids),
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
    print(
        f"[info] Tokenized {len(token_ids)} tokens into {len(spans)} chunks "
        f"(chunk_size={args.chunk_size}, overlap={args.chunk_overlap})",
        file=sys.stderr,
        flush=True,
    )

    write_json(
        run_dir / "config.json",
        {
            "input_file": str(input_path),
            "question": args.question,
            "model": args.model,
            "tokenizer": tokenizer_name,
            "chunk_size": args.chunk_size,
            "chunk_overlap": args.chunk_overlap,
            "reduce_fan_in": args.reduce_fan_in,
            "map_max_tokens": args.map_max_tokens,
            "reduce_max_tokens": args.reduce_max_tokens,
            "final_max_tokens": args.final_max_tokens,
            "max_kv_size": args.max_kv_size,
            "dry_run": args.dry_run,
            "trust_remote_code": args.trust_remote_code,
            "ignore_chat_template": args.ignore_chat_template,
            "artifacts_dir": str(run_dir),
            "timestamp_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        },
    )
    write_json(
        run_dir / "token_stats.json",
        {
            "input_token_count": len(token_ids),
            "chunk_count": len(spans),
            "chunk_size": args.chunk_size,
            "chunk_overlap": args.chunk_overlap,
            "step_size": args.chunk_size - args.chunk_overlap,
        },
    )

    map_records: list[dict[str, Any]] = []
    summary_nodes: list[dict[str, Any]] = []

    try:
        for idx, (start, end) in enumerate(spans, start=1):
            chunk_tokens = token_ids[start:end]
            chunk_text = tokenizer.decode(
                chunk_tokens,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            map_prompt = build_map_prompt(
                objective=args.question,
                chunk_text=chunk_text,
                chunk_idx=idx,
                chunk_total=len(spans),
                token_start=start,
                token_end=end,
            )
            dry_run_summary = (
                f"[dry-run] Chunk {idx}/{len(spans)} summary candidate: "
                f"{shorten(chunk_text, 220)}"
            )
            map_command = build_mlx_command(
                mlx_generate_bin=mlx_generate_bin,
                model=args.model,
                max_tokens=args.map_max_tokens,
                max_kv_size=args.max_kv_size,
                trust_remote_code=args.trust_remote_code,
                ignore_chat_template=args.ignore_chat_template,
                temperature=args.temperature,
                top_p=args.top_p,
                seed=args.seed,
            )
            print(f"[map] chunk {idx}/{len(spans)}", file=sys.stderr, flush=True)
            output = run_generation(
                stage_label=f"map chunk {idx}/{len(spans)}",
                stage_dir=map_dir,
                artifact_prefix=f"chunk_{idx:04d}",
                prompt=map_prompt,
                dry_run_output=dry_run_summary,
                dry_run=args.dry_run,
                command=map_command,
            )
            if args.write_chunk_text:
                map_chunks_dir.mkdir(parents=True, exist_ok=True)
                (map_chunks_dir / f"chunk_{idx:04d}.txt").write_text(chunk_text, encoding="utf-8")

            map_record = {
                "id": f"chunk_{idx:04d}",
                "chunk_index": idx,
                "chunk_count": len(spans),
                "token_start": start,
                "token_end": end,
                "token_count": end - start,
                "summary": output,
            }
            map_records.append(map_record)
            summary_nodes.append({"id": map_record["id"], "text": output})
    except Exception as exc:
        run_error_path.write_text(f"Map stage failed: {exc}\n", encoding="utf-8")
        print(f"ERROR: Map stage failed: {exc}", file=sys.stderr)
        return 1

    jsonl_dump(run_dir / "chunk_summaries.jsonl", map_records)

    reduce_records: list[dict[str, Any]] = []
    level = 1
    try:
        while len(summary_nodes) > 1:
            groups = [
                summary_nodes[i : i + args.reduce_fan_in]
                for i in range(0, len(summary_nodes), args.reduce_fan_in)
            ]
            next_nodes: list[dict[str, Any]] = []
            print(
                f"[reduce] level {level} has {len(summary_nodes)} summaries, "
                f"{len(groups)} group(s)",
                file=sys.stderr,
                flush=True,
            )
            for group_idx, group in enumerate(groups, start=1):
                reduce_prompt = build_reduce_prompt(
                    objective=args.question,
                    level=level,
                    group_idx=group_idx,
                    group_total=len(groups),
                    summaries=group,
                )
                dry_run_reduce = (
                    f"[dry-run] Reduce level {level} group {group_idx}/{len(groups)} "
                    f"merged {len(group)} summaries:\n"
                    + "\n".join(f"- {shorten(item['text'], 120)}" for item in group)
                )
                reduce_command = build_mlx_command(
                    mlx_generate_bin=mlx_generate_bin,
                    model=args.model,
                    max_tokens=args.reduce_max_tokens,
                    max_kv_size=args.max_kv_size,
                    trust_remote_code=args.trust_remote_code,
                    ignore_chat_template=args.ignore_chat_template,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    seed=args.seed,
                )
                output = run_generation(
                    stage_label=f"reduce level {level} group {group_idx}",
                    stage_dir=reduce_dir / f"level_{level:02d}",
                    artifact_prefix=f"group_{group_idx:03d}",
                    prompt=reduce_prompt,
                    dry_run_output=dry_run_reduce,
                    dry_run=args.dry_run,
                    command=reduce_command,
                )
                node_id = f"level_{level:02d}_group_{group_idx:03d}"
                next_nodes.append({"id": node_id, "text": output})
                reduce_records.append(
                    {
                        "level": level,
                        "group_index": group_idx,
                        "group_count": len(groups),
                        "input_ids": [item["id"] for item in group],
                        "output_id": node_id,
                        "output": output,
                    }
                )
            summary_nodes = next_nodes
            level += 1
    except Exception as exc:
        run_error_path.write_text(f"Reduce stage failed: {exc}\n", encoding="utf-8")
        print(f"ERROR: Reduce stage failed: {exc}", file=sys.stderr)
        return 1

    write_json(run_dir / "reduce_levels.json", {"levels": reduce_records})

    final_prompt = build_final_prompt(args.question, summary_nodes)
    dry_run_final = (
        "[dry-run] Final synthesized answer from reduced summaries:\n"
        + "\n".join(f"- {shorten(node['text'], 150)}" for node in summary_nodes)
    )
    final_command = build_mlx_command(
        mlx_generate_bin=mlx_generate_bin,
        model=args.model,
        max_tokens=args.final_max_tokens,
        max_kv_size=args.max_kv_size,
        trust_remote_code=args.trust_remote_code,
        ignore_chat_template=args.ignore_chat_template,
        temperature=args.temperature,
        top_p=args.top_p,
        seed=args.seed,
    )
    try:
        final_answer = run_generation(
            stage_label="final synthesis",
            stage_dir=final_dir,
            artifact_prefix="final",
            prompt=final_prompt,
            dry_run_output=dry_run_final,
            dry_run=args.dry_run,
            command=final_command,
        )
    except Exception as exc:
        run_error_path.write_text(f"Final stage failed: {exc}\n", encoding="utf-8")
        print(f"ERROR: Final stage failed: {exc}", file=sys.stderr)
        return 1

    (run_dir / "final_answer.txt").write_text(final_answer + "\n", encoding="utf-8")
    write_json(
        run_dir / "final_answer.json",
        {
            "question": args.question,
            "final_answer": final_answer,
            "source_summary_ids": [node["id"] for node in summary_nodes],
        },
    )
    write_json(
        run_dir / "run_summary.json",
        {
            "artifacts_dir": str(run_dir),
            "input_file": str(input_path),
            "input_token_count": len(token_ids),
            "chunk_count": len(spans),
            "reduce_call_count": len(reduce_records),
            "final_answer_file": str(run_dir / "final_answer.txt"),
            "dry_run": args.dry_run,
        },
    )

    print(f"[done] artifacts: {run_dir}", file=sys.stderr)
    print(f"[done] final answer: {run_dir / 'final_answer.txt'}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
