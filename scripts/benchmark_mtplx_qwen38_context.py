#!/usr/bin/env python3
"""Benchmark an MTPLX-served Qwen3.8 model across actual prompt sizes."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = Path(
    "/Users/satan/.cache/huggingface/hub/"
    "models--Youssofal--Qwen3.8-27B-MTPLX-Optimized-Speed/"
    "snapshots/57c0ede09cec77a02ff05f19cea5d81df7a20da6"
)
DEFAULT_CONTEXTS = (512, 2048, 8192, 32768, 65536)
DEFAULT_AR_CONTEXTS = (512, 8192)
SYSTEM_PROMPT = (
    "You are a precise benchmark participant. Read the complete supplied context, "
    "follow the requested output contract, and never invent the benchmark needle."
)
CSV_FIELDS = (
    "run_id",
    "mode",
    "target_context",
    "repeat",
    "status",
    "finish_reason",
    "quality_pass",
    "needle",
    "prompt_tokens",
    "completion_tokens",
    "prompt_tps",
    "decode_tps",
    "end_to_end_tps",
    "ttft_s",
    "prompt_elapsed_s",
    "decode_elapsed_s",
    "request_elapsed_s",
    "active_memory_gb",
    "peak_memory_gb",
    "cache_memory_gb",
    "accepted_depth_1",
    "accepted_depth_2",
    "drafted_depth_1",
    "drafted_depth_2",
    "acceptance_depth_1",
    "acceptance_depth_2",
    "fan_rpm_before",
    "fan_rpm_after",
    "hottest_c_before",
    "hottest_c_after",
    "battery_c_before",
    "battery_c_after",
    "load_1m_before",
    "load_1m_after",
    "prompt_sha256",
    "response_file",
    "output_file",
    "error",
)


def parse_int_list(value: str) -> tuple[int, ...]:
    values = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("expected a comma-separated list of positive integers")
    return values


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def json_request(url: str, payload: dict[str, Any] | None, timeout: float) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=data)
    if data is not None:
        request.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def thermal_snapshot() -> dict[str, float | int | None]:
    binary = Path("/Users/satan/.mtplx/bin/thermalforge")
    if not binary.is_file():
        return {"fan_rpm": None, "hottest_c": None, "battery_c": None}
    result = subprocess.run(
        [str(binary), "status"],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    if result.returncode != 0:
        return {"fan_rpm": None, "hottest_c": None, "battery_c": None}
    try:
        payload = json.loads(result.stdout)
        fans = payload.get("fans") or []
        temperatures = payload.get("temperatures") or {}
        return {
            "fan_rpm": int(fans[0]["actual_rpm"]) if fans else None,
            "hottest_c": max(float(value) for value in temperatures.values())
            if temperatures
            else None,
            "battery_c": float(temperatures["TB0T"]) if "TB0T" in temperatures else None,
        }
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return {"fan_rpm": None, "hottest_c": None, "battery_c": None}


def wait_for_start_temperature(maximum_c: float, timeout: float) -> dict[str, float | int | None]:
    deadline = time.monotonic() + timeout
    while True:
        snapshot = thermal_snapshot()
        hottest = snapshot["hottest_c"]
        if hottest is None or float(hottest) <= maximum_c:
            return snapshot
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"hottest sensor stayed at {hottest} C, above start limit {maximum_c} C"
            )
        print(
            f"[cooldown] hottest={hottest}C battery={snapshot['battery_c']}C; "
            f"waiting for <= {maximum_c}C",
            flush=True,
        )
        time.sleep(10)


def command_version(command: list[str]) -> str | None:
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    value = (result.stdout or result.stderr).strip()
    return value or None


def build_filler_tokens(tokenizer: Any, required_tokens: int) -> list[int]:
    record_count = max(512, required_tokens // 40 + 128)
    while True:
        records = []
        for index in range(record_count):
            digest = hashlib.blake2s(f"qwen-context-record-{index}".encode(), digest_size=8).hexdigest()
            subsystem = (index * 37 + 11) % 997
            records.append(
                f"Record {index:06d} | checksum {digest} | subsystem {subsystem:03d} | "
                "Observation: measure latency, memory, thermals, and output quality independently.\n"
            )
        token_ids = tokenizer.encode("".join(records), add_special_tokens=False)
        if len(token_ids) >= required_tokens:
            return token_ids
        record_count *= 2


def build_prompt(
    tokenizer: Any,
    filler_tokens: list[int],
    target_context: int,
    repeat: int,
) -> tuple[str, str, int, str]:
    needle_hash = hashlib.sha256(f"qwen38-{target_context}-{repeat}".encode()).hexdigest()[:12]
    needle = f"CINDER-{target_context}-{repeat}-{needle_hash}"
    intro = (
        "Audit the synthetic records below. Exactly one line is marked BENCHMARK_NEEDLE. "
        "Other checksums are distractors.\n\n"
    )
    needle_line = f'\nBENCHMARK_NEEDLE = "{needle}"\n'
    instruction = (
        "\nReturn the following without markdown:\n"
        "First line: NEEDLE=<exact BENCHMARK_NEEDLE value>\n"
        "Then give a concise original engineering assessment of local-LLM benchmarking "
        "covering latency, throughput, memory, thermals, and quality. Do not quote a Record line."
    )
    fixed_tokens = len(
        tokenizer.encode(intro + needle_line + instruction, add_special_tokens=False)
    )
    content_budget = max(64, target_context - fixed_tokens - 32)
    prefix_count = int(content_budget * 0.73)
    suffix_count = content_budget - prefix_count
    prefix = tokenizer.decode(filler_tokens[:prefix_count], skip_special_tokens=True)
    suffix = tokenizer.decode(
        filler_tokens[prefix_count : prefix_count + suffix_count],
        skip_special_tokens=True,
    )
    content = intro + prefix + needle_line + suffix + instruction
    raw_tokens = len(tokenizer.encode(content, add_special_tokens=False))
    prompt_hash = hashlib.sha256(content.encode()).hexdigest()
    return content, needle, raw_tokens, prompt_hash


def acceptance_ratio(accepted: list[int], drafted: list[int], depth: int) -> float | None:
    index = depth - 1
    if index >= len(accepted) or index >= len(drafted) or not drafted[index]:
        return None
    return accepted[index] / drafted[index]


def gibibytes(value: Any) -> float | None:
    if value is None:
        return None
    return float(value) / (1024**3)


def result_row(
    *,
    run_id: str,
    mode: str,
    target_context: int,
    repeat: int,
    needle: str,
    prompt_hash: str,
    payload: dict[str, Any] | None,
    response_file: str,
    output_file: str,
    before: dict[str, float | int | None],
    after: dict[str, float | int | None],
    load_before: float,
    load_after: float,
    error: str = "",
) -> dict[str, Any]:
    if payload is None:
        return {
            **{field: "" for field in CSV_FIELDS},
            "run_id": run_id,
            "mode": mode,
            "target_context": target_context,
            "repeat": repeat,
            "status": "failed",
            "quality_pass": False,
            "needle": needle,
            "prompt_sha256": prompt_hash,
            "response_file": response_file,
            "output_file": output_file,
            "fan_rpm_before": before["fan_rpm"],
            "fan_rpm_after": after["fan_rpm"],
            "hottest_c_before": before["hottest_c"],
            "hottest_c_after": after["hottest_c"],
            "battery_c_before": before["battery_c"],
            "battery_c_after": after["battery_c"],
            "load_1m_before": load_before,
            "load_1m_after": load_after,
            "error": error,
        }

    choice = payload.get("choices", [{}])[0]
    message = choice.get("message") or {}
    output = message.get("content") or ""
    timings = payload.get("timings") or {}
    stats = payload.get("mtplx_stats") or {}
    accepted = stats.get("accepted_by_depth") or []
    drafted = stats.get("drafted_by_depth") or []
    return {
        "run_id": run_id,
        "mode": mode,
        "target_context": target_context,
        "repeat": repeat,
        "status": "ok",
        "finish_reason": choice.get("finish_reason", ""),
        "quality_pass": needle in output,
        "needle": needle,
        "prompt_tokens": timings.get("prompt_n", stats.get("prompt_tokens")),
        "completion_tokens": timings.get("predicted_n", stats.get("completion_tokens")),
        "prompt_tps": timings.get("prompt_per_second", stats.get("prompt_tps")),
        "decode_tps": timings.get("predicted_per_second", stats.get("decode_tps")),
        "end_to_end_tps": stats.get("request_tok_s", stats.get("end_to_end_tok_s")),
        "ttft_s": stats.get("ttft_s"),
        "prompt_elapsed_s": stats.get("prompt_eval_time_s"),
        "decode_elapsed_s": stats.get("decode_elapsed_s"),
        "request_elapsed_s": stats.get("request_elapsed_s"),
        "active_memory_gb": gibibytes(stats.get("active_memory_bytes")),
        "peak_memory_gb": gibibytes(stats.get("peak_memory_bytes")),
        "cache_memory_gb": gibibytes(stats.get("cache_memory_bytes")),
        "accepted_depth_1": accepted[0] if len(accepted) > 0 else "",
        "accepted_depth_2": accepted[1] if len(accepted) > 1 else "",
        "drafted_depth_1": drafted[0] if len(drafted) > 0 else "",
        "drafted_depth_2": drafted[1] if len(drafted) > 1 else "",
        "acceptance_depth_1": acceptance_ratio(accepted, drafted, 1),
        "acceptance_depth_2": acceptance_ratio(accepted, drafted, 2),
        "fan_rpm_before": before["fan_rpm"],
        "fan_rpm_after": after["fan_rpm"],
        "hottest_c_before": before["hottest_c"],
        "hottest_c_after": after["hottest_c"],
        "battery_c_before": before["battery_c"],
        "battery_c_after": after["battery_c"],
        "load_1m_before": load_before,
        "load_1m_after": load_after,
        "prompt_sha256": prompt_hash,
        "response_file": response_file,
        "output_file": output_file,
        "error": error,
    }


def write_summary(rows: list[dict[str, Any]], path: Path) -> None:
    lines = [
        "# MTPLX Qwen3.8 context benchmark",
        "",
        "| Mode | Target | Prompt tokens | Prefill tok/s | Decode tok/s | TTFT s | Active GB | Peak GB | D1 accept | D2 accept | Quality | Status |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for row in rows:
        def number(key: str, digits: int = 2) -> str:
            value = row.get(key)
            return "—" if value in (None, "") else f"{float(value):.{digits}f}"

        formatted = dict(row)
        formatted.update(
            prompt_tokens=row.get("prompt_tokens") or "—",
            prompt_tps=number("prompt_tps"),
            decode_tps=number("decode_tps"),
            ttft_s=number("ttft_s", 3),
            active_memory_gb=number("active_memory_gb"),
            peak_memory_gb=number("peak_memory_gb"),
            acceptance_depth_1=number("acceptance_depth_1", 3),
            acceptance_depth_2=number("acceptance_depth_2", 3),
        )
        lines.append(
            "| {mode} | {target_context} | {prompt_tokens} | {prompt_tps} | "
            "{decode_tps} | {ttft_s} | {active_memory_gb} | {peak_memory_gb} | "
            "{acceptance_depth_1} | {acceptance_depth_2} | {quality_pass} | {status} |".format(
                **formatted
            )
        )
    path.write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default="http://127.0.0.1:11500")
    parser.add_argument("--model", default="mtplx-qwen38-27b-optimized-speed")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--contexts", type=parse_int_list, default=DEFAULT_CONTEXTS)
    parser.add_argument("--ar-contexts", type=parse_int_list, default=DEFAULT_AR_CONTEXTS)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--short-repeats", type=int, default=2)
    parser.add_argument("--long-repeats", type=int, default=1)
    parser.add_argument("--repeat-cutoff", type=int, default=8192)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=1200)
    parser.add_argument("--cooldown-seconds", type=float, default=2)
    parser.add_argument("--max-start-temp", type=float, default=65)
    parser.add_argument("--cooldown-timeout", type=float, default=900)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--no-ar-reference", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.max_tokens <= 0 or args.short_repeats <= 0 or args.long_repeats <= 0:
        parser.error("token and repeat counts must be positive")
    return args


def main() -> int:
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        local_files_only=True,
        trust_remote_code=True,
    )
    all_contexts = args.contexts if args.no_ar_reference else args.contexts + args.ar_contexts
    max_context = max(all_contexts)
    filler_tokens = build_filler_tokens(tokenizer, max_context + 2048)

    planned: list[tuple[str, int, int]] = []
    for context in args.contexts:
        repeats = args.short_repeats if context <= args.repeat_cutoff else args.long_repeats
        planned.extend(("mtp", context, repeat) for repeat in range(1, repeats + 1))
    if not args.no_ar_reference:
        planned.extend(("ar", context, 1) for context in args.ar_contexts)

    prompt_cache: dict[tuple[int, int], tuple[str, str, int, str]] = {}
    for _, context, repeat in planned:
        prompt_cache[(context, repeat)] = build_prompt(tokenizer, filler_tokens, context, repeat)

    if args.dry_run:
        for mode, context, repeat in planned:
            _, needle, raw_tokens, prompt_hash = prompt_cache[(context, repeat)]
            print(
                json.dumps(
                    {
                        "mode": mode,
                        "target_context": context,
                        "repeat": repeat,
                        "raw_user_tokens": raw_tokens,
                        "needle": needle,
                        "prompt_sha256": prompt_hash,
                    }
                )
            )
        return 0

    health = json_request(f"{args.endpoint}/health", None, 30)
    server_context = int(health.get("context_window") or 0)
    required_context = max_context + args.max_tokens
    if server_context and server_context < required_context:
        raise SystemExit(
            f"server context window {server_context} is smaller than required {required_context}"
        )

    output_dir = args.output_dir or (
        ROOT / "artifacts" / "benchmarks" / f"qwen38-mtplx-context-{utc_stamp()}"
    )
    raw_dir = output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=False)
    config = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": sys.argv,
        "endpoint": args.endpoint,
        "model": args.model,
        "model_path": str(args.model_path),
        "model_revision": args.model_path.name,
        "contexts": args.contexts,
        "ar_contexts": () if args.no_ar_reference else args.ar_contexts,
        "max_tokens": args.max_tokens,
        "short_repeats": args.short_repeats,
        "long_repeats": args.long_repeats,
        "repeat_cutoff": args.repeat_cutoff,
        "depth": args.depth,
        "max_start_temp": args.max_start_temp,
        "cooldown_timeout": args.cooldown_timeout,
        "system_prompt": SYSTEM_PROMPT,
        "prompt_generator": "unique indexed records, BLAKE2s checksums, needle at 73%",
        "server_health": health,
        "platform": platform.platform(),
        "python": sys.version,
        "mtplx_version": command_version(["mtplx", "--version"]),
        "git_commit": command_version(["git", "rev-parse", "HEAD"]),
    }
    (output_dir / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True))
    (output_dir / "prompt-spec.txt").write_text(
        "Deterministic unique indexed records with BLAKE2s checksums.\n"
        "One BENCHMARK_NEEDLE line is inserted after 73% of filler tokens.\n"
        "The exact generated prompt is identified by prompt_sha256 in results.csv.\n"
    )

    rows: list[dict[str, Any]] = []
    results_csv = output_dir / "results.csv"
    with results_csv.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for index, (mode, context, repeat) in enumerate(planned, start=1):
            run_id = f"{index:04d}_{mode}_ctx{context}_rep{repeat}"
            content, needle, raw_tokens, prompt_hash = prompt_cache[(context, repeat)]
            response_rel = f"raw/{run_id}.response.json"
            output_rel = f"raw/{run_id}.output.txt"
            response_path = output_dir / response_rel
            output_path = output_dir / output_rel
            request_meta = {
                "run_id": run_id,
                "mode": mode,
                "target_context": context,
                "raw_user_tokens": raw_tokens,
                "prompt_sha256": prompt_hash,
                "needle": needle,
                "max_tokens": args.max_tokens,
                "depth": args.depth if mode == "mtp" else 0,
            }
            (raw_dir / f"{run_id}.request-meta.json").write_text(
                json.dumps(request_meta, indent=2, sort_keys=True)
            )
            payload = {
                "model": args.model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": content},
                ],
                "max_tokens": args.max_tokens,
                "temperature": 0,
                "top_p": 1,
                "seed": context * 10 + repeat,
                "stream": False,
                "generation_mode": mode,
                "reasoning": "off",
            }
            if mode == "mtp":
                payload["depth"] = args.depth

            print(
                f"[{run_id}] start raw_user_tokens={raw_tokens} max_tokens={args.max_tokens}",
                flush=True,
            )
            try:
                before = wait_for_start_temperature(args.max_start_temp, args.cooldown_timeout)
            except KeyboardInterrupt:
                print(f"[{run_id}] cancelled during thermal cooldown", flush=True)
                return 130
            load_before = os.getloadavg()[0]
            response_payload: dict[str, Any] | None = None
            error = ""
            interrupted = False
            try:
                response_payload = json_request(
                    f"{args.endpoint}/v1/chat/completions", payload, args.timeout
                )
                response_path.write_text(json.dumps(response_payload, indent=2))
                output = (
                    response_payload.get("choices", [{}])[0]
                    .get("message", {})
                    .get("content", "")
                )
                output_path.write_text(output)
            except urllib.error.HTTPError as exc:
                error = f"HTTP {exc.code}: {exc.read().decode('utf-8', errors='replace')}"
                response_path.write_text(json.dumps({"error": error}, indent=2))
            except KeyboardInterrupt:
                interrupted = True
                error = "cancelled by operator"
                response_path.write_text(json.dumps({"error": error}, indent=2))
            except Exception as exc:  # preserve a failed matrix row and continue probing
                error = f"{type(exc).__name__}: {exc}"
                response_path.write_text(json.dumps({"error": error}, indent=2))
            after = thermal_snapshot()
            load_after = os.getloadavg()[0]
            row = result_row(
                run_id=run_id,
                mode=mode,
                target_context=context,
                repeat=repeat,
                needle=needle,
                prompt_hash=prompt_hash,
                payload=response_payload,
                response_file=response_rel,
                output_file=output_rel,
                before=before,
                after=after,
                load_before=load_before,
                load_after=load_after,
                error=error,
            )
            rows.append(row)
            writer.writerow(row)
            csv_file.flush()
            (output_dir / "results.json").write_text(json.dumps(rows, indent=2))
            write_summary(rows, output_dir / "summary.md")
            print(
                f"[{run_id}] {row['status']} prompt={row.get('prompt_tokens') or 'NA'} "
                f"prefill={row.get('prompt_tps') or 'NA'} decode={row.get('decode_tps') or 'NA'} "
                f"quality={row.get('quality_pass')}",
                flush=True,
            )
            if args.cooldown_seconds:
                time.sleep(args.cooldown_seconds)
            if interrupted:
                print(f"[{run_id}] cancelled; partial artifacts retained", flush=True)
                return 130

    print(f"Benchmark artifacts: {output_dir}")
    return 0 if all(row["status"] == "ok" for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
