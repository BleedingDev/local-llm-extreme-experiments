#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
import tomllib
import uuid
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TASK_ROOT = ROOT / "bench" / "datasets" / "tb21-probe" / "terminal-bench-2-1"
LC_REPO = ROOT / "bench" / "vendor" / "little-coder"
DEFAULT_TASKS = [
    "regex-log",
    "log-summary-date-ranges",
    "openssl-selfsigned-cert",
    "sqlite-db-truncate",
    "count-dataset-tokens",
]
TIMING_RE = re.compile(
    r"(?P<kind>prompt eval|eval) time =\s+(?P<ms>[0-9.]+) ms /\s+"
    r"(?P<tokens>[0-9]+) tokens .*?,\s+(?P<tps>[0-9.]+) tokens per second"
)


def run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, **kwargs)


def require_ok(cp: subprocess.CompletedProcess[str], label: str) -> None:
    if cp.returncode != 0:
        raise RuntimeError(f"{label} failed ({cp.returncode})\nSTDOUT:\n{cp.stdout}\nSTDERR:\n{cp.stderr}")


def parse_task(task_dir: Path) -> dict:
    data = tomllib.loads((task_dir / "task.toml").read_text())
    return data


def parse_timings(log_text: str) -> dict:
    proxy_records = []
    for line in log_text.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if record.get("path", "").rstrip("/").endswith("/chat/completions"):
            proxy_records.append(record)

    if proxy_records:
        prompt_tokens = sum(int(r.get("prompt_tokens") or 0) for r in proxy_records)
        completion_tokens = sum(
            int(r.get("completion_tokens") or r.get("estimated_completion_tokens") or 0)
            for r in proxy_records
        )
        elapsed = sum(float(r.get("elapsed_s") or 0.0) for r in proxy_records)
        completion_rates = [
            float(r.get("completion_tps") or r.get("estimated_completion_tps") or 0.0)
            for r in proxy_records
            if int(r.get("completion_tokens") or r.get("estimated_completion_tokens") or 0) > 0
        ]
        server_rates = [
            float(r["server_completion_tps"])
            for r in proxy_records
            if r.get("server_completion_tps") is not None
        ]
        finish_reasons: dict[str, int] = {}
        for record in proxy_records:
            reason = str(record.get("finish_reason"))
            finish_reasons[reason] = finish_reasons.get(reason, 0) + 1
        return {
            "source": "openai_proxy_jsonl",
            "requests": len(proxy_records),
            "prompt_tokens": prompt_tokens,
            "generation_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "prompt_seconds": None,
            "generation_seconds": round(elapsed, 3),
            "prompt_tps": None,
            "generation_tps": round(completion_tokens / elapsed, 3) if elapsed else None,
            "mean_generation_tps_per_request": (
                round(sum(completion_rates) / len(completion_rates), 3)
                if completion_rates
                else None
            ),
            "mean_server_generation_tps_per_request": (
                round(sum(server_rates) / len(server_rates), 3) if server_rates else None
            ),
            "finish_reasons": finish_reasons,
        }

    prompt_tokens = prompt_ms = gen_tokens = gen_ms = 0.0
    prompt_rates: list[float] = []
    gen_rates: list[float] = []
    requests = 0
    for line in log_text.splitlines():
        m = TIMING_RE.search(line)
        if not m:
            continue
        tokens = int(m.group("tokens"))
        ms = float(m.group("ms"))
        tps = float(m.group("tps"))
        if m.group("kind") == "prompt eval":
            requests += 1
            prompt_tokens += tokens
            prompt_ms += ms
            prompt_rates.append(tps)
        else:
            gen_tokens += tokens
            gen_ms += ms
            gen_rates.append(tps)
    return {
        "source": "llama_log",
        "requests": requests,
        "prompt_tokens": int(prompt_tokens),
        "generation_tokens": int(gen_tokens),
        "total_tokens": int(prompt_tokens + gen_tokens),
        "prompt_seconds": round(prompt_ms / 1000.0, 3),
        "generation_seconds": round(gen_ms / 1000.0, 3),
        "prompt_tps": round(prompt_tokens / (prompt_ms / 1000.0), 3) if prompt_ms else None,
        "generation_tps": round(gen_tokens / (gen_ms / 1000.0), 3) if gen_ms else None,
        "mean_prompt_tps_per_request": round(sum(prompt_rates) / len(prompt_rates), 3) if prompt_rates else None,
        "mean_generation_tps_per_request": round(sum(gen_rates) / len(gen_rates), 3) if gen_rates else None,
    }


class DockerShell:
    def __init__(self, container: str):
        self.container = container
        self.cwd = "/app"

    def run(self, command: str, timeout: int) -> str:
        sentinel = f"__LC_END_{uuid.uuid4().hex[:8]}__"
        wrapped = (
            f"{command}\n"
            "__lc_rc=$?\n"
            f"printf '\\n{sentinel}:%s:' \"$__lc_rc\"\n"
            "pwd\n"
        )
        try:
            cp = run(
                ["docker", "exec", "-i", "-w", self.cwd, self.container, "bash", "-lc", wrapped],
                timeout=timeout,
            )
            out = cp.stdout or ""
            err = cp.stderr or ""
            code = cp.returncode
        except subprocess.TimeoutExpired as exc:
            out = exc.stdout or ""
            err = (exc.stderr or "") + "\ncommand timed out"
            code = -1
            return self._format(out, err, code, True)

        marker = out.rfind(sentinel + ":")
        if marker >= 0:
            body = out[:marker].rstrip()
            tail = out[marker + len(sentinel) + 1 :]
            parts = tail.split(":", 1)
            try:
                code = int(parts[0])
            except (ValueError, IndexError):
                pass
            if len(parts) > 1:
                cwd_line = parts[1].lstrip("\r\n").splitlines()
                if cwd_line and cwd_line[0].strip():
                    self.cwd = cwd_line[0].strip()
            out = body
        return self._format(out, err, code, False)

    def reset(self) -> str:
        self.cwd = "/app"
        return "shell reset (cwd -> /app)"

    def _format(self, stdout: str, stderr: str, code: int, timed_out: bool) -> str:
        body = stdout or ""
        if stderr:
            body += ("\n[stderr]\n" if body else "[stderr]\n") + stderr
        footer = f"[exit={code} cwd={self.cwd} timed_out={'true' if timed_out else 'false'} backend=docker-direct]"
        return f"{body.rstrip()}\n{footer}" if body.strip() else footer


def start_container(task: str, image: str, task_dir: Path, out_dir: Path) -> str:
    name = f"ornith-tb21-{task}-{uuid.uuid4().hex[:6]}".replace("_", "-")
    cp = run(["docker", "run", "-d", "--rm", "--name", name, "-w", "/app", image, "sleep", "infinity"], timeout=900)
    require_ok(cp, f"docker run {image}")
    run(["docker", "exec", name, "bash", "-lc", "rm -rf /tests /logs && mkdir -p /tests /logs/verifier"], timeout=60)
    for child in sorted((task_dir / "tests").iterdir()):
        cp_tests = run(["docker", "cp", str(child), f"{name}:/tests/{child.name}"], timeout=120)
        require_ok(cp_tests, f"copy {child.name} for {task}")
    cp_check = run(["docker", "exec", name, "bash", "-lc", "test -f /tests/test.sh && test -f /tests/test_outputs.py"], timeout=30)
    require_ok(cp_check, f"verify tests copied for {task}")
    (out_dir / "container.txt").write_text(name + "\n")
    return name


def verify(container: str, timeout: int) -> dict:
    started = time.time()
    cp = run(["docker", "exec", "-w", "/app", container, "bash", "/tests/test.sh"], timeout=timeout)
    elapsed = time.time() - started
    reward_cp = run(["docker", "exec", container, "bash", "-lc", "cat /logs/verifier/reward.txt 2>/dev/null || true"], timeout=30)
    reward_txt = (reward_cp.stdout or "").strip()
    try:
        reward = float(reward_txt)
    except ValueError:
        reward = None
    return {
        "returncode": cp.returncode,
        "seconds": round(elapsed, 3),
        "reward": reward,
        "stdout_tail": (cp.stdout or "")[-4000:],
        "stderr_tail": (cp.stderr or "")[-4000:],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--llama-log", required=True)
    parser.add_argument("--model", default=os.environ.get("TB_MODEL", "omlx/mlx-community/Ornith-1.0-35B-4bit"))
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--verify-timeout", type=int, default=900)
    parser.add_argument("--max-turns", type=int, default=20)
    parser.add_argument("--max-output-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--thinking", choices=["true", "false"], default=None)
    parser.add_argument("tasks", nargs="*", default=DEFAULT_TASKS)
    args = parser.parse_args()

    sys.path.insert(0, str(LC_REPO))
    from benchmarks.rpc_client import PiRpc

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    llama_log = Path(args.llama_log)

    chat_template_kwargs = os.environ.get("LITTLE_CODER_CHAT_TEMPLATE_KWARGS")
    if args.thinking is not None:
        chat_template_kwargs = json.dumps({"enable_thinking": args.thinking == "true"})

    env = {
        "OMLX_API_KEY": "noop",
        "ORNITHCPP_API_KEY": "noop",
        "LITTLE_CODER_TEMPERATURE_PROVIDERS": "llamacpp,ollama,lmstudio,ornithcpp,omlx",
        "LITTLE_CODER_CHAT_TEMPLATE_KWARGS": os.environ.get(
            "LITTLE_CODER_CHAT_TEMPLATE_KWARGS",
            '{"enable_thinking":false}',
        ),
        "LITTLE_CODER_MAX_TOKENS": str(args.max_output_tokens),
    }
    if chat_template_kwargs:
        env["LITTLE_CODER_CHAT_TEMPLATE_KWARGS"] = chat_template_kwargs
    if args.temperature is not None:
        env["LITTLE_CODER_TEMPERATURE"] = str(args.temperature)
    os.environ.update(env)

    results = []
    for task in args.tasks:
        task_slug = task.removeprefix("terminal-bench/")
        task_dir = TASK_ROOT / task_slug
        out_dir = run_dir / task_slug
        out_dir.mkdir(parents=True, exist_ok=True)
        meta = parse_task(task_dir)
        image = meta["environment"]["docker_image"]
        instruction = (task_dir / "instruction.md").read_text()
        container = None
        log_start = llama_log.stat().st_size if llama_log.exists() else 0
        started = time.time()
        print(f"=== {task_slug} ===", flush=True)
        try:
            container = start_container(task_slug, image, task_dir, out_dir)
            shell = DockerShell(container)

            def shell_handler(payload: dict) -> str:
                if payload.get("op") == "run":
                    return shell.run(payload.get("command", ""), int(payload.get("timeout", 30)))
                if payload.get("op") == "reset":
                    return shell.reset()
                return f"Error: unknown ShellSession op {payload.get('op')!r}"

            prompt = (
                "You are solving a Terminal-Bench task inside a Linux container.\n"
                "The ONLY way to interact with the container is the ShellSession tool; "
                "its cwd persists between calls. Default working directory is /app.\n"
                "File tools are NOT available; use shell commands through ShellSession.\n\n"
                "On your first assistant turn, call ShellSession. Do not provide a prose-only "
                "analysis before using ShellSession. You must create or modify files in /app "
                "with shell commands; prose-only answers fail.\n\n"
                f"TASK:\n{instruction}\n\n"
                "When the task is complete, stop calling tools and say 'done'."
            )
            rpc = PiRpc(
                model=args.model,
                cwd=str(ROOT),
                benchmark="terminal_bench",
                allowed_tools=["ShellSession", "ShellSessionCwd", "ShellSessionReset"],
                session_id=f"direct-{task_slug}-{uuid.uuid4().hex[:6]}",
                tb_mode=True,
                max_turns=args.max_turns,
                tb_shell_handler=shell_handler,
                env=env,
            )
            try:
                agent_result = rpc.prompt_and_collect(prompt, timeout=args.timeout)
                stderr = "\n".join(rpc.stderr())
            finally:
                rpc.close(3)

            verify_result = verify(container, args.verify_timeout)
            elapsed = time.time() - started
            with llama_log.open("r", errors="replace") as f:
                f.seek(log_start)
                log_segment = f.read()
            timing = parse_timings(log_segment)
            result = {
                "task": f"terminal-bench/{task_slug}",
                "image": image,
                "seconds": round(elapsed, 3),
                "agent_text": agent_result.assistant_text.strip(),
                "turn_count": agent_result.turn_count,
                "tool_call_count": len(agent_result.tool_calls),
                "tool_calls": agent_result.tool_calls,
                "verifier": verify_result,
                "timing": timing,
                "pi_stderr_tail": stderr[-4000:],
            }
            (out_dir / "result.json").write_text(json.dumps(result, indent=2))
            print(json.dumps(result, indent=2), flush=True)
            results.append(result)
        except Exception as exc:
            elapsed = time.time() - started
            result = {
                "task": f"terminal-bench/{task_slug}",
                "image": image if "image" in locals() else None,
                "seconds": round(elapsed, 3),
                "error": str(exc),
            }
            (out_dir / "result.json").write_text(json.dumps(result, indent=2))
            print(json.dumps(result, indent=2), flush=True)
            results.append(result)
        finally:
            if container:
                run(["docker", "rm", "-f", container], timeout=60)

        summary = {"tasks": results}
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
