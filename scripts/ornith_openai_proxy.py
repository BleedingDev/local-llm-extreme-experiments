#!/usr/bin/env python3
"""Small OpenAI-compatible proxy for Ornith MLX/VLM.

It fixes role compatibility between Little Coder/pi and Ornith's chat template:
OpenAI-style `developer`/late `system` messages are merged into the first
system message because the template accepts `system` only at the beginning.
It also records request timing and token usage to JSONL.
"""

from __future__ import annotations

import argparse
import json
import os
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


SUPPORTED_ROLES = {"system", "user", "assistant", "tool"}


def content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks = []
        for item in content:
            if isinstance(item, dict):
                if "text" in item:
                    chunks.append(str(item["text"]))
                elif item.get("type") in {"text", "input_text"}:
                    chunks.append(str(item.get("text", "")))
            else:
                chunks.append(str(item))
        return "\n".join(chunk for chunk in chunks if chunk)
    return str(content)


def normalize_messages(messages: Any) -> tuple[Any, list[str]]:
    if not isinstance(messages, list):
        return messages, []

    system_chunks: list[str] = []
    normalized: list[dict[str, Any]] = []
    role_changes: list[str] = []

    for message in messages:
        if not isinstance(message, dict):
            normalized.append(message)
            continue

        role = message.get("role")
        if role in {"system", "developer"}:
            text = content_to_text(message.get("content"))
            if text:
                system_chunks.append(text)
            if role == "developer":
                role_changes.append("developer->system")
            elif normalized:
                role_changes.append("late-system->leading-system")
            continue

        next_message = dict(message)
        if role == "toolResult":
            next_message["role"] = "tool"
            role_changes.append("toolResult->tool")
        elif role not in SUPPORTED_ROLES:
            next_message["role"] = "user"
            next_message["content"] = content_to_text(message.get("content"))
            role_changes.append(f"{role}->user")
        normalized.append(next_message)

    if system_chunks:
        system_message = {"role": "system", "content": "\n\n".join(system_chunks)}
        if normalized and isinstance(normalized[0], dict) and normalized[0].get("role") == "system":
            existing = content_to_text(normalized[0].get("content"))
            normalized[0] = {
                **normalized[0],
                "content": "\n\n".join(chunk for chunk in [system_message["content"], existing] if chunk),
            }
        else:
            normalized.insert(0, system_message)

    return normalized, role_changes


def normalize_responses_input(input_items: Any) -> tuple[Any, list[str]]:
    if not isinstance(input_items, list):
        return input_items, []

    system_chunks: list[str] = []
    normalized: list[dict[str, Any]] = []
    role_changes: list[str] = []

    for item in input_items:
        if not isinstance(item, dict) or "role" not in item:
            normalized.append(item)
            continue

        role = item.get("role")
        if role in {"system", "developer"}:
            text = content_to_text(item.get("content"))
            if text:
                system_chunks.append(text)
            if role == "developer":
                role_changes.append("responses-developer->system")
            elif normalized:
                role_changes.append("responses-late-system->leading-system")
            continue

        next_item = dict(item)
        if role == "toolResult":
            next_item["role"] = "tool"
            role_changes.append("responses-toolResult->tool")
        elif role not in SUPPORTED_ROLES:
            next_item["role"] = "user"
            next_item["content"] = content_to_text(item.get("content"))
            role_changes.append(f"responses-{role}->user")
        normalized.append(next_item)

    if system_chunks:
        system_item = {"role": "system", "content": "\n\n".join(system_chunks)}
        if normalized and isinstance(normalized[0], dict) and normalized[0].get("role") == "system":
            existing = content_to_text(normalized[0].get("content"))
            normalized[0] = {
                **normalized[0],
                "content": "\n\n".join(chunk for chunk in [system_item["content"], existing] if chunk),
            }
        else:
            normalized.insert(0, system_item)

    return normalized, role_changes


def prepend_responses_system(input_items: Any, text: str) -> Any:
    if not text:
        return input_items
    if isinstance(input_items, str):
        return [
            {"role": "system", "content": text},
            {"role": "user", "content": input_items},
        ]
    if not isinstance(input_items, list):
        return input_items

    next_items = list(input_items)
    if next_items and isinstance(next_items[0], dict) and next_items[0].get("role") == "system":
        existing = content_to_text(next_items[0].get("content"))
        next_items[0] = {
            **next_items[0],
            "content": "\n\n".join(chunk for chunk in [text, existing] if chunk),
        }
    else:
        next_items.insert(0, {"role": "system", "content": text})
    return next_items


def summarize_payload_shape(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {"type": type(payload).__name__}

    summary: dict[str, Any] = {
        "keys": sorted(payload.keys()),
        "model": payload.get("model"),
        "temperature": payload.get("temperature"),
        "max_tokens": payload.get("max_tokens"),
        "max_completion_tokens": payload.get("max_completion_tokens"),
        "chat_template_kwargs": payload.get("chat_template_kwargs"),
    }
    messages = payload.get("messages")
    if isinstance(messages, list):
        summary["message_roles"] = [
            item.get("role") if isinstance(item, dict) else type(item).__name__
            for item in messages
        ]
    input_items = payload.get("input")
    if isinstance(input_items, list):
        summary["input_roles"] = [
            item.get("role") if isinstance(item, dict) else type(item).__name__
            for item in input_items
        ]
        summary["input_types"] = [
            item.get("type") if isinstance(item, dict) else type(item).__name__
            for item in input_items
        ]
    elif input_items is not None:
        summary["input_type"] = type(input_items).__name__
    return summary


def normalize_payload(payload: Any, default_model: str) -> tuple[Any, list[str]]:
    if not isinstance(payload, dict):
        return payload, []

    next_payload = dict(payload)
    changes: list[str] = []
    if default_model and next_payload.get("model") != default_model:
        changes.append(f"model:{next_payload.get('model')}->{default_model}")
        next_payload["model"] = default_model

    if "messages" in next_payload:
        messages, role_changes = normalize_messages(next_payload.get("messages"))
        next_payload["messages"] = messages
        changes.extend(role_changes)

    instructions = next_payload.pop("instructions", None)
    if isinstance(instructions, str) and instructions:
        next_payload["input"] = prepend_responses_system(next_payload.get("input"), instructions)
        changes.append("responses-instructions->leading-system")

    if "input" in next_payload:
        input_items, input_changes = normalize_responses_input(next_payload.get("input"))
        next_payload["input"] = input_items
        changes.extend(input_changes)

    return next_payload, changes


def parse_response_metrics(body: bytes) -> dict[str, Any]:
    usage: dict[str, Any] = {}
    timings: dict[str, Any] = {}
    finish_reason = None
    streamed_text = ""
    is_stream = False

    try:
        parsed = json.loads(body.decode("utf-8"))
        usage = parsed.get("usage") or {}
        timings = parsed.get("timings") or {}
        choices = parsed.get("choices") or []
        if choices:
            finish_reason = choices[0].get("finish_reason")
        if parsed.get("object") == "response":
            finish_reason = parsed.get("status") or finish_reason
            if usage:
                usage = {
                    "prompt_tokens": usage.get("prompt_tokens") or usage.get("input_tokens") or 0,
                    "completion_tokens": usage.get("completion_tokens") or usage.get("output_tokens") or 0,
                    "total_tokens": usage.get("total_tokens") or usage.get("input_tokens", 0) + usage.get("output_tokens", 0),
                }
        return {
            "usage": usage,
            "timings": timings,
            "finish_reason": finish_reason,
            "streamed_text": "",
            "is_stream": False,
        }
    except Exception:
        pass

    for raw_line in body.decode("utf-8", "replace").splitlines():
        line = raw_line.strip()
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if not data or data == "[DONE]":
            continue
        try:
            chunk = json.loads(data)
        except Exception:
            continue
        is_stream = True
        event_type = chunk.get("type")
        if event_type == "response.output_text.delta" and isinstance(chunk.get("delta"), str):
            streamed_text += chunk["delta"]
            continue
        if event_type == "response.completed":
            finish_reason = "completed"
            response = chunk.get("response") or {}
            response_usage = response.get("usage") or {}
            if response_usage:
                usage = {
                    "prompt_tokens": response_usage.get("prompt_tokens") or response_usage.get("input_tokens") or 0,
                    "completion_tokens": response_usage.get("completion_tokens") or response_usage.get("output_tokens") or 0,
                    "total_tokens": response_usage.get("total_tokens") or response_usage.get("input_tokens", 0) + response_usage.get("output_tokens", 0),
                }
            continue
        if chunk.get("usage"):
            usage = chunk.get("usage") or usage
        choices = chunk.get("choices") or []
        if not choices:
            continue
        choice = choices[0]
        finish_reason = choice.get("finish_reason") or finish_reason
        delta = choice.get("delta") or {}
        content = delta.get("content")
        if isinstance(content, str):
            streamed_text += content

    return {
        "usage": usage,
        "timings": timings,
        "finish_reason": finish_reason,
        "streamed_text": streamed_text,
        "is_stream": is_stream,
    }


def estimate_tokens(text: str) -> int:
    # Conservative lightweight proxy estimate for streaming responses that omit
    # usage. The exact non-streaming metrics remain preferred when available.
    return max(1, round(len(text) / 4)) if text else 0


class ProxyHandler(BaseHTTPRequestHandler):
    upstream: str
    log_path: Path
    default_model: str
    exit_on_upstream_error: bool
    exit_code: int

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        return

    def do_GET(self) -> None:  # noqa: N802
        self.forward(None)

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("content-length") or 0)
        body = self.rfile.read(length) if length else b""
        self.forward(body)

    def forward(self, body: bytes | None) -> None:
        started = time.perf_counter()
        path = self.path
        upstream_url = self.upstream.rstrip("/") + path
        request_body = body
        changes: list[str] = []
        request_shape: dict[str, Any] = {}

        if body:
            try:
                payload = json.loads(body.decode("utf-8"))
                payload, changes = normalize_payload(payload, self.default_model)
                request_shape = summarize_payload_shape(payload)
                request_body = json.dumps(payload).encode("utf-8")
            except Exception as exc:
                changes = [f"normalization-error:{exc!r}"]

        headers = {
            key: value
            for key, value in self.headers.items()
            if key.lower() not in {"host", "content-length", "accept-encoding"}
        }
        if request_body is not None:
            headers["content-length"] = str(len(request_body))

        req = urllib.request.Request(
            upstream_url,
            data=request_body,
            headers=headers,
            method=self.command,
        )

        status = 502
        response_body = b""
        response_headers: dict[str, str] = {}
        error = ""
        fatal_upstream_error = False
        try:
            with urllib.request.urlopen(req, timeout=900) as resp:
                status = resp.status
                response_body = resp.read()
                response_headers = dict(resp.headers.items())
        except urllib.error.HTTPError as exc:
            status = exc.code
            response_body = exc.read()
            response_headers = dict(exc.headers.items())
            error = response_body.decode("utf-8", "replace")[:1000]
        except Exception as exc:
            response_body = json.dumps({"error": repr(exc)}).encode("utf-8")
            response_headers = {"content-type": "application/json"}
            error = repr(exc)
            fatal_upstream_error = True

        elapsed = time.perf_counter() - started
        self.send_response(status)
        for key, value in response_headers.items():
            if key.lower() in {"content-length", "transfer-encoding", "connection", "content-encoding"}:
                continue
            self.send_header(key, value)
        self.send_header("content-length", str(len(response_body)))
        self.end_headers()
        self.wfile.write(response_body)

        metrics = parse_response_metrics(response_body)
        usage = metrics["usage"]
        timings = metrics["timings"]
        finish_reason = metrics["finish_reason"]

        completion_tokens = int(usage.get("completion_tokens") or 0)
        estimated_completion_tokens = 0
        if completion_tokens == 0 and metrics["streamed_text"]:
            estimated_completion_tokens = estimate_tokens(metrics["streamed_text"])
        record = {
            "ts": time.time(),
            "method": self.command,
            "path": path,
            "status": status,
            "elapsed_s": elapsed,
            "prompt_tokens": int(usage.get("prompt_tokens") or 0),
            "completion_tokens": completion_tokens,
            "estimated_completion_tokens": estimated_completion_tokens,
            "total_tokens": int(usage.get("total_tokens") or 0),
            "completion_tps": completion_tokens / elapsed if elapsed and completion_tokens else 0.0,
            "estimated_completion_tps": estimated_completion_tokens / elapsed if elapsed and estimated_completion_tokens else 0.0,
            "server_prompt_tps": timings.get("prompt_per_second"),
            "server_completion_tps": timings.get("predicted_per_second"),
            "server_peak_memory_gb": timings.get("peak_memory"),
            "is_stream": metrics["is_stream"],
            "finish_reason": finish_reason,
            "changes": changes,
            "request_shape": request_shape,
            "error": error,
        }
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
            f.flush()

        if fatal_upstream_error and self.exit_on_upstream_error:
            def _exit_after_response() -> None:
                time.sleep(0.2)
                os._exit(self.exit_code)

            threading.Thread(target=_exit_after_response, daemon=True).start()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--upstream", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="mlx-community/Ornith-1.0-35B-4bit")
    parser.add_argument("--log", default="bench/runs/ornith_openai_proxy.jsonl")
    parser.add_argument(
        "--exit-on-upstream-error",
        action="store_true",
        help="Exit the proxy after logging a transport-level upstream failure.",
    )
    parser.add_argument("--exit-code", type=int, default=75)
    args = parser.parse_args()

    ProxyHandler.upstream = args.upstream
    ProxyHandler.log_path = Path(args.log)
    ProxyHandler.default_model = args.model
    ProxyHandler.exit_on_upstream_error = args.exit_on_upstream_error
    ProxyHandler.exit_code = args.exit_code

    server = ThreadingHTTPServer((args.host, args.port), ProxyHandler)
    print(f"Proxying http://{args.host}:{args.port} -> {args.upstream}", flush=True)
    print(f"Logging to {args.log}", flush=True)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
