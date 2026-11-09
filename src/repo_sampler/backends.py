from __future__ import annotations

import asyncio
import json
import re
import shutil
import tempfile
import weakref
from pathlib import Path

import httpx
from loguru import logger

from .config import Settings


class AuthError(Exception):
    pass


class BackendError(RuntimeError):
    """A model backend failed in a way the caller should report."""


def effective_model(settings: Settings) -> str:
    """Name of the model the chosen backend will actually use."""
    if settings.agent_backend == "claude":
        return settings.agent_cli_model or settings.anonymizer_model
    if settings.agent_backend == "codex":
        return settings.agent_cli_model or "codex-default"
    return settings.agent_model


def check_backend(settings: Settings) -> None:
    """Fail fast, before any clone, when the chosen backend cannot run."""
    backend = settings.agent_backend
    if backend == "openrouter":
        if not settings.openrouter_api_key:
            raise BackendError("OPENROUTER_API_KEY is not set (needed for --backend openrouter)")
    elif backend in ("claude", "codex"):
        if shutil.which(backend) is None:
            raise BackendError(f"the '{backend}' CLI was not found on PATH (needed for --backend {backend})")
    else:
        raise BackendError(f"unknown backend {backend!r}")


# ---------------------------------------------------------------------------
# HTTP call to OpenRouter
# ---------------------------------------------------------------------------


async def _call_openrouter(
    messages: list[dict],
    tools: list[dict],
    settings: Settings,
    client: httpx.AsyncClient,
) -> dict:
    payload = {
        "model": settings.agent_model,
        "messages": messages,
        "tools": tools,
        "tool_choice": "auto",
        "max_tokens": 1024,
    }
    headers = {
        "Authorization": f"Bearer {settings.openrouter_api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/repo-sampler",
        "X-Title": "repo-sampler",
    }

    for attempt in range(3):
        try:
            resp = await client.post(
                f"{settings.openrouter_base_url}/chat/completions",
                json=payload,
                headers=headers,
                timeout=120.0,
            )
        except httpx.TimeoutException:
            wait = 2 ** attempt
            logger.warning(f"OpenRouter timeout, retrying in {wait}s")
            await asyncio.sleep(wait)
            continue

        if resp.status_code == 401:
            raise AuthError("HTTP 401: invalid API key")
        if resp.status_code in (402, 403):
            # OpenRouter returns these when the key is blocked or its spending
            # limit is exhausted — every subsequent call will fail the same way.
            raise AuthError(
                f"HTTP {resp.status_code}: OpenRouter key blocked or spending limit reached"
            )
        if resp.status_code in (429, 500, 502, 503):
            wait = 2 ** attempt
            logger.warning(f"HTTP {resp.status_code}, retrying in {wait}s")
            await asyncio.sleep(wait)
            continue

        resp.raise_for_status()
        try:
            return resp.json()
        except (json.JSONDecodeError, ValueError):
            # HTTP 200 but the body is not JSON (HTML error page, empty body,
            # or an SSE/stream). Without logging the raw body this surfaces only
            # as an opaque "Expecting value: line N column 1" parser error in
            # errors.jsonl, with no way to tell what the API actually returned.
            # Log the full body so it can be investigated, then retry like a
            # transient 5xx. These responses are rare (a handful per run), so
            # the full dump does not flood the log. A generous cap is kept only
            # as a guard against a pathological multi-MB body, and any trim is
            # made explicit (body_len + dropped count) so nothing is lost
            # silently.
            wait = 2 ** attempt
            body = resp.text or ""
            cap = 20000
            shown = body if len(body) <= cap else body[:cap] + f"…[+{len(body) - cap} more chars]"
            ctype = resp.headers.get("content-type", "")
            rid = (resp.headers.get("x-request-id")
                   or resp.headers.get("x-openrouter-request-id", ""))
            logger.warning(
                f"OpenRouter returned non-JSON body (status={resp.status_code} "
                f"content-type={ctype!r} request-id={rid!r} body_len={len(body)}), "
                f"retrying in {wait}s; body={shown!r}"
            )
            await asyncio.sleep(wait)
            continue

    raise RuntimeError("OpenRouter call failed after 3 attempts")


async def complete_turn(
    messages: list[dict],
    tools: list[dict],
    settings: Settings,
    client: httpx.AsyncClient,
) -> dict:
    """One model turn; returns an OpenAI-style chat completion dict."""
    if settings.agent_backend == "openrouter":
        return await _call_openrouter(messages, tools, settings, client)
    if settings.agent_backend in ("claude", "codex"):
        return await _run_cli_turn(messages, tools, settings)
    raise BackendError(f"unknown backend {settings.agent_backend!r}")


# ---------------------------------------------------------------------------
# Single-turn protocol for CLI backends
# ---------------------------------------------------------------------------


def turn_schema(tools: list[dict]) -> dict:
    """JSON schema of one model turn.

    Every object sets additionalProperties=false and lists all its properties
    as required, which both Claude's --json-schema and Codex's strict
    --output-schema accept.
    """
    names = [t["function"]["name"] for t in tools]
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "content": {"type": "string"},
            "tool_calls": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "name": {"type": "string", "enum": names},
                        "arguments_json": {"type": "string"},
                    },
                    "required": ["name", "arguments_json"],
                },
            },
        },
        "required": ["content", "tool_calls"],
    }


def render_transcript(messages: list[dict], tools: list[dict]) -> str:
    """Render the conversation as text for a stateless CLI model."""
    parts = ["You can call these tools:"]
    for t in tools:
        fn = t["function"]
        parts.append(
            f"- {fn['name']}: {fn.get('description', '')}\n"
            f"  parameters: {json.dumps(fn.get('parameters', {}))}"
        )
    parts.append("\nConversation so far:")
    for m in messages:
        role = m.get("role", "")
        if role == "tool":
            parts.append(f"[tool result for {m.get('tool_call_id', '')}]\n{m.get('content') or ''}")
            continue
        parts.append(f"[{role}]\n{m.get('content') or ''}")
        for tc in m.get("tool_calls") or []:
            fn = tc.get("function", {})
            parts.append(
                f"[{role} tool call {tc.get('id', '')}] "
                f"{fn.get('name', '')}({fn.get('arguments', '')})"
            )
    parts.append(
        "\nReply with the next assistant turn only, as JSON matching the given "
        "schema: put text in content and each tool call in tool_calls, with "
        "arguments_json holding the arguments as a JSON object string. "
        "Use an empty tool_calls list when you are done."
    )
    return "\n".join(parts)


def parse_turn(payload: dict) -> dict:
    """Convert a schema-shaped turn into an OpenAI-style chat completion."""
    if not isinstance(payload, dict) or not isinstance(payload.get("tool_calls"), list):
        raise BackendError("model turn is missing tool_calls")
    calls = []
    for i, raw in enumerate(payload["tool_calls"]):
        if not isinstance(raw, dict):
            raise BackendError(f"tool call {i} is not an object: {raw!r}")
        name = raw.get("name")
        if not isinstance(name, str) or not name:
            raise BackendError(f"tool call {i} has no valid name: {name!r}")
        args = raw.get("arguments_json")
        try:
            decoded = json.loads(args)
        except (TypeError, ValueError):
            raise BackendError(f"tool call {i} has invalid arguments_json: {args!r}") from None
        if not isinstance(decoded, dict):
            raise BackendError(f"tool call {i} arguments_json is not a JSON object: {args!r}")
        calls.append({
            "id": f"call_{i}",
            "type": "function",
            "function": {"name": name, "arguments": args},
        })
    message: dict = {"role": "assistant", "content": payload.get("content") or ""}
    if calls:
        message["tool_calls"] = calls
    return {
        "choices": [{
            "message": message,
            "finish_reason": "tool_calls" if calls else "stop",
        }],
    }


# ---------------------------------------------------------------------------
# Claude Code and Codex CLI backends (one stateless subprocess per turn)
# ---------------------------------------------------------------------------

# Matched case-insensitively, on error text only and only after a failure is
# established. Status codes need word boundaries ("14030" is not a 403).
_AUTH_PATTERN = re.compile(
    r"not logged in|unauthorized|\b401\b|\b403\b|credit balance|usage limit|"
    r"model_not_found|issue with the selected model|unrecognized_model|hit your limit",
    re.IGNORECASE,
)
_AUTH_STATUSES = (401, 403, 404)
_KILL_WAIT_SECONDS = 5
_ATTEMPTS = 3
_MAX_ERROR_CHARS = 300

# One semaphore per event loop, sized from the first caller's settings.
_semaphores: "weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, asyncio.Semaphore]" = (
    weakref.WeakKeyDictionary()
)


def _semaphore(settings: Settings) -> asyncio.Semaphore:
    loop = asyncio.get_running_loop()
    sem = _semaphores.get(loop)
    if sem is None:
        sem = _semaphores[loop] = asyncio.Semaphore(max(1, settings.agent_cli_concurrency))
    return sem


def build_claude_argv(settings: Settings, schema: dict) -> list[str]:
    """Claude Code argv; the prompt is sent on stdin, never in argv."""
    return [
        "claude", "-p",
        "--model", settings.agent_cli_model or settings.anonymizer_model,
        "--tools", "",
        "--no-session-persistence",
        "--setting-sources", "",
        "--strict-mcp-config",
        "--disable-slash-commands",
        "--output-format", "json",
        "--json-schema", json.dumps(schema),
    ]


def build_codex_argv(settings: Settings, cwd: Path, schema_path: Path, output_path: Path) -> list[str]:
    """Codex argv; the trailing "-" makes codex read the prompt from stdin."""
    argv = [
        "codex", "exec", "--ephemeral", "--skip-git-repo-check",
        "--ignore-user-config", "--ignore-rules",
        "-c", "features.shell_tool=false",
        "-s", "read-only", "-C", str(cwd),
    ]
    if settings.agent_cli_model:
        argv += ["-m", settings.agent_cli_model]
    argv += ["--output-schema", str(schema_path), "-o", str(output_path), "-"]
    return argv


def _short(text: str) -> str:
    text = " ".join(text.split())
    return text if len(text) <= _MAX_ERROR_CHARS else text[:_MAX_ERROR_CHARS] + "..."


def _raise_if_auth(error_text: str, status: object = None) -> None:
    """Raise AuthError when an already-failed turn failed for access reasons.

    Only call this with error text (never with model output or an echoed prompt).
    """
    if status in _AUTH_STATUSES:
        raise AuthError(f"CLI backend access failure (HTTP {status}): {_short(error_text)}")
    match = _AUTH_PATTERN.search(error_text)
    if match:
        raise AuthError(f"CLI backend authentication or access failure ({match.group(0)}): {_short(error_text)}")


async def _kill(proc) -> None:
    try:
        proc.kill()
    except ProcessLookupError:
        return
    try:
        await asyncio.wait_for(proc.communicate(), timeout=_KILL_WAIT_SECONDS)
    except (asyncio.TimeoutError, ProcessLookupError):
        logger.warning("CLI process did not exit after kill")


async def _run_process(argv: list[str], prompt: str, cwd: str, timeout: int) -> tuple[int, str, str]:
    proc = await asyncio.create_subprocess_exec(
        *argv,
        cwd=cwd,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        try:
            out, err = await asyncio.wait_for(proc.communicate(prompt.encode()), timeout=timeout)
        except asyncio.TimeoutError:
            raise BackendError(f"timed out after {timeout}s") from None
    finally:
        # Timeout, cancellation (a sibling raised AuthError) or any error:
        # never leave a paid CLI process running.
        if proc.returncode is None:
            await _kill(proc)
    return proc.returncode or 0, out.decode(errors="replace"), err.decode(errors="replace")


async def _attempt_claude(prompt: str, schema: dict, settings: Settings, cwd: str) -> dict:
    code, out, err = await _run_process(
        build_claude_argv(settings, schema), prompt, cwd, settings.agent_cli_timeout
    )
    try:
        data = json.loads(out)
    except ValueError:
        if code != 0:
            _raise_if_auth(f"{err}\n{out}")
        raise BackendError(f"exit {code}, output is not JSON: {_short(err or out)}") from None
    if not isinstance(data, dict):
        raise BackendError(f"unexpected output: {_short(out)}")
    if data.get("is_error") or code != 0:
        text = f"{data.get('result') or ''}\n{err}"
        _raise_if_auth(text, data.get("api_error_status"))
        raise BackendError(f"exit {code}: {_short(text)}")
    payload = data.get("structured_output")
    if not isinstance(payload, dict):
        raise BackendError("no structured_output in response")
    result = parse_turn(payload)
    usage = data.get("usage") or {}
    result["usage"] = {
        "prompt_tokens": usage.get("input_tokens", 0),
        "completion_tokens": usage.get("output_tokens", 0),
    }
    return result


def _codex_error_text(stderr: str) -> str:
    """Error lines of codex stderr. Codex echoes the whole prompt there, so
    prefer its "ERROR:" lines; without them use only the tail, where the
    error is and the fixed end of the prompt carries no markers."""
    lines = stderr.splitlines()
    errors = [ln for ln in lines if ln.lstrip().startswith("ERROR:")]
    return "\n".join(errors or lines[-5:])


async def _attempt_codex(prompt: str, schema: dict, settings: Settings, cwd: str) -> dict:
    schema_path = Path(cwd) / "schema.json"
    output_path = Path(cwd) / "output.json"
    schema_path.write_text(json.dumps(schema))
    code, out, err = await _run_process(
        build_codex_argv(settings, Path(cwd), schema_path, output_path),
        prompt, cwd, settings.agent_cli_timeout,
    )
    if code != 0:
        text = _codex_error_text(err)
        _raise_if_auth(text)
        raise BackendError(f"exit {code}: {_short(text or out)}")
    try:
        payload = json.loads(output_path.read_text())
    except (OSError, ValueError):
        raise BackendError(f"no valid JSON in output file: {_short(err or out)}") from None
    return parse_turn(payload)


async def _run_cli_turn(messages: list[dict], tools: list[dict], settings: Settings) -> dict:
    backend = settings.agent_backend
    attempt_fn = _attempt_claude if backend == "claude" else _attempt_codex
    schema = turn_schema(tools)
    prompt = render_transcript(messages, tools)
    last = ""
    for attempt in range(_ATTEMPTS):
        try:
            async with _semaphore(settings):
                with tempfile.TemporaryDirectory(prefix="repo-sampler-") as cwd:
                    return await attempt_fn(prompt, schema, settings, cwd)
        except AuthError:
            raise
        except BackendError as e:
            last = str(e)
        except OSError as e:
            last = f"cannot run {backend}: {e}"
        if attempt < _ATTEMPTS - 1:
            wait = 2 ** attempt
            logger.warning(f"{backend} turn failed ({_short(last)}), retrying in {wait}s")
            await asyncio.sleep(wait)
    raise BackendError(f"{backend} backend failed after {_ATTEMPTS} attempts: {_short(last)}")
