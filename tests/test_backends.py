import asyncio
import json
import shutil

import httpx
import pytest

from repo_sampler.backends import (
    AuthError,
    BackendError,
    build_claude_argv,
    build_codex_argv,
    check_backend,
    effective_model,
    complete_turn,
    parse_turn,
    render_transcript,
    turn_schema,
)
from repo_sampler.config import Settings


def settings_with_key() -> Settings:
    return Settings(_env_file=None, openrouter_api_key="test-key")


def test_default_backend_is_openrouter():
    assert Settings(_env_file=None).agent_backend == "openrouter"


async def test_openrouter_response_is_returned_unchanged(respx_mock):
    body = {"choices": [{"message": {"content": "x"}, "finish_reason": "stop"}]}
    respx_mock.post("https://openrouter.ai/api/v1/chat/completions").respond(json=body)
    async with httpx.AsyncClient() as client:
        assert await complete_turn([], [], settings_with_key(), client) == body


async def test_openrouter_401_raises_auth_error(respx_mock):
    respx_mock.post("https://openrouter.ai/api/v1/chat/completions").respond(401)
    async with httpx.AsyncClient() as client:
        with pytest.raises(AuthError):
            await complete_turn([], [], settings_with_key(), client)


TOOLS = [{"type": "function", "function": {"name": "bash", "description": "d", "parameters": {"type": "object"}}}]
MESSAGES = [{"role": "user", "content": "hi"}]
VALID_CLAUDE_OUTPUT = json.dumps({
    "is_error": False,
    "structured_output": {"content": "ok", "tool_calls": []},
    "usage": {"input_tokens": 7, "output_tokens": 3},
})
VALID_CODEX_OUTPUT = json.dumps({"content": "ok", "tool_calls": []})


def settings(**overrides) -> Settings:
    return Settings(_env_file=None, openrouter_api_key="k", **overrides)


real_sleep = asyncio.sleep  # captured before no_sleep patches it


class Calls:
    def __init__(self):
        self.count = 0
        self.running = 0
        self.max_running = 0
        self.killed = False
        self.last_argv: list[str] = []
        self.last_stdin = b""
        self.last_cwd = None


@pytest.fixture
def no_sleep(monkeypatch):
    """Record backoff pauses without sleeping."""
    slept: list[float] = []

    async def fake_sleep(delay, *a, **kw):
        slept.append(delay)
        await real_sleep(0)

    monkeypatch.setattr("repo_sampler.backends.asyncio.sleep", fake_sleep)
    return slept


@pytest.fixture
def fake_cli(monkeypatch, no_sleep):
    """Replace create_subprocess_exec; record calls, write the -o file for codex."""
    calls = Calls()

    def install(stdout="", stderr="", returncode=0, delay=0.0, hang=False, codex_output=None):
        class Proc:
            def __init__(self, argv):
                self.argv = argv
                self.returncode = None
                self.killed = False

            async def communicate(self, data=None):
                if self.killed:
                    return b"", b""
                calls.last_stdin = data or b""
                if hang:
                    await real_sleep(3600)
                calls.running += 1
                calls.max_running = max(calls.max_running, calls.running)
                try:
                    if delay:
                        await real_sleep(delay)
                finally:
                    calls.running -= 1
                if "-o" in self.argv and codex_output is not None:
                    with open(self.argv[self.argv.index("-o") + 1], "w") as f:
                        f.write(codex_output)
                self.returncode = returncode
                return stdout.encode(), stderr.encode()

            def kill(self):
                self.killed = True
                calls.killed = True

        async def fake_exec(*argv, **kwargs):
            calls.count += 1
            calls.last_argv = list(argv)
            calls.last_cwd = kwargs.get("cwd")
            return Proc(list(argv))

        monkeypatch.setattr("repo_sampler.backends.asyncio.create_subprocess_exec", fake_exec)
        return calls

    return install


def test_schema_is_strict_and_limits_tool_names():
    schema = turn_schema(TOOLS)
    assert schema["additionalProperties"] is False
    assert schema["required"] == ["content", "tool_calls"]
    item = schema["properties"]["tool_calls"]["items"]
    assert item["additionalProperties"] is False
    assert item["properties"]["name"]["enum"] == ["bash"]
    assert item["required"] == ["name", "arguments_json"]


def test_parse_turn_returns_openai_shape():
    data = parse_turn({"content": "c", "tool_calls": [{"name": "bash", "arguments_json": "{\"command\": \"ls\"}"}]})
    choice = data["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert choice["message"]["role"] == "assistant"
    call = choice["message"]["tool_calls"][0]
    assert call["id"] == "call_0" and call["type"] == "function"
    assert call["function"] == {"name": "bash", "arguments": "{\"command\": \"ls\"}"}


def test_parse_turn_numbers_calls_in_order():
    calls = [{"name": "bash", "arguments_json": "{}"}] * 2
    ids = [c["id"] for c in parse_turn({"content": "", "tool_calls": calls})["choices"][0]["message"]["tool_calls"]]
    assert ids == ["call_0", "call_1"]


def test_parse_turn_without_calls_stops():
    data = parse_turn({"content": "done", "tool_calls": []})
    assert data["choices"][0]["finish_reason"] == "stop"
    assert data["choices"][0]["message"]["content"] == "done"


@pytest.mark.parametrize("payload", [
    {"content": ""},
    {"content": "", "tool_calls": [{"name": "bash", "arguments_json": "{oops"}]},
    {"content": "", "tool_calls": [{"name": "bash", "arguments_json": "[1]"}]},
    {"content": "", "tool_calls": [{"arguments_json": "{}"}]},
    {"content": "", "tool_calls": [{"name": "", "arguments_json": "{}"}]},
    {"content": "", "tool_calls": [{"name": 5, "arguments_json": "{}"}]},
    {"content": "", "tool_calls": ["bash"]},
])
def test_parse_turn_rejects_bad_payload(payload):
    with pytest.raises(BackendError):
        parse_turn(payload)


def test_transcript_contains_roles_tool_results_and_tools():
    text = render_transcript([{"role": "system", "content": "S"}, {"role": "tool", "tool_call_id": "call_0", "content": "R"}], TOOLS)
    assert "S" in text and "call_0" in text and "R" in text and "bash" in text


def test_transcript_includes_assistant_text_and_calls():
    assistant = {"role": "assistant", "content": "looking", "tool_calls": [
        {"id": "call_3", "type": "function", "function": {"name": "bash", "arguments": "{\"command\": \"ls\"}"}}]}
    text = render_transcript([assistant], TOOLS)
    assert "looking" in text and "call_3" in text and "bash" in text and "ls" in text


def test_claude_argv_isolated_and_prompt_not_in_args():
    argv = build_claude_argv(settings(agent_cli_model=""), {"type": "object"})
    assert argv[:2] == ["claude", "-p"]
    assert argv[argv.index("--model") + 1] == settings().anonymizer_model
    assert argv[argv.index("--tools") + 1] == ""
    assert argv[argv.index("--setting-sources") + 1] == ""
    for flag in ("--no-session-persistence", "--strict-mcp-config", "--disable-slash-commands", "--json-schema"):
        assert flag in argv


def test_codex_argv_uses_config_model_when_empty(tmp_path):
    argv = build_codex_argv(settings(agent_cli_model=""), tmp_path, tmp_path / "s.json", tmp_path / "o.json")
    assert "-m" not in argv and argv[-1] == "-"
    assert argv[argv.index("-s") + 1] == "read-only"
    assert "--ephemeral" in argv and "--skip-git-repo-check" in argv
    assert "--ignore-user-config" in argv and "--ignore-rules" in argv
    assert "features.shell_tool=false" in argv


def test_codex_argv_passes_explicit_model(tmp_path):
    argv = build_codex_argv(settings(agent_cli_model="m1"), tmp_path, tmp_path / "s.json", tmp_path / "o.json")
    assert argv[argv.index("-m") + 1] == "m1"
    assert argv[argv.index("-C") + 1] == str(tmp_path)


async def test_claude_success_maps_usage(fake_cli):
    fake_cli(stdout=VALID_CLAUDE_OUTPUT)
    data = await complete_turn(MESSAGES, TOOLS, settings(agent_backend="claude"), None)
    assert data["choices"][0]["message"]["content"] == "ok"
    assert data["usage"] == {"prompt_tokens": 7, "completion_tokens": 3}


async def test_codex_success_reads_output_file(fake_cli):
    calls = fake_cli(codex_output=VALID_CODEX_OUTPUT)
    data = await complete_turn(MESSAGES, TOOLS, settings(agent_backend="codex"), None)
    assert data["choices"][0]["message"]["content"] == "ok"
    assert calls.last_cwd == calls.last_argv[calls.last_argv.index("-C") + 1]
    assert b"hi" in calls.last_stdin


async def test_claude_auth_failure_raises_auth_error(fake_cli):
    fake_cli(stdout='{"is_error": true, "result": "Not logged in"}', returncode=1)
    with pytest.raises(AuthError):
        await complete_turn(MESSAGES, TOOLS, settings(agent_backend="claude"), None)


async def test_claude_is_error_with_exit_zero_checks_auth(fake_cli):
    fake_cli(stdout='{"is_error": true, "result": "Credit balance is too low"}', returncode=0)
    with pytest.raises(AuthError):
        await complete_turn(MESSAGES, TOOLS, settings(agent_backend="claude"), None)


async def test_claude_is_error_without_marker_is_retried(fake_cli):
    calls = fake_cli(stdout='{"is_error": true, "result": "overloaded"}', returncode=0)
    with pytest.raises(BackendError, match="claude"):
        await complete_turn(MESSAGES, TOOLS, settings(agent_backend="claude"), None)
    assert calls.count == 3


async def test_missing_structured_output_is_retried(fake_cli):
    calls = fake_cli(stdout='{"is_error": false}', returncode=0)
    with pytest.raises(BackendError):
        await complete_turn(MESSAGES, TOOLS, settings(agent_backend="claude"), None)
    assert calls.count == 3


async def test_codex_unsupported_model_raises_auth_error(fake_cli):
    fake_cli(stderr='{"code":"model_not_found"}', returncode=1)
    with pytest.raises(AuthError):
        await complete_turn(MESSAGES, TOOLS, settings(agent_backend="codex"), None)


async def test_auth_error_is_not_retried(fake_cli):
    calls = fake_cli(stderr="Unauthorized", returncode=1)
    with pytest.raises(AuthError):
        await complete_turn(MESSAGES, TOOLS, settings(agent_backend="codex"), None)
    assert calls.count == 1


async def test_invalid_output_retries_three_times(fake_cli, no_sleep):
    calls = fake_cli(stdout="not json", returncode=0)
    with pytest.raises(BackendError):
        await complete_turn(MESSAGES, TOOLS, settings(agent_backend="claude"), None)
    assert calls.count == 3
    assert no_sleep == [1, 2]


async def test_timeout_kills_process_and_retries(fake_cli):
    calls = fake_cli(hang=True)
    s = settings(agent_backend="claude")
    s.agent_cli_timeout = 0.01  # the field is an int; bypass for a fast test
    with pytest.raises(BackendError, match="timed out"):
        await complete_turn(MESSAGES, TOOLS, s, None)
    assert calls.count == 3 and calls.killed


async def test_missing_binary_is_backend_error(monkeypatch, no_sleep):
    async def boom(*a, **kw):
        raise FileNotFoundError("claude")

    monkeypatch.setattr("repo_sampler.backends.asyncio.create_subprocess_exec", boom)
    with pytest.raises(BackendError, match="cannot run claude"):
        await complete_turn(MESSAGES, TOOLS, settings(agent_backend="claude"), None)


async def test_long_history_goes_through_stdin(fake_cli):
    calls = fake_cli(stdout=VALID_CLAUDE_OUTPUT, returncode=0)
    big = [{"role": "user", "content": "x" * 3_000_000}]
    await complete_turn(big, TOOLS, settings(agent_backend="claude"), None)
    assert all(len(arg) < 100_000 for arg in calls.last_argv)
    assert len(calls.last_stdin) > 3_000_000


async def test_concurrency_limit(fake_cli):
    calls = fake_cli(stdout=VALID_CLAUDE_OUTPUT, returncode=0, delay=0.02)
    s = settings(agent_backend="claude", agent_cli_concurrency=2)
    await asyncio.gather(*(complete_turn(MESSAGES, TOOLS, s, None) for _ in range(10)))
    assert calls.max_running == 2


@pytest.mark.parametrize("marker", ["401", "403", "unauthorized", "Not logged in", "usage limit", "14030"])
async def test_successful_claude_turn_containing_marker_does_not_raise(fake_cli, marker):
    out = json.dumps({
        "is_error": False, "session_id": f"s-{marker}", "duration_ms": 14030,
        "structured_output": {"content": f"grep -rn {marker} src", "tool_calls": [
            {"name": "bash", "arguments_json": json.dumps({"command": f"grep -rn {marker} src"})}]},
        "usage": {"input_tokens": 1, "output_tokens": 1},
    })
    fake_cli(stdout=out)
    data = await complete_turn(MESSAGES, TOOLS, settings(agent_backend="claude"), None)
    assert data["choices"][0]["finish_reason"] == "tool_calls"


async def test_successful_codex_turn_with_marker_in_echoed_prompt(fake_cli):
    fake_cli(stderr="user\nrun grep unauthorized 401 403", codex_output=VALID_CODEX_OUTPUT)
    data = await complete_turn(MESSAGES, TOOLS, settings(agent_backend="codex"), None)
    assert data["choices"][0]["message"]["content"] == "ok"


async def test_failed_codex_turn_with_marker_in_echoed_prompt_is_retried(fake_cli):
    calls = fake_cli(stderr="user\ngrep unauthorized 401 403\nERROR: stream disconnected", returncode=1)
    with pytest.raises(BackendError) as info:
        await complete_turn(MESSAGES, TOOLS, settings(agent_backend="codex"), None)
    assert not isinstance(info.value, AuthError)
    assert calls.count == 3


async def test_claude_unsupported_model_raises_auth_error(fake_cli):
    fake_cli(
        stdout=json.dumps({
            "is_error": True, "api_error_status": 404,
            "result": "There's an issue with the selected model (no-such-model-xyz). "
                      "It may not exist or you may not have access to it.",
        }),
        stderr="unrecognized_model",
        returncode=1,
    )
    with pytest.raises(AuthError):
        await complete_turn(MESSAGES, TOOLS, settings(agent_backend="claude"), None)


async def test_cancellation_kills_process(fake_cli):
    calls = fake_cli(hang=True)
    task = asyncio.ensure_future(complete_turn(MESSAGES, TOOLS, settings(agent_backend="claude"), None))
    while not calls.last_stdin:
        await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert calls.killed


def _backend_settings(**overrides) -> Settings:
    return Settings(_env_file=None, **overrides)


def test_openrouter_requires_key():
    with pytest.raises(BackendError, match="OPENROUTER_API_KEY"):
        check_backend(_backend_settings(agent_backend="openrouter", openrouter_api_key=""))


def test_cli_backend_requires_binary(monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda name: None)
    with pytest.raises(BackendError, match="codex"):
        check_backend(_backend_settings(agent_backend="codex", openrouter_api_key=""))


def test_claude_backend_does_not_need_openrouter_key(monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda name: "/bin/" + name)
    check_backend(_backend_settings(agent_backend="claude", openrouter_api_key=""))


def test_effective_model_per_backend():
    assert effective_model(_backend_settings(agent_backend="openrouter")) == "deepseek/deepseek-v4-flash"
    assert effective_model(_backend_settings(agent_backend="claude")) == "claude-haiku-4-5"
    assert effective_model(_backend_settings(agent_backend="claude", agent_cli_model="m1")) == "m1"
    assert effective_model(_backend_settings(agent_backend="codex")) == "codex-default"
    assert effective_model(_backend_settings(agent_backend="codex", agent_cli_model="m2")) == "m2"
