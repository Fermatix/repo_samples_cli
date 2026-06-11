import json
import tempfile
from pathlib import Path

import httpx
import pytest
import respx

from repo_sampler.agent import AuthError, run_agent
from repo_sampler.config import Settings


def _make_settings(**kwargs) -> Settings:
    base = {"openrouter_api_key": "test-key", "agent_max_iterations": 10}
    base.update(kwargs)
    return Settings(**base)


def _tool_call(call_id: str, name: str, args: dict) -> dict:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(args)},
    }


def _assistant_with_tools(tool_calls: list[dict]) -> dict:
    return {
        "choices": [{
            "message": {"role": "assistant", "content": None, "tool_calls": tool_calls},
            "finish_reason": "tool_calls",
        }]
    }


def _assistant_stop() -> dict:
    return {
        "choices": [{
            "message": {"role": "assistant", "content": "Done.", "tool_calls": None},
            "finish_reason": "stop",
        }]
    }


@pytest.mark.asyncio
async def test_happy_path_creates_deliverable():
    """Agent explores, saves files, writes summary, calls finish."""
    with tempfile.TemporaryDirectory() as tmp:
        repo = Path(tmp) / "repo"
        out = Path(tmp) / "out"
        repo.mkdir()
        (repo / "src").mkdir()
        (repo / "src" / "engine.py").write_text("x = 1\ny = 2\nz = 3\n")
        (repo / "tests").mkdir()
        (repo / "tests" / "test_engine.py").write_text("def test_x(): assert 1 == 1\n")

        settings = _make_settings()
        responses = iter([
            _assistant_with_tools([_tool_call("c1", "bash", {"command": "ls"})]),
            _assistant_with_tools([_tool_call("c2", "save_sample", {"path": "src/engine.py", "layer": "business"})]),
            _assistant_with_tools([_tool_call("c3", "write_summary", {"content": "## Repo\n\nA test repo.\n\n## Structure\n\nsrc/\n\n## What was sampled\n\ncore logic\n\n## Notes\n\nnone"})]),
            _assistant_with_tools([_tool_call("c4", "finish", {"message": "done", "total_loc": 3, "file_count": 1})]),
        ])

        with respx.mock:
            respx.post("https://openrouter.ai/api/v1/chat/completions").mock(
                side_effect=lambda req: httpx.Response(200, json=next(responses))
            )
            async with httpx.AsyncClient() as client:
                result = await run_agent(
                    repo_path=repo,
                    repo_url="https://github.com/owner/repo",
                    output_dir=out,
                    settings=settings,
                    client=client,
                )

        assert result.repo_name == "repo"
        assert result.folder_name == "github.com__owner__repo"
        assert len(result.files) == 1
        assert result.files[0].path == "src/engine.py"
        assert result.files[0].layer == "business"
        assert result.total_loc > 0
        assert result.summary_md != ""

        # Deliverable files exist on disk under folder_name
        folder = out / result.folder_name
        assert (folder / "samples" / "src" / "engine.py").exists()
        assert (folder / "repo_summary.md").exists()
        assert (folder / "agent_log.json").exists()


@pytest.mark.asyncio
async def test_max_iterations_guard_writes_fallback_summary():
    """Agent never calls finish — loop terminates and writes a fallback summary."""
    with tempfile.TemporaryDirectory() as tmp:
        repo = Path(tmp) / "repo"
        out = Path(tmp) / "out"
        repo.mkdir()
        (repo / "f.py").write_text("x = 1\n")

        settings = _make_settings(agent_max_iterations=3)

        # Always return a bash call — never finish
        always_bash = _assistant_with_tools([_tool_call("c1", "bash", {"command": "ls"})])

        with respx.mock:
            respx.post("https://openrouter.ai/api/v1/chat/completions").mock(
                return_value=httpx.Response(200, json=always_bash)
            )
            async with httpx.AsyncClient() as client:
                result = await run_agent(
                    repo_path=repo,
                    repo_url="https://github.com/owner/repo",
                    output_dir=out,
                    settings=settings,
                    client=client,
                )

        assert result.agent_iterations == 3
        assert result.folder_name == "github.com__owner__repo"
        # Fallback summary is written
        assert (out / result.folder_name / "repo_summary.md").exists()
        assert result.summary_md != ""


@pytest.mark.asyncio
async def test_auth_error_on_401():
    with tempfile.TemporaryDirectory() as tmp:
        repo = Path(tmp) / "repo"
        out = Path(tmp) / "out"
        repo.mkdir()

        settings = _make_settings()

        with respx.mock:
            respx.post("https://openrouter.ai/api/v1/chat/completions").mock(
                return_value=httpx.Response(401)
            )
            async with httpx.AsyncClient() as client:
                with pytest.raises(AuthError):
                    await run_agent(
                        repo_path=repo,
                        repo_url="https://github.com/owner/repo",
                        output_dir=out,
                        settings=settings,
                        client=client,
                    )


@pytest.mark.asyncio
async def test_retry_on_429():
    """Loop retries the stalled HTTP call, not the whole agent session."""
    with tempfile.TemporaryDirectory() as tmp:
        repo = Path(tmp) / "repo"
        out = Path(tmp) / "out"
        repo.mkdir()
        (repo / "f.py").write_text("x = 1\n")

        settings = _make_settings()
        call_count = 0
        responses_iter = iter([
            _assistant_with_tools([_tool_call("c1", "bash", {"command": "ls"})]),
            _assistant_with_tools([_tool_call("c2", "write_summary", {"content": "## Repo\n\nA\n\n## Structure\n\nB\n\n## What was sampled\n\nC\n\n## Notes\n\nD"})]),
            _assistant_with_tools([_tool_call("c3", "finish", {"message": "ok", "total_loc": 0, "file_count": 0})]),
        ])

        def handler(request):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                return httpx.Response(429)
            return httpx.Response(200, json=next(responses_iter))

        with respx.mock:
            respx.post("https://openrouter.ai/api/v1/chat/completions").mock(side_effect=handler)
            async with httpx.AsyncClient() as client:
                result = await run_agent(
                    repo_path=repo,
                    repo_url="https://github.com/owner/repo",
                    output_dir=out,
                    settings=settings,
                    client=client,
                )

        # call_count > 3 because 429 caused a retry
        assert call_count > 3
        assert result.repo_name == "repo"


@pytest.mark.asyncio
async def test_save_sample_bad_path_does_not_crash_loop():
    """Agent calls save_sample with a nonexistent path — gets error, continues."""
    with tempfile.TemporaryDirectory() as tmp:
        repo = Path(tmp) / "repo"
        out = Path(tmp) / "out"
        repo.mkdir()

        settings = _make_settings()
        responses = iter([
            _assistant_with_tools([
                _tool_call("c1", "save_sample", {"path": "does_not_exist.py", "layer": "other"})
            ]),
            _assistant_with_tools([
                _tool_call("c2", "write_summary", {"content": "## Repo\n\nA\n\n## Structure\n\nB\n\n## What was sampled\n\nC\n\n## Notes\n\nD"})
            ]),
            _assistant_with_tools([
                _tool_call("c3", "finish", {"message": "done", "total_loc": 0, "file_count": 0})
            ]),
        ])

        with respx.mock:
            respx.post("https://openrouter.ai/api/v1/chat/completions").mock(
                side_effect=lambda req: httpx.Response(200, json=next(responses))
            )
            async with httpx.AsyncClient() as client:
                result = await run_agent(
                    repo_path=repo,
                    repo_url="https://github.com/owner/repo",
                    output_dir=out,
                    settings=settings,
                    client=client,
                )

        # Loop completed without raising
        assert result.repo_name == "repo"
        # No files were saved (path was bad)
        assert len(result.files) == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("status_code", [402, 403])
async def test_auth_error_on_spending_limit(status_code):
    """402/403 (key blocked / spending cap) must abort like 401, not crash per-repo."""
    with tempfile.TemporaryDirectory() as tmp:
        repo = Path(tmp) / "repo"
        out = Path(tmp) / "out"
        repo.mkdir()

        settings = _make_settings()

        with respx.mock:
            respx.post("https://openrouter.ai/api/v1/chat/completions").mock(
                return_value=httpx.Response(status_code)
            )
            async with httpx.AsyncClient() as client:
                with pytest.raises(AuthError):
                    await run_agent(
                        repo_path=repo,
                        repo_url="https://github.com/owner/repo",
                        output_dir=out,
                        settings=settings,
                        client=client,
                    )
