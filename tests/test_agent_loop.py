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


# ---------------------------------------------------------------------------
# Language-aware sampling
# ---------------------------------------------------------------------------

def _summary_call(call_id: str) -> dict:
    return _tool_call(call_id, "write_summary", {
        "content": "## Repo\n\nA\n\n## Structure\n\nB\n\n## What was sampled\n\nC\n\n## Notes\n\nD"
    })


@pytest.mark.asyncio
async def test_language_distribution_injected_into_first_message():
    """The agent's first user message carries the distribution + PRIMARY LANGUAGE."""
    with tempfile.TemporaryDirectory() as tmp:
        repo = Path(tmp) / "repo"
        out = Path(tmp) / "out"
        (repo / "src").mkdir(parents=True)
        (repo / "src" / "app.js").write_text("var x;\n" * 80)
        (repo / "src" / "style.css").write_text("a {}\n" * 20)

        settings = _make_settings(lang_scan_use_scc=False)
        captured: list[dict] = []

        def handler(request):
            captured.append(json.loads(request.content))
            return httpx.Response(200, json=_assistant_stop())

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

        first_user = captured[0]["messages"][1]["content"]
        assert "PRIMARY LANGUAGE: JavaScript" in first_user
        assert "- JavaScript: 80.0%" in first_user
        assert "- CSS: 20.0%" in first_user
        assert "at least 20%" in first_user
        # system prompt carries the static rules
        assert "Language coverage — HARD REQUIREMENT" in captured[0]["messages"][0]["content"]
        assert result.primary_language == "JavaScript"
        assert result.lang_stats_source == "walk"
        assert result.primary_forced is False
        assert result.repo_lang_distribution == {"JavaScript": 0.8, "CSS": 0.2}


@pytest.mark.asyncio
async def test_saved_files_carry_language():
    with tempfile.TemporaryDirectory() as tmp:
        repo = Path(tmp) / "repo"
        out = Path(tmp) / "out"
        repo.mkdir()
        (repo / "engine.py").write_text("x = 1\ny = 2\nz = 3\n")

        settings = _make_settings(lang_scan_use_scc=False)
        responses = iter([
            _assistant_with_tools([_tool_call("c1", "save_sample", {"path": "engine.py", "layer": "business"})]),
            _assistant_with_tools([_summary_call("c2")]),
            _assistant_with_tools([_tool_call("c3", "finish", {"message": "done", "total_loc": 3, "file_count": 1})]),
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

        assert result.files[0].language == "Python"
        assert result.primary_language == "Python"
        log = json.loads((out / result.folder_name / "agent_log.json").read_text())
        assert log["stats"]["primary_language"] == "Python"
        assert log["stats"]["sample_lang_distribution"] == {"Python": 3}
        assert log["saved_files"][0]["language"] == "Python"


@pytest.mark.asyncio
async def test_primary_language_nudge_fires():
    """Sample dominated by a non-primary language triggers the language nudge."""
    with tempfile.TemporaryDirectory() as tmp:
        repo = Path(tmp) / "repo"
        out = Path(tmp) / "out"
        repo.mkdir()
        (repo / "main.py").write_text("x = 1\n" * 100)          # primary: Python
        (repo / "notes.md").write_text("plain line\n" * 30)     # what the agent saves

        settings = _make_settings(
            lang_scan_use_scc=False, target_loc=40, loc_tolerance=5, max_total_loc=200,
        )
        responses = iter([
            _assistant_with_tools([_tool_call("c1", "save_sample", {"path": "notes.md", "layer": "util"})]),
            _assistant_with_tools([_summary_call("c2")]),
            _assistant_with_tools([_tool_call("c3", "finish", {"message": "done", "total_loc": 30, "file_count": 1})]),
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

        log = json.loads((out / result.folder_name / "agent_log.json").read_text())
        nudges = [e["nudge"] for e in log["agent_log"] if "nudge" in e]
        assert any("LANGUAGE REQUIREMENT FAILING" in n for n in nudges)
        assert any("Python" in n for n in nudges)


@pytest.mark.asyncio
async def test_primary_language_override_wins():
    with tempfile.TemporaryDirectory() as tmp:
        repo = Path(tmp) / "repo"
        out = Path(tmp) / "out"
        repo.mkdir()
        (repo / "main.py").write_text("x = 1\n" * 50)

        settings = _make_settings(
            lang_scan_use_scc=False, primary_language_override="SQL",
        )
        captured: list[dict] = []

        def handler(request):
            captured.append(json.loads(request.content))
            return httpx.Response(200, json=_assistant_stop())

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

        assert "PRIMARY LANGUAGE: SQL" in captured[0]["messages"][1]["content"]
        assert result.primary_language == "SQL"
        assert result.primary_forced is True


@pytest.mark.asyncio
async def test_untrackable_scc_primary_falls_back_to_trackable(monkeypatch):
    """An scc-detected primary our path-tagging can't recognize must not be
    enforced (it would count 0 LOC forever) — fall back to the best trackable."""
    from repo_sampler import agent as agent_mod
    from repo_sampler.languages import LanguageStats

    with tempfile.TemporaryDirectory() as tmp:
        repo = Path(tmp) / "repo"
        out = Path(tmp) / "out"
        repo.mkdir()
        (repo / "a.js").write_text("var x;\n")

        monkeypatch.setattr(
            agent_mod, "compute_language_stats",
            lambda *a, **k: LanguageStats(
                counts={"Solidity": 900, "JavaScript": 100},
                total=1000, primary="Solidity", source="scc",
            ),
        )
        settings = _make_settings()
        captured: list[dict] = []

        def handler(request):
            captured.append(json.loads(request.content))
            return httpx.Response(200, json=_assistant_stop())

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

        assert result.primary_language == "JavaScript"
        assert "PRIMARY LANGUAGE: JavaScript" in captured[0]["messages"][1]["content"]


@pytest.mark.asyncio
async def test_no_primary_drops_language_section_from_system_prompt():
    with tempfile.TemporaryDirectory() as tmp:
        repo = Path(tmp) / "repo"
        out = Path(tmp) / "out"
        repo.mkdir()  # empty repo -> no distribution

        settings = _make_settings(lang_scan_use_scc=False)
        captured: list[dict] = []

        def handler(request):
            captured.append(json.loads(request.content))
            return httpx.Response(200, json=_assistant_stop())

        with respx.mock:
            respx.post("https://openrouter.ai/api/v1/chat/completions").mock(side_effect=handler)
            async with httpx.AsyncClient() as client:
                await run_agent(
                    repo_path=repo,
                    repo_url="https://github.com/owner/repo",
                    output_dir=out,
                    settings=settings,
                    client=client,
                )

        assert "Language coverage" not in captured[0]["messages"][0]["content"]
        assert "could not be determined" in captured[0]["messages"][1]["content"]


@pytest.mark.asyncio
async def test_unrecoverable_language_state_ends_run_early():
    """When the cap headroom can no longer fix the primary share, the loop
    aborts instead of thrashing through contradictory nudges."""
    with tempfile.TemporaryDirectory() as tmp:
        repo = Path(tmp) / "repo"
        out = Path(tmp) / "out"
        repo.mkdir()
        (repo / "main.py").write_text("x = 1\n" * 100)       # primary: Python
        (repo / "notes.md").write_text("plain line\n" * 40)  # non-primary filler

        settings = _make_settings(
            lang_scan_use_scc=False, target_loc=40, loc_tolerance=5, max_total_loc=45,
        )
        calls = 0

        def handler(request):
            nonlocal calls
            calls += 1
            return httpx.Response(200, json=_assistant_with_tools(
                [_tool_call(f"c{calls}", "save_sample", {"path": "notes.md", "layer": "util"})]
            ))

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

        # one save (40 LOC of md), then the loop detects unrecoverability:
        # need = ceil((0.2*40 - 0)/0.8) = 10 > headroom 5 -> abort
        assert calls == 1
        log = json.loads((out / result.folder_name / "agent_log.json").read_text())
        aborted = [e for e in log["agent_log"] if "aborted" in e]
        assert aborted and "unrecoverable" in aborted[0]["aborted"]
