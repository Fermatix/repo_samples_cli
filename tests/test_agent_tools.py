import asyncio
import tempfile
from pathlib import Path

import pytest

from repo_sampler.agent import (
    _ToolCtx,
    _count_loc_lines,
    _exec_bash,
    url_to_folder_name,
    _exec_save_sample,
    _exec_write_summary,
    _truncate_old_tool_results,
)
from repo_sampler.config import Settings


def _make_ctx(repo_path: Path, deliverable_dir: Path) -> _ToolCtx:
    return _ToolCtx(
        repo_path=repo_path,
        deliverable_dir=deliverable_dir,
        settings=Settings(openrouter_api_key="x"),
    )


# ---------------------------------------------------------------------------
# save_sample
# ---------------------------------------------------------------------------

def test_save_sample_verbatim_full_file():
    with tempfile.TemporaryDirectory() as tmp:
        repo = Path(tmp) / "repo"
        out = Path(tmp) / "out"
        repo.mkdir()
        content = "x = 1\ny = 2\nz = 3\n"
        (repo / "src").mkdir()
        (repo / "src" / "app.py").write_text(content)

        ctx = _make_ctx(repo, out)
        result = _exec_save_sample(ctx, "src/app.py", "business")

        assert result["saved"] is True
        dest = out / "samples" / "src" / "app.py"
        assert dest.exists()
        assert dest.read_text() == content  # verbatim
        assert len(ctx.saved_files) == 1
        assert ctx.saved_files[0].is_partial is False


def test_save_sample_partial_slice():
    with tempfile.TemporaryDirectory() as tmp:
        repo = Path(tmp) / "repo"
        out = Path(tmp) / "out"
        repo.mkdir()
        lines = [f"line_{i}\n" for i in range(20)]
        (repo / "f.py").write_text("".join(lines))

        ctx = _make_ctx(repo, out)
        result = _exec_save_sample(ctx, "f.py", "util", start_line=5, end_line=10)

        assert result["saved"] is True
        saved = (out / "samples" / "f.py").read_text()
        assert "line_4" in saved   # 1-indexed line 5 = line_4
        assert "line_9" in saved   # 1-indexed line 10 = line_9
        assert "line_0" not in saved
        assert ctx.saved_files[0].is_partial is True


def test_save_sample_running_total():
    with tempfile.TemporaryDirectory() as tmp:
        repo = Path(tmp) / "repo"
        out = Path(tmp) / "out"
        repo.mkdir()
        (repo / "a.py").write_text("x = 1\ny = 2\n")
        (repo / "b.py").write_text("a = 1\nb = 2\nc = 3\n")

        ctx = _make_ctx(repo, out)
        r1 = _exec_save_sample(ctx, "a.py", "util")
        r2 = _exec_save_sample(ctx, "b.py", "util")

        assert r1["running_total_loc"] == r1["loc"]
        assert r2["running_total_loc"] == r1["loc"] + r2["loc"]


def test_save_sample_nonexistent_returns_error():
    with tempfile.TemporaryDirectory() as tmp:
        ctx = _make_ctx(Path(tmp) / "repo", Path(tmp) / "out")
        result = _exec_save_sample(ctx, "does_not_exist.py", "other")
        assert result["saved"] is False
        assert "error" in result


def test_save_sample_out_of_bounds_lines():
    with tempfile.TemporaryDirectory() as tmp:
        repo = Path(tmp) / "repo"
        out = Path(tmp) / "out"
        repo.mkdir()
        (repo / "f.py").write_text("a\nb\nc\n")

        ctx = _make_ctx(repo, out)
        result = _exec_save_sample(ctx, "f.py", "util", start_line=10, end_line=5)
        assert result["saved"] is False


# ---------------------------------------------------------------------------
# bash tool
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_bash_runs_in_repo_cwd():
    with tempfile.TemporaryDirectory() as tmp:
        repo = Path(tmp) / "repo"
        repo.mkdir()
        (repo / "hello.py").write_text("print('hi')")

        ctx = _make_ctx(repo, Path(tmp) / "out")
        result = await _exec_bash(ctx, "ls")

        assert "hello.py" in result
        assert ctx.bash_calls == 1


@pytest.mark.asyncio
async def test_bash_captures_stderr():
    with tempfile.TemporaryDirectory() as tmp:
        repo = Path(tmp) / "repo"
        repo.mkdir()
        ctx = _make_ctx(repo, Path(tmp) / "out")
        result = await _exec_bash(ctx, "cat nonexistent_file.txt")
        # stderr is merged with stdout
        assert "nonexistent" in result or "No such file" in result


@pytest.mark.asyncio
async def test_bash_truncates_long_output():
    with tempfile.TemporaryDirectory() as tmp:
        repo = Path(tmp) / "repo"
        repo.mkdir()
        settings = Settings(openrouter_api_key="x", agent_bash_output_limit=100)
        ctx = _ToolCtx(
            repo_path=repo, deliverable_dir=Path(tmp) / "out",
            settings=settings,
        )
        # Generate output longer than 100 chars
        result = await _exec_bash(ctx, "python3 -c \"print('x' * 500)\"")
        assert "truncated" in result
        # Output is capped + truncation msg + progress footer
        assert result.count("x") <= 101


@pytest.mark.asyncio
async def test_bash_timeout_returns_message():
    with tempfile.TemporaryDirectory() as tmp:
        repo = Path(tmp) / "repo"
        repo.mkdir()
        settings = Settings(openrouter_api_key="x", agent_bash_timeout=1)
        ctx = _ToolCtx(
            repo_path=repo, deliverable_dir=Path(tmp) / "out",
            settings=settings,
        )
        result = await _exec_bash(ctx, "sleep 5")
        assert "timed out" in result.lower()


# ---------------------------------------------------------------------------
# count_loc_lines
# ---------------------------------------------------------------------------

def test_count_loc_lines_skips_blank_and_comments():
    lines = ["x = 1\n", "\n", "# comment\n", "y = 2\n"]
    assert _count_loc_lines(lines, "python") == 2


def test_count_loc_lines_go():
    lines = ["// comment\n", "func main() {}\n", "\n", "/* block */\n"]
    assert _count_loc_lines(lines, "go") == 1


# ---------------------------------------------------------------------------
# write_summary
# ---------------------------------------------------------------------------

def test_write_summary_creates_file():
    with tempfile.TemporaryDirectory() as tmp:
        ctx = _make_ctx(Path(tmp) / "repo", Path(tmp) / "out")
        _exec_write_summary(ctx, "# Hello\n\nContent here.\n")
        dest = Path(tmp) / "out" / "repo_summary.md"
        assert dest.exists()
        assert "Hello" in dest.read_text()
        assert ctx.summary_md == "# Hello\n\nContent here.\n"


def test_write_summary_overwrites():
    with tempfile.TemporaryDirectory() as tmp:
        ctx = _make_ctx(Path(tmp) / "repo", Path(tmp) / "out")
        _exec_write_summary(ctx, "first")
        _exec_write_summary(ctx, "second")
        dest = Path(tmp) / "out" / "repo_summary.md"
        assert dest.read_text() == "second"


# ---------------------------------------------------------------------------
# Context truncation
# ---------------------------------------------------------------------------

def test_truncate_old_tool_results_truncates_old():
    messages = [
        {"role": "tool", "tool_call_id": f"id_{i}", "content": "x" * 500}
        for i in range(6)
    ]
    _truncate_old_tool_results(messages, keep_last=2)
    # First 4 should be truncated
    for m in messages[:4]:
        assert len(m["content"]) < 300
        assert "truncated" in m["content"]
    # Last 2 stay intact
    for m in messages[4:]:
        assert m["content"] == "x" * 500


def test_truncate_old_tool_results_short_content_unchanged():
    messages = [
        {"role": "tool", "tool_call_id": "id_0", "content": "short"},
        {"role": "tool", "tool_call_id": "id_1", "content": "also short"},
    ]
    _truncate_old_tool_results(messages, keep_last=1)
    # Short content (< 300 chars) is not truncated even if old
    assert messages[0]["content"] == "short"


# ---------------------------------------------------------------------------
# url_to_folder_name
# ---------------------------------------------------------------------------

def test_url_to_folder_name_github():
    assert url_to_folder_name("https://github.com/owner/repo") == "github.com__owner__repo"

def test_url_to_folder_name_gitlab_deep():
    assert url_to_folder_name(
        "https://gitlab.com/org/subgroup/project/repo"
    ) == "gitlab.com__org__subgroup__project__repo"

def test_url_to_folder_name_strips_git_suffix():
    assert url_to_folder_name("https://github.com/owner/repo.git") == "github.com__owner__repo"

def test_url_to_folder_name_strips_trailing_slash():
    assert url_to_folder_name("https://github.com/owner/repo/") == "github.com__owner__repo"

def test_url_to_folder_name_ssh():
    assert url_to_folder_name("git@github.com:owner/repo.git") == "github.com__owner__repo"
