import tempfile
from pathlib import Path

import pytest

from repo_sampler.anonymizer import (
    _build_claude_argv,
    _extract_cost,
    _parse_unified_diff,
    compute_diff,
    discover_sample_dirs,
    is_anonymized,
    snapshot_targets,
)
from repo_sampler.config import Settings


def _make_sample_dir(root: Path, name: str, with_summary: bool = True) -> Path:
    d = root / name
    (d / "samples" / "src").mkdir(parents=True)
    (d / "samples" / "src" / "app.py").write_text("x = 1\n")
    if with_summary:
        (d / "repo_summary.md").write_text("# Overview\n")
    return d


# ---------------------------------------------------------------------------
# discover_sample_dirs
# ---------------------------------------------------------------------------

def test_discover_finds_dirs_with_summary():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _make_sample_dir(root, "github.com__a__b")
        _make_sample_dir(root, "gitlab.com__c__d")
        dirs = discover_sample_dirs(root)
        names = {d.name for d in dirs}
        assert names == {"github.com__a__b", "gitlab.com__c__d"}


def test_discover_skips_archive_and_files():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _make_sample_dir(root, "github.com__a__b")
        _make_sample_dir(root, "archive")  # excluded by name
        (root / "samples.jsonl").write_text("{}\n")  # top-level file
        (root / "no_summary").mkdir()  # dir without repo_summary.md
        dirs = discover_sample_dirs(root)
        names = {d.name for d in dirs}
        assert names == {"github.com__a__b"}


def test_discover_missing_output_returns_empty():
    with tempfile.TemporaryDirectory() as tmp:
        assert discover_sample_dirs(Path(tmp) / "nope") == []


# ---------------------------------------------------------------------------
# resume marker
# ---------------------------------------------------------------------------

def test_is_anonymized_marker():
    with tempfile.TemporaryDirectory() as tmp:
        d = _make_sample_dir(Path(tmp), "repo")
        assert is_anonymized(d) is False
        (d / "anonymization_report.json").write_text("{}")
        assert is_anonymized(d) is True


# ---------------------------------------------------------------------------
# argv builder
# ---------------------------------------------------------------------------

def test_argv_basic_flags():
    settings = Settings(anonymizer_model="claude-sonnet-4-6")
    argv = _build_claude_argv(settings)
    assert argv[0] == "claude"
    assert "-p" in argv
    assert "--model" in argv and "claude-sonnet-4-6" in argv
    assert "--output-format" in argv and "json" in argv
    assert "--no-session-persistence" in argv
    assert "--permission-mode" in argv
    # No budget flag by default
    assert "--max-budget-usd" not in argv


def test_argv_includes_effort():
    settings = Settings(anonymizer_effort="high")
    argv = _build_claude_argv(settings)
    assert "--effort" in argv
    assert argv[argv.index("--effort") + 1] == "high"


def test_argv_omits_effort_when_empty():
    settings = Settings(anonymizer_effort="")
    assert "--effort" not in _build_claude_argv(settings)


def test_argv_includes_budget_when_set():
    settings = Settings(anonymizer_max_budget_usd=2.5)
    argv = _build_claude_argv(settings)
    assert "--max-budget-usd" in argv
    idx = argv.index("--max-budget-usd")
    assert argv[idx + 1] == "2.5"


def test_argv_omits_budget_when_zero():
    settings = Settings(anonymizer_max_budget_usd=0.0)
    assert "--max-budget-usd" not in _build_claude_argv(settings)


# ---------------------------------------------------------------------------
# cost extraction
# ---------------------------------------------------------------------------

def test_extract_cost_from_json():
    assert _extract_cost('{"result": "ok", "total_cost_usd": 0.0123}') == 0.0123


def test_extract_cost_missing_returns_none():
    assert _extract_cost('{"result": "ok"}') is None
    assert _extract_cost("not json") is None


# ---------------------------------------------------------------------------
# diff parsing
# ---------------------------------------------------------------------------

def test_parse_unified_diff_counts():
    diff = (
        "--- a/samples/app.py\t2020\n"
        "+++ /out/repo/samples/app.py\t2020\n"
        "@@ -1,2 +1,2 @@\n"
        "-name = 'ACME Corp'\n"
        "+name = 'Acme'\n"
        " other = 1\n"
    )
    stats = _parse_unified_diff(diff)
    assert len(stats) == 1
    assert stats[0]["hunks"] == 1
    assert stats[0]["additions"] == 1
    assert stats[0]["deletions"] == 1


@pytest.mark.asyncio
async def test_compute_diff_detects_changes():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        d = _make_sample_dir(root, "repo")
        # Snapshot the original, then edit a file
        snap = snapshot_targets(d)
        (d / "samples" / "src" / "app.py").write_text("x = 2\ny = 3\n")

        diff_text, stats = await compute_diff(snap, d)

        assert "app.py" in diff_text
        assert len(stats) == 1
        assert stats[0]["path"] == "samples/src/app.py"
        assert stats[0]["additions"] >= 1


@pytest.mark.asyncio
async def test_compute_diff_no_changes():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        d = _make_sample_dir(root, "repo")
        snap = snapshot_targets(d)  # no edits

        diff_text, stats = await compute_diff(snap, d)
        assert stats == []
