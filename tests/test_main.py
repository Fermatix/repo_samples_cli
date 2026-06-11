import json
import tempfile
from pathlib import Path

from repo_sampler.main import _load_processed, _load_repos


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n")


def test_load_processed_skips_zero_loc_records():
    """Zero-LOC records are failed runs — re-runs must retry them."""
    with tempfile.TemporaryDirectory() as tmp:
        jsonl = Path(tmp) / "samples.jsonl"
        _write_jsonl(jsonl, [
            {"repo_url": "https://h/a/good", "total_loc": 5000},
            {"repo_url": "https://h/a/empty", "total_loc": 0},
            {"repo_url": "https://h/a/missing-field"},
        ])
        processed = _load_processed(jsonl)
        assert processed == {"https://h/a/good"}


def test_load_processed_missing_file():
    assert _load_processed(Path("/nonexistent/samples.jsonl")) == set()


def test_load_repos_dedupes_preserving_order():
    with tempfile.TemporaryDirectory() as tmp:
        repos = Path(tmp) / "repos.txt"
        repos.write_text(
            "https://h/o/r1\n"
            "# comment\n"
            "https://h/o/r2\n"
            "https://h/o/r1\n"
            "\n"
            "https://h/o/r3\n"
        )
        assert _load_repos(repos) == [
            "https://h/o/r1",
            "https://h/o/r2",
            "https://h/o/r3",
        ]


# ---------------------------------------------------------------------------
# _result_failure: zero-LOC and primary-language validation
# ---------------------------------------------------------------------------

from repo_sampler.agent import AgentResult, AgentSavedFile
from repo_sampler.config import Settings
from repo_sampler.main import _result_failure


def _result_with(files: list[tuple[str, int, str]], primary: str) -> AgentResult:
    saved = [
        AgentSavedFile(path=p, layer="business", loc_taken=loc, is_partial=False,
                       rank=i + 1, language=lang)
        for i, (p, loc, lang) in enumerate(files)
    ]
    return AgentResult(
        repo_url="https://h.com/o/r", repo_name="r", folder_name="h.com__o__r",
        files=saved, total_loc=sum(f.loc_taken for f in saved),
        primary_language=primary,
    )


def test_result_failure_zero_loc():
    settings = Settings(openrouter_api_key="k")
    result = _result_with([], primary="Python")
    assert _result_failure(result, settings) == ("agent_empty", "agent saved 0 LOC")


def test_result_failure_primary_missing_entirely():
    settings = Settings(openrouter_api_key="k")
    result = _result_with([("a.php", 5000, "PHP")], primary="JavaScript")
    stage, msg = _result_failure(result, settings)
    assert stage == "agent_no_primary_lang"
    assert "JavaScript" in msg and "0%" in msg


def test_result_failure_primary_below_minimum():
    settings = Settings(openrouter_api_key="k", primary_share_min=0.20)
    result = _result_with(
        [("a.php", 4500, "PHP"), ("b.js", 500, "JavaScript")], primary="JavaScript"
    )
    stage, _ = _result_failure(result, settings)
    assert stage == "agent_no_primary_lang"


def test_result_failure_primary_met():
    settings = Settings(openrouter_api_key="k", primary_share_min=0.20)
    result = _result_with(
        [("a.php", 3500, "PHP"), ("b.js", 1500, "JavaScript")], primary="JavaScript"
    )
    assert _result_failure(result, settings) is None


def test_result_failure_no_primary_language_known():
    settings = Settings(openrouter_api_key="k")
    result = _result_with([("a.weird", 100, "")], primary="")
    assert _result_failure(result, settings) is None
