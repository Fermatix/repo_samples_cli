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
