import jsonlines

from repo_sampler.agent import AgentResult
from repo_sampler.writer import append_jsonl_with_meta, remove_record


def _result(url: str, total_loc: int) -> AgentResult:
    name = url.rstrip("/").split("/")[-1]
    return AgentResult(
        repo_url=url,
        repo_name=name,
        folder_name=f"h.com__o__{name}",
        total_loc=total_loc,
    )


def _read(path):
    with jsonlines.open(path) as reader:
        return list(reader)


def test_append_creates_file_and_appends_distinct_repos(tmp_path):
    path = tmp_path / "samples.jsonl"
    append_jsonl_with_meta(_result("https://h.com/o/a", 5000), path, model="m")
    append_jsonl_with_meta(_result("https://h.com/o/b", 4800), path, model="m")

    records = _read(path)
    assert [r["repo_url"] for r in records] == ["https://h.com/o/a", "https://h.com/o/b"]


def test_rerun_replaces_existing_record_instead_of_duplicating(tmp_path):
    """--force re-collection must not leave two records for the same repo."""
    path = tmp_path / "samples.jsonl"
    append_jsonl_with_meta(_result("https://h.com/o/a", 5000), path, model="m")
    append_jsonl_with_meta(_result("https://h.com/o/b", 4800), path, model="m")
    append_jsonl_with_meta(_result("https://h.com/o/a", 6100), path, model="m")

    records = _read(path)
    assert len(records) == 2
    by_url = {r["repo_url"]: r for r in records}
    assert by_url["https://h.com/o/a"]["total_loc"] == 6100
    assert by_url["https://h.com/o/b"]["total_loc"] == 4800
    assert not path.with_name(path.name + ".tmp").exists()


def test_corrupt_line_does_not_block_future_writes(tmp_path):
    """A truncated/garbage line (e.g. kill mid-append in old versions) must not
    poison every subsequent manifest write."""
    path = tmp_path / "samples.jsonl"
    append_jsonl_with_meta(_result("https://h.com/o/a", 5000), path, model="m")
    with open(path, "a") as f:
        f.write('{"truncated: \n')
    append_jsonl_with_meta(_result("https://h.com/o/b", 4800), path, model="m")

    records = _read(path)
    assert [r["repo_url"] for r in records] == ["https://h.com/o/a", "https://h.com/o/b"]


def test_remove_record(tmp_path):
    path = tmp_path / "samples.jsonl"
    append_jsonl_with_meta(_result("https://h.com/o/a", 5000), path, model="m")
    append_jsonl_with_meta(_result("https://h.com/o/b", 4800), path, model="m")

    assert remove_record(path, "https://h.com/o/a") is True
    assert [r["repo_url"] for r in _read(path)] == ["https://h.com/o/b"]
    assert remove_record(path, "https://h.com/o/a") is False  # already gone
    assert remove_record(tmp_path / "missing.jsonl", "x") is False
