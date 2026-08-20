import json
import tempfile
from pathlib import Path

from repo_sampler.main import _load_processed, _load_repos, _migrate_root_meta_files


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n")


def test_migrate_root_meta_files_moves_legacy_manifest(tmp_path):
    """Pre-meta-layout runs left samples.jsonl etc. at the output root; a new
    run must relocate them so resume state is found and output stays clean."""
    output = tmp_path / "output"
    meta = tmp_path / "output_meta"
    output.mkdir()
    _write_jsonl(output / "samples.jsonl", [{"repo_url": "https://h/o/r", "total_loc": 10}])
    (output / "errors.jsonl").write_text("{}\n")
    (output / "keep-me.txt").write_text("not a meta file")

    _migrate_root_meta_files(output, meta)

    assert not (output / "samples.jsonl").exists()
    assert not (output / "errors.jsonl").exists()
    assert (output / "keep-me.txt").exists()
    assert _load_processed(meta / "samples.jsonl") == {"https://h/o/r"}


def test_migrate_root_meta_files_parks_on_collision(tmp_path):
    """If both layouts have the file, the meta one stays authoritative and the
    legacy copy is parked, not merged over it."""
    output = tmp_path / "output"
    meta = tmp_path / "output_meta"
    output.mkdir()
    meta.mkdir()
    (output / "samples.jsonl").write_text("legacy\n")
    (meta / "samples.jsonl").write_text("current\n")

    _migrate_root_meta_files(output, meta)

    assert not (output / "samples.jsonl").exists()
    assert (meta / "samples.jsonl").read_text() == "current\n"
    assert (meta / "samples.jsonl.pre-meta").read_text() == "legacy\n"


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


def _result_with(files: list[tuple[str, int, str]], primary: str,
                 forced: bool = True) -> AgentResult:
    saved = [
        AgentSavedFile(path=p, layer="business", loc_taken=loc, is_partial=False,
                       rank=i + 1, language=lang)
        for i, (p, loc, lang) in enumerate(files)
    ]
    return AgentResult(
        repo_url="https://h.com/o/r", repo_name="r", folder_name="h.com__o__r",
        files=saved, total_loc=sum(f.loc_taken for f in saved),
        primary_language=primary, primary_forced=forced,
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


def test_rejected_deliverable_dir_is_deleted(tmp_path, monkeypatch):
    """A sample that fails primary-language validation must not leave a
    populated deliverable folder behind — the anonymize step would ship it."""
    import asyncio

    import httpx

    from repo_sampler import main as main_mod

    url = "https://h.com/o/r"
    folder = "h.com__o__r"
    output = tmp_path / "out"
    output.mkdir()

    async def fake_clone(u, dest, timeout=0):
        dest.mkdir(parents=True, exist_ok=True)

    failing = _result_with([("a.php", 5000, "PHP")], primary="JavaScript")
    failing.folder_name = folder

    async def fake_run_agent(repo_path, repo_url, output_dir, settings, client):
        d = output_dir / folder
        (d / "samples").mkdir(parents=True, exist_ok=True)
        (d / "samples" / "a.php").write_text("<?php\n")
        (d / "repo_summary.md").write_text("summary")
        return failing

    monkeypatch.setattr(main_mod, "clone_repo", fake_clone)
    monkeypatch.setattr(main_mod, "run_agent", fake_run_agent)
    monkeypatch.setattr(main_mod, "cleanup_repo", lambda p: None)
    monkeypatch.setattr(
        main_mod,
        "collect_repo_identity",
        lambda path, repo_url: {"repo_url": repo_url},
    )

    settings = Settings(openrouter_api_key="k", clone_dir=str(tmp_path / "clones"))

    async def go():
        async with httpx.AsyncClient() as client:
            return await main_mod._process_repo(
                url, output, settings, client,
                keep_clones=True, dry_run=False,
                clone_sem=asyncio.Semaphore(1),
                errors_path=output / "errors.jsonl",
            )

    result = asyncio.run(go())
    assert "error" in result
    assert not (output / folder).exists()          # rejected deliverable removed
    assert (output / "errors.jsonl").exists()
    assert "agent_no_primary_lang" in (output / "errors.jsonl").read_text()


def test_successful_process_writes_identity_before_manifest(tmp_path, monkeypatch):
    import asyncio

    import httpx

    from repo_sampler import main as main_mod

    url = "https://h.com/o/r"
    folder = "h.com__o__r"
    output = tmp_path / "out"
    output.mkdir()
    result = _result_with([("a.php", 5000, "PHP")], primary="PHP")
    result.repo_url = url
    result.folder_name = folder

    async def fake_clone(_url, destination, timeout=0):
        destination.mkdir(parents=True, exist_ok=True)

    async def fake_checkout(_path):
        return "origin/main"

    async def fake_run_agent(repo_path, repo_url, output_dir, settings, client):
        sample_dir = output_dir / folder / "samples"
        sample_dir.mkdir(parents=True)
        (sample_dir / "a.php").write_text("<?php\n")
        (output_dir / folder / "repo_summary.md").write_text("summary")
        meta_dir = main_mod.default_meta_dir(output_dir) / folder
        meta_dir.mkdir(parents=True)
        (meta_dir / "agent_log.json").write_text("{}")
        return result

    async def fake_commit_sha(_path):
        return "abcdef0"

    events = []
    write_identity = main_mod.write_repo_identity
    append_manifest = main_mod.append_jsonl_with_meta

    def tracked_identity(path, identity):
        events.append("identity")
        write_identity(path, identity)

    def tracked_manifest(agent_result, path, **kwargs):
        identity_path = main_mod.default_meta_dir(output) / folder / "repo_identity.json"
        assert identity_path.exists()
        events.append("manifest")
        append_manifest(agent_result, path, **kwargs)

    monkeypatch.setattr(main_mod, "clone_repo", fake_clone)
    monkeypatch.setattr(main_mod, "checkout_latest_branch", fake_checkout)
    monkeypatch.setattr(main_mod, "run_agent", fake_run_agent)
    monkeypatch.setattr(main_mod, "_get_commit_sha", fake_commit_sha)
    monkeypatch.setattr(main_mod, "cleanup_repo", lambda _path: None)
    monkeypatch.setattr(
        main_mod,
        "collect_repo_identity",
        lambda _path, repo_url: {"repo_url": repo_url, "head_commit_sha": "a" * 40},
    )
    monkeypatch.setattr(main_mod, "write_repo_identity", tracked_identity)
    monkeypatch.setattr(main_mod, "append_jsonl_with_meta", tracked_manifest)
    settings = Settings(openrouter_api_key="k", clone_dir=str(tmp_path / "clones"))

    async def go():
        async with httpx.AsyncClient() as client:
            return await main_mod._process_repo(
                url,
                output,
                settings,
                client,
                keep_clones=True,
                dry_run=False,
                clone_sem=asyncio.Semaphore(1),
                errors_path=output / "errors.jsonl",
            )

    processed = asyncio.run(go())
    assert "error" not in processed
    assert events == ["identity", "manifest"]
    assert (main_mod.default_meta_dir(output) / "samples.jsonl").exists()


def test_result_failure_soft_share_not_rejected():
    """Auto-detected primary is a soft goal: low share is recorded, not rejected."""
    settings = Settings(openrouter_api_key="k", primary_share_min=0.20)
    result = _result_with(
        [("a.php", 4900, "PHP"), ("b.js", 100, "JavaScript")],
        primary="JavaScript", forced=False,
    )
    assert _result_failure(result, settings) is None
