from __future__ import annotations

import csv
import io
import json
import zipfile
from pathlib import Path

import pytest

from repo_sampler.packager import INDEX_COLUMNS, build_samples_archive


def _write_run(tmp_path: Path, *, with_identity: bool = True) -> tuple[Path, str]:
    output = tmp_path / "output"
    meta = tmp_path / "output_meta"
    folder = "git.example.com__group__repo"
    sample = output / folder
    (sample / "samples" / "src").mkdir(parents=True)
    (sample / "samples" / "src" / "app.py").write_text("print('ok')\n")
    (sample / "repo_summary.md").write_text("# Summary\n")
    meta.mkdir()
    manifest = {
        "folder_name": folder,
        "repo_name": "repo",
        "repo_url": "https://git.example.com/group/repo.git",
        "total_loc": 5012,
        "file_count": 17,
        "meta": {"sampled_at": "2026-08-20T10:11:12Z", "commit_sha": "abcdef0"},
    }
    (meta / "samples.jsonl").write_text(json.dumps(manifest) + "\n")
    if with_identity:
        identity_dir = meta / folder
        identity_dir.mkdir()
        (identity_dir / "repo_identity.json").write_text(
            json.dumps(
                {
                    "first_commit_hash": "first",
                    "early_commit_hashes": "early-1,early-2",
                    "commit_minhash": "minhash",
                    "repo_url": manifest["repo_url"],
                    "repo_org": "group",
                    "repo_name": "repo",
                    "head_commit_sha": "abcdef0123456789",
                }
            )
        )
    return output, folder


def test_build_samples_archive(tmp_path: Path) -> None:
    output, folder = _write_run(tmp_path)
    archive_path = tmp_path / "samples.zip"

    assert build_samples_archive(output, archive_path) == 1

    with zipfile.ZipFile(archive_path) as archive:
        assert set(archive.namelist()) == {
            "samples_index.csv",
            f"samples/{folder}/repo_summary.md",
            f"samples/{folder}/samples/src/app.py",
        }
        rows = list(
            csv.DictReader(io.StringIO(archive.read("samples_index.csv").decode("utf-8")))
        )
    assert list(rows[0]) == INDEX_COLUMNS
    assert rows[0] == {
        "folder": folder,
        "repo_name": "repo",
        "repo_org": "group",
        "repo_url": "https://git.example.com/group/repo.git",
        "first_commit_hash": "first",
        "early_commit_hashes": "early-1,early-2",
        "commit_minhash": "minhash",
        "sample_loc": "5012",
        "sample_files": "17",
        "sampled_at": "2026-08-20T10:11:12Z",
        "head_commit_sha": "abcdef0123456789",
        "sampler_version": "0.1.0",
    }


def test_build_legacy_run_without_identity(tmp_path: Path) -> None:
    output, folder = _write_run(tmp_path, with_identity=False)
    archive_path = tmp_path / "legacy.zip"

    build_samples_archive(output, archive_path)

    with zipfile.ZipFile(archive_path) as archive:
        row = next(
            csv.DictReader(io.StringIO(archive.read("samples_index.csv").decode("utf-8")))
        )
    assert row["folder"] == folder
    assert row["repo_url"] == "https://git.example.com/group/repo.git"
    assert row["first_commit_hash"] == ""
    assert row["early_commit_hashes"] == ""
    assert row["commit_minhash"] == ""
    assert row["head_commit_sha"] == ""


def test_missing_sample_folder_does_not_replace_archive(tmp_path: Path) -> None:
    output, folder = _write_run(tmp_path)
    for path in sorted((output / folder).rglob("*"), reverse=True):
        if path.is_file():
            path.unlink()
        else:
            path.rmdir()
    (output / folder).rmdir()
    archive_path = tmp_path / "samples.zip"
    archive_path.write_bytes(b"previous archive")

    with pytest.raises(ValueError, match="is missing"):
        build_samples_archive(output, archive_path)

    assert archive_path.read_bytes() == b"previous archive"
