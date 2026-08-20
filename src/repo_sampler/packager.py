from __future__ import annotations

import csv
import io
import json
import os
import tempfile
import zipfile
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from loguru import logger

from .config import default_meta_dir


INDEX_COLUMNS = [
    "folder",
    "repo_name",
    "repo_org",
    "repo_url",
    "first_commit_hash",
    "early_commit_hashes",
    "commit_minhash",
    "sample_loc",
    "sample_files",
    "sampled_at",
    "head_commit_sha",
    "sampler_version",
]


def _sampler_version() -> str:
    try:
        return version("repo-sampler")
    except PackageNotFoundError:
        return "0.1.0"


def _read_manifest(output_dir: Path) -> list[dict]:
    preferred = default_meta_dir(output_dir) / "samples.jsonl"
    legacy = output_dir / "samples.jsonl"
    path = preferred if preferred.exists() else legacy
    if not path.exists():
        raise ValueError(f"Sample manifest not found: {preferred}")

    records: list[dict] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid JSON on line {line_number} of {path}") from error
        if isinstance(record, dict):
            records.append(record)
    return records


def _safe_folder(record: dict) -> str:
    folder = str(record.get("folder_name") or record.get("folder") or "").strip()
    if not folder or folder in {".", ".."} or Path(folder).name != folder:
        raise ValueError(f"Invalid sample folder in manifest: {folder!r}")
    return folder


def _identity(meta_dir: Path, folder: str) -> dict[str, str]:
    path = meta_dir / folder / "repo_identity.json"
    if not path.exists():
        logger.warning(
            f"[{folder}] repo_identity.json is missing; packing a legacy run "
            "with empty repository fingerprints"
        )
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid repository identity file: {path}") from error
    if not isinstance(data, dict):
        raise ValueError(f"Invalid repository identity file: {path}")
    return {str(key): str(value) for key, value in data.items() if value is not None}


def _index_bytes(output_dir: Path, records: list[dict]) -> tuple[bytes, list[tuple[str, Path]]]:
    meta_dir = default_meta_dir(output_dir)
    rows: list[dict[str, object]] = []
    folders: list[tuple[str, Path]] = []
    seen: set[str] = set()
    sampler_version = _sampler_version()

    for record in records:
        folder = _safe_folder(record)
        if folder in seen:
            raise ValueError(f"Duplicate sample folder in manifest: {folder}")
        seen.add(folder)
        source = output_dir / folder
        if not source.is_dir():
            raise ValueError(f"Sample folder listed in manifest is missing: {source}")
        identity = _identity(meta_dir, folder)
        metadata = record.get("meta") if isinstance(record.get("meta"), dict) else {}
        rows.append(
            {
                "folder": folder,
                "repo_name": identity.get("repo_name", record.get("repo_name", "")),
                "repo_org": identity.get("repo_org", ""),
                "repo_url": identity.get("repo_url", record.get("repo_url", "")),
                "first_commit_hash": identity.get("first_commit_hash", ""),
                "early_commit_hashes": identity.get("early_commit_hashes", ""),
                "commit_minhash": identity.get("commit_minhash", ""),
                "sample_loc": record.get("total_loc", ""),
                "sample_files": record.get("file_count", ""),
                "sampled_at": metadata.get("sampled_at", ""),
                "head_commit_sha": identity.get("head_commit_sha", ""),
                "sampler_version": sampler_version,
            }
        )
        folders.append((folder, source))

    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=INDEX_COLUMNS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8"), folders


def build_samples_archive(output_dir: Path, archive_path: Path) -> int:
    output_dir = output_dir.resolve()
    if not output_dir.is_dir():
        raise ValueError(f"Output directory not found: {output_dir}")
    records = _read_manifest(output_dir)
    if not records:
        raise ValueError("Sample manifest is empty")
    index_bytes, folders = _index_bytes(output_dir, records)

    archive_path = archive_path.resolve()
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=archive_path.name + ".",
        suffix=".tmp",
        dir=archive_path.parent,
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        with zipfile.ZipFile(
            temporary_path, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True
        ) as archive:
            archive.writestr("samples_index.csv", index_bytes)
            for folder, source in folders:
                for path in sorted(source.rglob("*")):
                    if path.is_symlink():
                        logger.warning(f"[{folder}] skipping symlink: {path.relative_to(source)}")
                        continue
                    if path.is_file():
                        relative = path.relative_to(source).as_posix()
                        archive.write(path, f"samples/{folder}/{relative}")
        os.replace(temporary_path, archive_path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise
    return len(folders)
