from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import jsonlines
from loguru import logger

from .agent import AgentResult


def _read_manifest(path: Path) -> list[dict]:
    # skip_invalid: one corrupt line (e.g. from a kill mid-append in old
    # versions) must not poison every future manifest write.
    if not path.exists():
        return []
    with jsonlines.open(path) as reader:
        records = [r for r in reader.iter(type=dict, skip_invalid=True)]
    raw_lines = sum(1 for line in path.read_text(errors="replace").splitlines() if line.strip())
    if raw_lines != len(records):
        logger.warning(f"Dropped {raw_lines - len(records)} corrupt line(s) from {path}")
    return records


def _rewrite_manifest(path: Path, records: list[dict]) -> None:
    # Write-then-rename: a kill mid-rewrite must never truncate the manifest.
    tmp = path.with_name(path.name + ".tmp")
    with jsonlines.open(tmp, mode="w") as writer:
        for r in records:
            writer.write(r)
    os.replace(tmp, path)


def append_jsonl_with_meta(
    result: AgentResult,
    path: Path,
    model: str,
    commit_sha: str = "unknown",
) -> None:
    test_loc = sum(f.loc_taken for f in result.files if f.layer == "test")
    test_share = test_loc / result.total_loc if result.total_loc else 0.0

    sample_lang_distribution: dict[str, float] = {}
    if result.total_loc:
        by_lang: dict[str, int] = {}
        for f in result.files:
            key = f.language or "Unknown"
            by_lang[key] = by_lang.get(key, 0) + f.loc_taken
        sample_lang_distribution = {
            k: round(v / result.total_loc, 4) for k, v in by_lang.items()
        }

    record = {
        "repo_url": result.repo_url,
        "repo_name": result.repo_name,
        "folder_name": result.folder_name,
        "language": result.primary_language,
        "total_loc": result.total_loc,
        "file_count": len(result.files),
        "test_share": round(test_share, 4),
        "sample_lang_distribution": sample_lang_distribution,
        "files": [
            {
                "path": f.path,
                "layer": f.layer,
                "rank": f.rank,
                "loc_taken": f.loc_taken,
                "is_partial": f.is_partial,
                "language": f.language,
            }
            for f in result.files
        ],
        "repo_summary": result.summary_md,
        "meta": {
            "model": model,
            "agent_iterations": result.agent_iterations,
            "bash_calls": result.bash_calls,
            "sampled_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "commit_sha": commit_sha,
            "repo_lang_distribution": result.repo_lang_distribution,
            "lang_stats_source": result.lang_stats_source,
            "primary_forced": result.primary_forced,
        },
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    # Re-runs (--force, zero-LOC retries) must replace the repo's earlier
    # record, not append a duplicate line. NOTE: correctness of the
    # read-filter-rewrite relies on it staying synchronous — concurrent asyncio
    # workers call it directly and cannot interleave a no-await block. Do not
    # move it to a thread or executor.
    existing = [r for r in _read_manifest(path) if r.get("repo_url") != result.repo_url]
    _rewrite_manifest(path, existing + [record])


def remove_record(path: Path, repo_url: str) -> bool:
    """Drop a repo's manifest record, e.g. when a re-run cleared its
    deliverable folder but then produced nothing. Returns True if removed."""
    records = _read_manifest(path)
    kept = [r for r in records if r.get("repo_url") != repo_url]
    if len(kept) == len(records):
        return False
    _rewrite_manifest(path, kept)
    return True


def write_parquet(output_dir: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    jsonl_path = output_dir / "samples.jsonl"
    if not jsonl_path.exists():
        return

    records = []
    with jsonlines.open(jsonl_path) as reader:
        for record in reader:
            records.append(record)

    if not records:
        return

    flat_records = [
        {
            "repo_url": r.get("repo_url", ""),
            "repo_name": r.get("repo_name", ""),
            "language": r.get("language", ""),
            "total_loc": r.get("total_loc", 0),
            "file_count": r.get("file_count", 0),
            "test_share": r.get("test_share", 0.0),
            "repo_summary": r.get("repo_summary", ""),
            "files_json": json.dumps(r.get("files", [])),
            "meta_json": json.dumps(r.get("meta", {})),
            "lang_distribution_json": json.dumps(r.get("sample_lang_distribution", {})),
        }
        for r in records
    ]

    table = pa.table({
        "repo_url": [r["repo_url"] for r in flat_records],
        "repo_name": [r["repo_name"] for r in flat_records],
        "language": [r["language"] for r in flat_records],
        "total_loc": pa.array([r["total_loc"] for r in flat_records], type=pa.int64()),
        "file_count": pa.array([r["file_count"] for r in flat_records], type=pa.int64()),
        "test_share": pa.array([r["test_share"] for r in flat_records], type=pa.float64()),
        "repo_summary": [r["repo_summary"] for r in flat_records],
        "files_json": [r["files_json"] for r in flat_records],
        "meta_json": [r["meta_json"] for r in flat_records],
        "lang_distribution_json": [r["lang_distribution_json"] for r in flat_records],
    })

    pq.write_table(table, output_dir / "samples.parquet")
