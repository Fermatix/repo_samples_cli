from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import jsonlines

from .selector import RepoSample


def write_deliverable(sample: RepoSample, summary_md: str, output_dir: Path) -> None:
    repo_dir = output_dir / sample.repo_name
    repo_dir.mkdir(parents=True, exist_ok=True)

    for sf in sample.files:
        dest = repo_dir / "samples" / sf.path
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(sf.content, encoding="utf-8")

    (repo_dir / "repo_summary.md").write_text(summary_md, encoding="utf-8")


def append_jsonl(
    sample: RepoSample,
    summary_md: str,
    path: Path,
    model: str,
    files_found: int,
    files_scored: int,
    commit_sha: str = "unknown",
) -> None:
    test_loc = sum(sf.loc_taken for sf in sample.files if sf.layer == "test")
    test_share = test_loc / sample.total_loc if sample.total_loc else 0.0

    record = {
        "repo_url": sample.repo_url,
        "repo_name": sample.repo_name,
        "language": sample.language,
        "total_loc": sample.total_loc,
        "file_count": len(sample.files),
        "test_share": round(test_share, 4),
        "files": [
            {
                "path": sf.path,
                "layer": sf.layer,
                "rank": sf.rank,
                "loc_taken": sf.loc_taken,
                "is_partial": sf.is_partial,
            }
            for sf in sample.files
        ],
        "repo_summary": summary_md,
        "meta": {
            "files_found": files_found,
            "files_scored": files_scored,
            "model": model,
            "sampled_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "commit_sha": commit_sha,
        },
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    with jsonlines.open(path, mode="a") as writer:
        writer.write(record)


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

    flat_records = []
    for r in records:
        flat_records.append({
            "repo_url": r.get("repo_url", ""),
            "repo_name": r.get("repo_name", ""),
            "language": r.get("language", ""),
            "total_loc": r.get("total_loc", 0),
            "file_count": r.get("file_count", 0),
            "test_share": r.get("test_share", 0.0),
            "repo_summary": r.get("repo_summary", ""),
            "files_json": json.dumps(r.get("files", [])),
            "meta_json": json.dumps(r.get("meta", {})),
        })

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
    })

    pq.write_table(table, output_dir / "samples.parquet")
