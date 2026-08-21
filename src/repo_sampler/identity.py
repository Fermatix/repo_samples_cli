from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from pathlib import Path


_MINHASH_PERMS = 32
_MASK64 = (1 << 64) - 1


def _minhash_consts(index: int) -> tuple[int, int]:
    digest = hashlib.blake2b(str(index).encode(), digest_size=16).digest()
    multiplier = int.from_bytes(digest[:8], "big") | 1
    increment = int.from_bytes(digest[8:], "big")
    return multiplier, increment


_MINHASH_AB = [_minhash_consts(index) for index in range(_MINHASH_PERMS)]


def commit_minhash(hashes: list[str]) -> str:
    """Match repo_metadata_cli's 32-permutation commit MinHash exactly."""
    values = [int(commit[:16], 16) for commit in hashes if len(commit) >= 16]
    if not values:
        return ""
    signature = [
        min((multiplier * value + increment) & _MASK64 for value in values)
        for multiplier, increment in _MINHASH_AB
    ]
    return ",".join(format(component, "016x") for component in signature)


def parse_repo_org(url: str) -> str:
    """Return the namespace path, using repo_metadata_cli's URL rules."""
    value = url.strip().rstrip("/")
    if not value:
        return ""
    if value.endswith(".git"):
        value = value[:-4]
    value = re.sub(r"^[a-zA-Z]+://", "", value)
    value = value.replace(":", "/", 1)
    value = re.sub(r"^[^@/]+@", "", value)
    if "/" not in value:
        return ""
    path = value.split("/", 1)[1]
    segments = [segment for segment in path.split("/") if segment]
    if len(segments) <= 1:
        return ""
    return "/".join(segments[:-1])


def parse_repo_name(url: str) -> str:
    value = url.strip().rstrip("/")
    if value.endswith(".git"):
        value = value[:-4]
    return value.rsplit("/", 1)[-1].rsplit(":", 1)[-1]


def _git_lines(repo_path: Path, *args: str) -> list[str]:
    result = subprocess.run(
        ["git", "-C", str(repo_path), *args],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def collect_repo_identity(repo_path: Path, repo_url: str) -> dict[str, str]:
    """Collect stable repository identifiers while the clone still exists."""
    root_hashes = _git_lines(repo_path, "rev-list", "--max-parents=0", "HEAD")
    early_hashes = _git_lines(repo_path, "rev-list", "--reverse", "HEAD")[:10]
    head_hashes = _git_lines(repo_path, "rev-parse", "HEAD")
    all_hashes = _git_lines(repo_path, "rev-list", "--all")
    return {
        "first_commit_hash": ",".join(root_hashes),
        "early_commit_hashes": ",".join(early_hashes),
        "commit_minhash": commit_minhash(all_hashes),
        "repo_url": repo_url,
        "repo_org": parse_repo_org(repo_url),
        "repo_name": parse_repo_name(repo_url),
        "head_commit_sha": head_hashes[0] if head_hashes else "",
    }


def write_repo_identity(path: Path, identity: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(identity, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)
