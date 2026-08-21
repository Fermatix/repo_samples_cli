from __future__ import annotations

import asyncio
import subprocess
from pathlib import Path

from repo_sampler.cloner import clone_repo
from repo_sampler.identity import (
    collect_repo_identity,
    commit_minhash,
    parse_repo_name,
    parse_repo_org,
)


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _commit(repo: Path, number: int) -> str:
    (repo / "history.txt").write_text("\n".join(map(str, range(number + 1))) + "\n")
    _git(repo, "add", "history.txt")
    _git(
        repo,
        "-c",
        "user.name=Test User",
        "-c",
        "user.email=test@example.com",
        "commit",
        "--quiet",
        "-m",
        f"commit {number}",
    )
    return _git(repo, "rev-parse", "HEAD")


def test_commit_minhash_matches_repo_metadata_cli_golden() -> None:
    hashes = [
        "0123456789abcdef0123456789abcdef01234567",
        "fedcba9876543210fedcba9876543210fedcba98",
    ]
    expected = (
        "4ff4619f13d13bfb,68954ce1856a9e84,bb38e99b1b63da17,2a24b9a2b613afed,"
        "21b19f0ce5c063e5,56c5eda5e3d9a16e,a69f7c2d9dc02c51,4642f39adc0c8c6e,"
        "2b1f207f47edddd1,8bdbedfb741cb278,1a14805cf66b6aae,3cb0f18eeee4ace1,"
        "4a662ae69936e050,6c8ceb987f821ff1,210f2ba7042699d4,33e2feaae1678ddc,"
        "4c2d7bf7d5564fa8,5c213d497c667df4,89718cb863e1b6a1,41457cc539ee68e4,"
        "a690c21e32ff95e6,522bdf6e2345329f,6fc80c5d280ad463,97f2aac373533c80,"
        "aaa7d341ca182fe2,9947357f94c5ba91,207dc8bcf5ab3f56,989ed1b417e0269b,"
        "206b0342c7d51f0c,43eb468a96b2b1cd,6338bb8d80404e73,35eca2ad2a17c762"
    )
    assert commit_minhash(hashes) == expected
    assert commit_minhash([]) == ""


def test_collect_identity_matches_git_history(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "--quiet", "-b", "main")
    commits = [_commit(repo, number) for number in range(12)]

    identity = collect_repo_identity(
        repo, "ssh://git@git.example.com:2222/group/sub/repo.git"
    )

    assert identity["first_commit_hash"] == commits[0]
    assert identity["early_commit_hashes"] == ",".join(commits[:10])
    assert identity["head_commit_sha"] == commits[-1]
    assert identity["commit_minhash"] == commit_minhash(list(reversed(commits)))
    # repo_metadata_cli keeps the leading SSH port segment in repo_org.
    assert identity["repo_org"] == "2222/group/sub"
    assert identity["repo_name"] == "repo"


def test_clone_keeps_full_history_for_identity(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    _git(source, "init", "--quiet", "-b", "main")
    commits = [_commit(source, number) for number in range(12)]
    _git(source, "checkout", "--quiet", "-b", "side")
    side_tip = _commit(source, 99)
    _git(source, "checkout", "--quiet", "main")
    destination = tmp_path / "clone"

    asyncio.run(clone_repo(str(source), destination))
    identity = collect_repo_identity(destination, "https://git.example.com/group/repo.git")

    assert _git(destination, "rev-list", "--count", "HEAD") == "12"
    assert side_tip in _git(destination, "rev-list", "--all").splitlines()
    assert identity["first_commit_hash"] == commits[0]
    assert identity["early_commit_hashes"] == ",".join(commits[:10])


def test_shallow_cached_clone_is_unshallowed(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    _git(source, "init", "--quiet", "-b", "main")
    commits = [_commit(source, number) for number in range(12)]
    _git(source, "checkout", "--quiet", "-b", "side")
    side_tip = _commit(source, 99)
    _git(source, "checkout", "--quiet", "main")
    destination = tmp_path / "clone"
    subprocess.run(
        ["git", "clone", "--quiet", "--depth=1", source.as_uri(), str(destination)],
        check=True,
    )
    assert _git(destination, "rev-parse", "--is-shallow-repository") == "true"

    asyncio.run(clone_repo(str(source), destination))

    assert _git(destination, "rev-parse", "--is-shallow-repository") == "false"
    assert _git(destination, "rev-list", "--count", "HEAD") == "12"
    assert side_tip in _git(destination, "rev-list", "--all").splitlines()
    assert collect_repo_identity(destination, str(source))["first_commit_hash"] == commits[0]


def test_repo_url_parts() -> None:
    assert parse_repo_org("https://git.example.com/group/sub/repo.git") == "group/sub"
    assert parse_repo_org("git@git.example.com:group/repo.git") == "group"
    assert parse_repo_org("https://git.example.com/repo.git") == ""
    assert parse_repo_name("git@git.example.com:group/repo.git") == "repo"
