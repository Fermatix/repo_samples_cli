import asyncio
import subprocess
import tempfile
from pathlib import Path

import pytest

from repo_sampler.agent import url_to_folder_name
from repo_sampler.cloner import CloneError, clone_repo, rewrite_url


# ---------------------------------------------------------------------------
# rewrite_url
# ---------------------------------------------------------------------------

def test_rewrite_as_is_passthrough():
    url = "https://git.example.com/owner/repo"
    assert rewrite_url(url, "as-is") == url


def test_rewrite_https_to_ssh():
    assert rewrite_url("https://git.example.com/owner/repo", "ssh") == "git@git.example.com:owner/repo.git"
    assert rewrite_url("https://git.example.com/owner/repo.git", "ssh") == "git@git.example.com:owner/repo.git"
    assert rewrite_url("https://git.example.com/group/sub/repo/", "ssh") == "git@git.example.com:group/sub/repo.git"


def test_rewrite_ssh_to_https():
    assert rewrite_url("git@git.example.com:owner/repo.git", "https") == "https://git.example.com/owner/repo.git"
    assert rewrite_url("ssh://git@git.example.com/owner/repo", "https") == "https://git.example.com/owner/repo.git"


def test_rewrite_leaves_local_paths_alone():
    assert rewrite_url("/root/repos/repo-1.bundle", "ssh") == "/root/repos/repo-1.bundle"
    assert rewrite_url("/root/repos/repo-1.bundle", "https") == "/root/repos/repo-1.bundle"


def test_rewrite_already_target_scheme_unchanged():
    assert rewrite_url("git@h.com:o/r.git", "ssh") == "git@h.com:o/r.git"
    assert rewrite_url("https://h.com/o/r", "https") == "https://h.com/o/r"


def test_rewrite_unknown_scheme_raises():
    with pytest.raises(ValueError):
        rewrite_url("https://h.com/o/r", "carrier-pigeon")


def test_rewrite_does_not_change_folder_name():
    """Output naming must be stable across clone schemes."""
    url = "https://git.example.com/owner/repo"
    assert url_to_folder_name(url) == url_to_folder_name(rewrite_url(url, "ssh"))


# ---------------------------------------------------------------------------
# clone_repo
# ---------------------------------------------------------------------------

def _make_git_repo(path: Path) -> None:
    path.mkdir(parents=True)
    (path / "f.py").write_text("x = 1\n")
    subprocess.run(["git", "init", "--quiet"], cwd=path, check=True)
    subprocess.run(["git", "add", "."], cwd=path, check=True)
    subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "--quiet", "-m", "init"],
        cwd=path, check=True,
    )


def test_clone_repo_sets_noninteractive_env(monkeypatch):
    """GIT_TERMINAL_PROMPT=0 and batch-mode ssh must be passed to the subprocess."""
    captured = {}
    real_exec = asyncio.create_subprocess_exec

    async def fake_exec(*args, **kwargs):
        captured["env"] = kwargs.get("env")
        return await real_exec(*args, **kwargs)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)

    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "src-repo"
        _make_git_repo(src)
        dest = Path(tmp) / "clones" / "dest"
        asyncio.run(clone_repo(str(src), dest))

    assert captured["env"]["GIT_TERMINAL_PROMPT"] == "0"
    assert "BatchMode=yes" in captured["env"]["GIT_SSH_COMMAND"]


def test_clone_repo_clones_local_repo():
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "src-repo"
        _make_git_repo(src)
        dest = Path(tmp) / "clones" / "dest"
        result = asyncio.run(clone_repo(str(src), dest))
        assert result == dest
        assert (dest / "f.py").exists()


def test_clone_repo_fails_fast_without_credentials():
    """A private HTTPS URL with no credentials must raise CloneError, not hang."""
    with tempfile.TemporaryDirectory() as tmp:
        dest = Path(tmp) / "clones" / "dest"
        with pytest.raises(CloneError):
            asyncio.run(clone_repo("https://127.0.0.1:1/owner/repo.git", dest))


def test_same_leaf_name_distinct_clone_dests():
    """Repos sharing a leaf name must get distinct clone destinations."""
    a = url_to_folder_name("https://h.com/team-a/android")
    b = url_to_folder_name("https://h.com/team-b/android")
    assert a != b
