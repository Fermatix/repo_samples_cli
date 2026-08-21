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


def test_rewrite_ssh_with_custom_port():
    assert rewrite_url("https://git.example.com/group/sub/repo.git", "ssh", ssh_port=10022) == \
        "ssh://git@git.example.com:10022/group/sub/repo.git"
    # no port -> scp form
    assert rewrite_url("https://git.example.com/group/sub/repo.git", "ssh") == \
        "git@git.example.com:group/sub/repo.git"


def test_clone_repo_timeout_is_configurable(monkeypatch, tmp_path):
    """Giant repos need more than the old hardcoded 120s; timeout is a param now."""
    class FakeProc:
        pid = -1  # getpgid(-1) fails -> falls back to kill()

        def __init__(self):
            self._killed = asyncio.Event()

        async def communicate(self):
            await self._killed.wait()  # hangs until killed
            return b"", b""

        def kill(self):
            self._killed.set()

    async def fake_exec(*args, **kwargs):
        return FakeProc()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    with pytest.raises(CloneError, match="timeout after 1s"):
        asyncio.run(clone_repo("git@h.com:o/r.git", tmp_path / "clones" / "d", timeout=1))


def test_clone_timeout_setting_default(monkeypatch):
    monkeypatch.delenv("CLONE_TIMEOUT", raising=False)
    from repo_sampler.config import Settings
    assert Settings(_env_file=None).clone_timeout == 900


def test_clone_env_accepts_new_host_keys(monkeypatch):
    monkeypatch.delenv("GIT_SSH_COMMAND", raising=False)
    from repo_sampler.cloner import _clone_env
    cmd = _clone_env()["GIT_SSH_COMMAND"]
    assert "BatchMode=yes" in cmd
    assert "StrictHostKeyChecking=accept-new" in cmd


# ---------------------------------------------------------------------------
# checkout_latest_branch — move off an empty/README-only default branch
# ---------------------------------------------------------------------------

import os  # noqa: E402

from repo_sampler.cloner import checkout_latest_branch  # noqa: E402


def _git(cwd: Path, *args: str, date: str | None = None) -> None:
    env = {**os.environ}
    if date:
        env["GIT_AUTHOR_DATE"] = date
        env["GIT_COMMITTER_DATE"] = date
    subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", *args],
        cwd=cwd, check=True, env=env,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )


def test_checkout_latest_branch_moves_off_readme_only_default():
    """Default branch has only a README (old); the real code lives on a feature
    branch with a newer commit. After clone+checkout we must land on the code."""
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "src-repo"
        src.mkdir(parents=True)
        _git(src, "init", "--quiet", "-b", "main")
        # Default branch: README only, older commit.
        (src / "README.md").write_text("# Just a readme\n")
        _git(src, "add", ".")
        _git(src, "commit", "--quiet", "-m", "readme", date="2020-01-01T00:00:00")
        # Feature branch: real code, newer commit.
        _git(src, "checkout", "--quiet", "-b", "feature")
        (src / "engine.py").write_text("def run():\n    return 42\n")
        _git(src, "add", ".")
        _git(src, "commit", "--quiet", "-m", "code", date="2022-01-01T00:00:00")
        # Leave the default branch checked out, as a real remote would serve it.
        _git(src, "checkout", "--quiet", "main")

        dest = Path(tmp) / "clones" / "dest"

        async def _run():
            await clone_repo(str(src), dest)
            # Fresh clone is on the default (README-only) branch.
            assert (dest / "README.md").exists()
            assert not (dest / "engine.py").exists()
            selected = await checkout_latest_branch(dest)
            return selected

        selected = asyncio.run(_run())
        assert selected == "refs/remotes/origin/feature"
        assert (dest / "engine.py").exists()        # switched to the code branch


def test_checkout_latest_branch_reaches_remote_tracking_branch_in_mirror():
    """Regression: a local *mirror* (made by `git clone`) keeps non-default
    branches as remote-tracking refs (refs/remotes/origin/*), which a plain
    `git clone <mirror>` does NOT copy. The code branch must still be reached."""
    with tempfile.TemporaryDirectory() as tmp:
        upstream = Path(tmp) / "upstream"
        upstream.mkdir(parents=True)
        _git(upstream, "init", "--quiet", "-b", "main")
        (upstream / "README.md").write_text("# readme\n")
        _git(upstream, "add", ".")
        _git(upstream, "commit", "--quiet", "-m", "readme", date="2020-01-01T00:00:00")
        _git(upstream, "checkout", "--quiet", "-b", "feature/code")
        (upstream / "engine.py").write_text("def run():\n    return 42\n")
        _git(upstream, "add", ".")
        _git(upstream, "commit", "--quiet", "-m", "code", date="2022-01-01T00:00:00")
        _git(upstream, "checkout", "--quiet", "main")

        # Mirror = plain clone of upstream: 'main' is a local head, 'feature/code'
        # exists only under refs/remotes/origin/* — exactly like our partner mirrors.
        mirror = Path(tmp) / "mirror"
        subprocess.run(["git", "clone", "--quiet", str(upstream), str(mirror)], check=True)

        dest = Path(tmp) / "clones" / "dest"

        async def _run():
            await clone_repo(str(mirror), dest)
            return await checkout_latest_branch(dest)

        selected = asyncio.run(_run())
        assert selected == "refs/remotes/origin/feature/code"
        assert (dest / "engine.py").exists()   # reached the code branch, not empty main


def test_checkout_latest_branch_single_branch_is_stable():
    """A repo with only the default branch must still resolve and stay valid."""
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "src-repo"
        _make_git_repo(src)
        dest = Path(tmp) / "clones" / "dest"

        async def _run():
            await clone_repo(str(src), dest)
            return await checkout_latest_branch(dest)

        selected = asyncio.run(_run())
        assert selected is not None
        assert (dest / "f.py").exists()
