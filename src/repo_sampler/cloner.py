import asyncio
import contextlib
import os
import re
import shutil
import signal
from pathlib import Path

from loguru import logger


class CloneError(Exception):
    def __init__(self, url: str, stderr: str) -> None:
        self.url = url
        self.stderr = stderr
        super().__init__(f"Failed to clone {url}: {stderr}")


_HTTPS_URL_RE = re.compile(r"^https?://([^/]+)/(.+?)(\.git)?/?$")
_SSH_URL_RE = re.compile(r"^(?:ssh://)?git@([^:/]+)[:/](.+?)(\.git)?/?$")


def rewrite_url(url: str, scheme: str, ssh_port: int | None = None) -> str:
    """Rewrite a repo URL to the requested clone scheme.

    scheme: "ssh" | "https" | "as-is". Local paths and unrecognized URLs are
    returned unchanged. Only affects cloning — output folder naming must keep
    using the original URL. ssh_port: self-hosted GitLabs often serve SSH on
    a non-standard port; when given, the ssh:// URL form carries it.
    """
    if scheme == "as-is":
        return url
    if scheme == "ssh":
        m = _HTTPS_URL_RE.match(url)
        if m:
            if ssh_port:
                return f"ssh://git@{m.group(1)}:{ssh_port}/{m.group(2)}.git"
            return f"git@{m.group(1)}:{m.group(2)}.git"
        return url
    if scheme == "https":
        m = _SSH_URL_RE.match(url)
        if m:
            return f"https://{m.group(1)}/{m.group(2)}.git"
        return url
    raise ValueError(f"Unknown url scheme: {scheme!r} (expected ssh|https|as-is)")


def _clone_env() -> dict:
    # Never prompt interactively: with parallel workers, credential prompts
    # garble the terminal and hang clones until timeout. Fail fast instead so
    # the error lands in errors.jsonl with a clear message. accept-new keeps
    # first contact with a self-hosted git server from failing on the host-key
    # prompt (still fails loudly if a known host key changes).
    return {
        **os.environ,
        "GIT_TERMINAL_PROMPT": "0",
        "GIT_SSH_COMMAND": os.environ.get(
            "GIT_SSH_COMMAND",
            "ssh -oBatchMode=yes -oStrictHostKeyChecking=accept-new -oConnectTimeout=10",
        ),
    }


def _kill_proc_tree(proc) -> None:
    # git spawns ssh/credential helpers that inherit the stdout/stderr pipes;
    # killing only git leaves communicate() blocked until the child exits on
    # its own (minutes for a dead host). Kill the whole process group.
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except Exception:
        with contextlib.suppress(ProcessLookupError):
            proc.kill()


async def clone_repo(url: str, dest: Path, timeout: int = 900) -> Path:
    if dest.exists():
        logger.debug(f"Cache hit, skipping clone: {dest}")
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)

    proc = await asyncio.create_subprocess_exec(
        # --no-single-branch fetches every branch tip (still shallow at depth 1)
        # so checkout_latest_branch can move off an empty/README-only default
        # branch to wherever the real code lives.
        "git", "clone", "--depth=1", "--no-single-branch", "--quiet", url, str(dest),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=_clone_env(),
        start_new_session=True,
    )
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        _kill_proc_tree(proc)
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(proc.communicate(), timeout=5)
        raise CloneError(url, f"timeout after {timeout}s")

    if proc.returncode != 0:
        raise CloneError(url, stderr.decode(errors="replace"))

    return dest


async def _run_git(args: list[str], cwd: Path, timeout: int = 60) -> tuple[int, str, str]:
    """Run a git command in *cwd*, returning (returncode, stdout, stderr)."""
    proc = await asyncio.create_subprocess_exec(
        "git", "-C", str(cwd), *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=_clone_env(),
        start_new_session=True,
    )
    try:
        out, err = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        _kill_proc_tree(proc)
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(proc.communicate(), timeout=5)
        return 1, "", f"git {args[0]} timed out after {timeout}s"
    return proc.returncode or 0, out.decode(errors="replace"), err.decode(errors="replace")


async def checkout_latest_branch(repo_path: Path, timeout: int = 60) -> str | None:
    """Check out the branch with the most recent commit across all branches.

    The default branch is sometimes empty or README-only; the branch whose tip
    has the newest committer date is almost always where the real code lives.
    This mirrors repo_metadata_cli's ``latest_branch_by_commit`` so the sampled
    tree matches the branch the repo's metadata was computed on.

    Returns the checked-out ref (e.g. ``origin/develop``), or None when no ref
    could be selected or the checkout failed — in which case the caller keeps
    the default checkout untouched.
    """
    rc, out, _ = await _run_git(
        ["for-each-ref", "--sort=-committerdate", "--format=%(refname)",
         "refs/remotes/origin"],
        cwd=repo_path,
        timeout=timeout,
    )
    if rc != 0:
        return None
    ref = next(
        (line.strip() for line in out.splitlines()
         if line.strip() and not line.strip().endswith("/HEAD")),
        None,
    )
    if not ref:
        return None
    rc, _, _ = await _run_git(
        ["checkout", "--force", "--quiet", "--detach", ref],
        cwd=repo_path,
        timeout=timeout,
    )
    return ref if rc == 0 else None


def cleanup_repo(path: Path) -> None:
    try:
        shutil.rmtree(path)
    except Exception:
        pass
