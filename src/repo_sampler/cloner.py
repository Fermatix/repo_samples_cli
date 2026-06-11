import asyncio
import os
import re
import shutil
from pathlib import Path

from loguru import logger


class CloneError(Exception):
    def __init__(self, url: str, stderr: str) -> None:
        self.url = url
        self.stderr = stderr
        super().__init__(f"Failed to clone {url}: {stderr}")


_HTTPS_URL_RE = re.compile(r"^https?://([^/]+)/(.+?)(\.git)?/?$")
_SSH_URL_RE = re.compile(r"^(?:ssh://)?git@([^:/]+)[:/](.+?)(\.git)?/?$")


def rewrite_url(url: str, scheme: str) -> str:
    """Rewrite a repo URL to the requested clone scheme.

    scheme: "ssh" | "https" | "as-is". Local paths and unrecognized URLs are
    returned unchanged. Only affects cloning — output folder naming must keep
    using the original URL.
    """
    if scheme == "as-is":
        return url
    if scheme == "ssh":
        m = _HTTPS_URL_RE.match(url)
        if m:
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
    # the error lands in errors.jsonl with a clear message.
    return {
        **os.environ,
        "GIT_TERMINAL_PROMPT": "0",
        "GIT_SSH_COMMAND": os.environ.get("GIT_SSH_COMMAND", "ssh -oBatchMode=yes"),
    }


async def clone_repo(url: str, dest: Path) -> Path:
    if dest.exists():
        logger.debug(f"Cache hit, skipping clone: {dest}")
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)

    proc = await asyncio.create_subprocess_exec(
        "git", "clone", "--depth=1", "--quiet", url, str(dest),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=_clone_env(),
    )
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=120)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.communicate()
        raise CloneError(url, "timeout after 120s")

    if proc.returncode != 0:
        raise CloneError(url, stderr.decode(errors="replace"))

    return dest


def cleanup_repo(path: Path) -> None:
    try:
        shutil.rmtree(path)
    except Exception:
        pass
