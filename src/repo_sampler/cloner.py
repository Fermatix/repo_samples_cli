import asyncio
import shutil
from pathlib import Path

from loguru import logger


class CloneError(Exception):
    def __init__(self, url: str, stderr: str) -> None:
        self.url = url
        self.stderr = stderr
        super().__init__(f"Failed to clone {url}: {stderr}")


async def clone_repo(url: str, dest: Path) -> Path:
    if dest.exists():
        logger.debug(f"Cache hit, skipping clone: {dest}")
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)

    proc = await asyncio.create_subprocess_exec(
        "git", "clone", "--depth=1", "--quiet", "--filter=blob:none", url, str(dest),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
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
