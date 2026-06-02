from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path

from .tree import LANG_EXTENSIONS, FileEntry

COMMENT_PREFIXES: dict[str, list[str]] = {
    "python":     ["#"],
    "typescript": ["//", "*", "/*", "*/"],
    "javascript": ["//", "*", "/*", "*/"],
    "go":         ["//", "*", "/*", "*/"],
    "rust":       ["//", "///", "*", "/*", "*/"],
    "java":       ["//", "*", "/*", "*/"],
    "kotlin":     ["//", "*", "/*", "*/"],
    "ruby":       ["#"],
    "cpp":        ["//", "*", "/*", "*/"],
}

TEST_PATH_MARKERS = {
    "test_", "_test", "test/", "tests/", "spec/", "specs/",
    "__tests__/", ".test.", ".spec.",
}


@dataclass
class FileInfo:
    path: str
    loc: int
    raw_lines: int
    is_test: bool
    days_since_modified: int
    top_level_dir: str


@dataclass
class RepoInventory:
    repo_path: Path
    language: str
    top_dirs: list[str]
    lang_share: dict[str, float]
    test_file_count: int
    prod_file_count: int
    has_multiple_services: bool
    build_system: str
    files: list[FileInfo] = field(default_factory=list)


def count_loc(file_path: Path, language: str) -> int:
    prefixes = COMMENT_PREFIXES.get(language, ["#", "//"])
    count = 0
    for line in file_path.open(errors="ignore"):
        stripped = line.strip()
        if not stripped:
            continue
        if any(stripped.startswith(p) for p in prefixes):
            continue
        count += 1
    return count


def is_test_file(path: str) -> bool:
    path_lower = path.lower()
    return any(m in path_lower for m in TEST_PATH_MARKERS)


async def get_git_ages(repo_path: Path, paths: list[str]) -> dict[str, int]:
    import time

    now = int(time.time())
    result: dict[str, int] = {p: 0 for p in paths}

    batch_size = 50
    for i in range(0, len(paths), batch_size):
        batch = paths[i:i + batch_size]
        try:
            proc = await asyncio.create_subprocess_exec(
                "git", "-C", str(repo_path), "log", "--format=%H %ad", "--date=unix", "--",
                *batch,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=30)
            lines = stdout.decode(errors="replace").splitlines()

            # git log for multiple files returns interleaved; use per-file approach
            for path in batch:
                proc2 = await asyncio.create_subprocess_exec(
                    "git", "-C", str(repo_path), "log", "-1", "--format=%ad", "--date=unix",
                    "--", path,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                out2, _ = await asyncio.wait_for(proc2.communicate(), timeout=10)
                ts_str = out2.decode().strip()
                if ts_str.isdigit():
                    days = (now - int(ts_str)) // 86400
                    result[path] = days
        except Exception:
            pass

    return result


def detect_build_system(repo_path: Path) -> str:
    checks = [
        ("pyproject.toml", "uv/poetry"),
        ("setup.py", "setuptools"),
        ("Cargo.toml", "cargo"),
        ("go.mod", "go"),
        ("pom.xml", "maven"),
        ("build.gradle", "gradle"),
        ("build.gradle.kts", "gradle"),
    ]
    for filename, name in checks:
        if (repo_path / filename).exists():
            return name
    return "unknown"


def detect_lang_share(repo_path: Path) -> dict[str, float]:
    ext_to_lang: dict[str, str] = {}
    for lang, exts in LANG_EXTENSIONS.items():
        for ext in exts:
            ext_to_lang[ext] = lang

    loc_by_lang: dict[str, int] = {}
    for f in repo_path.rglob("*"):
        if not f.is_file():
            continue
        lang = ext_to_lang.get(f.suffix)
        if not lang:
            continue
        lines = sum(1 for _ in f.open(errors="ignore"))
        loc_by_lang[lang] = loc_by_lang.get(lang, 0) + lines

    total = sum(loc_by_lang.values())
    if total == 0:
        return {}
    return {lang: round(lines / total, 4) for lang, lines in sorted(
        loc_by_lang.items(), key=lambda x: x[1], reverse=True
    )}


async def analyze_repo(
    repo_path: Path, language: str, files: list[FileEntry]
) -> RepoInventory:
    paths = [f.path for f in files]
    git_ages = await get_git_ages(repo_path, paths)

    file_infos: list[FileInfo] = []
    for fe in files:
        abs_path = repo_path / fe.path
        loc = count_loc(abs_path, language)
        top_level_dir = fe.path.split("/")[0] if "/" in fe.path else "."
        file_infos.append(FileInfo(
            path=fe.path,
            loc=loc,
            raw_lines=fe.raw_lines,
            is_test=is_test_file(fe.path),
            days_since_modified=git_ages.get(fe.path, 0),
            top_level_dir=top_level_dir,
        ))

    build_system = detect_build_system(repo_path)
    lang_share = detect_lang_share(repo_path)

    top_dirs_set: list[str] = []
    seen: set[str] = set()
    for fi in file_infos:
        if fi.top_level_dir not in seen:
            seen.add(fi.top_level_dir)
            top_dirs_set.append(fi.top_level_dir)

    top_dirs = top_dirs_set[:10]

    has_multiple_services = _check_monorepo(repo_path)

    test_count = sum(1 for fi in file_infos if fi.is_test)
    prod_count = len(file_infos) - test_count

    return RepoInventory(
        repo_path=repo_path,
        language=language,
        top_dirs=top_dirs,
        lang_share=lang_share,
        test_file_count=test_count,
        prod_file_count=prod_count,
        has_multiple_services=has_multiple_services,
        build_system=build_system,
        files=file_infos,
    )


def _check_monorepo(repo_path: Path) -> bool:
    service_markers = {"pyproject.toml", "package.json", "go.mod"}
    count = 0
    for child in repo_path.iterdir():
        if not child.is_dir():
            continue
        for marker in service_markers:
            if (child / marker).exists():
                count += 1
                break
    return count > 3
