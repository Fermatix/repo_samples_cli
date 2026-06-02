from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

SKIP_DIRS = {
    ".git", ".svn",
    "__pycache__", ".mypy_cache", ".ruff_cache", ".pytest_cache",
    "dist", "build", ".next", ".nuxt", "out",
    "node_modules", "vendor", "venv", ".venv", "env", ".env",
    "generated", "gen", "pb", "proto_gen", "openapi_gen",
    "migrations",
    "fixtures", "testdata", "mocks", "assets", "static",
}

LANG_EXTENSIONS: dict[str, list[str]] = {
    "python":     [".py"],
    "typescript": [".ts", ".tsx"],
    "javascript": [".js", ".jsx"],
    "go":         [".go"],
    "rust":       [".rs"],
    "java":       [".java"],
    "kotlin":     [".kt"],
    "ruby":       [".rb"],
    "cpp":        [".cpp", ".cc", ".cxx", ".hpp", ".h"],
}


@dataclass
class FileEntry:
    path: str
    raw_lines: int


def get_repo_tree(repo_path: Path, language: str, max_files: int) -> list[FileEntry]:
    extensions = set(LANG_EXTENSIONS.get(language, []))
    entries: list[FileEntry] = []

    for f in repo_path.rglob("*"):
        if not f.is_file():
            continue
        parts = set(f.relative_to(repo_path).parts[:-1])
        if parts & SKIP_DIRS:
            continue
        if f.suffix not in extensions:
            continue

        raw_lines = sum(1 for _ in f.open(errors="ignore"))
        if raw_lines < 10:
            continue

        entries.append(FileEntry(
            path=str(f.relative_to(repo_path)),
            raw_lines=raw_lines,
        ))

    if len(entries) > max_files:
        entries.sort(key=lambda e: e.raw_lines, reverse=True)
        entries = entries[:max_files]

    entries.sort(key=lambda e: e.path)
    return entries


def format_tree_for_prompt(files: list) -> str:
    """Format FileInfo list into grouped tree string for LLM prompt."""
    from collections import defaultdict

    groups: dict[str, list] = defaultdict(list)
    for fi in files:
        groups[fi.top_level_dir].append(fi)

    lines: list[str] = []
    for dir_name in sorted(groups):
        lines.append(f"{dir_name}/")
        for fi in sorted(groups[dir_name], key=lambda x: x.path):
            tags = ""
            if fi.is_test:
                tags += "  [TEST]"
            if fi.days_since_modified > 180:
                tags += "  [OLD]"
            # Show full relative path so LLM can return it verbatim
            lines.append(f"  {fi.path} [{fi.loc} LOC]{tags}")
        lines.append("")

    result = "\n".join(lines)
    limit = 100000
    if len(result) > limit:
        truncated = result[:limit]
        last_newline = truncated.rfind("\n")
        truncated = truncated[:last_newline]
        remaining = len(files) - truncated.count("\n")
        result = truncated + f"\n... [{remaining} files not shown]"

    return result
