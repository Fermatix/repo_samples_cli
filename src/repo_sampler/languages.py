"""Language detection for sampled repositories.

Language names follow scc's naming exactly ("JavaScript", "C Header",
"Twig Template", "Plain Text", ...) because downstream metadata pipelines
compute `primary_language` with scc and the sample is validated against that
string. When the `scc` binary is available the distribution comes straight
from it (same exclude dirs as the metadata pipeline); otherwise a fast
newline-counting walk approximates the shares.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from loguru import logger

# Same exclusions the metadata pipeline passes to scc, plus .git.
DEFAULT_EXCLUDE_DIRS = frozenset(
    {"node_modules", "vendor", "dist", "build", "bower_components", ".git"}
)

_MAX_FILE_BYTES = 20 * 1024 * 1024  # walk skips files larger than this
_SNIFF_BYTES = 8192                 # null-byte binary sniff window

# Extension -> scc language name. Names verified against scc output in the
# metadata pipeline (lang_distribution values). Multi-suffix patterns like
# ".d.ts" / ".blade.php" are handled before the plain suffix lookup.
EXT_TO_LANG: dict[str, str] = {
    ".py": "Python", ".pyw": "Python",
    ".js": "JavaScript", ".mjs": "JavaScript", ".cjs": "JavaScript",
    ".jsx": "JSX",
    ".ts": "TypeScript", ".tsx": "TypeScript",
    ".vue": "Vue",
    ".svelte": "Svelte",
    ".php": "PHP", ".phtml": "PHP",
    ".html": "HTML", ".htm": "HTML",
    ".css": "CSS",
    ".scss": "Sass", ".sass": "Sass",
    ".less": "LESS",
    ".styl": "Stylus",
    ".xml": "XML",
    ".xsd": "XML Schema",
    ".json": "JSON",
    ".yml": "YAML", ".yaml": "YAML",
    ".sql": "SQL",
    ".md": "Markdown", ".markdown": "Markdown",
    ".mdx": "MDX",
    ".txt": "Plain Text",
    ".csv": "CSV",
    ".svg": "SVG",
    ".mustache": "Mustache",
    ".twig": "Twig Template",
    ".tpl": "Smarty Template",
    ".hbs": "Handlebars", ".handlebars": "Handlebars",
    ".jade": "Jade", ".pug": "Jade",
    ".haml": "HAML",
    ".erb": "Ruby HTML",
    ".razor": "Razor", ".cshtml": "Razor",
    ".c": "C",
    ".h": "C Header",
    ".cpp": "C++", ".cc": "C++", ".cxx": "C++",
    ".hpp": "C++ Header", ".hh": "C++ Header", ".hxx": "C++ Header",
    ".cs": "C#",
    ".java": "Java",
    ".kt": "Kotlin", ".kts": "Kotlin",
    ".go": "Go",
    ".rs": "Rust",
    ".rb": "Ruby",
    ".swift": "Swift",
    ".m": "Objective C",
    ".mm": "Objective C++",
    ".dart": "Dart",
    ".lua": "Lua",
    ".scala": "Scala",
    ".pl": "Perl", ".pm": "Perl",
    ".r": "R",
    ".sh": "Shell",
    ".bash": "BASH",
    ".zsh": "Zsh",
    ".bat": "Batch", ".cmd": "Batch",
    ".ps1": "Powershell",
    ".gradle": "Gradle",
    ".groovy": "Groovy",
    ".toml": "TOML",
    ".ini": "INI",
    ".properties": "Properties File",
    ".proto": "Protocol Buffers",
    ".tf": "Terraform",
    ".hcl": "HCL",
    ".cmake": "CMake",
    ".graphql": "GraphQL", ".gql": "GraphQL",
    ".coffee": "CoffeeScript",
    ".ipynb": "Jupyter",
    ".qml": "QML",
    ".xaml": "XAML",
    ".ex": "Elixir", ".exs": "Elixir",
    ".elm": "Elm",
    ".clj": "Clojure",
    ".fs": "F#",
    ".vb": "Visual Basic",
}

# Patterns checked against the lowercased file NAME before the suffix lookup.
_NAME_ENDSWITH_TO_LANG: list[tuple[str, str]] = [
    (".d.ts", "TypeScript Typings"),
    (".blade.php", "Blade template"),
    (".spec.ts", "TypeScript"),  # keep after .d.ts; plain suffix would match anyway
]

_FILENAME_TO_LANG: dict[str, str] = {
    "dockerfile": "Dockerfile",
    ".dockerignore": "Docker ignore",
    "makefile": "Makefile",
    "cmakelists.txt": "CMake",
    "gemfile": "Gemfile",
    "rakefile": "Rakefile",
    "license": "License",
}

# Comment prefixes for LOC counting of saved chunks. Keyed by scc names;
# lookup is case-insensitive so legacy lowercase callers keep working.
COMMENT_PREFIXES: dict[str, list[str]] = {
    "python": ["#"],
    "ruby": ["#"],
    "shell": ["#"],
    "bash": ["#"],
    "yaml": ["#"],
    "typescript": ["//", "*", "/*", "*/"],
    "typescript typings": ["//", "*", "/*", "*/"],
    "javascript": ["//", "*", "/*", "*/"],
    "jsx": ["//", "*", "/*", "*/"],
    "go": ["//", "*", "/*", "*/"],
    "rust": ["//", "///", "*", "/*", "*/"],
    "java": ["//", "*", "/*", "*/"],
    "kotlin": ["//", "*", "/*", "*/"],
    "c": ["//", "*", "/*", "*/"],
    "c header": ["//", "*", "/*", "*/"],
    "c++": ["//", "*", "/*", "*/"],
    "c++ header": ["//", "*", "/*", "*/"],
    "c#": ["//", "*", "/*", "*/"],
    "cpp": ["//", "*", "/*", "*/"],  # legacy key
    "swift": ["//", "*", "/*", "*/"],
    "objective c": ["//", "*", "/*", "*/"],
    "objective c++": ["//", "*", "/*", "*/"],
    "dart": ["//", "*", "/*", "*/"],
    "scala": ["//", "*", "/*", "*/"],
    "php": ["//", "#", "*", "/*", "*/"],
}

_FALLBACK_PREFIXES = ["#", "//"]


def lang_from_path(path: str | Path) -> str:
    """scc language name for a path, or "" when unknown."""
    name = Path(path).name.lower()
    for ending, lang in _NAME_ENDSWITH_TO_LANG:
        if name.endswith(ending):
            return lang
    if name in _FILENAME_TO_LANG:
        return _FILENAME_TO_LANG[name]
    return EXT_TO_LANG.get(Path(name).suffix, "")


def count_code_lines(lines: list[str], language: str) -> int:
    """Non-blank, non-comment-prefixed lines (for saved-chunk LOC)."""
    prefixes = COMMENT_PREFIXES.get(language.lower(), _FALLBACK_PREFIXES)
    count = 0
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if any(stripped.startswith(p) for p in prefixes):
            continue
        count += 1
    return count


_CANONICAL = {name.lower(): name for name in EXT_TO_LANG.values()}
_CANONICAL.update({lang.lower(): lang for _, lang in _NAME_ENDSWITH_TO_LANG})
_CANONICAL.update({lang.lower(): lang for lang in _FILENAME_TO_LANG.values()})

# Every name lang_from_path() can produce — only these can be tracked per-save
# and therefore enforced as a primary language.
TRACKABLE_LANGS = frozenset(_CANONICAL.values())


def canonicalize(name: str) -> str | None:
    """Map a case-insensitive language name to its scc spelling, or None."""
    return _CANONICAL.get(name.strip().lower())


def is_trackable(name: str) -> bool:
    """True when saved files of this language can be recognized by path."""
    return name in TRACKABLE_LANGS


def extensions_for(language: str) -> list[str]:
    exts = [ext for ext, lang in EXT_TO_LANG.items() if lang == language]
    exts += [end for end, lang in _NAME_ENDSWITH_TO_LANG if lang == language]
    exts += [fn for fn, lang in _FILENAME_TO_LANG.items() if lang == language]
    return sorted(exts)


@dataclass
class LanguageStats:
    counts: dict[str, int] = field(default_factory=dict)  # scc name -> lines
    total: int = 0
    primary: str | None = None
    source: str = "empty"  # "scc" | "walk" | "empty"


def _finalize(counts: dict[str, int], source: str) -> LanguageStats:
    counts = {k: v for k, v in counts.items() if v > 0}
    if not counts:
        return LanguageStats(source="empty")
    primary = max(sorted(counts), key=lambda k: counts[k])
    return LanguageStats(
        counts=counts, total=sum(counts.values()), primary=primary, source=source
    )


def _scc_language_counts(
    repo_path: Path, exclude_dirs: frozenset[str], timeout: int
) -> dict[str, int] | None:
    if not shutil.which("scc"):
        return None
    try:
        # --exclude-dir REPLACES scc's default [.git,.hg,.svn] — keep them so
        # the scc path and the walk fallback see the same tree.
        proc = subprocess.run(
            [
                "scc", "--format", "json", "--no-complexity",
                "--exclude-dir", ",".join(sorted(exclude_dirs | {".hg", ".svn"})),
                str(repo_path),
            ],
            capture_output=True,
            timeout=timeout,
        )
        if proc.returncode != 0:
            return None
        data = json.loads(proc.stdout)
        return {
            entry["Name"]: int(entry.get("Code", 0))
            for entry in data
            if entry.get("Name")
        }
    except Exception as e:
        logger.debug(f"scc language scan failed, falling back to walk: {e}")
        return None


def _walk_language_counts(
    repo_path: Path, exclude_dirs: frozenset[str]
) -> dict[str, int]:
    """Fast share estimate: count raw newlines in bytes, no decoding.

    Exact code-line counts are unnecessary for a distribution — relative
    shares are what matter, and byte-level newline counting is ~5-10x faster
    than decoding + comment parsing on multi-million-LOC repos.
    """
    counts: dict[str, int] = {}
    for root, dirs, files in os.walk(repo_path, topdown=True):
        dirs[:] = [d for d in dirs if d not in exclude_dirs]
        for fname in files:
            lang = lang_from_path(fname)
            if not lang:
                continue
            fpath = os.path.join(root, fname)
            try:
                size = os.path.getsize(fpath)
                if size == 0 or size > _MAX_FILE_BYTES:
                    continue
                lines = 0
                last = b""
                with open(fpath, "rb") as fh:
                    head = fh.read(_SNIFF_BYTES)
                    if b"\0" in head:
                        continue
                    lines += head.count(b"\n")
                    last = head[-1:]
                    while chunk := fh.read(1 << 20):
                        lines += chunk.count(b"\n")
                        last = chunk[-1:]
                if last and last != b"\n":
                    lines += 1  # final line without trailing newline
            except OSError:
                continue
            if lines:
                counts[lang] = counts.get(lang, 0) + lines
    return counts


def compute_language_stats(
    repo_path: Path,
    use_scc: bool = True,
    timeout: int = 120,
    exclude_dirs: frozenset[str] = DEFAULT_EXCLUDE_DIRS,
) -> LanguageStats:
    """Language distribution of a repo, scc-compatible names.

    Prefers the scc binary (exact parity with the metadata pipeline that
    computes primary_language); falls back to a fast newline-counting walk.
    """
    if use_scc:
        counts = _scc_language_counts(repo_path, exclude_dirs, timeout)
        if counts is not None:
            return _finalize(counts, "scc")
    return _finalize(_walk_language_counts(repo_path, exclude_dirs), "walk")


def format_distribution(stats: LanguageStats, top_n: int = 8) -> str:
    """Markdown-ish list of language shares for the agent prompt."""
    if not stats.counts:
        return "(language distribution unavailable)"
    ranked = sorted(stats.counts.items(), key=lambda kv: -kv[1])
    lines = []
    for i, (lang, n) in enumerate(ranked):
        share = n / stats.total
        if i >= top_n and share < 0.01:
            break
        lines.append(f"- {lang}: {share:.1%}")
    return "\n".join(lines)
