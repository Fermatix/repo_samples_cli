from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from loguru import logger

from .analyzer import COMMENT_PREFIXES, RepoInventory
from .config import Settings
from .scorer import RankedFile

# Files below this LOC threshold are taken verbatim (no extraction LLM call needed)
VERBATIM_THRESHOLD = 150


@dataclass
class SelectedFile:
    path: str
    rank: int           # 1 = most important
    layer: str
    loc_taken: int
    is_partial: bool    # True if only a chunk was extracted
    content: str


@dataclass
class RepoSample:
    repo_url: str
    repo_name: str
    language: str
    files: list[SelectedFile] = field(default_factory=list)
    total_loc: int = 0
    inventory: RepoInventory = None


def count_loc_lines(lines: list[str], language: str) -> int:
    prefixes = COMMENT_PREFIXES.get(language, ["#", "//"])
    count = 0
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if any(stripped.startswith(p) for p in prefixes):
            continue
        count += 1
    return count


def pick_files(
    ranked: list[RankedFile],
    settings: Settings,
) -> tuple[list[tuple[RankedFile, int]], list[tuple[RankedFile, int]]]:
    """Return (prod_picks, test_picks) each as [(RankedFile, target_loc), ...].

    target_loc is the LOC budget allocated to each file for the extraction step.
    """
    test_budget = int(settings.target_loc * settings.test_share_max)
    prod_budget = settings.target_loc - test_budget

    prod_ranked = [f for f in ranked if not f.is_test]
    test_ranked = [f for f in ranked if f.is_test]

    prod_picks = _allocate(prod_ranked, prod_budget, settings)
    test_picks = _allocate(test_ranked, test_budget, settings)
    return prod_picks, test_picks


def _allocate(
    files: list[RankedFile],
    budget: int,
    settings: Settings,
) -> list[tuple[RankedFile, int]]:
    """Greedy: walk ranked list, allocate min(file.loc, remaining) to each file."""
    picks: list[tuple[RankedFile, int]] = []
    remaining = budget
    for rf in files:
        if remaining < settings.partial_extract_min_loc:
            break
        target = min(rf.loc, remaining)
        if target < settings.partial_extract_min_loc:
            continue
        picks.append((rf, target))
        remaining -= target
    return picks


def make_selected_file(
    rf: RankedFile,
    repo_path: Path,
    rank: int,
    start_line: int | None,
    end_line: int | None,
    language: str,
) -> SelectedFile | None:
    """Read file content and produce a SelectedFile.

    If start_line/end_line are provided, extract that range; otherwise take verbatim.
    """
    abs_path = repo_path / rf.path
    if not abs_path.exists():
        logger.warning(f"File not found: {rf.path}")
        return None

    full_lines = abs_path.read_text(errors="ignore").splitlines(keepends=True)

    if start_line is not None and end_line is not None:
        chunk = full_lines[start_line - 1: end_line]
        content = "".join(chunk)
        loc_taken = count_loc_lines(chunk, language)
        is_partial = (start_line > 1 or end_line < len(full_lines))
    else:
        content = "".join(full_lines)
        loc_taken = rf.loc
        is_partial = False

    return SelectedFile(
        path=rf.path,
        rank=rank,
        layer=rf.layer,
        loc_taken=loc_taken,
        is_partial=is_partial,
        content=content,
    )


def needs_extraction(rf: RankedFile) -> bool:
    return rf.loc > VERBATIM_THRESHOLD
