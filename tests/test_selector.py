import tempfile
from pathlib import Path

import pytest

from repo_sampler.analyzer import FileInfo, RepoInventory
from repo_sampler.config import Settings
from repo_sampler.scorer import RankedFile
from repo_sampler.selector import (
    make_selected_file,
    needs_extraction,
    pick_files,
    VERBATIM_THRESHOLD,
)


def _make_ranked(path: str, loc: int, layer: str = "business") -> RankedFile:
    is_test = layer == "test"
    top = path.split("/")[0] if "/" in path else "."
    return RankedFile(
        path=path,
        loc=loc,
        raw_lines=loc + 10,
        layer=layer,
        is_test=is_test,
        top_level_dir=top,
    )


def _make_inventory(repo_path: Path, files_info: list[tuple[str, int, bool]]) -> RepoInventory:
    fis = []
    for path, loc, is_test in files_info:
        top = path.split("/")[0] if "/" in path else "."
        fis.append(FileInfo(
            path=path, loc=loc, raw_lines=loc + 10,
            is_test=is_test, days_since_modified=10, top_level_dir=top,
        ))
    return RepoInventory(
        repo_path=repo_path, language="python",
        top_dirs=["src", "tests"], lang_share={"python": 1.0},
        test_file_count=sum(1 for _, _, t in files_info if t),
        prod_file_count=sum(1 for _, _, t in files_info if not t),
        has_multiple_services=False, build_system="uv/poetry", files=fis,
    )


def _write_files(repo_path: Path, names: list[tuple[str, int]]) -> None:
    for path, loc in names:
        abs_path = repo_path / path
        abs_path.parent.mkdir(parents=True, exist_ok=True)
        abs_path.write_text("\n".join(f"x_{i} = {i}" for i in range(loc)))


def test_pick_files_total_loc_within_budget():
    settings = Settings(openrouter_api_key="x")
    ranked = [
        _make_ranked("src/a.py", 800),
        _make_ranked("src/b.py", 800),
        _make_ranked("src/c.py", 800),
        _make_ranked("tests/test_a.py", 400, layer="test"),
    ]
    prod_picks, test_picks = pick_files(ranked, settings)
    prod_total = sum(t for _, t in prod_picks)
    test_total = sum(t for _, t in test_picks)

    assert prod_total <= int(settings.target_loc * (1 - settings.test_share_max)) + 1
    assert test_total <= int(settings.target_loc * settings.test_share_max) + 1


def test_pick_files_test_budget_not_exceeded():
    settings = Settings(openrouter_api_key="x")
    ranked = [_make_ranked(f"tests/test_{i}.py", 300, layer="test") for i in range(20)]
    _, test_picks = pick_files(ranked, settings)
    test_total = sum(t for _, t in test_picks)
    assert test_total <= int(settings.target_loc * settings.test_share_max) + 1


def test_make_selected_file_verbatim():
    with tempfile.TemporaryDirectory() as tmp:
        repo = Path(tmp)
        content = "x = 1\ny = 2\nz = 3\n"
        (repo / "src").mkdir()
        (repo / "src" / "app.py").write_text(content)

        rf = _make_ranked("src/app.py", 3)
        sel = make_selected_file(rf, repo, rank=1, start_line=None, end_line=None, language="python")

        assert sel is not None
        assert sel.content == content
        assert sel.is_partial is False
        assert "===" not in sel.content


def test_make_selected_file_chunk():
    with tempfile.TemporaryDirectory() as tmp:
        repo = Path(tmp)
        lines = [f"x_{i} = {i}\n" for i in range(100)]
        (repo / "src").mkdir()
        (repo / "src" / "big.py").write_text("".join(lines))

        rf = _make_ranked("src/big.py", 100)
        sel = make_selected_file(rf, repo, rank=1, start_line=10, end_line=40, language="python")

        assert sel is not None
        assert sel.is_partial is True
        assert "x_9" in sel.content  # line 10 (0-indexed line 9)
        assert "x_39" in sel.content  # line 40
        assert "x_0 " not in sel.content  # before start


def test_needs_extraction_threshold():
    small = _make_ranked("src/tiny.py", VERBATIM_THRESHOLD - 1)
    large = _make_ranked("src/big.py", VERBATIM_THRESHOLD + 1)
    assert needs_extraction(small) is False
    assert needs_extraction(large) is True
