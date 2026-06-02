import tempfile
from pathlib import Path

import pytest

from repo_sampler.analyzer import FileInfo
from repo_sampler.tree import SKIP_DIRS, format_tree_for_prompt, get_repo_tree


def _make_file(path: Path, lines: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(f"line {i}" for i in range(lines)))


def test_get_repo_tree_filters_skip_dirs():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _make_file(root / "src" / "app.py", 20)
        _make_file(root / "__pycache__" / "app.cpython-311.pyc", 20)
        _make_file(root / "node_modules" / "lib.py", 20)

        entries = get_repo_tree(root, "python", 300)
        paths = [e.path for e in entries]

        assert any("app.py" in p for p in paths)
        assert not any("__pycache__" in p for p in paths)
        assert not any("node_modules" in p for p in paths)


def test_get_repo_tree_filters_by_extension():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _make_file(root / "app.py", 20)
        _make_file(root / "app.go", 20)
        _make_file(root / "style.css", 20)

        entries = get_repo_tree(root, "python", 300)
        paths = [e.path for e in entries]

        assert any("app.py" in p for p in paths)
        assert not any("app.go" in p for p in paths)
        assert not any("style.css" in p for p in paths)


def test_get_repo_tree_skips_tiny_files():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _make_file(root / "tiny.py", 5)
        _make_file(root / "normal.py", 20)

        entries = get_repo_tree(root, "python", 300)
        paths = [e.path for e in entries]

        assert not any("tiny.py" in p for p in paths)
        assert any("normal.py" in p for p in paths)


def test_get_repo_tree_truncates_at_max_files():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for i in range(20):
            _make_file(root / f"module_{i}.py", 20 + i)

        entries = get_repo_tree(root, "python", 10)
        assert len(entries) == 10


def test_get_repo_tree_sorted_by_path():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _make_file(root / "z_module.py", 20)
        _make_file(root / "a_module.py", 20)

        entries = get_repo_tree(root, "python", 300)
        paths = [e.path for e in entries]
        assert paths == sorted(paths)


def _make_file_info(path: str, loc: int, is_test: bool = False, days: int = 0) -> FileInfo:
    top = path.split("/")[0] if "/" in path else "."
    return FileInfo(
        path=path,
        loc=loc,
        raw_lines=loc + 10,
        is_test=is_test,
        days_since_modified=days,
        top_level_dir=top,
    )


def test_format_tree_groups_by_directory():
    files = [
        _make_file_info("api/handlers.py", 100),
        _make_file_info("api/router.py", 80),
        _make_file_info("core/engine.py", 200),
    ]
    result = format_tree_for_prompt(files)
    assert "api/" in result
    assert "core/" in result
    assert "api/handlers.py" in result
    assert "core/engine.py" in result


def test_format_tree_test_label():
    files = [
        _make_file_info("tests/test_engine.py", 100, is_test=True),
        _make_file_info("src/engine.py", 200),
    ]
    result = format_tree_for_prompt(files)
    assert "[TEST]" in result


def test_format_tree_old_label():
    files = [
        _make_file_info("src/old.py", 100, days=200),
        _make_file_info("src/new.py", 100, days=10),
    ]
    result = format_tree_for_prompt(files)
    assert "[OLD]" in result
    assert result.count("[OLD]") == 1


def test_format_tree_respects_char_limit():
    # Each entry ~30 chars; need to exceed the 100 000 char limit → use 4000 files
    files = [_make_file_info(f"src/module_{i:05d}.py", 100) for i in range(4000)]
    result = format_tree_for_prompt(files)
    assert len(result) <= 100_100
    assert "not shown" in result
