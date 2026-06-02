import tempfile
from pathlib import Path

import pytest

from repo_sampler.analyzer import COMMENT_PREFIXES, count_loc, is_test_file
from repo_sampler.selector import count_loc_lines


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def test_count_loc_skips_blank_lines():
    with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
        f.write("x = 1\n\n\ny = 2\n")
        fname = f.name
    assert count_loc(Path(fname), "python") == 2


def test_count_loc_skips_comment_lines_python():
    with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
        f.write("# comment\nx = 1\n# another comment\n")
        fname = f.name
    assert count_loc(Path(fname), "python") == 1


def test_count_loc_skips_comment_lines_go():
    with tempfile.NamedTemporaryFile(suffix=".go", mode="w", delete=False) as f:
        f.write("// comment\nfunc main() {}\n/* block */\n")
        fname = f.name
    assert count_loc(Path(fname), "go") == 1


def test_count_loc_counts_code_lines():
    with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
        f.write("a = 1\nb = 2\nc = 3\n")
        fname = f.name
    assert count_loc(Path(fname), "python") == 3


def test_is_test_file_test_prefix():
    assert is_test_file("tests/test_engine.py") is True
    assert is_test_file("test_utils.py") is True


def test_is_test_file_spec():
    assert is_test_file("spec/user_spec.rb") is True
    assert is_test_file("__tests__/app.test.ts") is True


def test_is_test_file_negative():
    assert is_test_file("src/engine.py") is False
    assert is_test_file("core/processor.go") is False


def test_is_test_file_case_insensitive():
    assert is_test_file("Tests/TestEngine.py") is True


def test_count_loc_lines_matches_count_loc():
    code = "x = 1\n# comment\n\ny = 2\n"
    with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
        f.write(code)
        fname = f.name

    file_loc = count_loc(Path(fname), "python")
    lines = code.splitlines(keepends=True)
    lines_loc = count_loc_lines(lines, "python")

    assert file_loc == lines_loc
