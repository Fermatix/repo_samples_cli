import json
import subprocess
from pathlib import Path

import pytest

from repo_sampler import languages
from repo_sampler.languages import (
    LanguageStats,
    canonicalize,
    compute_language_stats,
    count_code_lines,
    extensions_for,
    format_distribution,
    lang_from_path,
)


# ---------------------------------------------------------------------------
# Extension -> scc name mapping (names must match the metadata pipeline)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("path,lang", [
    ("a.py", "Python"),
    ("a.js", "JavaScript"),
    ("a.jsx", "JSX"),                      # scc counts JSX separately
    ("a.ts", "TypeScript"),
    ("a.d.ts", "TypeScript Typings"),      # before plain .ts lookup
    ("a.vue", "Vue"),
    ("a.twig", "Twig Template"),
    ("a.mustache", "Mustache"),
    ("a.tpl", "Smarty Template"),
    ("view.blade.php", "Blade template"),  # before plain .php lookup
    ("a.php", "PHP"),
    ("a.cpp", "C++"),
    ("a.h", "C Header"),                   # scc parity: NOT C++
    ("a.hpp", "C++ Header"),
    ("a.txt", "Plain Text"),
    ("a.scss", "Sass"),
    ("a.sql", "SQL"),
    ("a.xml", "XML"),
    ("a.json", "JSON"),
    ("a.m", "Objective C"),
    ("Dockerfile", "Dockerfile"),
    ("Makefile", "Makefile"),
    ("a.unknown-ext", ""),
    ("noext", ""),
])
def test_lang_from_path(path, lang):
    assert lang_from_path(path) == lang


def test_lang_from_path_case_insensitive():
    assert lang_from_path("A.PY") == "Python"
    assert lang_from_path("DOCKERFILE") == "Dockerfile"


def test_canonicalize():
    assert canonicalize("javascript") == "JavaScript"
    assert canonicalize("  c++ ") == "C++"
    assert canonicalize("twig template") == "Twig Template"
    assert canonicalize("Klingon") is None


def test_extensions_for():
    assert ".cpp" in extensions_for("C++")
    assert ".h" not in extensions_for("C++")
    assert ".vue" in extensions_for("Vue")


def test_count_code_lines_case_insensitive_and_fallback():
    lines = ["x = 1\n", "# comment\n", "\n", "y = 2\n"]
    assert count_code_lines(lines, "Python") == 2
    assert count_code_lines(lines, "python") == 2     # legacy lowercase callers
    js = ["// c\n", "var x;\n"]
    assert count_code_lines(js, "SomethingUnknown") == 1  # fallback ["#", "//"]


# ---------------------------------------------------------------------------
# Walk-based distribution
# ---------------------------------------------------------------------------

def _make_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    (repo / "src").mkdir(parents=True)
    (repo / "src" / "app.php").write_text("<?php\n" + "echo 1;\n" * 50)
    (repo / "src" / "main.js").write_text("var x;\n" * 10)
    (repo / "node_modules" / "lib").mkdir(parents=True)
    (repo / "node_modules" / "lib" / "huge.js").write_text("x;\n" * 100000)
    (repo / ".git").mkdir()
    (repo / ".git" / "config.xml").write_text("<x/>\n" * 500)
    (repo / "logo.png").write_bytes(b"\x89PNG\x00\x00binary")
    (repo / "data.bin").write_text("not detected\n")  # unknown ext
    return repo


def test_walk_counts_and_exclusions(tmp_path):
    stats = compute_language_stats(_make_repo(tmp_path), use_scc=False)
    assert stats.source == "walk"
    assert stats.primary == "PHP"
    assert stats.counts["PHP"] == 51        # newline count incl. <?php line
    assert stats.counts["JavaScript"] == 10  # node_modules pruned
    assert "XML" not in stats.counts         # .git pruned
    assert stats.total == 61


def test_walk_skips_binary(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "img.json").write_bytes(b'{"x": 1}\n\x00\x00garbage\n' * 3)
    stats = compute_language_stats(repo, use_scc=False)
    assert stats.counts == {}
    assert stats.primary is None
    assert stats.source == "empty"


def test_empty_repo(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    stats = compute_language_stats(repo, use_scc=False)
    assert stats.primary is None
    assert stats.source == "empty"


def test_primary_tie_break_alphabetical(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "a.py").write_text("x\n" * 5)
    (repo / "b.go").write_text("y\n" * 5)
    stats = compute_language_stats(repo, use_scc=False)
    assert stats.primary == "Go"  # Go < Python alphabetically


# ---------------------------------------------------------------------------
# scc preference and fallback
# ---------------------------------------------------------------------------

def test_scc_preferred_when_available(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "a.py").write_text("x\n")

    monkeypatch.setattr(languages.shutil, "which", lambda _: "/usr/bin/scc")

    def fake_run(argv, capture_output, timeout):
        class P:
            returncode = 0
            stdout = json.dumps([
                {"Name": "JavaScript", "Code": 900},
                {"Name": "PHP", "Code": 100},
            ]).encode()
        return P()

    monkeypatch.setattr(languages.subprocess, "run", fake_run)
    stats = compute_language_stats(repo, use_scc=True)
    assert stats.source == "scc"
    assert stats.primary == "JavaScript"
    assert stats.counts == {"JavaScript": 900, "PHP": 100}


def test_scc_failure_falls_back_to_walk(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "a.py").write_text("x\n" * 3)

    monkeypatch.setattr(languages.shutil, "which", lambda _: "/usr/bin/scc")

    def boom(*args, **kwargs):
        raise subprocess.TimeoutExpired("scc", 1)

    monkeypatch.setattr(languages.subprocess, "run", boom)
    stats = compute_language_stats(repo, use_scc=True)
    assert stats.source == "walk"
    assert stats.primary == "Python"


def test_format_distribution():
    stats = LanguageStats(
        counts={"JavaScript": 600, "PHP": 350, "CSS": 50},
        total=1000, primary="JavaScript", source="walk",
    )
    text = format_distribution(stats)
    assert text.splitlines()[0] == "- JavaScript: 60.0%"
    assert "- PHP: 35.0%" in text
    assert format_distribution(LanguageStats()) == "(language distribution unavailable)"


def test_walk_counts_final_line_without_trailing_newline(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "one.py").write_text("x = 1")            # 1 line, no trailing \n
    (repo / "two.py").write_text("a\nb")             # 2 lines, no trailing \n
    stats = compute_language_stats(repo, use_scc=False)
    assert stats.counts == {"Python": 3}


def test_is_trackable():
    from repo_sampler.languages import is_trackable
    assert is_trackable("JavaScript")
    assert is_trackable("Blade template")
    assert not is_trackable("Solidity")
    assert not is_trackable("")


def test_pick_code_primary_skips_markup():
    from repo_sampler.languages import pick_code_primary
    stats = LanguageStats(
        counts={"CSS": 600, "SVG": 200, "PHP": 150, "JavaScript": 50},
        total=1000, primary="CSS", source="walk",
    )
    assert pick_code_primary(stats) == "PHP"


def test_pick_code_primary_skips_untrackable_and_data():
    from repo_sampler.languages import pick_code_primary
    stats = LanguageStats(
        counts={"JSON": 500, "Solidity": 300, "Markdown": 150, "Go": 50},
        total=1000, primary="JSON", source="scc",
    )
    assert pick_code_primary(stats) == "Go"


def test_pick_code_primary_none_when_no_code():
    from repo_sampler.languages import pick_code_primary
    stats = LanguageStats(
        counts={"CSS": 600, "HTML": 400}, total=1000, primary="CSS", source="walk",
    )
    assert pick_code_primary(stats) is None


def test_pick_code_primary_tie_break_and_floor():
    from repo_sampler.languages import pick_code_primary
    # tie -> alphabetical
    stats = LanguageStats(counts={"PHP": 300, "Go": 300, "CSS": 400},
                          total=1000, primary="CSS", source="walk")
    assert pick_code_primary(stats) == "Go"
    # below the 5% floor -> not a focus candidate
    stats = LanguageStats(counts={"JSON": 970, "Go": 30},
                          total=1000, primary="JSON", source="walk")
    assert pick_code_primary(stats) is None


def test_logicless_templates_are_not_code():
    from repo_sampler.languages import is_code_language
    assert not is_code_language("Mustache")
    assert not is_code_language("Handlebars")
    assert not is_code_language("Jupyter")
    assert is_code_language("Twig Template")
    assert is_code_language("Blade template")
    assert is_code_language("SQL")
