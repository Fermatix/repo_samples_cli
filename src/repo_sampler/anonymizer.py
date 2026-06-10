from __future__ import annotations

import asyncio
import json
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from loguru import logger

from .config import Settings

# Directories under output_dir that are never sample dirs.
SKIP_DIRS = {"archive"}

# Files inside a sample dir that must not be touched / not counted in the diff.
_DIFF_EXCLUDES = ["agent_log.json", "anonymization.*", ".*"]


ANONYMIZE_PROMPT = """\
You are anonymizing a code sample deliverable in the current directory. Process EVERY file under \
`samples/` (all source code) AND `repo_summary.md`. Do NOT modify or create any other files \
(`agent_log.json`, `anonymization.diff`, `anonymization_report.json`, dotfiles).

In every processed file, find and replace ONLY identifying content of these kinds:
- Company / organisation / legal-entity names (ООО, ОАО, ЗАО, LLC, Inc, GmbH, etc.) — including ones written in Cyrillic
- Brand / product / service / application names
- Personal names of individuals, usernames, handles, author signatures
- Emails, phone numbers, postal / physical addresses
- Websites, domains, URLs, social-media links tied to a specific company or person
- API keys, tokens, passwords, secrets, account IDs

Cyrillic / non-English text is FINE and must be LEFT AS-IS — do NOT translate or rewrite comments or \
strings just because they are in another language. Only neutralise the specific company / personal \
identifiers above, wherever they appear (including inside Russian text).

Replace with neutral, generic, syntactically valid placeholders, consistent within each file:
company → `Acme`, product → `AppName`, person → `John Doe`, email → `user@example.com`, \
domain → `example.com`, phone → `+1-555-0100`, secret → `REDACTED`, etc.

CRITICAL — this is a code-quality sample:
- Preserve code correctness, structure, control flow, formatting, indentation, language, and coding \
style EXACTLY. Only swap identifying tokens; never rewrite, translate, or "improve" anything else.
- Keep each file syntactically valid.
- Edit files IN PLACE. Do not rename, move, add, or delete files.
- Do NOT add headers, banners, comments, or "anonymized" markers.
- Be exhaustive — open and check every file; do not skip any.

When finished, report a one-line summary of how many files you changed.
"""


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def discover_sample_dirs(output_dir: Path, skip: set[str] | None = None) -> list[Path]:
    """Direct children of output_dir that look like sample deliverables."""
    skip = skip if skip is not None else SKIP_DIRS
    dirs: list[Path] = []
    if not output_dir.exists():
        return dirs
    for child in sorted(output_dir.iterdir()):
        if not child.is_dir():
            continue
        if child.name in skip:
            continue
        if (child / "repo_summary.md").exists():
            dirs.append(child)
    return dirs


def is_anonymized(sample_dir: Path) -> bool:
    return (sample_dir / "anonymization_report.json").exists()


# ---------------------------------------------------------------------------
# Snapshot + diff (deterministic audit, no LLM)
# ---------------------------------------------------------------------------

def snapshot_targets(sample_dir: Path) -> Path:
    """Copy the to-be-anonymized files into a fresh temp dir; return its path."""
    snap = Path(tempfile.mkdtemp(prefix="anon-snap-"))
    samples = sample_dir / "samples"
    if samples.is_dir():
        shutil.copytree(samples, snap / "samples")
    summary = sample_dir / "repo_summary.md"
    if summary.exists():
        shutil.copy2(summary, snap / "repo_summary.md")
    return snap


def _parse_unified_diff(diff_text: str) -> list[dict]:
    """Parse `diff -ruN` output into per-file {path, hunks, additions, deletions}."""
    stats: list[dict] = []
    current: dict | None = None
    for line in diff_text.splitlines():
        if line.startswith("+++ "):
            # "+++ <sample_dir>/<relpath>\t<timestamp>" — take the path token, strip dir prefix later
            raw = line[4:].split("\t", 1)[0].strip()
            current = {"path": raw, "hunks": 0, "additions": 0, "deletions": 0}
            stats.append(current)
        elif current is None:
            continue
        elif line.startswith("@@"):
            current["hunks"] += 1
        elif line.startswith("+") and not line.startswith("+++"):
            current["additions"] += 1
        elif line.startswith("-") and not line.startswith("---"):
            current["deletions"] += 1
    return stats


def _normalize_paths(stats: list[dict], sample_dir: Path) -> list[dict]:
    """Trim the diff's right-side path down to a path relative to the sample dir."""
    prefix = str(sample_dir).rstrip("/") + "/"
    out: list[dict] = []
    for s in stats:
        p = s["path"]
        if p.startswith(prefix):
            p = p[len(prefix):]
        out.append({**s, "path": p})
    return out


async def compute_diff(snapshot_dir: Path, sample_dir: Path) -> tuple[str, list[dict]]:
    """Run `diff -ruN` between the snapshot and the (now edited) sample dir."""
    args = ["diff", "-ruN"]
    for pat in _DIFF_EXCLUDES:
        args += ["--exclude", pat]
    args += [str(snapshot_dir), str(sample_dir)]

    proc = await asyncio.create_subprocess_exec(
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()
    # diff exit codes: 0 = identical, 1 = differences (normal), >=2 = error
    if proc.returncode is not None and proc.returncode >= 2:
        logger.warning(f"diff error for {sample_dir.name}: {stderr.decode(errors='replace')[:200]}")
    diff_text = stdout.decode(errors="replace")
    stats = _normalize_paths(_parse_unified_diff(diff_text), sample_dir)
    return diff_text, stats


# ---------------------------------------------------------------------------
# Claude subprocess
# ---------------------------------------------------------------------------

def _build_claude_argv(settings: Settings) -> list[str]:
    argv = [
        "claude", "-p", ANONYMIZE_PROMPT,
        "--model", settings.anonymizer_model,
        "--permission-mode", settings.anonymizer_permission_mode,
        "--tools", "Read,Edit,Write,Glob,Grep",
        "--output-format", "json",
        "--no-session-persistence",
    ]
    if settings.anonymizer_effort:
        argv += ["--effort", settings.anonymizer_effort]
    if settings.anonymizer_max_budget_usd and settings.anonymizer_max_budget_usd > 0:
        argv += ["--max-budget-usd", str(settings.anonymizer_max_budget_usd)]
    return argv


def _extract_cost(stdout: str) -> float | None:
    try:
        data = json.loads(stdout)
    except (json.JSONDecodeError, ValueError):
        return None
    if isinstance(data, dict):
        for key in ("total_cost_usd", "cost_usd"):
            if isinstance(data.get(key), (int, float)):
                return float(data[key])
    return None


async def anonymize_dir(
    sample_dir: Path,
    settings: Settings,
    sem: asyncio.Semaphore,
) -> dict:
    folder = sample_dir.name
    async with sem:
        snap = snapshot_targets(sample_dir)
        try:
            logger.info(f"[{folder}] anonymizing ({settings.anonymizer_model})...")
            argv = _build_claude_argv(settings)
            proc = await asyncio.create_subprocess_exec(
                *argv,
                cwd=str(sample_dir),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                stdout_b, stderr_b = await asyncio.wait_for(
                    proc.communicate(), timeout=settings.anonymizer_timeout
                )
            except asyncio.TimeoutError:
                try:
                    proc.kill()
                    await proc.communicate()
                except Exception:
                    pass
                logger.error(f"[{folder}] timed out after {settings.anonymizer_timeout}s")
                return {"folder": folder, "status": "error", "error": "timeout"}

            if proc.returncode != 0:
                err = stderr_b.decode(errors="replace")[:300]
                logger.error(f"[{folder}] claude exited {proc.returncode}: {err}")
                return {"folder": folder, "status": "error", "error": err or "non-zero exit"}

            stdout = stdout_b.decode(errors="replace")
            cost = _extract_cost(stdout)

            diff_text, stats = await compute_diff(snap, sample_dir)
            _write_artifacts(sample_dir, diff_text, stats, settings.anonymizer_model, cost)

            files_changed = len(stats)
            logger.info(
                f"[{folder}] done: {files_changed} files changed"
                f"{f', cost ${cost:.4f}' if cost is not None else ''}"
            )
            return {
                "folder": folder,
                "status": "ok",
                "cost": cost,
                "files_changed": files_changed,
            }

        except Exception as e:
            logger.error(f"[{folder}] anonymization failed: {e}")
            return {"folder": folder, "status": "error", "error": str(e)}
        finally:
            shutil.rmtree(snap, ignore_errors=True)


def _write_artifacts(
    sample_dir: Path,
    diff_text: str,
    stats: list[dict],
    model: str,
    cost: float | None,
) -> None:
    (sample_dir / "anonymization.diff").write_text(diff_text, encoding="utf-8")

    report = {
        "anonymized_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "model": model,
        "cost_usd": cost,
        "files_changed": len(stats),
        "total_additions": sum(s["additions"] for s in stats),
        "total_deletions": sum(s["deletions"] for s in stats),
        "changes": stats,
    }
    (sample_dir / "anonymization_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

async def run_anonymizer(
    output_dir: Path,
    settings: Settings,
    force: bool = False,
) -> list[dict]:
    dirs = discover_sample_dirs(output_dir)

    if not force:
        original = len(dirs)
        dirs = [d for d in dirs if not is_anonymized(d)]
        skipped = original - len(dirs)
        if skipped:
            logger.info(f"Skipping {skipped} already anonymized dirs (use --force to redo)")

    logger.info(
        f"Anonymizing {len(dirs)} dirs | "
        f"workers={settings.anonymizer_workers} | model={settings.anonymizer_model}"
    )

    sem = asyncio.Semaphore(settings.anonymizer_workers)
    tasks = [anonymize_dir(d, settings, sem) for d in dirs]
    results = await asyncio.gather(*tasks, return_exceptions=False)
    return list(results)
