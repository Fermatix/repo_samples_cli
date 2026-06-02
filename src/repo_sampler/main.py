from __future__ import annotations

import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import httpx
import typer
from loguru import logger
from rich.console import Console
from rich.table import Table

from .analyzer import analyze_repo
from .cloner import CloneError, cleanup_repo, clone_repo
from .config import Settings
from .scorer import AuthError, extract_chunk, rank_tree
from .selector import (
    RepoSample,
    SelectedFile,
    make_selected_file,
    needs_extraction,
    pick_files,
)
from .summarizer import _get_commit_sha, generate_summary
from .tree import format_tree_for_prompt, get_repo_tree
from .writer import append_jsonl, write_deliverable, write_parquet

app = typer.Typer(help="CLI tool for extracting representative code samples from git repositories.")
console = Console()

def _write_ranking_debug(
    output_dir: Path,
    repo_name: str,
    ranked,
    sample: "RepoSample",
) -> None:
    """Write per-repo structured debug log: full ranked list + selection details."""
    selected_map = {sf.path: sf for sf in sample.files}

    ranked_entries = []
    for i, rf in enumerate(ranked, start=1):
        sel = selected_map.get(rf.path)
        entry: dict = {
            "rank": i,
            "path": rf.path,
            "layer": rf.layer,
            "loc": rf.loc,
            "is_test": rf.is_test,
        }
        if sel is not None:
            entry["selected"] = True
            entry["loc_taken"] = sel.loc_taken
            entry["is_partial"] = sel.is_partial
        else:
            entry["selected"] = False
        ranked_entries.append(entry)

    selected_entries = [
        {
            "rank": sf.rank,
            "path": sf.path,
            "layer": sf.layer,
            "loc_taken": sf.loc_taken,
            "is_partial": sf.is_partial,
        }
        for sf in sorted(sample.files, key=lambda f: f.rank)
    ]

    test_loc = sum(sf.loc_taken for sf in sample.files if sf.layer == "test")
    layer_breakdown: dict[str, int] = {}
    for sf in sample.files:
        layer_breakdown[sf.layer] = layer_breakdown.get(sf.layer, 0) + sf.loc_taken

    debug = {
        "repo_name": repo_name,
        "sampled_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "summary": {
            "files_in_inventory": len(ranked),
            "files_selected": len(sample.files),
            "total_loc": sample.total_loc,
            "test_share_pct": round(test_loc / sample.total_loc * 100, 1) if sample.total_loc else 0,
            "layer_breakdown_loc": layer_breakdown,
        },
        "selected_files": selected_entries,
        "full_ranking": ranked_entries,
    }

    dest = output_dir / repo_name / "ranking_debug.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(debug, ensure_ascii=False, indent=2), encoding="utf-8")


def _setup_logging(output_dir: Path | None = None) -> None:
    logger.remove()
    logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {level:<8} | {message}")
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.add(
            output_dir / "run.log",
            level="DEBUG",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {message}",
            encoding="utf-8",
            rotation="100 MB",
        )


def _load_repos(repos_file: Path) -> list[str]:
    return [
        line.strip()
        for line in repos_file.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]


def _load_processed(jsonl_path: Path) -> set[str]:
    if not jsonl_path.exists():
        return set()
    processed: set[str] = set()
    try:
        import jsonlines
        with jsonlines.open(jsonl_path) as reader:
            for record in reader:
                url = record.get("repo_url", "")
                if url:
                    processed.add(url)
    except Exception:
        pass
    return processed


async def _build_sample(
    ranked,
    repo_path: Path,
    settings: Settings,
    inventory,
    client: httpx.AsyncClient,
    repo_url: str,
    llm_sem: asyncio.Semaphore,
) -> RepoSample:
    repo_name = repo_url.rstrip("/").split("/")[-1] if repo_url else repo_path.name
    prod_picks, test_picks = pick_files(ranked, settings)
    all_picks = prod_picks + test_picks

    async def _process(rf, target_loc: int, rank: int) -> SelectedFile | None:
        start_line = end_line = None
        if needs_extraction(rf):
            content_full = (repo_path / rf.path).read_text(errors="ignore")
            async with llm_sem:
                result = await extract_chunk(
                    path=rf.path,
                    content=content_full,
                    language=inventory.language,
                    target_loc=target_loc,
                    settings=settings,
                    client=client,
                )
            if result:
                start_line, end_line = result
        return make_selected_file(rf, repo_path, rank, start_line, end_line, inventory.language)

    tasks = [
        _process(rf, target_loc, rank + 1)
        for rank, (rf, target_loc) in enumerate(all_picks)
    ]
    results = await asyncio.gather(*tasks)

    selected = [s for s in results if s is not None]
    total_loc = sum(s.loc_taken for s in selected)

    if total_loc < (settings.target_loc - settings.loc_tolerance):
        logger.warning(f"LOC budget not reached: {total_loc} < {settings.target_loc - settings.loc_tolerance}")

    test_loc = sum(s.loc_taken for s in selected if s.layer == "test")
    test_pct = test_loc / total_loc * 100 if total_loc else 0
    logger.info(f"Selected: {len(selected)} files, {total_loc} LOC, test share: {test_pct:.0f}%")

    return RepoSample(
        repo_url=repo_url,
        repo_name=repo_name,
        language=inventory.language,
        files=selected,
        total_loc=total_loc,
        inventory=inventory,
    )


async def _process_repo(
    url: str,
    language: str,
    output_dir: Path,
    settings: Settings,
    client: httpx.AsyncClient,
    keep_clones: bool,
    dry_run: bool,
    clone_sem: asyncio.Semaphore,
    llm_sem: asyncio.Semaphore,
    errors_path: Path,
) -> dict | None:
    repo_name = url.rstrip("/").split("/")[-1]
    clone_dest = Path(settings.clone_dir) / repo_name

    async with clone_sem:
        stage = "clone"
        try:
            logger.info(f"[{repo_name}] cloning...")
            await clone_repo(url, clone_dest)

            stage = "analyze"
            logger.info(f"[{repo_name}] analyzing...")
            entries = get_repo_tree(clone_dest, language, settings.max_repo_files)
            inventory = await analyze_repo(clone_dest, language, entries)

            if dry_run:
                logger.info(f"[{repo_name}] dry-run: found {len(entries)} files")
                return {"repo_url": url, "repo_name": repo_name, "dry_run": True,
                        "files_found": len(entries)}

            stage = "rank"
            logger.info(f"[{repo_name}] ranking {len(entries)} files via LLM...")
            ranked = await rank_tree(inventory, settings, client)

            stage = "extract"
            logger.info(f"[{repo_name}] extracting chunks...")
            sample = await _build_sample(
                ranked, clone_dest, settings, inventory, client, url, llm_sem
            )

            _write_ranking_debug(output_dir, repo_name, ranked, sample)

            stage = "summarize"
            summary_md = await generate_summary(sample, settings, client)

            stage = "write"
            write_deliverable(sample, summary_md, output_dir)

            commit_sha = await _get_commit_sha(clone_dest)
            jsonl_path = output_dir / "samples.jsonl"
            append_jsonl(
                sample, summary_md, jsonl_path,
                model=settings.openrouter_model,
                files_found=len(entries),
                files_scored=len(ranked),
                commit_sha=commit_sha,
            )

            test_loc = sum(sf.loc_taken for sf in sample.files if sf.layer == "test")
            test_share = test_loc / sample.total_loc if sample.total_loc else 0.0

            logger.info(f"[{repo_name}] done: {sample.total_loc} LOC, {len(sample.files)} files")

            return {
                "repo_url": url,
                "repo_name": repo_name,
                "total_loc": sample.total_loc,
                "file_count": len(sample.files),
                "test_share": test_share,
            }

        except AuthError as e:
            logger.critical(f"Auth error: {e}")
            _write_error(errors_path, url, stage, str(e))
            raise typer.Exit(1)

        except CloneError as e:
            logger.error(f"[{repo_name}] clone failed: {e}")
            _write_error(errors_path, url, stage, str(e))
            return {"repo_url": url, "repo_name": repo_name, "error": str(e)}

        except Exception as e:
            logger.error(f"[{repo_name}] failed at {stage}: {e}")
            _write_error(errors_path, url, stage, str(e))
            return {"repo_url": url, "repo_name": repo_name, "error": str(e)}

        finally:
            if not keep_clones and clone_dest.exists():
                cleanup_repo(clone_dest)


def _write_error(path: Path, url: str, stage: str, error: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    import jsonlines
    with jsonlines.open(path, mode="a") as writer:
        writer.write({
            "repo_url": url,
            "stage": stage,
            "error": error,
            "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        })


@app.command()
def run(
    repos_file: Path = typer.Argument(..., help="File with repo URLs, one per line"),
    language: str = typer.Option("python", help="Language: python|typescript|go|..."),
    output: Path = typer.Option(Path("./output"), help="Output directory"),
    format: str = typer.Option("jsonl", help="jsonl|parquet"),
    workers: Optional[int] = typer.Option(None, help="Parallel clone workers"),
    llm_workers: Optional[int] = typer.Option(None, help="Parallel LLM extraction workers"),
    resume: bool = typer.Option(False, help="Skip already processed repos"),
    dry_run: bool = typer.Option(False, help="Clone + tree + analyze, no LLM"),
    keep_clones: bool = typer.Option(False, help="Keep clones after processing"),
) -> None:
    """Process a list of repositories and extract representative code samples."""
    output.mkdir(parents=True, exist_ok=True)
    _setup_logging(output)
    settings = Settings()

    if workers:
        settings.clone_workers = workers
    if llm_workers:
        settings.llm_workers = llm_workers

    repos = _load_repos(repos_file)

    jsonl_path = output / "samples.jsonl"
    errors_path = output / "errors.jsonl"

    if resume:
        processed = _load_processed(jsonl_path)
        original_count = len(repos)
        repos = [r for r in repos if r not in processed]
        logger.info(f"Resume: skipping {original_count - len(repos)} already processed repos")

    logger.info(f"Processing {len(repos)} repos with {settings.clone_workers} clone workers, "
                f"{settings.llm_workers} LLM workers")

    results = asyncio.run(
        _run_all(repos, language, output, settings, keep_clones, dry_run, errors_path)
    )

    if format == "parquet":
        write_parquet(output)

    _print_summary_table(results)


async def _run_all(
    repos: list[str],
    language: str,
    output_dir: Path,
    settings: Settings,
    keep_clones: bool,
    dry_run: bool,
    errors_path: Path,
) -> list[dict]:
    clone_sem = asyncio.Semaphore(settings.clone_workers)
    llm_sem = asyncio.Semaphore(settings.llm_workers)
    async with httpx.AsyncClient() as client:
        tasks = [
            _process_repo(
                url, language, output_dir, settings, client,
                keep_clones, dry_run, clone_sem, llm_sem, errors_path,
            )
            for url in repos
        ]
        results = await asyncio.gather(*tasks, return_exceptions=False)
    return [r for r in results if r is not None]


def _print_summary_table(results: list[dict]) -> None:
    table = Table(title="Results")
    table.add_column("Repo", style="cyan")
    table.add_column("LOC", justify="right")
    table.add_column("Files", justify="right")
    table.add_column("Test share", justify="right")

    errors = 0
    total_loc = 0

    for r in results:
        if "error" in r:
            table.add_row(r["repo_name"], "ERROR", "-", "-", style="red")
            errors += 1
        else:
            loc = r.get("total_loc", 0)
            files = r.get("file_count", 0)
            test_share = r.get("test_share", 0)
            table.add_row(r["repo_name"], str(loc), str(files), f"{test_share:.0%}")
            total_loc += loc

    console.print(table)
    console.print(
        f"Processed: {len(results) - errors}/{len(results)}  |  "
        f"Errors: {errors}  |  "
        f"Total LOC: {total_loc:,}"
    )


@app.command()
def estimate(
    repos_file: Path = typer.Argument(..., help="File with repo URLs, one per line"),
    language: str = typer.Option("python", help="Language"),
) -> None:
    """Estimate token usage and cost without running LLM calls."""
    _setup_logging()
    settings = Settings()
    repos = _load_repos(repos_file)[:5]

    logger.info(f"Estimating based on first {len(repos)} repos...")

    results = asyncio.run(_estimate_repos(repos, language, settings))

    table = Table(title="Token Estimate")
    table.add_column("Repo", style="cyan")
    table.add_column("Ranking tokens", justify="right")
    table.add_column("Cost @ Haiku", justify="right")

    total_tokens = 0
    for name, tokens in results:
        cost_haiku = tokens / 1_000_000 * 0.25
        table.add_row(name, str(tokens), f"${cost_haiku:.4f}")
        total_tokens += tokens

    n_repos = len(repos)
    if n_repos:
        avg = total_tokens / n_repos
        total_repos = _count_lines(repos_file)
        extrapolated = int(avg * total_repos)
        console.print(table)
        console.print(f"\nAvg tokens/repo: {avg:.0f}")
        console.print(f"Extrapolated for {total_repos} repos: {extrapolated:,} tokens")
        console.print(f"Est. cost @ Haiku: ${extrapolated / 1_000_000 * 0.25:.2f}")
    else:
        console.print(table)


async def _estimate_repos(repos: list[str], language: str, settings: Settings) -> list[tuple[str, int]]:
    results = []
    for url in repos:
        repo_name = url.rstrip("/").split("/")[-1]
        clone_dest = Path(settings.clone_dir) / repo_name
        try:
            await clone_repo(url, clone_dest)
            entries = get_repo_tree(clone_dest, language, settings.max_repo_files)
            inventory = await analyze_repo(clone_dest, language, entries)
            from .scorer import _format_file_list
            file_list_str = _format_file_list(inventory)
            tokens = len(file_list_str) // 4
            results.append((repo_name, tokens))
        except Exception as e:
            logger.error(f"[{repo_name}] estimate failed: {e}")
            results.append((repo_name, 0))
        finally:
            cleanup_repo(clone_dest)
    return results


def _count_lines(path: Path) -> int:
    try:
        return sum(
            1 for line in path.read_text().splitlines()
            if line.strip() and not line.startswith("#")
        )
    except Exception:
        return 0


@app.command("show-tree")
def show_tree(
    repo_url: str = typer.Argument(..., help="Repository URL"),
    language: str = typer.Option("python", help="Language"),
) -> None:
    """Clone repo, build tree, print formatted ranking prompt input to stdout."""
    _setup_logging()
    settings = Settings()

    repo_name = repo_url.rstrip("/").split("/")[-1]
    clone_dest = Path(settings.clone_dir) / repo_name

    async def _run():
        try:
            await clone_repo(repo_url, clone_dest)
            entries = get_repo_tree(clone_dest, language, settings.max_repo_files)
            inventory = await analyze_repo(clone_dest, language, entries)
            from .scorer import _format_file_list
            print(_format_file_list(inventory))
        finally:
            cleanup_repo(clone_dest)

    asyncio.run(_run())


@app.command("show-sample")
def show_sample(
    repo_url: str = typer.Argument(..., help="Repository URL"),
    language: str = typer.Option("python", help="Language"),
) -> None:
    """Full run for one repo, print selected files without writing to disk."""
    _setup_logging()
    settings = Settings()

    repo_name = repo_url.rstrip("/").split("/")[-1]
    clone_dest = Path(settings.clone_dir) / repo_name

    async def _run():
        async with httpx.AsyncClient() as client:
            llm_sem = asyncio.Semaphore(settings.llm_workers)
            try:
                await clone_repo(repo_url, clone_dest)
                entries = get_repo_tree(clone_dest, language, settings.max_repo_files)
                inventory = await analyze_repo(clone_dest, language, entries)
                ranked = await rank_tree(inventory, settings, client)
                sample = await _build_sample(
                    ranked, clone_dest, settings, inventory, client, repo_url, llm_sem
                )

                table = Table(title=f"Sample: {repo_name}")
                table.add_column("Rank", justify="right")
                table.add_column("Path", style="cyan")
                table.add_column("Layer")
                table.add_column("LOC", justify="right")
                table.add_column("Partial")

                for sf in sorted(sample.files, key=lambda f: f.rank):
                    table.add_row(
                        str(sf.rank), sf.path, sf.layer,
                        str(sf.loc_taken), "yes" if sf.is_partial else "no"
                    )

                console.print(table)
                console.print(f"\nTotal: {sample.total_loc} LOC, {len(sample.files)} files")
            finally:
                cleanup_repo(clone_dest)

    asyncio.run(_run())


if __name__ == "__main__":
    app()
