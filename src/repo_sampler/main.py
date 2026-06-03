from __future__ import annotations

import asyncio
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import httpx
import typer
from loguru import logger
from rich.console import Console
from rich.table import Table

from .agent import AgentResult, AuthError, run_agent, url_to_folder_name
from .cloner import CloneError, cleanup_repo, clone_repo
from .config import Settings
from .writer import append_jsonl_with_meta, write_parquet

app = typer.Typer(help="CLI tool for extracting representative code samples from git repositories.")
console = Console()


def _setup_logging(output_dir: Path | None = None) -> None:
    logger.remove()
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.add(
            output_dir / "run.log",
            level="INFO",
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


async def _get_commit_sha(repo_path: Path) -> str:
    try:
        proc = await asyncio.create_subprocess_exec(
            "git", "-C", str(repo_path), "rev-parse", "HEAD",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=10)
        return stdout.decode().strip()[:7]
    except Exception:
        return "unknown"


async def _process_repo(
    url: str,
    output_dir: Path,
    settings: Settings,
    client: httpx.AsyncClient,
    keep_clones: bool,
    dry_run: bool,
    clone_sem: asyncio.Semaphore,
    errors_path: Path,
) -> dict | None:
    repo_name = url.rstrip("/").split("/")[-1]
    folder_name = url_to_folder_name(url)
    clone_dest = Path(settings.clone_dir) / repo_name

    async with clone_sem:
        stage = "clone"
        try:
            logger.info(f"[{repo_name}] cloning...")
            await clone_repo(url, clone_dest)

            if dry_run:
                proc = await asyncio.create_subprocess_exec(
                    "bash", "-c", "find . -type f | wc -l",
                    cwd=str(clone_dest),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                stdout, _ = await proc.communicate()
                file_count = stdout.decode().strip()
                logger.info(f"[{repo_name}] dry-run: {file_count} files found")
                return {"repo_url": url, "repo_name": repo_name, "folder_name": folder_name, "dry_run": True}

            stage = "agent"
            logger.info(f"[{repo_name}] starting agent ({settings.agent_model})...")
            result = await run_agent(
                repo_path=clone_dest,
                repo_url=url,
                output_dir=output_dir,
                settings=settings,
                client=client,
            )

            stage = "write"
            commit_sha = await _get_commit_sha(clone_dest)
            jsonl_path = output_dir / "samples.jsonl"
            append_jsonl_with_meta(
                result,
                jsonl_path,
                model=settings.agent_model,
                commit_sha=commit_sha,
            )

            test_loc = sum(f.loc_taken for f in result.files if f.layer == "test")
            test_share = test_loc / result.total_loc if result.total_loc else 0.0

            return {
                "repo_url": url,
                "repo_name": repo_name,
                "folder_name": folder_name,
                "total_loc": result.total_loc,
                "file_count": len(result.files),
                "test_share": test_share,
                "iterations": result.agent_iterations,
            }

        except AuthError as e:
            logger.critical(f"Auth error: {e}")
            _write_error(errors_path, url, stage, str(e))
            raise typer.Exit(1)

        except CloneError as e:
            logger.error(f"[{repo_name}] clone failed: {e}")
            _write_error(errors_path, url, stage, str(e))
            return {"repo_url": url, "repo_name": repo_name, "folder_name": folder_name, "error": str(e)}

        except Exception as e:
            logger.error(f"[{repo_name}] failed at {stage}: {e}")
            _write_error(errors_path, url, stage, str(e))
            return {"repo_url": url, "repo_name": repo_name, "folder_name": folder_name, "error": str(e)}

        finally:
            if not keep_clones and clone_dest.exists():
                cleanup_repo(clone_dest)


@app.command()
def run(
    repos_file: Path = typer.Argument(..., help="File with repo URLs, one per line"),
    output: Path = typer.Option(Path("./output"), help="Output directory"),
    format: str = typer.Option("jsonl", help="jsonl|parquet"),
    workers: Optional[int] = typer.Option(None, help="Parallel clone workers"),
    force: bool = typer.Option(False, help="Re-process already completed repos"),
    dry_run: bool = typer.Option(False, help="Clone only, no agent"),
    keep_clones: bool = typer.Option(False, help="Keep clones after processing"),
) -> None:
    """Process repos from file. Already completed repos are skipped automatically (use --force to override)."""
    output.mkdir(parents=True, exist_ok=True)
    _setup_logging(output)
    settings = Settings()

    if workers:
        settings.clone_workers = workers

    repos = _load_repos(repos_file)
    jsonl_path = output / "samples.jsonl"
    errors_path = output / "errors.jsonl"

    if not force:
        processed = _load_processed(jsonl_path)
        original_count = len(repos)
        repos = [r for r in repos if r not in processed]
        skipped = original_count - len(repos)
        if skipped:
            logger.info(f"Skipping {skipped} already processed repos (use --force to reprocess)")

    logger.info(
        f"Processing {len(repos)} repos | "
        f"workers={settings.clone_workers} | model={settings.agent_model}"
    )

    results = asyncio.run(
        _run_all(repos, output, settings, keep_clones, dry_run, errors_path)
    )

    if format == "parquet":
        write_parquet(output)

    _print_summary_table(results)


async def _run_all(
    repos: list[str],
    output_dir: Path,
    settings: Settings,
    keep_clones: bool,
    dry_run: bool,
    errors_path: Path,
) -> list[dict]:
    clone_sem = asyncio.Semaphore(settings.clone_workers)
    async with httpx.AsyncClient() as client:
        tasks = [
            _process_repo(
                url, output_dir, settings, client,
                keep_clones, dry_run, clone_sem, errors_path,
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
    table.add_column("Iters", justify="right")

    errors = 0
    total_loc = 0

    for r in results:
        display = r.get("folder_name") or r.get("repo_name", "?")
        if "error" in r:
            table.add_row(display, "ERROR", "-", "-", "-", style="red")
            errors += 1
        elif r.get("dry_run"):
            table.add_row(display, "dry-run", "-", "-", "-", style="dim")
        else:
            loc = r.get("total_loc", 0)
            files = r.get("file_count", 0)
            test_share = r.get("test_share", 0)
            iters = r.get("iterations", "-")
            table.add_row(display, str(loc), str(files), f"{test_share:.0%}", str(iters))
            total_loc += loc

    console.print(table)
    console.print(
        f"Processed: {len(results) - errors}/{len(results)}  |  "
        f"Errors: {errors}  |  "
        f"Total LOC: {total_loc:,}"
    )


@app.command("show-sample")
def show_sample(
    repo_url: str = typer.Argument(..., help="Repository URL"),
    output: Path = typer.Option(Path("./output"), help="Output directory"),
    keep_clones: bool = typer.Option(False, help="Keep clone after run"),
) -> None:
    """Full agent run for one repo. Writes deliverable to output/ and prints summary."""
    output.mkdir(parents=True, exist_ok=True)
    _setup_logging(output)
    settings = Settings()

    repo_name = repo_url.rstrip("/").split("/")[-1]
    clone_dest = Path(settings.clone_dir) / repo_name

    async def _run():
        async with httpx.AsyncClient() as client:
            try:
                await clone_repo(repo_url, clone_dest)
                result = await run_agent(
                    repo_path=clone_dest,
                    repo_url=repo_url,
                    output_dir=output,
                    settings=settings,
                    client=client,
                )

                table = Table(title=f"Sample: {repo_name}")
                table.add_column("Rank", justify="right")
                table.add_column("Path", style="cyan")
                table.add_column("Layer")
                table.add_column("LOC", justify="right")
                table.add_column("Partial")

                for sf in sorted(result.files, key=lambda f: f.rank):
                    table.add_row(
                        str(sf.rank), sf.path, sf.layer,
                        str(sf.loc_taken), "yes" if sf.is_partial else "no",
                    )

                console.print(table)
                console.print(
                    f"\nTotal: {result.total_loc} LOC, {len(result.files)} files | "
                    f"iterations: {result.agent_iterations} | bash calls: {result.bash_calls}"
                )
            finally:
                if not keep_clones and clone_dest.exists():
                    cleanup_repo(clone_dest)

    asyncio.run(_run())


if __name__ == "__main__":
    app()
