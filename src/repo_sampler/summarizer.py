from __future__ import annotations

import asyncio
from collections import Counter

import httpx
from loguru import logger

from .config import Settings
from .prompts import REPO_SUMMARY_PROMPT
from .scorer import AuthError
from .selector import RepoSample


async def generate_summary(
    sample: RepoSample,
    settings: Settings,
    client: httpx.AsyncClient,
) -> str:
    file_list = "\n".join(
        f"  {sf.path} [{sf.layer}, {sf.loc_taken} LOC]"
        for sf in sample.files
    )

    layer_counts: Counter[str] = Counter()
    for sf in sample.files:
        layer_counts[sf.layer] += sf.loc_taken
    total_loc = sum(layer_counts.values()) or 1

    layer_breakdown = ", ".join(
        f"{layer}: {round(loc / total_loc * 100)}%"
        for layer, loc in layer_counts.most_common()
    )

    test_loc = sum(sf.loc_taken for sf in sample.files if sf.layer == "test")
    test_share_pct = round(test_loc / total_loc * 100)

    analysis_notes_parts: list[str] = []
    if sample.inventory and sample.inventory.has_multiple_services:
        analysis_notes_parts.append("This appears to be a monorepo with multiple services.")
    if test_share_pct < 15:
        analysis_notes_parts.append(
            "Test coverage is sparse; test share in sample was limited accordingly."
        )
    partial_count = sum(1 for sf in sample.files if sf.is_partial)
    if partial_count:
        analysis_notes_parts.append(
            f"{partial_count} files were partially extracted due to size."
        )
    analysis_notes = "\n".join(analysis_notes_parts) or "None."

    commit_sha = await _get_commit_sha(sample.inventory.repo_path) if sample.inventory else "unknown"

    lang_share_str = ""
    if sample.inventory:
        lang_share_str = ", ".join(
            f"{lang.capitalize()} {round(share * 100)}%"
            for lang, share in sample.inventory.lang_share.items()
        )

    top_dirs = ", ".join(sample.inventory.top_dirs) if sample.inventory else ""
    build_system = sample.inventory.build_system if sample.inventory else "unknown"

    prompt = REPO_SUMMARY_PROMPT.format(
        repo_name=sample.repo_name,
        repo_url=sample.repo_url,
        language=sample.language,
        lang_share=lang_share_str,
        build_system=build_system,
        top_dirs=top_dirs,
        commit_sha=commit_sha,
        file_count=len(sample.files),
        total_loc=sample.total_loc,
        layer_breakdown=layer_breakdown,
        test_share_pct=test_share_pct,
        file_list=file_list,
        analysis_notes=analysis_notes,
    )

    payload = {
        "model": settings.openrouter_model,
        "max_tokens": 1500,
        "temperature": 0.2,
        "messages": [{"role": "user", "content": prompt}],
    }

    headers = {
        "Authorization": f"Bearer {settings.openrouter_api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/repo-sampler",
        "X-Title": "repo-sampler",
    }

    for attempt in range(3):
        try:
            resp = await client.post(
                f"{settings.openrouter_base_url}/chat/completions",
                json=payload,
                headers=headers,
                timeout=60.0,
            )
        except httpx.TimeoutException:
            wait = 2 ** attempt
            logger.warning(f"Summary LLM timeout, retrying in {wait}s")
            await asyncio.sleep(wait)
            continue

        if resp.status_code == 401:
            raise AuthError("HTTP 401: invalid API key")

        if resp.status_code in (429, 500, 502, 503):
            wait = 2 ** attempt
            logger.warning(f"HTTP {resp.status_code} on summary, retrying in {wait}s")
            await asyncio.sleep(wait)
            continue

        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]

    logger.warning("Summary generation failed after 3 attempts, using placeholder")
    return f"# {sample.repo_name}\n\nSummary generation failed.\n"


async def _get_commit_sha(repo_path) -> str:
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
