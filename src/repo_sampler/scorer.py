from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass

import httpx
from loguru import logger

from .analyzer import RepoInventory
from .config import Settings
from .prompts import FILE_EXTRACTION_PROMPT, TREE_RANKING_PROMPT


class AuthError(Exception):
    pass


@dataclass
class RankedFile:
    path: str
    loc: int
    raw_lines: int
    layer: str
    is_test: bool
    top_level_dir: str = ""


def _format_file_list(inventory: RepoInventory) -> str:
    lines: list[str] = []
    for fi in sorted(inventory.files, key=lambda f: f.path):
        tags = ""
        if fi.is_test:
            tags += " [TEST]"
        if fi.days_since_modified > 180:
            tags += " [OLD]"
        lines.append(f"{fi.path} [{fi.loc} LOC]{tags}")
    return "\n".join(lines)


async def rank_tree(
    inventory: RepoInventory,
    settings: Settings,
    client: httpx.AsyncClient,
) -> list[RankedFile]:
    lang_share_str = ", ".join(
        f"{lang.capitalize()} {round(share * 100)}%"
        for lang, share in inventory.lang_share.items()
    )

    monorepo_note = ""
    if inventory.has_multiple_services:
        monorepo_note = "NOTE: This appears to be a monorepo with multiple services."

    file_list_str = _format_file_list(inventory)

    prompt = TREE_RANKING_PROMPT.format(
        language=inventory.language,
        lang_share=lang_share_str,
        build_system=inventory.build_system,
        total_count=len(inventory.files),
        test_count=inventory.test_file_count,
        prod_count=inventory.prod_file_count,
        monorepo_note=monorepo_note,
        file_list=file_list_str,
    )

    payload = {
        "model": settings.openrouter_model,
        "max_tokens": 8000,
        "temperature": 0.0,
        "messages": [{"role": "user", "content": prompt}],
    }

    headers = {
        "Authorization": f"Bearer {settings.openrouter_api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/repo-sampler",
        "X-Title": "repo-sampler",
    }

    raw_items = await _call_llm(
        client,
        f"{settings.openrouter_base_url}/chat/completions",
        payload,
        headers,
    )

    file_map = {fi.path: fi for fi in inventory.files}
    ranked: list[RankedFile] = []
    seen: set[str] = set()

    for item in raw_items:
        path = item.get("f", "")
        layer = item.get("l", "other")

        if path in seen:
            continue
        seen.add(path)

        fi = file_map.get(path)
        if fi is None:
            logger.warning(f"Ranked file not found in inventory: {path}")
            continue

        ranked.append(RankedFile(
            path=path,
            loc=fi.loc,
            raw_lines=fi.raw_lines,
            layer=layer,
            is_test=fi.is_test,
            top_level_dir=fi.top_level_dir,
        ))

    # Append any inventory files the LLM omitted (at the end, lowest priority)
    for fi in inventory.files:
        if fi.path not in seen:
            ranked.append(RankedFile(
                path=fi.path,
                loc=fi.loc,
                raw_lines=fi.raw_lines,
                layer="test" if fi.is_test else "other",
                is_test=fi.is_test,
                top_level_dir=fi.top_level_dir,
            ))

    # Guarantee tier ordering: push low-value layers to the end regardless of LLM rank
    LOW_VALUE_LAYERS = {"autogen", "infra"}
    high = [f for f in ranked if f.layer not in LOW_VALUE_LAYERS]
    low  = [f for f in ranked if f.layer in LOW_VALUE_LAYERS]
    ranked = high + low

    logger.info(f"Ranked: {len(ranked)} files ({len(inventory.files)} in inventory)")
    return ranked


async def _call_llm(
    client: httpx.AsyncClient,
    url: str,
    payload: dict,
    headers: dict,
) -> list[dict]:
    for attempt in range(3):
        try:
            resp = await client.post(url, json=payload, headers=headers, timeout=90.0)
        except httpx.TimeoutException:
            wait = 2 ** attempt
            logger.warning(f"LLM timeout, retrying in {wait}s (attempt {attempt + 1})")
            await asyncio.sleep(wait)
            continue

        if resp.status_code == 401:
            raise AuthError("HTTP 401: invalid API key")

        if resp.status_code in (429, 500, 502, 503):
            wait = 2 ** attempt
            logger.warning(f"HTTP {resp.status_code}, retrying in {wait}s (attempt {attempt + 1})")
            await asyncio.sleep(wait)
            continue

        resp.raise_for_status()
        data = resp.json()
        content = (data.get("choices", [{}])[0].get("message", {}).get("content") or "").strip()

        if not content:
            logger.warning("LLM returned empty content")
            return []

        if content.startswith("```"):
            lines = content.splitlines()
            end = next((i for i in range(len(lines) - 1, 0, -1) if lines[i].strip() == "```"), len(lines))
            content = "\n".join(lines[1:end])

        try:
            return json.loads(content)
        except json.JSONDecodeError:
            logger.warning(f"LLM returned invalid JSON: {content[:300]}")
            return []

    logger.warning("LLM call failed after 3 attempts")
    return []


async def extract_chunk(
    path: str,
    content: str,
    language: str,
    target_loc: int,
    settings: Settings,
    client: httpx.AsyncClient,
) -> tuple[int, int] | None:
    """Ask LLM to select the most representative line range from a file.

    Returns (start, end) as 1-indexed line numbers, or None on failure.
    """
    lines = content.splitlines()
    total_lines = len(lines)

    min_lines = max(10, int(target_loc * 0.5))
    max_lines = min(total_lines, int(target_loc * 2.5))

    prompt = FILE_EXTRACTION_PROMPT.format(
        language=language,
        path=path,
        total_lines=total_lines,
        content=content,
        target_loc=target_loc,
        min_lines=min_lines,
        max_lines=max_lines,
    )

    payload = {
        "model": settings.openrouter_model,
        "max_tokens": 64,
        "temperature": 0.0,
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
                timeout=30.0,
            )
        except httpx.TimeoutException:
            await asyncio.sleep(2 ** attempt)
            continue

        if resp.status_code == 401:
            raise AuthError("HTTP 401: invalid API key")
        if resp.status_code in (429, 500, 502, 503):
            await asyncio.sleep(2 ** attempt)
            continue

        resp.raise_for_status()
        data = resp.json()
        raw = (data.get("choices", [{}])[0].get("message", {}).get("content") or "").strip()

        if raw.startswith("```"):
            raw_lines = raw.splitlines()
            end_idx = next(
                (i for i in range(len(raw_lines) - 1, 0, -1) if raw_lines[i].strip() == "```"),
                len(raw_lines),
            )
            raw = "\n".join(raw_lines[1:end_idx])

        try:
            obj = json.loads(raw)
            start = int(obj["start"])
            end = int(obj["end"])
            start = max(1, min(start, total_lines))
            end = max(start, min(end, total_lines))
            return start, end
        except (json.JSONDecodeError, KeyError, ValueError, TypeError):
            logger.warning(f"Bad extraction response for {path}: {raw[:100]}")
            return None

    return None


# Keep old name as alias so summarizer.py import still works
score_tree = rank_tree
