from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import httpx
from loguru import logger

from .config import Settings

COMMENT_PREFIXES: dict[str, list[str]] = {
    "python":     ["#"],
    "typescript": ["//", "*", "/*", "*/"],
    "javascript": ["//", "*", "/*", "*/"],
    "go":         ["//", "*", "/*", "*/"],
    "rust":       ["//", "///", "*", "/*", "*/"],
    "java":       ["//", "*", "/*", "*/"],
    "kotlin":     ["//", "*", "/*", "*/"],
    "ruby":       ["#"],
    "cpp":        ["//", "*", "/*", "*/"],
}

_EXT_TO_LANG: dict[str, str] = {
    ".py": "python", ".pyw": "python",
    ".ts": "typescript", ".tsx": "typescript",
    ".js": "javascript", ".jsx": "javascript",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".kt": "kotlin", ".kts": "kotlin",
    ".rb": "ruby",
    ".cpp": "cpp", ".cc": "cpp", ".cxx": "cpp", ".hpp": "cpp", ".h": "cpp",
}


def _lang_from_path(path: str) -> str:
    return _EXT_TO_LANG.get(Path(path).suffix.lower(), "other")

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class AgentSavedFile:
    path: str
    layer: str
    loc_taken: int
    is_partial: bool
    rank: int


@dataclass
class AgentResult:
    repo_url: str
    repo_name: str       # last path segment, for display
    folder_name: str     # full URL-based name, used for output directory
    files: list[AgentSavedFile] = field(default_factory=list)
    total_loc: int = 0
    agent_iterations: int = 0
    bash_calls: int = 0
    summary_md: str = ""


class AuthError(Exception):
    pass


# ---------------------------------------------------------------------------
# Tool definitions (OpenAI-compatible format)
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Run a shell command in the repo root (find, cat, wc -l, git log, cloc, grep, tree…). Output capped. Do NOT use to copy files — use save_sample.",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "save_sample",
            "description": "Copy a file (or line range) verbatim from disk to deliverable/samples/. Returns {saved, loc, running_total_loc}.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Relative path from repo root."},
                    "layer": {
                        "type": "string",
                        "enum": ["business", "data", "api", "util", "test", "infra", "autogen"],
                    },
                    "start_line": {"type": "integer", "description": "1-indexed. Omit for full file."},
                    "end_line":   {"type": "integer", "description": "Inclusive. Omit for full file."},
                },
                "required": ["path", "layer"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_summary",
            "description": "Write repo_summary.md. Call after all save_sample calls.",
            "parameters": {
                "type": "object",
                "properties": {"content": {"type": "string"}},
                "required": ["content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "finish",
            "description": "Signal sampling is complete.",
            "parameters": {
                "type": "object",
                "properties": {
                    "message":    {"type": "string"},
                    "total_loc":  {"type": "integer"},
                    "file_count": {"type": "integer"},
                },
                "required": ["message", "total_loc", "file_count"],
            },
        },
    },
]


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

def _build_system_prompt(settings: Settings) -> str:
    return f"""\
## WORKFLOW — READ FIRST

Save files as you find them. Do not map the entire codebase before saving.

1. Quick scan (1–2 bash calls): top-level structure + language stats.
2. Pick a file → read it → if worth including, call save_sample immediately.
3. Repeat across directories/layers until ~{settings.target_loc} LOC saved.
4. write_summary → finish.

Hard rules:
- Call save_sample within your first 3 tool calls. Do not make 5+ bash calls before saving.
- Each bash result shows your LOC progress. React: if far from {settings.target_loc}, save more.
- You have {settings.agent_max_iterations} iterations. Do not spend them all on exploration.
- Once you have enough files, stop exploring and call write_summary + finish.

## Goal

Collect ~{settings.target_loc} LOC (±{settings.loc_tolerance}) of code samples that are representative of the codebase's typical quality — not cherry-picked best files. Include messy and boring parts proportionally.

## Select files

- Cover every significant module/layer: business logic, data access, API handlers, tests, utilities.
- Prefer medium-size files (100–500 LOC). Whole files preferred; partial extracts only for >800 LOC files.
- Tests: {int(settings.test_share_min * 100)}–{int(settings.test_share_max * 100)}% of total LOC.
- Stratify by module, layer, age (git log), and author when possible.
- **Quality over quantity of files**: if the repo has few substantive files, it is better to sample 1–2k LOC from the best available files than to pad the selection with many small or trivial files. Do NOT fill the budget with __init__.py, thin wrappers, or files under 30 LOC just to reach the LOC target. An honest under-sample of 1–2k LOC is preferable to a bloated sample full of noise.

## Exclude

- Auto-generated files: migrations, protobuf/OpenAPI stubs, ORM auto-gen, any "DO NOT EDIT" files.
- Vendored code, lock files, binary assets, pure-constant or fixture files.
- Files containing secrets, credentials, or PII — skip entirely.

## repo_summary.md format (four sections, ~1 page)

1. Repository overview — purpose, language, build system, main frameworks.
2. Repository structure — top-level tree 2–3 levels deep, one-line per directory.
3. What was sampled — which layers/modules covered, test vs prod ratio.
4. Notes — monorepo info, skipped areas, under-represented parts.

## Anonymization — ZERO TOLERANCE

ALL text you write (summary, notes, any output) must be fully anonymous:
- NO company, organisation, brand, or product names.
- NO personal names, emails, phone numbers, usernames, or any PII.
- NO geographic identifiers: country names, city names, region names, national languages (e.g. "Russian", "English", "Chinese"), cultural references, or any hint of origin country. This includes indirect references (e.g. Cyrillic variable names → say "non-ASCII identifiers").
- NO app/service/domain names visible in the code.
- Files with PII or secrets → skip with save_sample, do not include.
- Describe everything in neutral technical terms: "user management service", "payment module", "REST API backend".
- When in doubt → omit.

## Other rules

- No transformations on files ever. Files must be identical to a git checkout.
- Honesty over flattery — messy parts included proportionally.
- Monorepo → sample from each major project.
- When in doubt, prefer the boring/typical file over the impressive one.
"""


# ---------------------------------------------------------------------------
# Tool execution state
# ---------------------------------------------------------------------------

@dataclass
class _ToolCtx:
    repo_path: Path
    deliverable_dir: Path
    settings: Settings
    saved_files: list[AgentSavedFile] = field(default_factory=list)
    summary_md: str = ""
    finished: bool = False
    finish_args: dict = field(default_factory=dict)
    bash_calls: int = 0


def _count_loc_lines(lines: list[str], language: str) -> int:
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


async def _exec_bash(ctx: _ToolCtx, command: str) -> str:
    ctx.bash_calls += 1
    try:
        proc = await asyncio.create_subprocess_exec(
            "bash", "-c", command,
            cwd=str(ctx.repo_path),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        stdout, _ = await asyncio.wait_for(
            proc.communicate(), timeout=ctx.settings.agent_bash_timeout
        )
        output = stdout.decode(errors="replace")
    except asyncio.TimeoutError:
        try:
            proc.kill()
        except Exception:
            pass
        return f"[Command timed out after {ctx.settings.agent_bash_timeout}s]"
    except Exception as e:
        return f"[Error running command: {e}]"

    limit = ctx.settings.agent_bash_output_limit
    if len(output) > limit:
        output = output[:limit] + f"\n[...output truncated at {limit} chars...]"

    saved_loc = sum(f.loc_taken for f in ctx.saved_files)
    remaining = max(0, ctx.settings.target_loc - saved_loc)
    output += f"\n[LOC:{saved_loc}/{ctx.settings.target_loc} files:{len(ctx.saved_files)} need:{remaining}]"
    return output


def _exec_save_sample(
    ctx: _ToolCtx,
    path: str,
    layer: str,
    start_line: int | None = None,
    end_line: int | None = None,
) -> dict:
    src = ctx.repo_path / path
    if not src.exists():
        return {"saved": False, "error": f"File not found: {path}"}

    try:
        content = src.read_text(errors="ignore")
    except Exception as e:
        return {"saved": False, "error": str(e)}

    all_lines = content.splitlines(keepends=True)
    total_lines = len(all_lines)

    if start_line is not None and end_line is not None:
        s = max(1, start_line) - 1
        e = min(end_line, total_lines)
        if s >= e:
            return {"saved": False, "error": f"Invalid line range {start_line}–{end_line} for {total_lines}-line file"}
        chunk = all_lines[s:e]
        is_partial = (s > 0 or e < total_lines)
    else:
        chunk = all_lines
        is_partial = False

    dest = ctx.deliverable_dir / "samples" / path
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text("".join(chunk), encoding="utf-8")

    loc = _count_loc_lines(chunk, _lang_from_path(path))
    rank = len(ctx.saved_files) + 1
    running_total = sum(f.loc_taken for f in ctx.saved_files) + loc

    ctx.saved_files.append(AgentSavedFile(
        path=path,
        layer=layer,
        loc_taken=loc,
        is_partial=is_partial,
        rank=rank,
    ))

    return {"saved": True, "loc": loc, "running_total_loc": running_total}


def _exec_write_summary(ctx: _ToolCtx, content: str) -> dict:
    dest = ctx.deliverable_dir / "repo_summary.md"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(content, encoding="utf-8")
    ctx.summary_md = content
    return {"written": True}


def _exec_finish(ctx: _ToolCtx, message: str, total_loc: int, file_count: int) -> dict:
    ctx.finished = True
    ctx.finish_args = {"message": message, "total_loc": total_loc, "file_count": file_count}
    return {"ok": True}


async def _dispatch_tool(ctx: _ToolCtx, name: str, args: dict) -> str:
    if name == "bash":
        command = args.get("command", "")
        if not command:
            return json.dumps({"error": "bash called with empty command"})
        return await _exec_bash(ctx, command)
    elif name == "save_sample":
        path = args.get("path", "")
        layer = args.get("layer", "other")
        if not path:
            return json.dumps({"error": "save_sample called without 'path'"})
        return json.dumps(_exec_save_sample(
            ctx,
            path=path,
            layer=layer,
            start_line=args.get("start_line"),
            end_line=args.get("end_line"),
        ))
    elif name == "write_summary":
        content = args.get("content", "")
        if not content:
            return json.dumps({"error": "write_summary called with empty content"})
        return json.dumps(_exec_write_summary(ctx, content))
    elif name == "finish":
        return json.dumps(_exec_finish(
            ctx,
            message=args.get("message", ""),
            total_loc=args.get("total_loc", 0),
            file_count=args.get("file_count", 0),
        ))
    else:
        return json.dumps({"error": f"Unknown tool: {name}"})


# ---------------------------------------------------------------------------
# Context window management
# ---------------------------------------------------------------------------

def _truncate_old_tool_results(messages: list[dict], keep_last: int = 2) -> None:
    """Truncate old tool result messages to save context window space."""
    tool_indices = [i for i, m in enumerate(messages) if m.get("role") == "tool"]
    to_truncate = tool_indices[:-keep_last] if len(tool_indices) > keep_last else []
    for i in to_truncate:
        content = messages[i].get("content", "")
        if isinstance(content, str) and len(content) > 300:
            messages[i] = {
                **messages[i],
                "content": content[:150] + "\n[truncated]",
            }


def _prune_old_assistant_messages(messages: list[dict], keep_last: int = 3) -> None:
    """Remove text content from old assistant messages; keep tool_calls intact."""
    assistant_indices = [i for i, m in enumerate(messages) if m.get("role") == "assistant"]
    for i in assistant_indices[:-keep_last]:
        if messages[i].get("content"):
            messages[i] = {**messages[i], "content": None}


# ---------------------------------------------------------------------------
# HTTP call to OpenRouter
# ---------------------------------------------------------------------------

async def _call_openrouter(
    messages: list[dict],
    settings: Settings,
    client: httpx.AsyncClient,
) -> dict:
    payload = {
        "model": settings.agent_model,
        "messages": messages,
        "tools": TOOLS,
        "tool_choice": "auto",
        "max_tokens": 1024,
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
                timeout=120.0,
            )
        except httpx.TimeoutException:
            wait = 2 ** attempt
            logger.warning(f"OpenRouter timeout, retrying in {wait}s")
            await asyncio.sleep(wait)
            continue

        if resp.status_code == 401:
            raise AuthError("HTTP 401: invalid API key")
        if resp.status_code in (429, 500, 502, 503):
            wait = 2 ** attempt
            logger.warning(f"HTTP {resp.status_code}, retrying in {wait}s")
            await asyncio.sleep(wait)
            continue

        resp.raise_for_status()
        return resp.json()

    raise RuntimeError("OpenRouter call failed after 3 attempts")


# ---------------------------------------------------------------------------
# URL → folder name
# ---------------------------------------------------------------------------

def url_to_folder_name(url: str) -> str:
    """Convert a repo URL to a filesystem-safe folder name preserving the full path.

    Examples:
      https://github.com/owner/repo          -> github.com__owner__repo
      https://gitlab.com/org/sub/repo.git    -> gitlab.com__org__sub__repo
      git@github.com:owner/repo.git          -> github.com__owner__repo
    """
    url = url.strip().rstrip("/")
    if url.endswith(".git"):
        url = url[:-4]
    # Handle SSH git@ syntax
    if url.startswith("git@"):
        url = url[4:].replace(":", "/", 1)
    else:
        for prefix in ("https://", "http://", "git://", "ssh://"):
            if url.startswith(prefix):
                url = url[len(prefix):]
                break
    return url.replace("/", "__")


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

def _log_tool_call(repo_name: str, tool_name: str, args: dict) -> None:
    if tool_name == "bash":
        cmd = args.get("command", "")
        # Truncate long commands for readability
        display = cmd if len(cmd) <= 120 else cmd[:117] + "..."
        logger.info(f"[{repo_name}] bash: {display}")
    elif tool_name == "save_sample":
        path = args.get("path", "?")
        layer = args.get("layer", "?")
        extra = ""
        if args.get("start_line") is not None:
            extra = f" lines {args['start_line']}–{args.get('end_line', '?')}"
        logger.info(f"[{repo_name}] save_sample: {path} [{layer}]{extra}")
    elif tool_name == "write_summary":
        length = len(args.get("content", ""))
        logger.info(f"[{repo_name}] write_summary: {length} chars")
    elif tool_name == "finish":
        logger.info(
            f"[{repo_name}] finish: {args.get('total_loc', 0)} LOC, "
            f"{args.get('file_count', 0)} files — {args.get('message', '')}"
        )
    else:
        logger.debug(f"[{repo_name}] tool: {tool_name}({list(args.keys())})")


# ---------------------------------------------------------------------------
# Main agentic loop
# ---------------------------------------------------------------------------

async def run_agent(
    repo_path: Path,
    repo_url: str,
    output_dir: Path,
    settings: Settings,
    client: httpx.AsyncClient,
) -> AgentResult:
    folder_name = url_to_folder_name(repo_url) if repo_url else repo_path.name
    repo_name = repo_url.rstrip("/").split("/")[-1] if repo_url else repo_path.name
    deliverable_dir = output_dir / folder_name

    if deliverable_dir.exists():
        import shutil
        shutil.rmtree(deliverable_dir)
        logger.info(f"[{repo_name}] cleared stale deliverable dir: {deliverable_dir}")
    deliverable_dir.mkdir(parents=True, exist_ok=True)

    ctx = _ToolCtx(
        repo_path=repo_path,
        deliverable_dir=deliverable_dir,
        settings=settings,
    )

    messages: list[dict] = [
        {"role": "system", "content": _build_system_prompt(settings)},
        {
            "role": "user",
            "content": (
                f"Repository has been cloned to: {repo_path}\n\n"
                "Begin."
            ),
        },
    ]

    agent_log: list[dict] = []
    iterations = 0
    last_save_iteration = -1   # track when agent last called save_sample

    for iteration in range(settings.agent_max_iterations):
        iterations = iteration + 1

        # Nudge if agent has been exploring without saving for too long
        saves_so_far = len(ctx.saved_files)
        turns_since_save = iteration - (last_save_iteration + 1)
        if saves_so_far == 0 and iteration >= 3:
            nudge = (
                f"⚠️ URGENT: You have made {iteration} tool calls and saved 0 files. "
                f"You MUST call save_sample on your very next tool call. "
                f"Stop exploring — pick any good file you have already seen and save it NOW. "
                f"Target: {settings.target_loc} LOC total."
            )
            messages.append({"role": "user", "content": nudge})
            agent_log.append({"turn": iteration + 1, "nudge": nudge})
        elif saves_so_far > 0 and turns_since_save >= 4:
            saved_loc = sum(f.loc_taken for f in ctx.saved_files)
            remaining = max(0, settings.target_loc - saved_loc)
            nudge = (
                f"⚠️ You have saved {saves_so_far} files ({saved_loc} LOC) but haven't saved "
                f"anything in {turns_since_save} turns. You still need ~{remaining} more LOC. "
                f"Save more files now."
            )
            messages.append({"role": "user", "content": nudge})
            agent_log.append({"turn": iteration + 1, "nudge": nudge})

        data = await _call_openrouter(messages, settings, client)
        choices = data.get("choices") or []
        if not choices:
            logger.warning(f"[{repo_name}] empty choices in API response, skipping turn")
            continue
        choice = choices[0]
        message = choice.get("message")
        if message is None:
            logger.warning(f"[{repo_name}] no message in API response choice, skipping turn")
            continue
        finish_reason = choice.get("finish_reason", "")

        # Append assistant message to history
        messages.append(message)
        agent_log.append({"turn": iteration + 1, "assistant": message})

        if finish_reason == "length":
            logger.warning(f"[{repo_name}] agent hit context length limit at iteration {iteration + 1}")
            break

        tool_calls = message.get("tool_calls") or []

        if not tool_calls:
            logger.warning(f"[{repo_name}] agent stopped calling tools at iteration {iteration + 1}")
            break

        # Execute all tool calls in this turn
        for tc in tool_calls:
            func = tc.get("function") or {}
            tool_name = func.get("name", "")
            if not tool_name:
                logger.warning(f"[{repo_name}] tool call missing name, skipping")
                continue
            try:
                tool_args = json.loads(func.get("arguments") or "{}")
            except json.JSONDecodeError:
                tool_args = {}

            _log_tool_call(repo_name, tool_name, tool_args)
            files_before = len(ctx.saved_files)
            try:
                result_str = await _dispatch_tool(ctx, tool_name, tool_args)
            except Exception as exc:
                logger.warning(f"[{repo_name}] tool {tool_name} raised unexpectedly: {exc}")
                result_str = json.dumps({"error": str(exc)})
            if len(ctx.saved_files) > files_before:
                last_save_iteration = iteration
                new_file = ctx.saved_files[-1]
                logger.info(
                    f"[{repo_name}] saved: {new_file.path} "
                    f"({new_file.layer}, {new_file.loc_taken} LOC"
                    f"{', partial' if new_file.is_partial else ''})"
                )

            messages.append({
                "role": "tool",
                "tool_call_id": tc.get("id", f"call_{iteration}_{tool_name}"),
                "content": result_str,
            })
            agent_log.append({
                "turn": iteration + 1,
                "tool": tool_name,
                "args": {k: v for k, v in tool_args.items() if k != "content"},
                "result_preview": result_str[:200],
            })

        # Truncate old context to keep token count manageable
        _truncate_old_tool_results(messages, keep_last=2)
        _prune_old_assistant_messages(messages, keep_last=3)

        if ctx.finished:
            break

    # Write agent log for debugging
    _write_agent_log(deliverable_dir, agent_log, ctx)

    # Ensure summary exists even if agent forgot to write it
    if not ctx.summary_md:
        _write_fallback_summary(ctx, repo_name, repo_url)

    total_loc = sum(f.loc_taken for f in ctx.saved_files)
    if total_loc < (settings.target_loc - settings.loc_tolerance):
        logger.warning(
            f"[{repo_name}] LOC budget not reached: {total_loc} < "
            f"{settings.target_loc - settings.loc_tolerance}"
        )

    test_loc = sum(f.loc_taken for f in ctx.saved_files if f.layer == "test")
    test_pct = test_loc / total_loc * 100 if total_loc else 0
    logger.info(
        f"[{repo_name}] done: {total_loc} LOC, {len(ctx.saved_files)} files, "
        f"test share: {test_pct:.0f}%, iterations: {iterations}, "
        f"bash calls: {ctx.bash_calls}"
    )

    return AgentResult(
        repo_url=repo_url,
        repo_name=repo_name,
        folder_name=folder_name,
        files=ctx.saved_files,
        total_loc=total_loc,
        agent_iterations=iterations,
        bash_calls=ctx.bash_calls,
        summary_md=ctx.summary_md,
    )


def _write_agent_log(deliverable_dir: Path, agent_log: list[dict], ctx: _ToolCtx) -> None:
    deliverable_dir.mkdir(parents=True, exist_ok=True)
    log = {
        "sampled_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "stats": {
            "files_saved": len(ctx.saved_files),
            "total_loc": sum(f.loc_taken for f in ctx.saved_files),
            "bash_calls": ctx.bash_calls,
            "iterations": len({e["turn"] for e in agent_log}),
        },
        "saved_files": [
            {
                "rank": f.rank,
                "path": f.path,
                "layer": f.layer,
                "loc_taken": f.loc_taken,
                "is_partial": f.is_partial,
            }
            for f in ctx.saved_files
        ],
        "agent_log": agent_log,
    }
    (deliverable_dir / "agent_log.json").write_text(
        json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _write_fallback_summary(ctx: _ToolCtx, repo_name: str, repo_url: str) -> None:
    layer_counts: dict[str, int] = {}
    for f in ctx.saved_files:
        layer_counts[f.layer] = layer_counts.get(f.layer, 0) + f.loc_taken

    file_list = "\n".join(
        f"- `{f.path}` ({f.layer}, {f.loc_taken} LOC)"
        for f in ctx.saved_files
    )
    total = sum(layer_counts.values())
    breakdown = ", ".join(
        f"{k}: {round(v / total * 100)}%"
        for k, v in sorted(layer_counts.items(), key=lambda x: -x[1])
    ) if total else ""

    summary = f"""## Repository overview

A code repository. This summary was auto-generated because the agent did not produce one.

## Repository structure

See the `samples/` directory for the sampled files.

## What was sampled

{len(ctx.saved_files)} files, {total} LOC total.
Layer breakdown: {breakdown}

{file_list}

## Notes

Summary was automatically generated due to agent not completing the write_summary step.
"""
    _exec_write_summary(ctx, summary)
