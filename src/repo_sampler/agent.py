from __future__ import annotations

import asyncio
import json
import math
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import httpx
from loguru import logger

from .config import Settings
from .languages import (
    COMMENT_PREFIXES,  # noqa: F401  (re-export for legacy imports)
    compute_language_stats,
    count_code_lines,
    extensions_for,  # noqa: F401  (re-export for legacy imports)
    format_distribution,
    lang_from_path,
    pick_code_primary,
    primary_display,
    primary_extensions,
    same_language_bucket,
)

# Legacy aliases — older code and tests import these names from agent.
_lang_from_path = lang_from_path
_count_loc_lines = count_code_lines

# ---------------------------------------------------------------------------
# Layer value semantics
# ---------------------------------------------------------------------------
# The `layer` an agent tags each saved file with also encodes how much
# substantive, hand-written logic the file carries. We track the share of the
# sample that lands in the high-signal layers and steer the agent toward it,
# so the budget is spent on real business/domain logic and algorithms rather
# than boilerplate and generated filler.
SUBSTANCE_LAYERS = {"business", "algorithm", "data", "api"}
LOW_VALUE_LAYERS = {"boilerplate", "infra", "autogen"}

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
    language: str = ""   # scc language name, "" if unknown


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
    primary_language: str = ""                 # enforced language ("" = none)
    repo_lang_distribution: dict[str, float] = field(default_factory=dict)
    lang_stats_source: str = ""                # "scc" | "walk" | "empty"
    primary_forced: bool = False               # True when set via override


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
            "description": "Copy a file (or line range) verbatim from disk to deliverable/samples/. Returns {saved, loc, running_total_loc, logic_share}.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Relative path from repo root."},
                    "layer": {
                        "type": "string",
                        "description": (
                            "Tag the file honestly by what it mainly contains. "
                            "High-signal (favour these): business=domain rules/workflows/state machines; "
                            "algorithm=non-trivial algorithms/data processing; data=substantive queries/data logic; "
                            "api=API handlers with real logic. "
                            "Neutral: util, test. "
                            "Low-signal (sample sparingly): boilerplate=DTO/ORM scaffolding, DI/config wiring, "
                            "thin wrappers, presentation/markup; infra=build/ops; autogen=generated."
                        ),
                        "enum": ["business", "algorithm", "data", "api", "util", "test", "infra", "boilerplate", "autogen"],
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

def _build_system_prompt(
    settings: Settings, has_primary: bool = True, hard: bool = False
) -> str:
    if not has_primary:
        language_section = ""
    elif hard:
        language_section = f"""
## Language coverage — HARD REQUIREMENT

The first user message lists this repository's language distribution and its PRIMARY language.
- Your sample's language mix must roughly mirror that distribution.
- At least {int(settings.primary_share_min * 100)}% of saved LOC MUST be files in the PRIMARY language. A sample with little or no primary-language code is REJECTED ENTIRELY — the whole run is wasted.
- If most primary-language code is generated or minified (bundles, dumps, vendored builds), find the HAND-WRITTEN files in that language: component scripts, custom modules, queries, templates, central configs. Generated files stay excluded — but "hard to find" never justifies missing the minimum.
- If the primary language is markup/data (XML, JSON, SQL, templates), still satisfy the minimum: pick the most meaningful hand-written files (schemas, layouts, queries, central configs), not autogenerated dumps or lock files.
- Do not let a secondary language crowd out the primary one. Each tool result shows your primary-language share — react to it.
"""
    else:
        language_section = f"""
## Language focus

The first user message lists this repository's language distribution and the PRIMARY CODE language to focus on.
- Aim for at least {int(settings.primary_share_min * 100)}% of saved LOC in that language, and roughly mirror the rest of the distribution. Each tool result shows your share — react to it.
- Prefer hand-written files in the primary code language when choosing between candidates of similar value.
- Markup/style/data files (CSS, HTML, SVG, JSON, Markdown) matter much less than real code — include a few only when they are central to the project.
- If the repo genuinely lacks enough primary-language code (everything is generated/minified), sample the best real code available and explain it in the summary Notes — do NOT pad with junk to hit a number.
"""
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

Collect ~{settings.target_loc} LOC (±{settings.loc_tolerance}) of code samples that capture the repository's substantive, hand-written engineering — its real business and domain logic, non-trivial algorithms, and genuinely meaningful code. Aim for a high signal-to-noise sample: maximise the share of files dense with real logic, and keep boilerplate and machine-generated filler to a small, representative minimum.

HARD CAP: {settings.max_total_loc} LOC total. save_sample will reject any file that would push the total above it — do not try; once you are near {settings.target_loc}, stop saving and finish. Overshooting is as wrong as undershooting.

## Prioritise substance (favour these)

Spend most of the budget on files where most lines do real work. After every tool call you are shown your **logic share** — the % of saved LOC tagged business/algorithm/data/api. Aim for at least {int(settings.logic_share_min * 100)}% in these high-signal layers:
- **business** — domain rules, workflows, state machines, lifecycle/status transitions, calculations, domain formulas, validation with real branching.
- **algorithm** — non-trivial algorithms and data processing: parsing, scheduling/optimisation, transformations, concurrency, protocol handling, custom data structures.
- **data** — substantive queries and data-access logic (complex queries, aggregations), not plain ORM field declarations.
- **api** — the logic inside API handlers (orchestration, rules, error handling), not the routing skeleton.

Between two files of similar size, prefer the one with more real control flow and edge-case handling over the one that is mostly declarations.

## Sample sparingly — low signal

These carry little value. Include only a little, only when genuinely representative; keep the combined boilerplate/infra/autogen share under ~{int(settings.boilerplate_share_max * 100)}%:
- **boilerplate** — DTOs/models that are mostly getters/setters/annotations, ORM entity scaffolding, dependency-injection wiring, config modules, constant tables, thin pass-through controllers/repositories that only delegate, one-line CRUD endpoints, and presentation/UI (view templates, styling, component markup, layout). Favour the logic behind the UI over the UI itself.
- **infra** — build/CI/deploy scripts.
- Framework/CMS glue — files that are mostly calls into a framework/CMS with little original logic of their own.

## Select files

- Cover the significant modules, but weight coverage toward where the real logic lives.
- Prefer medium-size files (100–500 LOC). Whole files preferred; partial extracts only for >800 LOC files.
- Tests: {int(settings.test_share_min * 100)}–{int(settings.test_share_max * 100)}% of total LOC (tag them `test`).
- Stratify by module, layer, and age (git log) when possible.
- **Quality over quantity**: do NOT fill the budget with __init__.py, thin wrappers, boilerplate, or files under 30 LOC just to reach the LOC target. Under-sampling is acceptable ONLY when the repository genuinely lacks ~{settings.target_loc} LOC of substantive code. If it clearly has enough real source, keep saving substantive files until at least {settings.target_loc - settings.loc_tolerance} LOC — ordinary production logic counts even when unremarkable, but padding with boilerplate to hit the number does not.
{language_section}
## Exclude entirely

- Auto-generated files: migrations, protobuf/OpenAPI stubs, ORM auto-gen, any "DO NOT EDIT" files.
- Vendored/third-party code, lock files, minified bundles, binary assets, pure-constant or fixture files.
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
- Real over polished — include ordinary, even messy, production logic; just make sure it IS logic, not boilerplate.
- Monorepo → sample from each major project, focusing on its logic-bearing parts.
- When two candidates are equally representative, prefer the one carrying more substantive logic.
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
    primary_language: str | None = None
    primary_hard: bool = False   # True only with --primary-language override


def _primary_loc(ctx: _ToolCtx) -> int:
    if not ctx.primary_language:
        return 0
    # JS and TS count as one bucket — see languages.same_language_bucket.
    return sum(
        f.loc_taken for f in ctx.saved_files
        if same_language_bucket(f.language, ctx.primary_language)
    )


def _substance_loc(ctx: _ToolCtx) -> int:
    """LOC saved into the high-signal layers (real business logic & algorithms)."""
    return sum(f.loc_taken for f in ctx.saved_files if f.layer in SUBSTANCE_LAYERS)


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
    progress = f"[LOC:{saved_loc}/{ctx.settings.target_loc} files:{len(ctx.saved_files)} need:{remaining}"
    if saved_loc:
        # floor, not round: 59.6% must not read as a passing-looking 60%
        logic_pct = int(_substance_loc(ctx) / saved_loc * 100)
        progress += (
            f" | logic: {logic_pct}% — goal {int(ctx.settings.logic_share_min * 100)}%"
        )
    if ctx.primary_language:
        ploc = _primary_loc(ctx)
        # floor, not round: 19.6% must never display as a passing-looking 20%
        pct = int(ploc / saved_loc * 100) if saved_loc else 0
        kind = "min" if ctx.primary_hard else "goal"
        progress += (
            f" | {ctx.primary_language}: {ploc} LOC ({pct}%)"
            f" — {kind} {int(ctx.settings.primary_share_min * 100)}%"
        )
    output += f"\n{progress}]"
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

    norm = os.path.normpath(path)
    already = next((f for f in ctx.saved_files if os.path.normpath(f.path) == norm), None)
    if already:
        # Re-saving inflates the LOC counter without adding content (and a
        # different line range would overwrite the earlier chunk on disk).
        return {
            "saved": False,
            "error": (
                f"Already saved {path} ({already.loc_taken} LOC, rank {already.rank}). "
                f"Re-saving adds nothing — pick a DIFFERENT file."
            ),
        }

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

    language = lang_from_path(path)
    loc = count_code_lines(chunk, language)
    if loc == 0:
        return {
            "saved": False,
            "error": (
                f"File {path} has 0 substantive LOC (empty, binary, or only blanks/comments). "
                f"Pick a different file with actual code."
            ),
        }
    current_total = sum(f.loc_taken for f in ctx.saved_files)
    cap = ctx.settings.max_total_loc
    if current_total + loc > cap:
        if current_total >= ctx.settings.target_loc:
            share_failing = (
                ctx.primary_language
                and ctx.primary_hard
                and _primary_loc(ctx) < ctx.settings.primary_share_min * current_total
            )
            caveat = (
                f" WARNING: your {ctx.primary_language} share is still below the "
                f"{int(ctx.settings.primary_share_min * 100)}% minimum — any remaining "
                f"saves must be small {ctx.primary_language} files."
                if share_failing else ""
            )
            return {
                "saved": False,
                "error": (
                    f"LOC budget exhausted ({current_total}/{cap} max). You have enough — "
                    f"do not save more files. Call write_summary and finish now.{caveat}"
                ),
            }
        return {
            "saved": False,
            "error": (
                f"Saving this file ({loc} LOC) would exceed the {cap} LOC hard cap "
                f"(currently {current_total}). Save a smaller file, or a partial range "
                f"via start_line/end_line, to land near {ctx.settings.target_loc} total."
            ),
        }

    dest = ctx.deliverable_dir / "samples" / path
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text("".join(chunk), encoding="utf-8")

    rank = len(ctx.saved_files) + 1
    running_total = current_total + loc

    ctx.saved_files.append(AgentSavedFile(
        path=path,
        layer=layer,
        loc_taken=loc,
        is_partial=is_partial,
        rank=rank,
        language=language,
    ))

    result = {"saved": True, "loc": loc, "running_total_loc": running_total,
              "language": language or "unknown"}
    if running_total:
        # share of the sample so far that is substantive logic (floor to 2dp)
        result["logic_share"] = int(_substance_loc(ctx) / running_total * 100) / 100
    if ctx.primary_language and running_total:
        # floor to 2dp: 0.196 must not display as a passing-looking 0.2
        result["primary_share"] = int(_primary_loc(ctx) / running_total * 100) / 100
    return result


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
        if resp.status_code in (402, 403):
            # OpenRouter returns these when the key is blocked or its spending
            # limit is exhausted — every subsequent call will fail the same way.
            raise AuthError(
                f"HTTP {resp.status_code}: OpenRouter key blocked or spending limit reached"
            )
        if resp.status_code in (429, 500, 502, 503):
            wait = 2 ** attempt
            logger.warning(f"HTTP {resp.status_code}, retrying in {wait}s")
            await asyncio.sleep(wait)
            continue

        resp.raise_for_status()
        try:
            return resp.json()
        except (json.JSONDecodeError, ValueError):
            # HTTP 200 but the body is not JSON (HTML error page, empty body,
            # or an SSE/stream). Without logging the raw body this surfaces only
            # as an opaque "Expecting value: line N column 1" parser error in
            # errors.jsonl, with no way to tell what the API actually returned.
            # Log the full body so it can be investigated, then retry like a
            # transient 5xx. These responses are rare (a handful per run), so
            # the full dump does not flood the log. A generous cap is kept only
            # as a guard against a pathological multi-MB body, and any trim is
            # made explicit (body_len + dropped count) so nothing is lost
            # silently.
            wait = 2 ** attempt
            body = resp.text or ""
            cap = 20000
            shown = body if len(body) <= cap else body[:cap] + f"…[+{len(body) - cap} more chars]"
            ctype = resp.headers.get("content-type", "")
            rid = (resp.headers.get("x-request-id")
                   or resp.headers.get("x-openrouter-request-id", ""))
            logger.warning(
                f"OpenRouter returned non-JSON body (status={resp.status_code} "
                f"content-type={ctype!r} request-id={rid!r} body_len={len(body)}), "
                f"retrying in {wait}s; body={shown!r}"
            )
            await asyncio.sleep(wait)
            continue

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

    stats = await asyncio.to_thread(
        compute_language_stats,
        repo_path,
        use_scc=settings.lang_scan_use_scc,
        timeout=settings.lang_scan_timeout,
    )
    primary_forced = bool(settings.primary_language_override.strip())
    if primary_forced:
        primary = settings.primary_language_override.strip()
        if stats.counts.get(primary, 0) == 0:
            logger.warning(
                f"[{repo_name}] forced primary language '{primary}' not detected in "
                f"the repo scan ({stats.source}) — enforcing anyway"
            )
    else:
        # Focus on the largest trackable REAL-CODE language: markup/style/data
        # pluralities (CSS, SVG, JSON, ...) and languages we cannot tag by
        # path are skipped. Soft goal — prompts and nudges, no rejection.
        primary = pick_code_primary(stats) or ""
        raw = stats.primary or ""
        if primary and raw and primary != raw:
            logger.info(
                f"[{repo_name}] plurality language '{raw}' is markup/data or "
                f"untrackable — focusing on '{primary}' instead"
            )
    ctx.primary_language = primary or None
    ctx.primary_hard = primary_forced
    logger.info(
        f"[{repo_name}] language scan ({stats.source}): primary={primary or 'n/a'}"
        f"{' (forced)' if primary_forced else ''}, "
        f"{len(stats.counts)} languages, {stats.total:,} lines"
    )

    if primary:
        share_min_pct = int(settings.primary_share_min * 100)
        pdisp = primary_display(primary)
        exts = ", ".join(primary_extensions(primary)) or "n/a"
        if not primary_forced and stats.primary and stats.primary != primary:
            from .languages import NON_CODE_LANGS
            if stats.primary in NON_CODE_LANGS:
                # Trackable markup plurality: keep a token presence so the
                # client-side "sample contains primary_language lines" check
                # stays satisfiable.
                plurality_note = (
                    f"(The repo's plurality language {stats.primary} is "
                    f"markup/style/data — focus the code sampling on {primary} "
                    f"instead, but still include 1-2 representative "
                    f"{stats.primary} files so that language is present.)\n"
                )
            else:
                # Untrackable language: we cannot tag its files, so do not
                # promise presence — just redirect the focus.
                plurality_note = (
                    f"(The repo's plurality language {stats.primary} cannot be "
                    f"tracked here — focus the code sampling on {primary}.)\n"
                )
        else:
            plurality_note = ""
        if primary_forced:
            requirement = (
                f"PRIMARY LANGUAGE: {pdisp} (file extensions: {exts}).\n"
                f"HARD REQUIREMENT: at least {share_min_pct}% of the LOC you save must "
                f"be {pdisp} files. Track this in the LOC progress shown after each "
                f"tool call."
            )
        else:
            requirement = (
                f"PRIMARY CODE LANGUAGE: {pdisp} (file extensions: {exts}).\n"
                f"{plurality_note}"
                f"Aim for at least {share_min_pct}% of saved LOC in {pdisp}; prefer "
                f"it over markup/style/data files. Track this in the LOC progress "
                f"shown after each tool call."
            )
        first_user_msg = (
            f"Repository has been cloned to: {repo_path}\n\n"
            f"Repository language distribution (share of code lines):\n"
            f"{format_distribution(stats)}\n\n"
            f"{requirement}\n\nBegin."
        )
    elif stats.counts:
        # Repo has a distribution but no trackable code language (e.g. pure
        # markup/template sites) — show it, no focus requirement.
        first_user_msg = (
            f"Repository has been cloned to: {repo_path}\n\n"
            f"Repository language distribution (share of code lines):\n"
            f"{format_distribution(stats)}\n\n"
            "No dominant real code language was detected — sample the most "
            "substantive hand-written files this repo has.\n\nBegin."
        )
    else:
        first_user_msg = (
            f"Repository has been cloned to: {repo_path}\n\n"
            "Language distribution could not be determined — identify the dominant "
            "real source language yourself and focus the sample on it.\n\nBegin."
        )

    messages: list[dict] = [
        {
            "role": "system",
            "content": _build_system_prompt(
                settings, has_primary=bool(primary), hard=primary_forced
            ),
        },
        {"role": "user", "content": first_user_msg},
    ]

    agent_log: list[dict] = []
    iterations = 0
    last_save_iteration = -1   # track when agent last called save_sample
    last_primary_nudge = -10   # rate-limit the language-requirement nudge
    last_logic_nudge = -10     # rate-limit the logic-share nudge

    for iteration in range(settings.agent_max_iterations):
        iterations = iteration + 1

        # Nudge if agent has been exploring without saving for too long,
        # over-collecting, or running out of iterations under target.
        saves_so_far = len(ctx.saved_files)
        saved_loc = sum(f.loc_taken for f in ctx.saved_files)
        turns_since_save = iteration - (last_save_iteration + 1)
        primary_loc = _primary_loc(ctx)
        if saves_so_far == 0 and iteration >= 3:
            nudge = (
                f"⚠️ URGENT: You have made {iteration} tool calls and saved 0 files. "
                f"You MUST call save_sample on your very next tool call. "
                f"Stop exploring — pick any good file you have already seen and save it NOW. "
                f"Target: {settings.target_loc} LOC total."
            )
            messages.append({"role": "user", "content": nudge})
            agent_log.append({"turn": iteration + 1, "nudge": nudge})
        elif (
            ctx.primary_language
            and saved_loc >= settings.target_loc // 2
            # soft mode stops pushing the goal once the sample is over target —
            # alternating "save more X" / "stop saving" would be contradictory
            and (ctx.primary_hard
                 or saved_loc < settings.target_loc + settings.loc_tolerance)
            and primary_loc < settings.primary_share_min * saved_loc
            and iteration - last_primary_nudge >= 3
        ):
            # Outranks the over-target nudge: finishing while below the
            # primary-language goal wastes the run (hard mode: guarantees
            # post-run rejection).
            last_primary_nudge = iteration
            share_min = settings.primary_share_min
            pct = int(primary_loc / saved_loc * 100) if saved_loc else 0
            # Saving primary LOC also grows the total, so the true need is
            # (min*saved - ploc) / (1 - min), not the naive deficit.
            need = max(1, math.ceil((share_min * saved_loc - primary_loc) / (1 - share_min)))
            cap_headroom = settings.max_total_loc - saved_loc
            if need > cap_headroom:
                if ctx.primary_hard:
                    # Even filling all remaining cap headroom with primary
                    # files cannot reach the minimum — further iterations are
                    # wasted; end the run, the retry starts with a clean budget.
                    note = (
                        f"language requirement unrecoverable: need {need} more "
                        f"{ctx.primary_language} LOC but only {cap_headroom} LOC of cap "
                        f"headroom remains — ending run early"
                    )
                    logger.warning(f"[{repo_name}] {note}")
                    agent_log.append({"turn": iteration + 1, "aborted": note})
                    break
                # Soft mode: the goal is out of reach — stop nudging about it
                # and let the run finish normally.
                last_primary_nudge = settings.agent_max_iterations
            else:
                exts = ", ".join(primary_extensions(ctx.primary_language)) or "n/a"
                headroom = (
                    f" The {settings.max_total_loc} LOC hard cap still has "
                    f"{cap_headroom} LOC of room even though you are at target."
                    if saved_loc >= settings.target_loc else ""
                )
                if ctx.primary_hard:
                    nudge = (
                        f"⚠️ LANGUAGE REQUIREMENT FAILING: only {primary_loc} LOC ({pct}%) "
                        f"of your sample is {ctx.primary_language} — the minimum is "
                        f"{int(share_min * 100)}%. You need ~{need} more LOC of "
                        f"hand-written {ctx.primary_language} files ({exts}). "
                        f"Save those NOW.{headroom} "
                        f"A sample below the minimum is rejected entirely."
                    )
                else:
                    nudge = (
                        f"⚠️ LANGUAGE FOCUS: only {primary_loc} LOC ({pct}%) of your "
                        f"sample is {ctx.primary_language} — aim for at least "
                        f"{int(share_min * 100)}%. Prefer hand-written "
                        f"{ctx.primary_language} files ({exts}) for your remaining "
                        f"saves (~{need} LOC would close the gap).{headroom}"
                    )
                messages.append({"role": "user", "content": nudge})
                agent_log.append({"turn": iteration + 1, "nudge": nudge})
        elif saved_loc >= settings.target_loc + settings.loc_tolerance and not (
            ctx.primary_hard
            and ctx.primary_language
            and primary_loc < settings.primary_share_min * saved_loc
        ):
            # Suppressed (hard mode only) while the primary-language minimum
            # is unmet — telling the agent to finish in that state would
            # guarantee rejection.
            nudge = (
                f"⚠️ You have {saved_loc} LOC — the target is {settings.target_loc}. "
                f"STOP saving files. Call write_summary and finish now."
            )
            messages.append({"role": "user", "content": nudge})
            agent_log.append({"turn": iteration + 1, "nudge": nudge})
        elif saves_so_far > 0 and turns_since_save >= 4 and saved_loc < settings.target_loc:
            remaining = max(0, settings.target_loc - saved_loc)
            lang_caveat = (
                f" Your {ctx.primary_language} share is below the "
                f"{int(settings.primary_share_min * 100)}% "
                f"{'minimum' if ctx.primary_hard else 'goal'} — prioritize "
                f"{ctx.primary_language} files."
                if ctx.primary_language
                and last_primary_nudge < settings.agent_max_iterations  # not abandoned
                and primary_loc < settings.primary_share_min * saved_loc else ""
            )
            nudge = (
                f"⚠️ You have saved {saves_so_far} files ({saved_loc} LOC) but haven't saved "
                f"anything in {turns_since_save} turns. You still need ~{remaining} more LOC."
                f"{lang_caveat} "
                f"Save more files now. If the repo has no substantive files left, do NOT "
                f"repeat already-saved files — call write_summary and finish instead."
            )
            messages.append({"role": "user", "content": nudge})
            agent_log.append({"turn": iteration + 1, "nudge": nudge})
        elif (
            iteration >= settings.agent_max_iterations - 5
            and saved_loc < settings.target_loc - settings.loc_tolerance
        ):
            lang_caveat = (
                f" Your {ctx.primary_language} share is below the "
                f"{int(settings.primary_share_min * 100)}% "
                f"{'minimum — those last saves must be' if ctx.primary_hard else 'goal — prefer'} "
                f"{ctx.primary_language} files."
                if ctx.primary_language
                and last_primary_nudge < settings.agent_max_iterations  # not abandoned
                and primary_loc < settings.primary_share_min * saved_loc else ""
            )
            nudge = (
                f"⚠️ Only {settings.agent_max_iterations - iteration} iterations left and you "
                f"have {saved_loc}/{settings.target_loc} LOC. Stop exploring — save your best "
                f"remaining candidate files NOW, then write_summary and finish.{lang_caveat}"
            )
            messages.append({"role": "user", "content": nudge})
            agent_log.append({"turn": iteration + 1, "nudge": nudge})
        elif (
            saved_loc >= settings.target_loc // 2
            and saved_loc < settings.target_loc + settings.loc_tolerance
            and _substance_loc(ctx) < settings.logic_share_min * saved_loc
            and iteration - last_logic_nudge >= 4
        ):
            # Lowest-priority steer: only when not urgent/over-target/stalled and
            # the language goal is not itself unmet — push the remaining budget
            # toward substantive logic instead of boilerplate.
            last_logic_nudge = iteration
            sub = _substance_loc(ctx)
            pct = int(sub / saved_loc * 100) if saved_loc else 0
            nudge = (
                f"⚠️ LOGIC SHARE LOW: only {pct}% of your sample is substantive logic "
                f"(business/algorithm/data/api) — aim for at least "
                f"{int(settings.logic_share_min * 100)}%. For your remaining saves, prefer files "
                f"with real domain logic and algorithms; avoid boilerplate (DTO/ORM scaffolding, "
                f"DI/config wiring, thin wrappers, presentation/UI)."
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

        # DIAG: per-turn API outcome — finish_reason, token usage, tool-call count.
        usage = data.get("usage") or {}
        _tc = message.get("tool_calls") or []
        logger.info(
            f"[{repo_name}] DIAG turn {iteration + 1}: finish_reason={finish_reason!r} "
            f"tool_calls={len(_tc)} "
            f"tokens(in/out)={usage.get('prompt_tokens','?')}/{usage.get('completion_tokens','?')} "
            f"content_len={len(message.get('content') or '')}"
        )

        # Append assistant message to history
        messages.append(message)
        agent_log.append({
            "turn": iteration + 1,
            "assistant": message,
            "finish_reason": finish_reason,
            "usage": usage,
        })

        if finish_reason == "length":
            logger.warning(f"[{repo_name}] agent hit context length limit at iteration {iteration + 1}")
            break

        tool_calls = message.get("tool_calls") or []

        if not tool_calls:
            # DIAG: model returned prose instead of a tool call — log what it said,
            # this is the usual reason a repo ends with 0 saved files.
            content = (message.get("content") or "").strip()
            logger.warning(
                f"[{repo_name}] agent stopped calling tools at iteration {iteration + 1} "
                f"(finish_reason={finish_reason!r}); model said: {content[:500]!r}"
            )
            agent_log.append({"turn": iteration + 1, "stopped_no_tool_calls": content[:2000]})
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
            elif tool_name == "save_sample":
                # DIAG: a save that did NOT add a file — log the rejection reason.
                # When a repo ends at 0 LOC this shows exactly why every save failed
                # (file not found / already saved / 0 substantive LOC / over cap).
                reason = result_str
                try:
                    reason = json.loads(result_str).get("error", result_str)
                except Exception:
                    pass
                logger.warning(
                    f"[{repo_name}] DIAG save_sample REJECTED "
                    f"path={tool_args.get('path')!r}: {str(reason)[:300]}"
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
    # DIAG: when nothing was saved, summarize the run so the empty result is
    # explainable from the log alone (no need to dig into agent_log.json).
    if total_loc == 0:
        save_attempts = sum(1 for e in agent_log if e.get("tool") == "save_sample")
        bash_attempts = sum(1 for e in agent_log if e.get("tool") == "bash")
        stopped = next((e["stopped_no_tool_calls"] for e in agent_log
                        if "stopped_no_tool_calls" in e), None)
        logger.error(
            f"[{repo_name}] DIAG ZERO-LOC: iterations={iterations} "
            f"save_sample_attempts={save_attempts} bash_calls={ctx.bash_calls} "
            f"bash_log_entries={bash_attempts} saved_files=0"
            + (f" | last model prose: {stopped[:300]!r}" if stopped else "")
        )

    test_loc = sum(f.loc_taken for f in ctx.saved_files if f.layer == "test")
    test_pct = test_loc / total_loc * 100 if total_loc else 0
    primary_pct = _primary_loc(ctx) / total_loc * 100 if total_loc else 0
    logger.info(
        f"[{repo_name}] done: {total_loc} LOC, {len(ctx.saved_files)} files, "
        f"test share: {test_pct:.0f}%, "
        + (f"primary {primary}: {primary_pct:.0f}%, " if primary else "")
        + f"iterations: {iterations}, bash calls: {ctx.bash_calls}"
    )

    repo_lang_distribution = (
        {lang: round(n / stats.total, 4) for lang, n in stats.counts.items()}
        if stats.total else {}
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
        primary_language=primary,
        repo_lang_distribution=repo_lang_distribution,
        lang_stats_source=stats.source,
        primary_forced=primary_forced,
    )


def _write_agent_log(deliverable_dir: Path, agent_log: list[dict], ctx: _ToolCtx) -> None:
    deliverable_dir.mkdir(parents=True, exist_ok=True)
    total_loc = sum(f.loc_taken for f in ctx.saved_files)
    sample_lang_distribution: dict[str, int] = {}
    for f in ctx.saved_files:
        key = f.language or "Unknown"
        sample_lang_distribution[key] = sample_lang_distribution.get(key, 0) + f.loc_taken
    log = {
        "sampled_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "stats": {
            "files_saved": len(ctx.saved_files),
            "total_loc": total_loc,
            "bash_calls": ctx.bash_calls,
            "iterations": len({e["turn"] for e in agent_log}),
            "primary_language": ctx.primary_language or "",
            "primary_loc": _primary_loc(ctx),
            "sample_lang_distribution": sample_lang_distribution,
        },
        "saved_files": [
            {
                "rank": f.rank,
                "path": f.path,
                "layer": f.layer,
                "loc_taken": f.loc_taken,
                "is_partial": f.is_partial,
                "language": f.language,
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
