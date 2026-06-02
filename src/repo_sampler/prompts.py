TREE_RANKING_PROMPT = """\
You are a senior software engineer building a code quality sample \
from a {language} repository for external assessment.

REPOSITORY CONTEXT:
- Primary language: {language}
- Language distribution: {lang_share}
- Build system: {build_system}
- Total files: {total_count} ({test_count} test, {prod_count} production)
{monorepo_note}

FILE LIST (LOC = non-blank non-comment lines; [TEST] = test file; [OLD] = not modified in 6+ months):
{file_list}

TASK: Return ALL files ranked by value for assessing hand-written code quality.
Rank 1 = highest value. Include every file from the list — do not omit any.

RANKING TIERS — rank in this order:

TIER 1 — RANK HIGHEST (hand-written logic that reveals developer skill):
  • Business logic: domain rules, algorithms, workflows, state machines
  • Data layer: ORM models, schemas, DB queries, serializers, validators
  • API layer: request handlers, controllers, route definitions, RPC endpoints
  • Tests: unit tests, integration tests, e2e tests — rank proportionally
    (aim for 20-35% of the top selection by LOC)

TIER 2 — RANK MIDDLE (useful context, lower signal):
  • Utilities and shared helpers (formatting, parsing, decorators)
  • Service/client adapters written by hand (HTTP clients, queue consumers)
  • DI / dependency wiring written by hand

TIER 3 — RANK LOWEST (skip-worthy; rank at the very bottom):
  • Auto-generated files of any kind (migrations, protobuf stubs, OpenAPI clients,
    gRPC generated code, ORM auto-migrations, code-gen output)
  • Configuration files (settings.py, config.py, constants.py, env loaders,
    Docker/k8s configs embedded as Python, feature flags definitions)
  • Infrastructure / deployment glue (CI scripts, Makefiles converted to Python,
    health-check endpoints, WSGI/ASGI entry points with no logic)
  • Boilerplate with no meaningful logic (__init__.py with only imports,
    thin wrappers that just re-export, empty base classes)
  • Fixtures, seed data, test helpers with no assertions

LAYER TAGS (assign one per file):
  "business" — domain logic, algorithms, rules, state machines
  "data"     — models, schemas, DB access, serialization, validation
  "api"      — handlers, controllers, routes, RPC endpoints
  "util"     — shared helpers, formatting, parsing, decorators
  "test"     — any test file (unit, integration, e2e)
  "infra"    — config, deployment, clients, adapters, glue code
  "autogen"  — auto-generated or mostly auto-generated file

Return ONLY a JSON array of ALL files, most important first. No explanation, no markdown fences.

Format:
[
  {{"f": "path/to/file.py", "l": "business"}},
  {{"f": "tests/test_core.py", "l": "test"}},
  {{"f": "alembic/versions/0001_init.py", "l": "autogen"}}
]
"""

FILE_EXTRACTION_PROMPT = """\
You are reviewing a single {language} source file to select the most representative \
code excerpt for an external quality assessment.

FILE: {path}
TOTAL LINES IN FILE: {total_lines}

FILE CONTENT:
{content}

Select ONE continuous block of lines that best represents the quality and typical coding \
style of this file. Choose the section with the most meaningful logic — core algorithms, \
business rules, key abstractions, or representative implementation patterns. \
Avoid import blocks, configuration constants, and trivial boilerplate.

Target block size: approximately {target_loc} non-blank non-comment lines.
The block must be between {min_lines} and {max_lines} total lines (inclusive).

Return ONLY a JSON object with 1-indexed line numbers, no explanation:
{{"start": 42, "end": 89}}
"""

REPO_SUMMARY_PROMPT = """\
Write a concise `repo_summary.md` for a code sample deliverable.
Keep it to roughly one page. Use plain Markdown.

REPOSITORY INFORMATION:
- Name: {repo_name}
- URL: {repo_url}
- Primary language: {language} ({lang_share})
- Build system: {build_system}
- Top-level directories: {top_dirs}
- Commit analyzed: {commit_sha}

WHAT WAS SAMPLED:
- Files selected: {file_count} files, {total_loc} LOC
- Layer breakdown: {layer_breakdown}
- Test share: {test_share_pct}%
- File list (path, layer, LOC):
{file_list}

NOTES FROM ANALYSIS:
{analysis_notes}

ANONYMIZATION RULES — STRICTLY ENFORCED:
- Do NOT mention any company names, organisation names, brand names, or product names.
- Do NOT mention any end-user product or service (app name, platform name, SaaS name, etc.).
- Do NOT mention any geographic or national identifiers: country names, city names,
  human languages (e.g. "Russian", "English"), regional markets, or anything that could
  hint at the country or region where the software was developed or deployed.
- Do NOT reproduce or paraphrase repository-specific text that would violate the above
  (e.g. business domain names visible in package paths, module names, or string literals).
- Describe the codebase only in terms of technical purpose, architecture, and patterns.
  Use generic domain language: "user management", "subscription billing",
  "REST API backend", "authentication module", "content delivery service", etc.
- If you are unsure whether something is a brand, company, product, or geographic
  identifier — omit it entirely.

The document must contain exactly these four sections:

## Repository overview
One paragraph: what the repo does, primary language, build system, main framework(s).
Describe only technical purpose — no product names, no company names, no geography.

## Repository structure
Top-level directory tree (2-3 levels deep) with one-line description of each directory.

## What was sampled
Brief narrative: which parts of the codebase are covered, which layers, languages,
tests vs production code, roughly in what proportions. No exhaustive file lists.

## Notes
Anything the reviewer should know upfront: monorepo structure, deprecated areas,
files skipped due to size, areas under-represented and why.

Write in a neutral, factual tone. Do not praise or criticize the code.
"""
