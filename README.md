# Repository Samples CLI

Collect code samples from Git repositories. An agent clones each repository,
explores its structure and selects about 5,000 lines of code (hard cap 6,500).
The samples are then anonymized and packed into one archive for upload.

Sampling targets hand-written code: business and domain logic, non-trivial
algorithms and data handling. It avoids boilerplate such as DTO and ORM
skeletons, DI and configuration wiring, thin wrappers, presentation markup and
generated code. After each step the agent sees its logic share (the share of
lines in the business, algorithm, data and api layers) and keeps it above the
target level, so the sample carries as little noise as possible.

## Installation

Use macOS or Linux. Git and `uv` are required. Anonymization also needs the
Claude Code CLI (`claude`), installed and signed in.

**macOS**, with [Homebrew](https://brew.sh/) installed:

```bash
brew install git uv
```

**Ubuntu / Debian**:

```bash
sudo apt-get update
sudo apt-get install -y git curl
curl -LsSf https://astral.sh/uv/install.sh | sh
. "$HOME/.local/bin/env"
```

**Then, on either platform**:

```bash
git clone https://github.com/Fermatix/repo_samples_cli.git
cd repo_samples_cli
uv sync --locked
```

Run subsequent commands from this directory. `uv run` uses the project
environment; activation is unnecessary.

## Quickstart

Upload your repository metadata to the partner portal first. Samples are
matched against it, so samples for repositories without metadata cannot be
matched.

Sampling sends repository code to an LLM provider through OpenRouter. The
`anonymize` step sends every sample to Anthropic through Claude Code. Run both
only on repositories you are allowed to share this way.

Create `.env` in the project directory from the example and set your
OpenRouter key:

```bash
cp .env.example .env
```

```text
OPENROUTER_API_KEY=sk-or-...
```

Create `repos.txt` with one repository URL per line. This Quickstart assumes
your SSH key is configured and has read access to the repositories:

```text
# Blank lines and lines starting with # are ignored

git@git.example.com:group/service-api.git
git@git.example.com:group/mobile-app.git
```

`repos.example.txt` contains the same kind of list. Then run the three steps:

```bash
uv run repo-sampler run repos.txt --url-scheme ssh
uv run repo-sampler anonymize ./output
uv run repo-sampler pack ./output --archive repo_samples.zip
```

`run` collects the samples into `./output`. `anonymize` replaces names and
contact details in them. `pack` builds `repo_samples.zip`.

Upload `repo_samples.zip` (up to 1 GB) in the Samples tab of the partner
portal. The archive contains the samples in `samples/` and a `samples_index.csv`.
The index is built from the saved repository identities, so it holds real
identifying data. Its columns are `folder`, `repo_name`, `repo_org`, `repo_url`,
`first_commit_hash`, `early_commit_hashes`, `commit_minhash`, `sample_loc`,
`sample_files`, `sampled_at`, `head_commit_sha` and `sampler_version`.
`repo_url` is the real repository URL, or the absolute local path for a local
entry. Apart from that index, no files from `output_meta` are included.

---

## Optional reference

### Access to repositories

Cloning never asks for a login or password. If access is missing, the
repository goes straight to `errors.jsonl` with the error. Set up access
before the run, in one of these ways:

- SSH key (recommended). Make sure the key is added to GitHub or GitLab and run
  with `--url-scheme ssh`. Then `https://` URLs in `repos.txt` are cloned over
  SSH:

  ```bash
  uv run repo-sampler run repos.txt --url-scheme ssh
  ```

  If your GitLab serves SSH on a non-standard port, add `--ssh-port`:

  ```bash
  uv run repo-sampler run repos.txt --url-scheme ssh --ssh-port 10022
  ```

  A key with a passphrase must be loaded into `ssh-agent` (`ssh-add`), because
  the tool cannot ask for the passphrase.

- HTTPS. Sign in from the terminal with `gh auth login` (GitHub) or
  `glab auth login --hostname <your-gitlab>` (GitLab). Git then uses the saved
  credentials.

To check access before a run, this command must finish without any prompt:

```bash
git ls-remote git@<your-git-server>:<group>/<repository>.git
```

### Repository list

`repos.txt` holds one URL or absolute local path per line. Lines starting with
`#` are ignored and duplicate lines are dropped automatically:

```text
https://github.com/example-org/mobile-app.git
https://gitlab.example.com/group/subgroup/service-api.git
/home/user/repos/local-repo
# this line is ignored
```

### Reruns

Repositories that are already processed are skipped on a rerun. Repositories
whose sample came out empty (0 lines) or failed are retried on the next run, so
repeating the same command is enough.

If the OpenRouter key is blocked or reaches its spending limit (HTTP 402 or
403), the whole run stops at once. Raise the limit and run the command again.

### Anonymization and `<output>_meta`

Collection uses an inexpensive model and does not guarantee anonymity. The
`anonymize` command goes through every sample folder with a Claude agent. Only
the `claude` CLI runs on your machine; the sample contents are sent to Anthropic
through it.
It replaces mentions of companies, brands, products, legal entities, people,
names, phone numbers, emails and websites with neutral placeholders. Code and
its structure stay unchanged, and Cyrillic text is left as it is.

```bash
uv run repo-sampler anonymize ./output
```

It uses your current Claude Code sign-in, so the `claude` CLI must be installed
and signed in. No separate API key is needed.

For every folder the command saves an audit of the changes, made with `diff`
and without an LLM:

- `anonymization.diff` lists every changed section (before and after).
- `anonymization_report.json` summarizes the changes per file (number of edits,
  cost).

Everything that must not leave your company is written to a separate folder,
`<output>_meta`, next to the output folder. This covers `agent_log.json` with
raw, non-anonymized code fragments, the `samples.jsonl` manifest and
`errors.jsonl` with real repository addresses, run logs and the anonymization
audit. Only clean anonymized samples remain in `output`. A rerun of `run` or
`anonymize` also moves these files out of older exports, where they still sit
in the root of `output` or inside the sample folders.

The `<output>_meta` folder stays with your company and is not part of the
archive. It is your own audit record of sampling and anonymization:
`agent_log.json`, `anonymization.diff` and `anonymization_report.json` let you
check what was sampled and what was changed. Do not delete it.

Old runs without `repo_identity.json` can still be packed. `pack` warns about
empty identifiers, and the partner portal lists such rows under "Unmatched".

### Output layout

```text
output/
└── gitlab.com__owner__repo/
    ├── repo_summary.md              # repository overview
    └── samples/                     # selected files (verbatim)
        └── <original/path/to/file>

output_meta/                         # internal, not part of the archive
├── samples.jsonl                    # metadata of all repositories (real URLs)
├── errors.jsonl                     # repositories with errors (clone/agent)
├── run.log                          # full run log
└── gitlab.com__owner__repo/
    ├── agent_log.json               # collection agent actions (raw previews)
    ├── repo_identity.json           # identifiers for matching in the partner portal
    ├── anonymization.diff           # after anonymization: what changed
    └── anonymization_report.json    # after anonymization: summary
```

### Additional options

```bash
# Reprocess repositories that are already collected
uv run repo-sampler run repos.txt --force

# Clone only, without the LLM (checks that repositories are reachable)
uv run repo-sampler run repos.txt --dry-run

# Number of parallel clone workers (default 10)
uv run repo-sampler run repos.txt --workers 20

# Full run for one repository, printed to the console
uv run repo-sampler show-sample https://github.com/example-org/mobile-app

# Anonymization: parallelism, another model, reprocessing
uv run repo-sampler anonymize ./output --workers 5
uv run repo-sampler anonymize ./output --force
```

`run` also accepts `--output`, `--format jsonl|parquet`, `--keep-clones` and
`--primary-language` (an scc language name applied to all repositories in the
file). `anonymize` also accepts `--model`, `--effort low|medium|high|max` and
`--meta-dir`. For all options: `uv run repo-sampler <command> --help`.

### Sampling backends

By default the collection agent calls OpenRouter. The `claude` and `codex` backends call the model through a CLI you have already signed in to, so no OpenRouter key is needed:

```bash
uv run repo-sampler run repos.txt --backend openrouter
uv run repo-sampler run repos.txt --backend claude
uv run repo-sampler run repos.txt --backend codex
```

The same option works for `show-sample`. Without it, the `AGENT_BACKEND` setting decides. The tool checks the backend before cloning anything: it needs `OPENROUTER_API_KEY` for `openrouter`, and the `claude` or `codex` binary on `PATH` for the others.

With claude and codex, code is sent to Anthropic or OpenAI under your account. Only the CLI itself runs locally, not the model.

The Codex backend ignores your Codex config, rules and shell tool on purpose.

| Variable | Default | Description |
|---|---|---|
| `AGENT_BACKEND` | `openrouter` | `openrouter`, `claude` or `codex` |
| `AGENT_CLI_MODEL` | empty | Model for `claude` and `codex`. Empty means `ANONYMIZER_MODEL` for `claude` and Codex's built-in default for `codex` |
| `AGENT_CLI_TIMEOUT` | `300` | Timeout of one CLI turn, in seconds |
| `AGENT_CLI_CONCURRENCY` | `4` | Concurrent CLI processes for the whole run |

### Settings (`.env`)

| Variable | Default | Description |
|---|---|---|
| `OPENROUTER_API_KEY` | none | Required only for the `openrouter` backend. OpenRouter key |
| `AGENT_MODEL` | `deepseek/deepseek-v4-flash` | Model for the collection agent |
| `TARGET_LOC` | `5000` | Target sample size (lines of code) |
| `MAX_TOTAL_LOC` | `6500` | Hard cap on sample size; saves above it are rejected |
| `LOGIC_SHARE_MIN` | `0.6` | Target share of substantive logic (business/algorithm/data/api layers). A soft target: prompt and nudge, no hard rejection |
| `BOILERPLATE_SHARE_MAX` | `0.2` | Soft cap on the share of low-signal layers (boilerplate/infra/autogen), hinted in the prompt |
| `CLONE_WORKERS` | `10` | Parallel clones |
| `CLONE_TIMEOUT` | `900` | Timeout of one `git clone`, in seconds (large repositories may need more) |
| `CLONE_DIR` | `/tmp/repo-sampler/clones` | Directory for clones |
| `ANONYMIZER_MODEL` | `claude-haiku-4-5` | Model of the anonymization agent |
| `ANONYMIZER_WORKERS` | `12` | Parallel anonymization agents |
| `ANONYMIZER_TIMEOUT` | `900` | Timeout per folder, in seconds |
| `ANONYMIZER_MAX_BUDGET_USD` | `0` | Cost limit per folder in USD (0 = no limit) |
