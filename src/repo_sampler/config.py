from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_ignore_empty=True)

    openrouter_api_key: str = ""
    openrouter_base_url: str = "https://openrouter.ai/api/v1"

    # Model for the agentic sampling loop
    agent_model: str = "deepseek/deepseek-v4-flash"

    # Sample targets (injected into agent system prompt as instructions)
    target_loc: int = 5000
    loc_tolerance: int = 300
    # Hard ceiling enforced by save_sample itself: a save that would push the
    # running total above this is rejected, the agent is told to finish.
    max_total_loc: int = 6500
    test_share_min: float = 0.1
    test_share_max: float = 0.2

    # Sample quality: bias the budget toward substantive, hand-written logic
    # (business/domain rules, algorithms, data processing, real API logic) and
    # away from boilerplate (DTO/ORM scaffolding, DI/config wiring, thin
    # wrappers, presentation/markup, generated filler). The agent sees its
    # running "logic share" after every tool call and is steered toward this
    # goal; it is a soft target (prompt + nudge), never a hard rejection.
    logic_share_min: float = 0.6
    # Soft ceiling on the combined share of low-signal layers
    # (boilerplate/infra/autogen). Surfaced in the prompt as guidance.
    boilerplate_share_max: float = 0.2

    # Language coverage: the sample must contain at least this share of LOC
    # in the repo's primary language (the plurality language by code lines,
    # scc-compatible naming). ~1000 LOC of a 5000-LOC sample — clearly "well
    # represented", yet achievable even when the primary is markup/data or
    # mostly generated.
    primary_share_min: float = 0.20
    # Force a specific primary language (scc name, e.g. "JavaScript"); set
    # from the --primary-language CLI option. Empty = auto-detect.
    primary_language_override: str = ""
    # Prefer the scc binary for the repo language scan when installed
    # (exact parity with the metadata pipeline); tests disable it.
    lang_scan_use_scc: bool = True
    lang_scan_timeout: int = 120       # seconds for the scc subprocess

    # Agent loop limits
    agent_max_iterations: int = 50
    agent_bash_timeout: int = 30       # seconds per bash call
    agent_bash_output_limit: int = 8000  # chars, truncated if exceeded

    # Concurrency
    clone_workers: int = 10
    # Multi-million-LOC repos take well over 2 minutes even with
    # --depth=1 --filter=blob:none, so the default is generous.
    clone_timeout: int = 900           # seconds per git clone

    # Anonymization (local Claude Code agent)
    anonymizer_model: str = "claude-haiku-4-5"
    anonymizer_workers: int = 12            # parallel `claude -p` subprocesses
    anonymizer_timeout: int = 900          # seconds per directory
    anonymizer_permission_mode: str = "acceptEdits"
    anonymizer_effort: str = "low"        # thinking effort: low|medium|high|max ("" = model default)
    anonymizer_max_budget_usd: float = 0.0  # 0 = no cap; pass --max-budget-usd only if > 0

    # Paths
    clone_dir: str = "/tmp/repo-sampler/clones"
    output_dir: str = "./output"
    output_format: str = "jsonl"
