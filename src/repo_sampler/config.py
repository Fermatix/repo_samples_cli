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

    # Agent loop limits
    agent_max_iterations: int = 50
    agent_bash_timeout: int = 30       # seconds per bash call
    agent_bash_output_limit: int = 8000  # chars, truncated if exceeded

    # Concurrency
    clone_workers: int = 10

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
