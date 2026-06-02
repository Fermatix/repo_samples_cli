from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_ignore_empty=True)

    openrouter_api_key: str = ""
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_model: str = "deepseek/deepseek-v4-pro"

    target_loc: int = 5000
    loc_tolerance: int = 300
    target_file_count: tuple[int, int] = (15, 50)

    min_file_loc: int = 15
    max_file_loc: int = 3000
    partial_extract_min_loc: int = 50

    max_repo_files: int = 99999
    test_share_min: float = 0.15
    test_share_max: float = 0.40

    min_score: int = 0

    clone_workers: int = 10
    llm_workers: int = 20

    clone_dir: str = "/tmp/repo-sampler/clones"
    output_dir: str = "./output"
    output_format: str = "jsonl"
