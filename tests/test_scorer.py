import pytest
import respx
import httpx

from repo_sampler.scorer import AuthError, rank_tree, _call_llm, extract_chunk
from repo_sampler.config import Settings
from repo_sampler.analyzer import RepoInventory, FileInfo
from pathlib import Path


def _make_inventory() -> RepoInventory:
    return RepoInventory(
        repo_path=Path("/tmp/fake"),
        language="python",
        top_dirs=["src", "tests"],
        lang_share={"python": 1.0},
        test_file_count=1,
        prod_file_count=1,
        has_multiple_services=False,
        build_system="uv/poetry",
        files=[
            FileInfo(
                path="src/engine.py",
                loc=200,
                raw_lines=220,
                is_test=False,
                days_since_modified=10,
                top_level_dir="src",
            ),
            FileInfo(
                path="tests/test_engine.py",
                loc=80,
                raw_lines=90,
                is_test=True,
                days_since_modified=5,
                top_level_dir="tests",
            ),
        ],
    )


@pytest.mark.asyncio
async def test_rank_tree_parses_valid_json():
    settings = Settings(openrouter_api_key="test-key")
    inventory = _make_inventory()

    valid_response = '[{"f": "src/engine.py", "l": "business"}, {"f": "tests/test_engine.py", "l": "test"}]'

    with respx.mock:
        respx.post("https://openrouter.ai/api/v1/chat/completions").mock(
            return_value=httpx.Response(200, json={
                "choices": [{"message": {"content": valid_response}}]
            })
        )
        async with httpx.AsyncClient() as client:
            ranked = await rank_tree(inventory, settings, client)

    assert ranked[0].path == "src/engine.py"
    assert ranked[0].layer == "business"
    assert ranked[1].path == "tests/test_engine.py"
    assert ranked[1].is_test is True


@pytest.mark.asyncio
async def test_rank_tree_handles_json_fence():
    settings = Settings(openrouter_api_key="test-key")
    inventory = _make_inventory()

    fenced_response = '```json\n[{"f": "src/engine.py", "l": "business"}]\n```'

    with respx.mock:
        respx.post("https://openrouter.ai/api/v1/chat/completions").mock(
            return_value=httpx.Response(200, json={
                "choices": [{"message": {"content": fenced_response}}]
            })
        )
        async with httpx.AsyncClient() as client:
            ranked = await rank_tree(inventory, settings, client)

    assert ranked[0].path == "src/engine.py"


@pytest.mark.asyncio
async def test_rank_tree_appends_omitted_files():
    """Files the LLM omits should still appear at the end of the ranked list."""
    settings = Settings(openrouter_api_key="test-key")
    inventory = _make_inventory()

    # LLM returns only engine.py, omits test_engine.py
    partial_response = '[{"f": "src/engine.py", "l": "business"}]'

    with respx.mock:
        respx.post("https://openrouter.ai/api/v1/chat/completions").mock(
            return_value=httpx.Response(200, json={
                "choices": [{"message": {"content": partial_response}}]
            })
        )
        async with httpx.AsyncClient() as client:
            ranked = await rank_tree(inventory, settings, client)

    paths = [r.path for r in ranked]
    assert "src/engine.py" in paths
    assert "tests/test_engine.py" in paths
    assert paths.index("src/engine.py") < paths.index("tests/test_engine.py")


@pytest.mark.asyncio
async def test_rank_tree_handles_null_content():
    settings = Settings(openrouter_api_key="test-key")
    inventory = _make_inventory()

    with respx.mock:
        respx.post("https://openrouter.ai/api/v1/chat/completions").mock(
            return_value=httpx.Response(200, json={
                "choices": [{"message": {"content": None}}]
            })
        )
        async with httpx.AsyncClient() as client:
            ranked = await rank_tree(inventory, settings, client)

    # Falls back: all inventory files appended
    assert len(ranked) == 2


@pytest.mark.asyncio
async def test_retry_on_429():
    call_count = 0
    valid_response = '[{"f": "src/engine.py", "l": "business"}]'

    def handler(request):
        nonlocal call_count
        call_count += 1
        if call_count < 2:
            return httpx.Response(429)
        return httpx.Response(200, json={
            "choices": [{"message": {"content": valid_response}}]
        })

    with respx.mock:
        respx.post("https://openrouter.ai/api/v1/chat/completions").mock(side_effect=handler)
        async with httpx.AsyncClient() as client:
            result = await _call_llm(
                client,
                "https://openrouter.ai/api/v1/chat/completions",
                {"model": "test", "messages": []},
                {"Authorization": "Bearer test-key"},
            )

    assert call_count == 2
    assert len(result) == 1


@pytest.mark.asyncio
async def test_auth_error_on_401():
    with respx.mock:
        respx.post("https://openrouter.ai/api/v1/chat/completions").mock(
            return_value=httpx.Response(401)
        )
        async with httpx.AsyncClient() as client:
            with pytest.raises(AuthError):
                await _call_llm(
                    client,
                    "https://openrouter.ai/api/v1/chat/completions",
                    {"model": "test", "messages": []},
                    {"Authorization": "Bearer bad-key"},
                )


@pytest.mark.asyncio
async def test_extract_chunk_returns_line_range():
    settings = Settings(openrouter_api_key="test-key")
    content = "\n".join(f"line_{i} = {i}" for i in range(100))

    with respx.mock:
        respx.post("https://openrouter.ai/api/v1/chat/completions").mock(
            return_value=httpx.Response(200, json={
                "choices": [{"message": {"content": '{"start": 10, "end": 50}'}}]
            })
        )
        async with httpx.AsyncClient() as client:
            result = await extract_chunk(
                path="src/engine.py",
                content=content,
                language="python",
                target_loc=40,
                settings=settings,
                client=client,
            )

    assert result == (10, 50)


@pytest.mark.asyncio
async def test_extract_chunk_returns_none_on_bad_json():
    settings = Settings(openrouter_api_key="test-key")
    content = "\n".join(f"x = {i}" for i in range(50))

    with respx.mock:
        respx.post("https://openrouter.ai/api/v1/chat/completions").mock(
            return_value=httpx.Response(200, json={
                "choices": [{"message": {"content": "not json"}}]
            })
        )
        async with httpx.AsyncClient() as client:
            result = await extract_chunk("src/f.py", content, "python", 30, settings, client)

    assert result is None
