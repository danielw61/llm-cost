import pytest

from llm_cost.pricing import (
    UnknownModelError,
    default_pricing,
    parse_pricing,
)


def test_default_pricing_has_known_models():
    table = default_pricing()
    price = table.resolve("claude-opus-5")
    assert price.input == 5.00
    assert price.output == 25.00
    assert table.as_of == "2026-06-24"
    assert table.source == "built-in"


def test_resolve_is_case_and_whitespace_insensitive():
    table = default_pricing()
    assert table.resolve("  Claude-Opus-5  ").name == "claude-opus-5"


def test_resolve_tolerates_provider_prefix_and_date_suffix():
    table = default_pricing()
    price = table.resolve("anthropic.claude-opus-5-20260101")
    assert price.name == "claude-opus-5"


def test_resolve_tolerates_bedrock_style_prefix_and_version_tag():
    table = default_pricing()
    price = table.resolve("anthropic.claude-opus-5-20260101-v1:0")
    assert price.name == "claude-opus-5"


def test_resolve_tolerates_slash_prefix():
    table = default_pricing()
    price = table.resolve("openai/gpt-4o-mini")
    assert price.name == "gpt-4o-mini"


def test_resolve_unknown_model_raises():
    table = default_pricing()
    with pytest.raises(UnknownModelError):
        table.resolve("internal-router-v3")


def test_contains():
    table = default_pricing()
    assert "claude-opus-5" in table
    assert "internal-router-v3" not in table


def test_parse_pricing_fills_in_cache_prices_from_input_price():
    table = parse_pricing({"models": {"internal-router-v3": {"input": 0.2, "output": 0.8}}})
    price = table.resolve("internal-router-v3")
    assert price.cached_input == 0.2
    assert price.cache_write == 0.2


def test_parse_pricing_keeps_explicit_cache_prices():
    table = parse_pricing(
        {"models": {"claude-opus-5": {"input": 4.25, "output": 21.0, "cached_input": 0.42, "cache_write": 5.3}}}
    )
    price = table.resolve("claude-opus-5")
    assert price.cached_input == 0.42
    assert price.cache_write == 5.3


def test_parse_pricing_requires_input_and_output():
    with pytest.raises(ValueError):
        parse_pricing({"models": {"broken": {"input": 0.2}}})


def test_parse_pricing_records_as_of():
    table = parse_pricing({"as_of": "2026-07-01", "models": {}})
    assert table.as_of == "2026-07-01"
