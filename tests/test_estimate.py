import math

import pytest

from llm_cost.estimate import estimate_cost
from llm_cost.pricing import default_pricing


def test_estimate_matches_hand_computed_cost():
    table = default_pricing()
    price = table.resolve("claude-opus-5")
    result = estimate_cost(price, input_tokens=12000, output_tokens=800, cached_input_tokens=52000)
    assert math.isclose(result.total_cost, 0.106, rel_tol=1e-9, abs_tol=1e-9)
    assert math.isclose(result.cost_per_call, result.total_cost)


def test_calls_multiplies_total_but_not_per_call_cost():
    table = default_pricing()
    price = table.resolve("claude-haiku-4-5")
    result = estimate_cost(price, input_tokens=1000, output_tokens=1000, calls=50)
    assert math.isclose(result.total_cost, result.cost_per_call * 50)
    assert result.calls == 50


def test_unpriced_token_classes_default_to_zero_cost():
    table = default_pricing()
    price = table.resolve("claude-sonnet-5")
    result = estimate_cost(price, input_tokens=1000)
    assert result.output_cost == 0
    assert result.cached_input_cost == 0
    assert result.cache_write_cost == 0


def test_to_dict_matches_computed_fields():
    table = default_pricing()
    price = table.resolve("gpt-4o-mini")
    result = estimate_cost(price, input_tokens=100, output_tokens=50)
    data = result.to_dict()
    assert data["model"] == "gpt-4o-mini"
    assert data["calls"] == 1
    assert math.isclose(data["total_cost"], result.total_cost)
    assert math.isclose(data["cost_per_call"], result.cost_per_call)


def test_calls_must_be_at_least_one():
    table = default_pricing()
    price = table.resolve("claude-opus-5")
    with pytest.raises(ValueError):
        estimate_cost(price, input_tokens=100, calls=0)
