"""Cost of a single call, or a batch of identical calls."""

from dataclasses import dataclass

from .pricing import ModelPrice

_PER = 1_000_000


@dataclass(frozen=True)
class CostEstimate:
    model: str
    calls: int
    input_tokens: int
    output_tokens: int
    cached_input_tokens: int
    cache_write_tokens: int
    input_cost: float
    output_cost: float
    cached_input_cost: float
    cache_write_cost: float

    @property
    def cost_per_call(self) -> float:
        return self.input_cost + self.output_cost + self.cached_input_cost + self.cache_write_cost

    @property
    def total_cost(self) -> float:
        return self.cost_per_call * self.calls

    def to_dict(self) -> dict:
        return {
            "model": self.model,
            "calls": self.calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cached_input_tokens": self.cached_input_tokens,
            "cache_write_tokens": self.cache_write_tokens,
            "input_cost": self.input_cost,
            "output_cost": self.output_cost,
            "cached_input_cost": self.cached_input_cost,
            "cache_write_cost": self.cache_write_cost,
            "cost_per_call": self.cost_per_call,
            "total_cost": self.total_cost,
        }


def estimate_cost(
    price: ModelPrice,
    *,
    input_tokens: int = 0,
    output_tokens: int = 0,
    cached_input_tokens: int = 0,
    cache_write_tokens: int = 0,
    calls: int = 1,
) -> CostEstimate:
    """Cost of `calls` identical calls, each with the given token counts."""
    if calls < 1:
        raise ValueError("calls must be at least 1")

    return CostEstimate(
        model=price.name,
        calls=calls,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cached_input_tokens=cached_input_tokens,
        cache_write_tokens=cache_write_tokens,
        input_cost=input_tokens / _PER * price.input,
        output_cost=output_tokens / _PER * price.output,
        cached_input_cost=cached_input_tokens / _PER * price.cached_input,
        cache_write_cost=cache_write_tokens / _PER * price.cache_write,
    )
