"""The price table: built-in prices, override files, and name resolution.

Prices are USD per 1,000,000 tokens. A model's `cached_input` and
`cache_write` prices fall back to its `input` price when a table does not
set them explicitly - a provider that doesn't discount cached tokens still
needs a number to multiply, and assuming no discount over-states the bill
rather than under-stating it.
"""

import json
import re
from dataclasses import dataclass
from typing import Dict, Optional

# Public list prices as of this date. This will drift; override it with
# --pricing rather than editing it in place.
BUILTIN_AS_OF = "2026-06-24"

_BUILTIN_MODELS = {
    "claude-haiku-4-5": {"provider": "anthropic", "input": 1.00, "output": 5.00, "cached_input": 0.10, "cache_write": 1.25},
    "claude-sonnet-4-6": {"provider": "anthropic", "input": 3.00, "output": 15.00, "cached_input": 0.30, "cache_write": 3.75},
    "claude-sonnet-5": {"provider": "anthropic", "input": 3.00, "output": 15.00, "cached_input": 0.30, "cache_write": 3.75},
    "claude-opus-4-6": {"provider": "anthropic", "input": 5.00, "output": 25.00, "cached_input": 0.50, "cache_write": 6.25},
    "claude-opus-4-7": {"provider": "anthropic", "input": 5.00, "output": 25.00, "cached_input": 0.50, "cache_write": 6.25},
    "claude-opus-4-8": {"provider": "anthropic", "input": 5.00, "output": 25.00, "cached_input": 0.50, "cache_write": 6.25},
    "claude-opus-5": {"provider": "anthropic", "input": 5.00, "output": 25.00, "cached_input": 0.50, "cache_write": 6.25},
    "claude-fable-5": {"provider": "anthropic", "input": 10.00, "output": 50.00, "cached_input": 1.00, "cache_write": 12.50},
    "gpt-4o": {"provider": "openai", "input": 2.50, "output": 10.00, "cached_input": 1.25},
    "gpt-4o-mini": {"provider": "openai", "input": 0.15, "output": 0.60, "cached_input": 0.075},
    "gemini-2.5-flash": {"provider": "google", "input": 0.30, "output": 2.50},
    "gemini-2.5-pro": {"provider": "google", "input": 1.25, "output": 10.00},
}

_SUFFIX_PATTERNS = [
    re.compile(r":\d+$"),          # bedrock version tag, e.g. ":0"
    re.compile(r"-v\d+$"),         # "-v1"
    re.compile(r"-\d{8}$"),        # "-20260101"
    re.compile(r"-\d{4}-\d{2}-\d{2}$"),  # "-2026-06-24"
]


class UnknownModelError(Exception):
    """Raised when a pricing table has no entry for a resolved model name."""

    def __init__(self, name: str):
        self.name = name
        super().__init__(f"no price for model {name!r}")


@dataclass(frozen=True)
class ModelPrice:
    name: str
    input: float
    output: float
    cached_input: float
    cache_write: float
    provider: Optional[str] = None


def _build_price(name: str, spec: dict) -> ModelPrice:
    for field in ("input", "output"):
        if field not in spec:
            raise ValueError(f"model {name!r} is missing required field {field!r}")
    input_price = float(spec["input"])
    return ModelPrice(
        name=name.strip().lower(),
        input=input_price,
        output=float(spec["output"]),
        cached_input=float(spec.get("cached_input", input_price)),
        cache_write=float(spec.get("cache_write", input_price)),
        provider=spec.get("provider"),
    )


def _strip_provider_prefix(name: str) -> str:
    for sep in (".", "/"):
        if sep in name:
            _prefix, _sep, rest = name.partition(sep)
            if rest:
                return rest
    return name


def _strip_known_suffixes(name: str) -> str:
    changed = True
    while changed:
        changed = False
        for pattern in _SUFFIX_PATTERNS:
            replaced = pattern.sub("", name)
            if replaced != name:
                name = replaced
                changed = True
    return name


class PricingTable:
    """A set of model prices, plus the metadata needed to explain them."""

    def __init__(self, models: Dict[str, ModelPrice], as_of: Optional[str] = None, source: str = "custom"):
        self._models = dict(models)
        self.as_of = as_of
        self.source = source

    def __len__(self) -> int:
        return len(self._models)

    def __iter__(self):
        return iter(self._models.values())

    def __contains__(self, name: str) -> bool:
        try:
            self.resolve(name)
            return True
        except UnknownModelError:
            return False

    def resolve(self, name: str) -> ModelPrice:
        """Look up a model's price, tolerating provider prefixes and date suffixes.

        API-reported model names rarely match a price table's canonical keys
        exactly - Bedrock prefixes with "anthropic.", the Anthropic and OpenAI
        APIs both suffix with a release date. Try the name as given, then with
        a provider prefix removed, then with trailing version/date noise
        stripped, in whichever combination first hits the table.
        """
        normalized = name.strip().lower()
        candidates = []
        for candidate in (normalized, _strip_provider_prefix(normalized)):
            if candidate not in candidates:
                candidates.append(candidate)
            stripped = _strip_known_suffixes(candidate)
            if stripped not in candidates:
                candidates.append(stripped)

        for candidate in candidates:
            price = self._models.get(candidate)
            if price is not None:
                return price
        raise UnknownModelError(name)

    def get(self, name: str, default=None):
        try:
            return self.resolve(name)
        except UnknownModelError:
            return default


def parse_pricing(obj: dict) -> PricingTable:
    """Build a PricingTable from a pricing document's parsed JSON."""
    models = {}
    for name, spec in obj.get("models", {}).items():
        price = _build_price(name, spec)
        models[price.name] = price
    return PricingTable(models, as_of=obj.get("as_of"), source="override")


def default_pricing() -> PricingTable:
    models = {name: _build_price(name, spec) for name, spec in _BUILTIN_MODELS.items()}
    return PricingTable(models, as_of=BUILTIN_AS_OF, source="built-in")


def load_pricing(path: str) -> PricingTable:
    """Load an override file and merge it onto the built-in table.

    Set "replace": true in the file to start from an empty table instead, so
    a model missing from the file raises rather than silently pricing
    against a stale built-in default.
    """
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)

    override = parse_pricing(data)
    if data.get("replace", False):
        return override

    base = default_pricing()
    models = dict(base._models)
    models.update(override._models)
    return PricingTable(models, as_of=override.as_of or base.as_of, source="merged")
