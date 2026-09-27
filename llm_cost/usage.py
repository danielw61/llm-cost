"""Parsing usage logs: one JSON object per line, OpenAI- or Anthropic-shaped.

OpenAI's `prompt_tokens` counts the whole prompt including any cached prefix;
Anthropic's `input_tokens` counts only the uncached portion and reports the
cached prefix separately as `cache_read_input_tokens`. Both are normalised
here to the same shape - `input_tokens` excludes anything cached - so a
report can sum them without caring which API produced the line.
"""

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


class UsageParseError(Exception):
    """Raised in strict mode when a line cannot be parsed into a usage record."""

    def __init__(self, line_number: int, reason: str):
        self.line_number = line_number
        self.reason = reason
        super().__init__(f"line {line_number}: {reason}")


@dataclass(frozen=True)
class UsageProblem:
    line_number: int
    reason: str


@dataclass(frozen=True)
class UsageRecord:
    model: str
    date: Optional[str]
    input_tokens: int
    output_tokens: int
    cached_input_tokens: int
    cache_write_tokens: int
    fields: Dict[str, Any] = field(default_factory=dict)


def _tokens_from_anthropic_usage(usage: dict) -> Tuple[int, int, int, int]:
    input_tokens = int(usage.get("input_tokens", 0) or 0)
    output_tokens = int(usage.get("output_tokens", 0) or 0)
    cached_input_tokens = int(usage.get("cache_read_input_tokens", 0) or 0)
    cache_write_tokens = int(usage.get("cache_creation_input_tokens", 0) or 0)
    return input_tokens, output_tokens, cached_input_tokens, cache_write_tokens


def _tokens_from_openai_usage(usage: dict) -> Tuple[int, int, int, int]:
    prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
    output_tokens = int(usage.get("completion_tokens", 0) or 0)
    details = usage.get("prompt_tokens_details") or {}
    cached_input_tokens = int(details.get("cached_tokens", 0) or 0)
    # prompt_tokens includes the cached prefix; input_tokens should not.
    input_tokens = max(0, prompt_tokens - cached_input_tokens)
    return input_tokens, output_tokens, cached_input_tokens, 0


def _parse_line(line_number: int, line: str) -> UsageRecord:
    try:
        obj = json.loads(line)
    except json.JSONDecodeError as exc:
        raise UsageParseError(line_number, f"invalid JSON ({exc.msg})") from exc

    if not isinstance(obj, dict):
        raise UsageParseError(line_number, "line is not a JSON object")

    model = obj.get("model")
    if not isinstance(model, str) or not model.strip():
        raise UsageParseError(line_number, "missing or invalid 'model' field")

    usage = obj.get("usage")
    if not isinstance(usage, dict):
        raise UsageParseError(line_number, "missing or invalid 'usage' field")

    if "input_tokens" in usage:
        tokens = _tokens_from_anthropic_usage(usage)
    elif "prompt_tokens" in usage:
        tokens = _tokens_from_openai_usage(usage)
    else:
        raise UsageParseError(line_number, "unrecognized usage shape")

    input_tokens, output_tokens, cached_input_tokens, cache_write_tokens = tokens
    fields = {key: value for key, value in obj.items() if key != "usage"}

    return UsageRecord(
        model=model,
        date=obj.get("date"),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cached_input_tokens=cached_input_tokens,
        cache_write_tokens=cache_write_tokens,
        fields=fields,
    )


def load_usage(text: str, *, strict: bool = False) -> Tuple[List[UsageRecord], List[UsageProblem]]:
    """Parse a JSONL usage log into records, plus a list of unparseable lines.

    Blank lines are skipped silently. In strict mode the first malformed line
    raises `UsageParseError` instead of being recorded as a problem, so a CI
    check can fail loudly rather than silently under-reporting a log.
    """
    records: List[UsageRecord] = []
    problems: List[UsageProblem] = []

    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        try:
            records.append(_parse_line(line_number, line))
        except UsageParseError as exc:
            if strict:
                raise
            problems.append(UsageProblem(line_number=line_number, reason=exc.reason))

    return records, problems
