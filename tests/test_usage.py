import pytest

from llm_cost.usage import UsageParseError, load_usage


def test_parses_anthropic_shaped_record():
    line = (
        '{"model":"claude-opus-5","date":"2026-06-01T09:12:00Z","team":"agents",'
        '"usage":{"input_tokens":18400,"output_tokens":2100,'
        '"cache_read_input_tokens":52000,"cache_creation_input_tokens":9000}}'
    )
    records, problems = load_usage(line)
    assert problems == []
    assert len(records) == 1

    record = records[0]
    assert record.model == "claude-opus-5"
    assert record.date == "2026-06-01T09:12:00Z"
    assert record.input_tokens == 18400
    assert record.output_tokens == 2100
    assert record.cached_input_tokens == 52000
    assert record.cache_write_tokens == 9000
    assert record.fields["team"] == "agents"


def test_parses_openai_shaped_record_and_excludes_cached_from_input():
    line = (
        '{"model":"gpt-4o-mini","date":"2026-06-01T10:03:00Z",'
        '"usage":{"prompt_tokens":31000,"completion_tokens":420,'
        '"prompt_tokens_details":{"cached_tokens":24000}}}'
    )
    records, problems = load_usage(line)
    assert problems == []

    record = records[0]
    assert record.input_tokens == 7000
    assert record.output_tokens == 420
    assert record.cached_input_tokens == 24000
    assert record.cache_write_tokens == 0


def test_openai_shaped_record_without_cache_details_defaults_to_zero():
    line = '{"model":"gpt-4o","usage":{"prompt_tokens":1000,"completion_tokens":50}}'
    records, problems = load_usage(line)
    assert problems == []
    record = records[0]
    assert record.input_tokens == 1000
    assert record.cached_input_tokens == 0


def test_blank_lines_are_skipped():
    text = '\n{"model":"gpt-4o","usage":{"prompt_tokens":10,"completion_tokens":1}}\n\n'
    records, problems = load_usage(text)
    assert len(records) == 1
    assert problems == []


def test_invalid_json_is_reported_as_a_problem_with_line_number():
    text = 'not json\n{"model":"gpt-4o","usage":{"prompt_tokens":10,"completion_tokens":1}}'
    records, problems = load_usage(text)
    assert len(records) == 1
    assert len(problems) == 1
    assert problems[0].line_number == 1
    assert "invalid JSON" in problems[0].reason


def test_missing_model_is_reported_as_a_problem():
    text = '{"usage":{"prompt_tokens":10,"completion_tokens":1}}'
    records, problems = load_usage(text)
    assert records == []
    assert len(problems) == 1
    assert "model" in problems[0].reason


def test_unrecognized_usage_shape_is_reported_as_a_problem():
    text = '{"model":"internal-router-v3","usage":{"tokens_used":100}}'
    records, problems = load_usage(text)
    assert records == []
    assert len(problems) == 1
    assert "unrecognized usage shape" in problems[0].reason


def test_strict_mode_raises_on_first_bad_line():
    text = 'not json\n{"model":"gpt-4o","usage":{"prompt_tokens":10,"completion_tokens":1}}'
    with pytest.raises(UsageParseError) as excinfo:
        load_usage(text, strict=True)
    assert excinfo.value.line_number == 1


def test_line_number_counts_skipped_blank_lines():
    text = '\n\nnot json'
    _records, problems = load_usage(text)
    assert problems[0].line_number == 3
