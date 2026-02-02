"""Comprehensive tests for R_acc parser accuracy.

Validates the full accuracy scoring pipeline:
- extract_tool_calls_from_text: action/JSON format parsing
- _parse_args / _normalize_args: argument parsing
- score_tool_name_match: tool name comparison
- score_tool_args_match: argument matching
- score_response_overlap: token-level F1
- compute_accuracy: end-to-end accuracy scoring

Catches regressions in edge cases: nested args, malformed input,
mixed formats, Unicode, empty fields, and weight customization.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from entropy_reward.data.accuracy import (
    AccuracyResult,
    ACTION_PATTERN,
    FUNCTION_CALL_PATTERN,
    _normalize_args,
    _parse_args,
    compute_accuracy,
    extract_tool_calls_from_text,
    score_response_overlap,
    score_tool_args_match,
    score_tool_name_match,
)


# ═══════════════════════════════════════════════════════════════════════
#  extract_tool_calls_from_text
# ═══════════════════════════════════════════════════════════════════════


class TestExtractToolCallsActionFormat:
    """Tests for <action>tool_name(args)</action> extraction."""

    def test_simple_action(self):
        text = "<action>get_weather(city=London)</action>"
        calls = extract_tool_calls_from_text(text)
        assert len(calls) == 1
        assert calls[0]["name"] == "get_weather"
        assert calls[0]["arguments"]["city"] == "London"

    def test_action_with_json_args(self):
        text = '<action>place_order({"user_id": "u123", "item": "pizza"})</action>'
        calls = extract_tool_calls_from_text(text)
        assert len(calls) == 1
        assert calls[0]["name"] == "place_order"
        assert calls[0]["arguments"]["user_id"] == "u123"

    def test_action_no_args(self):
        text = "<action>get_status()</action>"
        calls = extract_tool_calls_from_text(text)
        assert len(calls) == 1
        assert calls[0]["name"] == "get_status"
        assert calls[0]["arguments"] == {}

    def test_action_multiple_kv_args(self):
        text = "<action>search_flights(origin=SVO, dest=LED, date=2025-07-01)</action>"
        calls = extract_tool_calls_from_text(text)
        assert len(calls) == 1
        assert calls[0]["name"] == "search_flights"
        assert calls[0]["arguments"]["origin"] == "SVO"
        assert calls[0]["arguments"]["dest"] == "LED"

    def test_action_quoted_values(self):
        text = """<action>get_weather(city="New York", units="imperial")</action>"""
        calls = extract_tool_calls_from_text(text)
        assert len(calls) == 1
        assert calls[0]["arguments"]["city"] == "New York"
        assert calls[0]["arguments"]["units"] == "imperial"

    def test_multiple_action_tags(self):
        text = (
            "<think>Need to search and then book.</think>\n"
            "<action>search_flights(origin=SVO, dest=LED)</action>\n"
            "<action>book_flight(flight_id=SU42)</action>"
        )
        calls = extract_tool_calls_from_text(text)
        assert len(calls) == 2
        assert calls[0]["name"] == "search_flights"
        assert calls[1]["name"] == "book_flight"

    def test_action_with_whitespace(self):
        text = "<action>  get_weather  ( city=London )  </action>"
        calls = extract_tool_calls_from_text(text)
        assert len(calls) == 1
        assert calls[0]["name"] == "get_weather"

    def test_action_multiline(self):
        text = (
            "<action>\n"
            "  get_weather(city=London)\n"
            "</action>"
        )
        calls = extract_tool_calls_from_text(text)
        assert len(calls) == 1
        assert calls[0]["name"] == "get_weather"

    def test_action_embedded_in_longer_text(self):
        text = (
            "<think>The user needs weather info for London. "
            "I'll use the weather tool.</think>\n"
            "<action>get_weather(city=London, units=metric)</action>\n"
            "Waiting for tool response..."
        )
        calls = extract_tool_calls_from_text(text)
        assert len(calls) == 1
        assert calls[0]["name"] == "get_weather"


class TestExtractToolCallsJSONFormat:
    """Tests for JSON function call block extraction."""

    def test_json_function_call(self):
        text = '{"name": "get_weather", "arguments": {"city": "London"}}'
        calls = extract_tool_calls_from_text(text)
        assert len(calls) == 1
        assert calls[0]["name"] == "get_weather"

    def test_json_with_string_arguments(self):
        text = '{"name": "search", "arguments": "{\\"query\\": \\"pizza\\"}"}'
        # The FUNCTION_CALL_PATTERN looks for "arguments": "{...}"
        calls = extract_tool_calls_from_text(text)
        # May or may not parse depending on escaping; test it doesn't crash
        assert isinstance(calls, list)

    def test_no_tool_calls_plain_text(self):
        text = "I think the answer is 42. No tools needed here."
        calls = extract_tool_calls_from_text(text)
        assert len(calls) == 0

    def test_no_tool_calls_empty_string(self):
        calls = extract_tool_calls_from_text("")
        assert len(calls) == 0

    def test_action_format_preferred_over_json(self):
        """When <action> tags are found, JSON format should be skipped."""
        text = (
            '<action>get_weather(city=London)</action>\n'
            '{"name": "book_hotel", "arguments": {"hotel": "Ritz"}}'
        )
        calls = extract_tool_calls_from_text(text)
        # Only action format should be returned (early return)
        assert len(calls) == 1
        assert calls[0]["name"] == "get_weather"


class TestExtractToolCallsEdgeCases:
    """Edge cases and adversarial inputs."""

    def test_unclosed_action_tag(self):
        """Unclosed <action> should not match."""
        text = "<action>get_weather(city=London)"
        calls = extract_tool_calls_from_text(text)
        assert len(calls) == 0

    def test_action_tag_without_parens(self):
        """<action> without function call pattern should not match."""
        text = "<action>just some text without tool call syntax</action>"
        calls = extract_tool_calls_from_text(text)
        assert len(calls) == 0

    def test_nested_parens_in_args_limitation(self):
        """Known limitation: nested parens break the regex [^)]*."""
        text = '<action>eval(expr=foo("bar"))</action>'
        calls = extract_tool_calls_from_text(text)
        # The regex [^)]* stops at the first ), so this won't capture correctly
        # This documents the known limitation
        assert len(calls) >= 0  # at least doesn't crash

    def test_unicode_in_args(self):
        text = "<action>send_message(text=Привет мир, user=Иван)</action>"
        calls = extract_tool_calls_from_text(text)
        assert len(calls) == 1
        assert calls[0]["name"] == "send_message"
        assert "Привет мир" in calls[0]["arguments"].get("text", "")

    def test_special_chars_in_tool_name(self):
        """Tool names should be word characters only (\w+)."""
        text = "<action>get-weather(city=London)</action>"
        calls = extract_tool_calls_from_text(text)
        # Hyphen isn't matched by \w+, so this won't match
        assert len(calls) == 0

    def test_tool_name_with_underscore(self):
        text = "<action>get_current_weather(city=London)</action>"
        calls = extract_tool_calls_from_text(text)
        assert len(calls) == 1
        assert calls[0]["name"] == "get_current_weather"


# ═══════════════════════════════════════════════════════════════════════
#  _parse_args
# ═══════════════════════════════════════════════════════════════════════


class TestParseArgs:
    def test_empty_string(self):
        assert _parse_args("") == {}

    def test_json_object(self):
        result = _parse_args('{"city": "London", "units": "metric"}')
        assert result == {"city": "London", "units": "metric"}

    def test_kv_pairs(self):
        result = _parse_args("city=London, units=metric")
        assert result["city"] == "London"
        assert result["units"] == "metric"

    def test_kv_quoted_values(self):
        result = _parse_args("city='New York', units=\"imperial\"")
        assert result["city"] == "New York"
        assert result["units"] == "imperial"

    def test_positional_args(self):
        result = _parse_args("London, metric")
        assert result == {"arg_0": "London", "arg_1": "metric"}

    def test_mixed_kv_and_positional(self):
        result = _parse_args("London, units=metric")
        assert "arg_0" in result
        assert result["units"] == "metric"

    def test_single_value(self):
        result = _parse_args("London")
        assert result == {"arg_0": "London"}

    def test_malformed_json_falls_through(self):
        """Invalid JSON should fall through to key=value parsing."""
        result = _parse_args("{invalid json}")
        assert isinstance(result, dict)

    def test_json_int_wrapped(self):
        """json.loads('123') returns int — must be wrapped in dict."""
        result = _parse_args("123")
        assert isinstance(result, dict)
        assert result == {"arg_0": 123}

    def test_json_float_wrapped(self):
        result = _parse_args("3.14")
        assert isinstance(result, dict)
        assert result == {"arg_0": 3.14}

    def test_json_list_wrapped(self):
        result = _parse_args('[1, 2, 3]')
        assert isinstance(result, dict)
        assert result == {"arg_0": [1, 2, 3]}

    def test_json_bool_wrapped(self):
        result = _parse_args("true")
        assert isinstance(result, dict)
        assert result == {"arg_0": True}

    def test_json_null_wrapped(self):
        result = _parse_args("null")
        assert isinstance(result, dict)


# ═══════════════════════════════════════════════════════════════════════
#  _normalize_args
# ═══════════════════════════════════════════════════════════════════════


class TestNormalizeArgs:
    def test_dict_passthrough(self):
        d = {"city": "London"}
        assert _normalize_args(d) == d

    def test_json_string(self):
        result = _normalize_args('{"city": "London"}')
        assert result == {"city": "London"}

    def test_invalid_json_string(self):
        result = _normalize_args("not json")
        assert result == {"raw": "not json"}

    def test_empty_dict(self):
        assert _normalize_args({}) == {}

    def test_none(self):
        assert _normalize_args(None) == {}

    def test_empty_string(self):
        result = _normalize_args("")
        # Empty string is not valid JSON, falls to {"raw": ""}
        assert result == {"raw": ""}

    def test_int_wrapped(self):
        """int arguments must be wrapped — caused AttributeError in production."""
        result = _normalize_args(42)
        assert isinstance(result, dict)
        assert result == {"arg_0": 42}

    def test_float_wrapped(self):
        result = _normalize_args(3.14)
        assert isinstance(result, dict)

    def test_list_wrapped(self):
        result = _normalize_args([1, 2, 3])
        assert isinstance(result, dict)
        assert result == {"arg_0": [1, 2, 3]}

    def test_bool_wrapped(self):
        result = _normalize_args(True)
        assert isinstance(result, dict)

    def test_json_string_with_int(self):
        """JSON string '123' should be wrapped, not returned as int."""
        result = _normalize_args("123")
        assert isinstance(result, dict)
        assert result == {"arg_0": 123}


# ═══════════════════════════════════════════════════════════════════════
#  score_tool_name_match
# ═══════════════════════════════════════════════════════════════════════


class TestScoreToolNameMatch:
    def test_perfect_match_single(self):
        pred = [{"name": "get_weather"}]
        ref = [{"function": {"name": "get_weather"}}]
        assert score_tool_name_match(pred, ref) == 1.0

    def test_no_match(self):
        pred = [{"name": "cancel_order"}]
        ref = [{"function": {"name": "place_order"}}]
        assert score_tool_name_match(pred, ref) == 0.0

    def test_partial_match_multiple_refs(self):
        pred = [{"name": "get_weather"}, {"name": "book_hotel"}]
        ref = [
            {"function": {"name": "get_weather"}},
            {"function": {"name": "search_flights"}},
        ]
        # 1 out of 2 reference names matched
        assert score_tool_name_match(pred, ref) == 0.5

    def test_no_reference_no_prediction(self):
        assert score_tool_name_match([], []) == 1.0

    def test_no_reference_with_prediction(self):
        assert score_tool_name_match([{"name": "foo"}], []) == 0.5

    def test_reference_with_flat_name(self):
        """Reference using 'name' key directly (not nested in 'function')."""
        pred = [{"name": "get_weather"}]
        ref = [{"name": "get_weather"}]
        # ref_names extraction: c.get("function", {}).get("name", c.get("name", ""))
        # This will get "" from function.name, but then c.get("name", "") fallback doesn't work
        # because the extraction logic tries function.name first
        score = score_tool_name_match(pred, ref)
        # With the current code: ref_name = {}.get("name", ref.get("name",""))
        # = {}.get("name", "get_weather") = "get_weather"? No.
        # c.get("function", {}) returns {} since no "function" key
        # {}.get("name", c.get("name", "")) = {}.get("name", "get_weather") = "get_weather"
        # Wait: c.get("function", {}).get("name", c.get("name", ""))
        # = {}.get("name", "get_weather") = "get_weather"  ✓
        assert score == 1.0

    def test_all_refs_matched(self):
        pred = [{"name": "a"}, {"name": "b"}, {"name": "c"}]
        ref = [
            {"function": {"name": "a"}},
            {"function": {"name": "b"}},
        ]
        assert score_tool_name_match(pred, ref) == 1.0

    def test_duplicate_pred_names(self):
        """If model calls same tool twice, both refs to that tool still count."""
        pred = [{"name": "get_weather"}, {"name": "get_weather"}]
        ref = [{"function": {"name": "get_weather"}}]
        assert score_tool_name_match(pred, ref) == 1.0

    def test_duplicate_ref_names(self):
        """Two refs expecting the same tool — prediction has it once."""
        pred = [{"name": "get_weather"}]
        ref = [
            {"function": {"name": "get_weather"}},
            {"function": {"name": "get_weather"}},
        ]
        # Both refs find "get_weather" in pred_names → 2/2 = 1.0
        assert score_tool_name_match(pred, ref) == 1.0


# ═══════════════════════════════════════════════════════════════════════
#  score_tool_args_match
# ═══════════════════════════════════════════════════════════════════════


class TestScoreToolArgsMatch:
    def test_perfect_match(self):
        pred = [{"name": "get_weather", "arguments": {"city": "London"}}]
        ref = [{"function": {"name": "get_weather", "arguments": '{"city": "London"}'}}]
        score = score_tool_args_match(pred, ref)
        assert score == 1.0

    def test_no_reference(self):
        assert score_tool_args_match([], []) == 1.0

    def test_wrong_tool_name_no_match(self):
        pred = [{"name": "cancel_order", "arguments": {"id": "123"}}]
        ref = [{"function": {"name": "place_order", "arguments": '{"id": "123"}'}}]
        score = score_tool_args_match(pred, ref)
        assert score == 0.0

    def test_matching_keys_wrong_values(self):
        pred = [{"name": "get_weather", "arguments": {"city": "Paris"}}]
        ref = [{"function": {"name": "get_weather", "arguments": '{"city": "London"}'}}]
        score = score_tool_args_match(pred, ref)
        # key_overlap = 1/1 = 1.0, val_matches = 0/1 = 0.0
        # best_score = 0.5 * 1.0 + 0.5 * 0.0 = 0.5
        assert score == 0.5

    def test_extra_keys_in_prediction(self):
        pred = [{"name": "get_weather", "arguments": {"city": "London", "extra": "yes"}}]
        ref = [{"function": {"name": "get_weather", "arguments": '{"city": "London"}'}}]
        score = score_tool_args_match(pred, ref)
        # key_overlap = 1/1 (ref has 1 key, pred covers it), val = 1/1
        # best_score = 0.5 * 1.0 + 0.5 * 1.0 = 1.0
        assert score == 1.0

    def test_missing_keys_in_prediction(self):
        pred = [{"name": "get_weather", "arguments": {"city": "London"}}]
        ref = [{"function": {"name": "get_weather", "arguments": '{"city": "London", "units": "metric"}'}}]
        score = score_tool_args_match(pred, ref)
        # ref_keys = {"city", "units"}, pred_keys = {"city"}
        # key_overlap = 1/2 = 0.5, val_matches(city) = 1/1 = 1.0
        # best_score = 0.5 * 0.5 + 0.5 * 1.0 = 0.75
        assert score == 0.75

    def test_empty_args_both(self):
        pred = [{"name": "get_status", "arguments": {}}]
        ref = [{"function": {"name": "get_status", "arguments": "{}"}}]
        score = score_tool_args_match(pred, ref)
        assert score == 1.0

    def test_case_insensitive_values(self):
        pred = [{"name": "get_weather", "arguments": {"city": "LONDON"}}]
        ref = [{"function": {"name": "get_weather", "arguments": '{"city": "london"}'}}]
        score = score_tool_args_match(pred, ref)
        # str comparison is case-insensitive: "LONDON".lower() == "london".lower()
        assert score == 1.0

    def test_multiple_refs_partial(self):
        pred = [
            {"name": "get_weather", "arguments": {"city": "London"}},
        ]
        ref = [
            {"function": {"name": "get_weather", "arguments": '{"city": "London"}'}},
            {"function": {"name": "book_hotel", "arguments": '{"hotel": "Ritz"}'}},
        ]
        score = score_tool_args_match(pred, ref)
        # First ref matched with score 1.0, second ref no matching pred → 0.0
        # total = (1.0 + 0.0) / 2 = 0.5
        assert score == 0.5

    def test_args_as_json_string_in_reference(self):
        """Reference arguments stored as JSON string (common in dataset)."""
        pred = [{"name": "search", "arguments": {"query": "pizza", "limit": "10"}}]
        ref = [{"function": {"name": "search", "arguments": '{"query": "pizza", "limit": "10"}'}}]
        score = score_tool_args_match(pred, ref)
        assert score == 1.0

    def test_int_args_no_crash(self):
        """Regression: int arguments caused 'int has no attribute keys'."""
        pred = [{"name": "get_item", "arguments": 42}]
        ref = [{"function": {"name": "get_item", "arguments": '{"id": "42"}'}}]
        score = score_tool_args_match(pred, ref)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_list_args_no_crash(self):
        pred = [{"name": "batch", "arguments": [1, 2, 3]}]
        ref = [{"function": {"name": "batch", "arguments": '{"items": [1,2,3]}'}}]
        score = score_tool_args_match(pred, ref)
        assert isinstance(score, float)

    def test_bool_args_no_crash(self):
        pred = [{"name": "toggle", "arguments": True}]
        ref = [{"function": {"name": "toggle", "arguments": '{"enabled": true}'}}]
        score = score_tool_args_match(pred, ref)
        assert isinstance(score, float)

    def test_ref_args_as_int_string(self):
        """Reference arguments stored as '123' string (non-dict JSON)."""
        pred = [{"name": "get_item", "arguments": {"id": "123"}}]
        ref = [{"function": {"name": "get_item", "arguments": "123"}}]
        score = score_tool_args_match(pred, ref)
        assert isinstance(score, float)


# ═══════════════════════════════════════════════════════════════════════
#  score_response_overlap
# ═══════════════════════════════════════════════════════════════════════


class TestScoreResponseOverlap:
    def test_identical(self):
        assert score_response_overlap("hello world", "hello world") == 1.0

    def test_no_overlap(self):
        assert score_response_overlap("abc def", "xyz uvw") == 0.0

    def test_partial_overlap(self):
        score = score_response_overlap("hello world foo", "hello world bar")
        # pred_tokens = {hello, world, foo}, ref_tokens = {hello, world, bar}
        # overlap = 2, precision = 2/3, recall = 2/3, F1 = 2/3
        assert abs(score - 2 / 3) < 0.01

    def test_empty_predicted(self):
        assert score_response_overlap("", "hello") == 0.0

    def test_empty_reference(self):
        assert score_response_overlap("hello", "") == 0.5  # neutral

    def test_both_empty(self):
        # reference is empty → return 0.5
        assert score_response_overlap("", "") == 0.5

    def test_case_insensitive(self):
        assert score_response_overlap("HELLO WORLD", "hello world") == 1.0

    def test_subset_of_reference(self):
        score = score_response_overlap("hello", "hello world foo bar")
        # pred_tokens = {hello}, ref_tokens = {hello, world, foo, bar}
        # overlap = 1, precision = 1/1 = 1.0, recall = 1/4 = 0.25
        # F1 = 2 * 1.0 * 0.25 / 1.25 = 0.4
        assert abs(score - 0.4) < 0.01

    def test_superset_of_reference(self):
        score = score_response_overlap("hello world foo bar", "hello")
        # pred_tokens = {hello, world, foo, bar}, ref_tokens = {hello}
        # overlap = 1, precision = 1/4, recall = 1/1
        # F1 = 2 * 0.25 * 1.0 / 1.25 = 0.4
        assert abs(score - 0.4) < 0.01

    def test_repeated_words_in_predicted(self):
        """Set-based overlap: repeats don't matter."""
        score = score_response_overlap("hello hello hello", "hello world")
        # pred_tokens = {hello}, ref_tokens = {hello, world}
        # overlap = 1, precision = 1/1, recall = 1/2
        # F1 = 2 * 1.0 * 0.5 / 1.5 = 2/3
        assert abs(score - 2 / 3) < 0.01

    def test_long_response(self):
        pred = " ".join([f"word{i}" for i in range(100)])
        ref = " ".join([f"word{i}" for i in range(50, 150)])
        score = score_response_overlap(pred, ref)
        # 50 overlapping words out of 100 in each
        # precision = 50/100, recall = 50/100, F1 = 0.5
        assert abs(score - 0.5) < 0.01


# ═══════════════════════════════════════════════════════════════════════
#  compute_accuracy (end-to-end)
# ═══════════════════════════════════════════════════════════════════════


class TestComputeAccuracy:
    def test_perfect_tool_call(self):
        generated = "<think>Place the order</think>\n<action>place_order(user_id=u123)</action>"
        ref_response = "Placing your order now."
        ref_calls = [{"function": {"name": "place_order", "arguments": '{"user_id": "u123"}'}}]
        result = compute_accuracy(generated, ref_response, ref_calls)
        assert result.tool_name_match == 1.0
        assert result.tool_args_match == 1.0
        assert result.score > 0.5

    def test_wrong_tool_name(self):
        generated = "<action>cancel_order(id=x)</action>"
        ref_calls = [{"function": {"name": "place_order", "arguments": "{}"}}]
        result = compute_accuracy(generated, "", ref_calls)
        assert result.tool_name_match == 0.0
        assert result.score < 0.5

    def test_no_tools_identical_response(self):
        generated = "The answer is 42."
        result = compute_accuracy(generated, "The answer is 42.", [])
        assert result.response_overlap == 1.0
        # No tools expected: overall = 0.3 * tool_score + 0.7 * resp_overlap
        # tool_score = 0.5 * 1.0 + 0.5 * 1.0 = 1.0 (no ref, no pred → 1.0 each)
        # overall = 0.3 * 1.0 + 0.7 * 1.0 = 1.0
        assert result.score == 1.0

    def test_no_tools_different_response(self):
        generated = "I don't know."
        result = compute_accuracy(generated, "The answer is 42.", [])
        assert result.response_overlap < 0.5
        assert result.score < 1.0

    def test_returns_accuracy_result_type(self):
        result = compute_accuracy("test", "test", [])
        assert isinstance(result, AccuracyResult)

    def test_custom_weights(self):
        generated = "<action>get_weather(city=London)</action>"
        ref_response = "The weather is sunny."
        ref_calls = [{"function": {"name": "get_weather", "arguments": '{"city": "London"}'}}]

        # High tool weight
        result_tool = compute_accuracy(
            generated, ref_response, ref_calls, tool_weight=0.9, response_weight=0.1
        )
        # High response weight
        result_resp = compute_accuracy(
            generated, ref_response, ref_calls, tool_weight=0.1, response_weight=0.9
        )

        # Tool match is perfect, response overlap is partial
        # So higher tool_weight should give higher score
        assert result_tool.score > result_resp.score

    def test_multiple_reference_tools(self):
        generated = (
            "<action>search_flights(origin=SVO, dest=LED)</action>\n"
            "<action>book_flight(flight_id=SU42)</action>"
        )
        ref_calls = [
            {"function": {"name": "search_flights", "arguments": '{"origin": "SVO", "dest": "LED"}'}},
            {"function": {"name": "book_flight", "arguments": '{"flight_id": "SU42"}'}},
        ]
        result = compute_accuracy(generated, "", ref_calls)
        assert result.tool_name_match == 1.0
        assert result.tool_args_match == 1.0

    def test_empty_generation(self):
        result = compute_accuracy("", "reference response", [])
        assert result.response_overlap == 0.0

    def test_empty_everything(self):
        result = compute_accuracy("", "", [])
        # No ref → resp_overlap = 0.5 (neutral)
        # No tools → name_match = 1.0, args_match = 1.0
        assert result.tool_name_match == 1.0
        assert result.response_overlap == 0.5

    def test_score_between_0_and_1(self):
        """Accuracy score should always be in [0, 1]."""
        test_cases = [
            ("", "", []),
            ("hello", "world", []),
            ("<action>foo()</action>", "bar", [{"function": {"name": "baz", "arguments": "{}"}}]),
            ("<action>place_order(id=1)</action>", "done", [{"function": {"name": "place_order", "arguments": '{"id": "1"}'}}]),
        ]
        for gen, ref, calls in test_cases:
            result = compute_accuracy(gen, ref, calls)
            assert 0.0 <= result.score <= 1.0, (
                f"Score {result.score} out of range for gen={gen!r}"
            )

    def test_tool_calls_present_but_not_expected(self):
        """Model called tools but reference has no tool calls."""
        generated = "<action>get_weather(city=London)</action>"
        result = compute_accuracy(generated, "It's sunny.", [])
        # tool_name_match: no ref, but pred exists → 0.5
        # tool_args_match: no ref → 1.0
        # tool_score = 0.5 * 0.5 + 0.5 * 1.0 = 0.75
        # No tools expected: overall = 0.3 * 0.75 + 0.7 * resp_overlap
        assert result.tool_name_match == 0.5


# ═══════════════════════════════════════════════════════════════════════
#  Regex patterns direct tests
# ═══════════════════════════════════════════════════════════════════════


class TestRegexPatterns:
    def test_action_pattern_basic(self):
        m = ACTION_PATTERN.search("<action>foo(bar=1)</action>")
        assert m is not None
        assert m.group(1) == "foo"
        assert m.group(2) == "bar=1"

    def test_action_pattern_no_match_on_bare_text(self):
        m = ACTION_PATTERN.search("foo(bar=1)")
        assert m is None

    def test_function_call_pattern(self):
        text = '{"name": "get_weather", "arguments": {"city": "London"}}'
        m = FUNCTION_CALL_PATTERN.search(text)
        assert m is not None
        assert m.group(1) == "get_weather"

    def test_action_pattern_empty_args(self):
        m = ACTION_PATTERN.search("<action>status()</action>")
        assert m is not None
        assert m.group(1) == "status"
        assert m.group(2).strip() == ""


# ═══════════════════════════════════════════════════════════════════════
#  Integration: realistic model outputs
# ═══════════════════════════════════════════════════════════════════════


class TestRealisticModelOutputs:
    """Test with outputs resembling actual model generations."""

    GOLDEN_TOOL_CALL = (
        "<think>The user wants the current weather in London. "
        "I should call the get_weather function.</think>\n"
        '<action>get_weather(city="London", units="metric")</action>'
    )

    GOLDEN_ANSWER = (
        "<think>The user asked for the capital of France. "
        "I know the answer without needing any tool.</think>\n"
        "<answer>The capital of France is Paris.</answer>"
    )

    BAD_NO_TAGS = (
        "Sure! Let me check the weather for you.\n"
        "The weather in London is 15°C with light rain."
    )

    def test_golden_tool_call_accuracy(self):
        ref_calls = [{"function": {"name": "get_weather", "arguments": '{"city": "London", "units": "metric"}'}}]
        result = compute_accuracy(
            self.GOLDEN_TOOL_CALL,
            "The weather in London is sunny.",
            ref_calls,
        )
        assert result.tool_name_match == 1.0
        assert result.score > 0.5

    def test_golden_answer_accuracy(self):
        result = compute_accuracy(
            self.GOLDEN_ANSWER,
            "The capital of France is Paris.",
            [],
        )
        # NOTE: response_overlap is ~0.4 (not >0.5) because score_response_overlap
        # operates on the raw generated text including <think>/<answer> tags.
        # Tags like "<think>the" and "paris.</answer>" are treated as separate
        # tokens that don't match reference words, reducing precision.
        # This is a known limitation — stripping tags before F1 would improve this.
        assert result.response_overlap > 0.3
        assert result.score > 0.5

    def test_bad_output_accuracy(self):
        ref_calls = [{"function": {"name": "get_weather", "arguments": '{"city": "London"}'}}]
        result = compute_accuracy(
            self.BAD_NO_TAGS,
            "The weather in London is 15°C.",
            ref_calls,
        )
        # No tool calls extracted → name_match = 0.0
        assert result.tool_name_match == 0.0
        # But there's some response overlap
        assert result.response_overlap > 0.0

    def test_partial_tool_match(self):
        """Model gets tool name right but wrong arguments."""
        generated = "<action>get_weather(city=Paris)</action>"
        ref_calls = [{"function": {"name": "get_weather", "arguments": '{"city": "London"}'}}]
        result = compute_accuracy(generated, "", ref_calls)
        assert result.tool_name_match == 1.0
        assert result.tool_args_match == 0.5  # key match but value mismatch

    def test_multitool_partial(self):
        """Model gets one tool right out of two."""
        generated = (
            "<action>get_weather(city=London)</action>\n"
            "<action>wrong_tool(x=1)</action>"
        )
        ref_calls = [
            {"function": {"name": "get_weather", "arguments": '{"city": "London"}'}},
            {"function": {"name": "book_hotel", "arguments": '{"hotel": "Ritz"}'}},
        ]
        result = compute_accuracy(generated, "", ref_calls)
        assert result.tool_name_match == 0.5  # 1 out of 2
