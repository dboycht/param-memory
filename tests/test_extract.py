"""Guards for the conversation-to-memory extraction parser.

An extraction error does not stay in a log: it is written into the weights and read back
as memory. So the parser refuses to guess -- anything it cannot parse returns no pairs,
and every shape a hosted model plausibly returns is pinned here.
"""

from __future__ import annotations

from parammem.bench.extract import build_extraction_prompt, flatten_turns, parse_pairs


def test_plain_json_array():
    text = '[{"question": "How long is my commute?", "answer": "45 minutes each way"}]'
    assert parse_pairs(text) == [{"question": "How long is my commute?",
                                  "answer": "45 minutes each way"}]


def test_fenced_json_with_a_preamble():
    text = ('Here are the items:\n```json\n'
            '[{"question": "What camera do I use?", "answer": "a Canon AE-1"}]\n```')
    assert parse_pairs(text) == [{"question": "What camera do I use?",
                                  "answer": "a Canon AE-1"}]


def test_single_object_instead_of_an_array():
    text = '{"question": "q", "answer": "a"}'
    assert parse_pairs(text) == [{"question": "q", "answer": "a"}]


def test_numbered_text_form():
    text = "1. Q: How long is my commute?\n   A: 45 minutes each way\n" \
           "2. Q: What is my dog's name?\n   A: Biscuit"
    assert parse_pairs(text) == [
        {"question": "How long is my commute?", "answer": "45 minutes each way"},
        {"question": "What is my dog's name?", "answer": "Biscuit"},
    ]


def test_items_missing_a_field_are_dropped_not_repaired():
    text = '[{"question": "kept", "answer": "yes"}, {"question": "no answer"}, ' \
           '{"answer": "no question"}, {"question": "  ", "answer": "blank"}]'
    assert parse_pairs(text) == [{"question": "kept", "answer": "yes"}]


def test_unparseable_reply_yields_nothing():
    """No guessing: an invented memory would be indistinguishable from a real one."""
    assert parse_pairs("I could not find any durable facts.") == []
    assert parse_pairs("") == []
    assert parse_pairs("```json\n{\"question\": \"truncated\"") == []


def test_truncated_json_falls_back_to_nothing_rather_than_a_partial_pair():
    text = '[{"question": "How long is my commute?", "answer": "45 min'
    assert parse_pairs(text) == []


def test_turns_are_flattened_with_roles_and_empty_turns_dropped():
    session = [{"role": "user", "content": " hello "},
               {"role": "assistant", "content": ""},
               {"role": "assistant", "content": "world"}]
    assert flatten_turns(session) == ["user: hello", "assistant: world"]


def test_prompt_carries_the_rules_and_the_excerpt():
    prompt = build_extraction_prompt(["user: I collect cameras"])
    assert "JSON array" in prompt
    assert "user: I collect cameras" in prompt
