"""Tests for the LLM judge (offline: the transport is injected)."""

from __future__ import annotations

import json

import pytest

from parammem.eval.judge import (
    JudgeVerdict,
    LLMJudge,
    build_prompt,
    load_api_key,
    parse_verdicts,
)

KEYS = ["weights", "context"]


# --------------------------------------------------------------- parsing

def test_parses_clean_json():
    reply = '{"weights": {"correct": true, "reason": "matches"}, "context": {"correct": false, "reason": "wrong"}}'
    verdicts = parse_verdicts(reply, KEYS)
    assert verdicts["weights"].correct is True
    assert verdicts["context"].correct is False
    assert verdicts["weights"].reason == "matches"


def test_parses_fenced_json_with_surrounding_prose():
    reply = (
        "Sure, here is the grading:\n```json\n"
        '{"weights": {"correct": true}, "context": {"correct": true}}'
        "\n```\nHope that helps."
    )
    verdicts = parse_verdicts(reply, KEYS)
    assert all(verdicts[k].correct is True for k in KEYS)


@pytest.mark.parametrize("value,expected", [
    (True, True), (False, False),
    ("true", True), ("False", False), ("correct", True), ("incorrect", False),
])
def test_coerces_boolean_spellings(value, expected):
    reply = json.dumps({"weights": {"correct": value}})
    assert parse_verdicts(reply, ["weights"])["weights"].correct is expected


def test_unreadable_reply_is_not_counted_as_either_outcome():
    for reply in ("", "I cannot grade this.", "{not json}", '{"other": 1}'):
        verdicts = parse_verdicts(reply, KEYS)
        assert all(v.correct is None for v in verdicts.values()), reply
        assert all(not v.usable for v in verdicts.values())


def test_missing_label_defaults_to_none_but_others_still_parse():
    reply = '{"weights": {"correct": true}}'
    verdicts = parse_verdicts(reply, KEYS)
    assert verdicts["weights"].correct is True
    assert verdicts["context"].correct is None
    assert "not returned" in verdicts["context"].reason


def test_verdict_usable_flag():
    assert JudgeVerdict(True).usable
    assert not JudgeVerdict(None).usable


# --------------------------------------------------------------- prompt

def test_prompt_carries_reference_and_every_candidate():
    prompt = build_prompt("the capital is Paris", {"weights": "Paris", "context": "Lyon"})
    assert "the capital is Paris" in prompt
    assert "weights: Paris" in prompt
    assert "context: Lyon" in prompt
    assert "JSON" in prompt


# ------------------------------------------------------------ credentials

def test_repr_never_leaks_the_key():
    judge = LLMJudge(base_url="https://api.example/v1", model="m",
                     api_key="sk-super-secret-value")
    text = repr(judge)
    assert "sk-super-secret-value" not in text
    assert "<set,21>" in text


def test_load_api_key_reads_and_validates(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"llm": {"api_key": "sk-abc"}}), encoding="utf-8")
    assert load_api_key(path) == "sk-abc"
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"llm": {}}), encoding="utf-8")
    with pytest.raises(ValueError, match="api_key"):
        load_api_key(bad)


# ------------------------------------------------------------- transport

def _judge(post, **kwargs) -> LLMJudge:
    return LLMJudge(base_url="https://api.example/v1", model="m", api_key="sk-x",
                    post=post, sleep=lambda _s: None, **kwargs)


def test_batch_grading_uses_one_call_for_all_candidates():
    calls = []

    def post(url, headers, payload):
        calls.append((url, headers, payload))
        return json.dumps({"choices": [{"message": {"content":
            json.dumps({k: {"correct": k == "weights"} for k in KEYS})}}]})

    judge = _judge(post)
    verdicts = judge.judge_batch("ref", {"weights": "a", "context": "b"})
    assert len(calls) == 1
    assert verdicts["weights"].correct is True
    assert verdicts["context"].correct is False
    url, headers, payload = calls[0]
    assert url.endswith("/chat/completions")
    assert headers["Authorization"] == "Bearer sk-x"
    assert payload["model"] == "m"
    # temperature is omitted by default: kimi-k2.6 rejects anything but its own value
    assert "temperature" not in payload


def test_temperature_is_sent_only_when_asked_for():
    calls = []

    def post(url, headers, payload):
        calls.append(payload)
        return json.dumps({"choices": [{"message": {"content": "{}"}}]})

    _judge(post).judge_batch("ref", {"a": "x"})
    _judge(post, temperature=0.0).judge_batch("ref", {"a": "x"})
    assert "temperature" not in calls[0]
    assert calls[1]["temperature"] == 0.0


def test_retries_then_succeeds():
    attempts = {"n": 0}

    def flaky_post(url, headers, payload):
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise urllib_error()
        return json.dumps({"choices": [{"message": {"content": '{"weights": {"correct": true}}'}}]})

    judge = _judge(flaky_post)
    assert judge.judge_batch("ref", {"weights": "a"})["weights"].correct is True
    assert attempts["n"] == 3


def test_gives_up_after_the_configured_attempts():
    def always_fail(url, headers, payload):
        raise urllib_error()

    judge = _judge(always_fail)
    with pytest.raises(RuntimeError, match="after 3 attempts"):
        judge.judge_batch("ref", {"weights": "a"})


def test_empty_candidate_set_makes_no_call():
    def post(url, headers, payload):  # pragma: no cover - must not run
        raise AssertionError("no request expected")

    assert _judge(post).judge_batch("ref", {}) == {}


def test_rate_limit_is_honoured():
    slept: list[float] = []

    def post(url, headers, payload):
        return json.dumps({"choices": [{"message": {"content": "{}"}}]})

    judge = LLMJudge(base_url="https://api.example/v1", model="m", api_key="sk-x",
                     post=post, sleep=slept.append, min_interval=5.0)
    judge.judge_batch("ref", {"a": "x"})      # first call: no wait
    judge.judge_batch("ref", {"a": "x"})      # second call: must wait out the interval
    assert len(slept) == 1
    assert 0 < slept[0] <= 5.0


def urllib_error() -> Exception:
    import urllib.error

    return urllib.error.URLError("boom")
