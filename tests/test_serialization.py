"""State serialization, validation, and benchmark-suite tests."""

import json

import pytest

from llmbuffer import PromptConfig, functional, loads, new_state
from llmbuffer.benchmark import run_simulated
from llmbuffer.state import dumps, validate_state


def test_round_trip():
    config = PromptConfig(transition_mode="manual")
    state = new_state()
    state = functional.append_message(state, {"role": "user", "content": "q"}, config)
    restored = loads(dumps(state))
    assert restored == state


def test_validation_rejects_bad_shapes():
    with pytest.raises(TypeError):
        validate_state([])
    with pytest.raises(ValueError):
        validate_state({"version": 99, "long_lived": [], "short_term": []})
    with pytest.raises(ValueError):
        validate_state({"version": 1, "long_lived": [{"no_role": 1}], "short_term": []})
    with pytest.raises(ValueError):
        validate_state({"version": 1, "long_lived": "nope", "short_term": []})


def test_state_is_plain_json():
    state = new_state()
    assert json.loads(json.dumps(state)) == state


def test_simulated_benchmark_reports_cache_hits():
    report, _ = run_simulated(n_turns=5)
    # First turn is always a miss; later turns must hit the stable prefix.
    assert report.turns[0].cache_hit is False
    assert all(t.cache_hit for t in report.turns[1:])
    assert 0 < report.cache_hit_ratio < 1
    costs = report.cost_estimate()
    assert costs["savings_usd"] > 0

    d = report.to_dict()
    assert d["total_cached_tokens"] > 0
    md = report.to_markdown()
    assert "Cache hit ratio" in md
    assert report.turns[0].dynamic_changed is True  # turn 1 always starts new context


def test_simulated_comparison_shows_naive_worse():
    llmbuf, naive = run_simulated(n_turns=10, compare=True)
    assert naive is not None
    # After dynamic context rotates (turn 4, 7, ...) naive should have more misses.
    assert llmbuf.cache_hit_ratio > naive.cache_hit_ratio
