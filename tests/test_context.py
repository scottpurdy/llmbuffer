"""Dynamic-context channel tests: append_context, consolidation at compaction."""

import json

import pytest

from llmbuffer import OpenAIAdapter, PromptManager, functional, new_state


def _msg(i, size=400):
    return {"role": "user" if i % 2 == 0 else "assistant", "content": f"m{i} " + "x" * size}


# -- append_context ----------------------------------------------------------


def test_initial_context_seeds_long_lived():
    state = new_state(initial_context="WORLD v1", context_key="world")
    assert len(state["long_lived"]) == 1
    msg = state["long_lived"][0]
    assert msg["role"] == "system"
    assert msg["content"] == "WORLD v1"
    assert msg["_llmbuffer"] == {"context_key": "world"}


def test_initial_context_identical_to_append_context_on_empty_state():
    seeded = new_state(initial_context="WORLD v1", context_key="world")
    appended = functional.append_context(new_state(), "WORLD v1", key="world")
    assert seeded == appended


def test_append_context_respects_transition_mode():
    state = new_state()
    state = functional.append_context(
        state, "delta 1", key="world", transition_mode="manual"
    )
    # Manual mode: parked in short-term, preserving temporal ordering
    assert state["long_lived"] == []
    assert len(state["short_term"]) == 1
    state = functional.transition(state)
    assert len(state["long_lived"]) == 1


def test_context_temporal_ordering_in_assembled_messages():
    state = new_state(initial_context="WORLD v1")
    state = functional.append_message(
        state, {"role": "user", "content": "q"}, transition_mode="manual"
    )
    state = functional.append_context(state, "delta", transition_mode="manual")
    messages = functional.build_messages(state, static_system_prompt="STATIC")
    # Delta arrives after the in-flight user message, not buried in the prefix
    assert [m["content"] for m in messages] == ["STATIC", "WORLD v1", "q", "delta"]


def test_build_messages_strips_llmbuffer_metadata():
    state = new_state(initial_context="WORLD v1")
    state = functional.append_context(state, "delta", transition_mode="manual")
    messages = functional.build_messages(state, static_system_prompt="S")
    assert all("_llmbuffer" not in m for m in messages)
    # State still carries the tags
    assert state["long_lived"][0]["_llmbuffer"]["context_key"] == "context"


def test_cache_prefix_strips_metadata_and_stays_stable_across_deltas():
    state = new_state(initial_context="WORLD v1")
    state = functional.append_message(state, {"role": "user", "content": "q1"})
    p1 = functional.cache_prefix(state, static_system_prompt="S")
    assert all("_llmbuffer" not in m for m in p1)
    # Appending a delta only appends — existing prefix is untouched
    state = functional.append_context(state, "delta")
    p2 = functional.cache_prefix(state, static_system_prompt="S")
    assert p2[: len(p1)] == p1


def test_context_tags_survive_serialization():
    from llmbuffer import dumps, loads

    state = new_state(initial_context="WORLD v1", context_key="world")
    restored = loads(dumps(state))
    assert restored == state


# -- consolidation at compaction ----------------------------------------------


def _state_with_context_and_history(n_msgs=20):
    state = new_state(initial_context="WORLD v1 " + "w" * 400, context_key="world")
    for i in range(n_msgs):
        state = functional.append_message(state, _msg(i))
        if i == 5:
            state = functional.append_context(state, "delta A", key="world")
        if i == 10:
            state = functional.append_context(state, "delta B", key="world")
    return state


def test_compaction_consolidates_context_in_order():
    seen = {}

    def consolidate(key, messages):
        seen[key] = [m["content"] for m in messages]
        return "WORLD v2 (rewritten)"

    state = _state_with_context_and_history()
    compacted = functional.compact(
        state, max_tokens=1000, context_consolidation_hook=consolidate
    )
    # Hook saw initial block first, then deltas in history order
    assert seen["world"][0].startswith("WORLD v1")
    assert seen["world"][1:] == ["delta A", "delta B"]
    # Consolidated block sits at the front, keyed for the next round
    first = compacted["long_lived"][0]
    assert first["content"] == "WORLD v2 (rewritten)"
    assert first["_llmbuffer"] == {"context_key": "world"}
    # No other keyed messages remain
    assert sum(1 for m in compacted["long_lived"] if "_llmbuffer" in m) == 1


def test_lossy_hook_never_sees_keyed_messages():
    seen_by_lossy = []

    def lossy(messages, target, adapter):
        seen_by_lossy.extend(messages)
        return messages[-2:]

    state = _state_with_context_and_history()
    functional.compact(
        state, max_tokens=1000, compaction_hook=lossy
    )
    assert all("_llmbuffer" not in m for m in seen_by_lossy)


def test_both_phases_always_run():
    # Even when consolidation alone would get under target, the lossy hook
    # still runs (compact all the way down to maximize stable stretch).
    lossy_ran = []

    def lossy(messages, target, adapter):
        lossy_ran.append(True)
        return messages

    state = _state_with_context_and_history()
    functional.compact(state, max_tokens=1000, compaction_hook=lossy)
    assert lossy_ran


def test_default_concat_consolidation_is_lossless():
    state = _state_with_context_and_history()
    compacted = functional.compact(state, max_tokens=1000)
    first = compacted["long_lived"][0]
    assert first["content"].startswith("WORLD v1")
    assert "delta A" in first["content"]
    assert "delta B" in first["content"]


def test_total_after_compaction_fits_target():
    adapter = OpenAIAdapter()
    state = _state_with_context_and_history(30)
    compacted = functional.compact(state, max_tokens=1500)
    # Consolidated blocks + compacted rest fit the post-compaction target
    assert adapter.count_tokens(compacted["long_lived"]) <= 750 + 110  # +1 min kept msg


def test_short_term_deltas_not_consolidated():
    state = _state_with_context_and_history()
    state = functional.append_context(
        state, "in-flight delta", key="world", transition_mode="manual"
    )
    compacted = functional.compact(state, max_tokens=1000)
    # The in-flight delta stays in short-term, untouched
    assert compacted["short_term"][-1]["content"] == "in-flight delta"
    # It consolidates after transitioning, at the next compaction
    state2 = functional.transition(compacted)
    compacted2 = functional.compact(state2, max_tokens=10)  # force trigger
    assert "in-flight delta" in compacted2["long_lived"][0]["content"]


def test_multiple_keys_consolidate_separately():
    state = new_state(initial_context="WORLD", context_key="world")
    state = functional.append_context(state, "PROFILE", key="profile")
    for i in range(20):
        state = functional.append_message(state, _msg(i))
    state = functional.append_context(state, "world delta", key="world")
    compacted = functional.compact(state, max_tokens=1000)
    keys = [
        m["_llmbuffer"]["context_key"]
        for m in compacted["long_lived"]
        if "_llmbuffer" in m
    ]
    assert keys == ["world", "profile"]  # ordered by first appearance
    assert "world delta" in compacted["long_lived"][0]["content"]
    assert compacted["long_lived"][1]["content"] == "PROFILE"


# -- manager integration -------------------------------------------------------


def test_manager_initial_context_and_append_context():
    manager = PromptManager(
        static_system_prompt="STATIC",
        initial_context="WORLD v1",
        context_key="world",
    )
    manager.append({"role": "user", "content": "q"})
    manager.append_context("delta A")  # defaults to manager's context_key
    messages = manager.build_messages()
    assert [m["content"] for m in messages] == ["STATIC", "WORLD v1", "q", "delta A"]
    assert all("_llmbuffer" not in m for m in messages)


def test_manager_rejects_state_plus_initial_context():
    with pytest.raises(ValueError, match="not both"):
        PromptManager(state=new_state(), initial_context="X")


def test_manager_consolidation_hook_wired_through():
    def consolidate(key, messages):
        return f"[{key}: {len(messages)} merged]"

    manager = PromptManager(
        max_tokens=800,
        initial_context="WORLD " + "w" * 400,
        context_key="world",
        context_consolidation_hook=consolidate,
    )
    for i in range(10):
        manager.append(_msg(i))
    contents = [m["content"] for m in manager.long_lived_history]
    assert any(c.startswith("[world:") for c in contents)
