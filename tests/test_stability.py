"""Cache-prefix stability tests.

These verify the spec's core promise: the static system prompt and
long-lived history are never modified, mutated, or re-ordered across
turns, so the provider cache prefix stays byte-stable.
"""

import json

from llmbuffer import PromptConfig, PromptManager, functional, new_state


def _snapshot_prefix(messages, prefix_len):
    return json.dumps(messages[:prefix_len], sort_keys=True)


def test_prefix_identical_across_turns():
    config = PromptConfig(static_system_prompt="STATIC", transition_mode="agent_cycle")
    manager = PromptManager(config)
    manager.append({"role": "user", "content": "q1"})
    manager.append({"role": "assistant", "content": "a1"})  # cycle ends -> transition

    baseline = manager.build_messages(apply_cache_markers=False)
    prefix_len = len(baseline)  # static + 2 long-lived

    # Several more turns; the original prefix must remain byte-identical.
    expected = _snapshot_prefix(baseline, prefix_len)
    for i in range(2, 6):
        manager.append({"role": "user", "content": f"q{i}"})
        mid_turn = manager.build_messages(
            dynamic_system_prompt=f"time={i}", apply_cache_markers=False
        )
        assert _snapshot_prefix(mid_turn, prefix_len) == expected
        manager.append({"role": "assistant", "content": f"a{i}"})
        post_turn = manager.build_messages(apply_cache_markers=False)
        assert _snapshot_prefix(post_turn, prefix_len) == expected


def test_dynamic_prompt_changes_do_not_touch_prefix():
    config = PromptConfig(static_system_prompt="STATIC")
    state = functional.append_message(
        new_state(), {"role": "user", "content": "q"}, config
    )
    a = functional.build_messages(state, config, dynamic_system_prompt="t=1")
    b = functional.build_messages(state, config, dynamic_system_prompt="t=2")
    assert a[:2] == b[:2]  # static + long-lived identical
    assert a[2] != b[2]


def test_build_messages_does_not_mutate_state():
    config = PromptConfig(static_system_prompt="STATIC")
    state = functional.append_message(
        new_state(), {"role": "user", "content": "q"}, config
    )
    before = json.dumps(state, sort_keys=True)
    messages = functional.build_messages(state, config)
    messages[0]["content"] = "TAMPERED"
    messages[-1]["content"] = "TAMPERED"
    assert json.dumps(state, sort_keys=True) == before


def test_cache_markers_do_not_mutate_state():
    from llmbuffer import AnthropicAdapter

    config = PromptConfig(static_system_prompt="STATIC", adapter=AnthropicAdapter())
    manager = PromptManager(config)
    manager.append({"role": "user", "content": "q"})
    before = manager.to_json()
    manager.build_messages(dynamic_system_prompt="dyn")
    assert manager.to_json() == before


def test_cache_prefix_helper_stable_identity():
    config = PromptConfig(static_system_prompt="STATIC", transition_mode="manual")
    manager = PromptManager(config)
    manager.append({"role": "user", "content": "q1"}).append(
        {"role": "assistant", "content": "a1"}
    ).transition()
    p1 = manager.cache_prefix()
    manager.append({"role": "user", "content": "q2"})
    p2 = manager.cache_prefix()
    assert p1 == p2


def test_serialization_round_trip_preserves_prefix():
    config = PromptConfig(static_system_prompt="STATIC", transition_mode="manual")
    manager = PromptManager(config)
    manager.append({"role": "user", "content": "q"}).transition()
    payload = manager.to_json()
    restored = PromptManager.from_json(payload, config)
    assert restored.build_messages() == manager.build_messages()
    assert restored.state == manager.state
