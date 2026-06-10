"""build_messages metadata and compact_for_request tests."""

import pytest

from llmbuffer import AnthropicAdapter, OpenAIAdapter, PromptManager, functional, new_state


def _msg(i, size=400):
    return {"role": "user" if i % 2 == 0 else "assistant", "content": f"m{i} " + "x" * size}


def _build_state(n=4):
    state = new_state()
    for i in range(n):
        state = functional.append_message(state, _msg(i))
    return state


# -- build_messages metadata -------------------------------------------------


def test_default_return_shape_unchanged():
    messages = functional.build_messages(_build_state(), static_system_prompt="S")
    assert isinstance(messages, list)
    assert all(isinstance(m, dict) for m in messages)


def test_metadata_boundaries_and_counts():
    state = _build_state(4)
    state = functional.append_message(
        state, {"role": "user", "content": "new q"}, transition_mode="manual"
    )
    messages, meta = functional.build_messages(
        state,
        static_system_prompt="STATIC",
        dynamic_system_prompt="DYN",
        with_metadata=True,
    )
    # Prefix = static system (1) + long-lived (4); suffix = dynamic + short-term
    assert meta["prefix_message_count"] == 5
    assert meta["boundaries"] == [0, 4]
    assert len(messages) == 7
    assert meta["prefix_tokens"] > 0
    assert meta["suffix_tokens"] > 0
    assert meta["total_tokens"] == meta["prefix_tokens"] + meta["suffix_tokens"]


def test_metadata_empty_state():
    messages, meta = functional.build_messages(new_state(), with_metadata=True)
    assert messages == []
    assert meta == {
        "boundaries": [],
        "prefix_message_count": 0,
        "prefix_tokens": 0,
        "suffix_tokens": 0,
        "total_tokens": 0,
    }


def test_metadata_tokens_counted_before_markers():
    # Marker injection must not affect the reported token estimates.
    state = _build_state(2)
    _, meta_marked = functional.build_messages(
        state, static_system_prompt="S", adapter=AnthropicAdapter(), with_metadata=True
    )
    _, meta_plain = functional.build_messages(
        state,
        static_system_prompt="S",
        adapter=AnthropicAdapter(),
        apply_cache_markers=False,
        with_metadata=True,
    )
    assert meta_marked == meta_plain


def test_manager_metadata_passthrough():
    manager = PromptManager(static_system_prompt="STATIC")
    manager.append({"role": "user", "content": "q"})
    messages, meta = manager.build_messages(with_metadata=True)
    assert meta["prefix_message_count"] == 2  # static + 1 long-lived
    assert len(messages) == 2


# -- compact_for_request -----------------------------------------------------


def test_compact_for_request_subtracts_static_and_reserved():
    adapter = OpenAIAdapter()
    state = _build_state(20)  # ~100 tokens per message
    static = "s" * 4000  # ~1000 tokens
    compacted = functional.compact_for_request(
        state,
        request_budget=2500,
        static_system_prompt=static,
        reserved_tokens=500,
    )
    # Long-lived budget = 2500 - ~1000 - 500 = ~1000; target = budget // 2
    long_lived_tokens = adapter.count_tokens(compacted["long_lived"])
    assert long_lived_tokens <= 1000
    assert len(compacted["long_lived"]) < 20
    assert compacted["long_lived"][-1]["content"].startswith("m19")


def test_compact_for_request_noop_under_budget():
    state = _build_state(3)
    result = functional.compact_for_request(state, request_budget=100_000)
    assert result == state


def test_compact_for_request_rejects_impossible_budget():
    state = _build_state(3)
    with pytest.raises(ValueError, match="leaves no room"):
        functional.compact_for_request(
            state,
            request_budget=1000,
            static_system_prompt="s" * 4000,  # ~1000 tokens on its own
            reserved_tokens=500,
        )


def test_compact_for_request_deterministic_across_turns():
    # Same budget + same reserved headroom -> same compaction decision,
    # regardless of what the current turn's dynamic content looks like.
    state = _build_state(20)
    once = functional.compact_for_request(state, request_budget=2000, reserved_tokens=300)
    twice = functional.compact_for_request(once, request_budget=2000, reserved_tokens=300)
    assert twice == once  # already under budget; prefix untouched


def test_manager_compact_for_request():
    manager = PromptManager(static_system_prompt="s" * 4000)
    for i in range(20):
        manager.append(_msg(i))
    manager.compact_for_request(request_budget=3000, reserved_tokens=500)
    adapter = manager.adapter
    # budget = 3000 - ~1000 static - 500 reserved = ~1500
    assert adapter.count_tokens(manager.long_lived_history) <= 1500
