"""Message-assembly ordering and cache-marker tests."""

from llmbuffer import (
    AnthropicAdapter,
    PromptConfig,
    PromptManager,
    functional,
    new_state,
)


def test_strict_ordering():
    config = PromptConfig(static_system_prompt="STATIC", transition_mode="manual")
    state = new_state()
    # Two committed turns
    state = functional.append_message(state, {"role": "user", "content": "old q"}, config)
    state = functional.append_message(state, {"role": "assistant", "content": "old a"}, config)
    state = functional.transition(state, config)
    # One short-term turn
    state = functional.append_message(state, {"role": "user", "content": "new q"}, config)

    messages = functional.build_messages(state, config, dynamic_system_prompt="DYNAMIC")
    assert [m["content"] for m in messages] == ["STATIC", "old q", "old a", "DYNAMIC", "new q"]
    assert messages[0]["role"] == "system"
    assert messages[3]["role"] == "system"


def test_no_static_no_dynamic():
    config = PromptConfig()
    state = functional.append_message(
        new_state(), {"role": "user", "content": "hi"}, config
    )
    assert functional.build_messages(state, config) == [{"role": "user", "content": "hi"}]


def test_dynamic_system_role_override():
    config = PromptConfig(dynamic_system_role="user")
    messages = functional.build_messages(new_state(), config, dynamic_system_prompt="ctx")
    assert messages == [{"role": "user", "content": "ctx"}]


def test_anthropic_cache_markers_at_boundaries():
    config = PromptConfig(
        static_system_prompt="STATIC",
        adapter=AnthropicAdapter(),
        transition_mode="manual",
    )
    manager = PromptManager(config)
    manager.append({"role": "user", "content": "q1"})
    manager.append({"role": "assistant", "content": "a1"})
    manager.transition()
    manager.append({"role": "user", "content": "q2"})

    messages = manager.build_messages(dynamic_system_prompt="DYN")
    # Marker on static system prompt
    assert messages[0]["content"][-1]["cache_control"] == {"type": "ephemeral"}
    # Marker on the last long-lived message (index 2 = "a1")
    assert messages[2]["content"][-1]["cache_control"] == {"type": "ephemeral"}
    # No markers on dynamic or short-term messages
    assert messages[3]["content"] == "DYN"
    assert messages[4]["content"] == "q2"


def test_openai_adapter_injects_no_markers():
    config = PromptConfig(static_system_prompt="STATIC")
    manager = PromptManager(config)
    manager.append({"role": "user", "content": "q"})
    for msg in manager.build_messages():
        assert "cache_control" not in str(msg)


def test_anthropic_markers_on_block_content():
    adapter = AnthropicAdapter()
    messages = [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]
    marked = adapter.apply_cache_markers(messages, [0])
    assert marked[0]["content"][-1]["cache_control"] == {"type": "ephemeral"}
    # Original untouched
    assert "cache_control" not in messages[0]["content"][-1]
