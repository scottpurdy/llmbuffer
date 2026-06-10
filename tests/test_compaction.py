"""Compaction threshold, target-size, and hook tests."""

from llmbuffer import OpenAIAdapter, PromptManager, functional, new_state


def _msg(i, size=400):
    return {"role": "user" if i % 2 == 0 else "assistant", "content": f"m{i} " + "x" * size}


def test_no_compaction_without_limits():
    manager = PromptManager()
    for i in range(50):
        manager.append(_msg(i))
    assert len(manager.long_lived_history) == 50


def test_manager_compaction_triggers_over_threshold():
    manager = PromptManager(
        max_tokens=1000,
        compaction_threshold=1000,
        post_compaction_token_threshold=500,
    )
    for i in range(20):
        manager.append(_msg(i))  # ~100 tokens each
    # Compaction happened (history was truncated) and never exceeds the trigger
    assert len(manager.long_lived_history) < 20
    tokens = manager.adapter.count_tokens(manager.long_lived_history)
    assert tokens <= 1000
    # Newest messages survive truncation
    assert manager.long_lived_history[-1]["content"].startswith("m19")


def test_functional_compact_is_explicit():
    # The functional API never compacts on append; compact() must be called.
    state = new_state()
    for i in range(20):
        state = functional.append_message(state, _msg(i))
    assert len(state["long_lived"]) == 20  # untouched despite being huge

    compacted = functional.compact(
        state, max_tokens=1000, post_compaction_token_threshold=500
    )
    adapter = OpenAIAdapter()
    assert adapter.count_tokens(compacted["long_lived"]) <= 500
    assert compacted["long_lived"][-1]["content"].startswith("m19")
    # Original state untouched (pure function)
    assert len(state["long_lived"]) == 20


def test_compact_defaults_derive_from_max_tokens():
    state = new_state()
    for i in range(20):
        state = functional.append_message(state, _msg(i))
    adapter = OpenAIAdapter()
    # threshold defaults to max_tokens; target defaults to max_tokens // 2
    compacted = functional.compact(state, max_tokens=1000)
    assert adapter.count_tokens(compacted["long_lived"]) <= 500


def test_compact_under_threshold_returns_state_unchanged():
    state = new_state()
    for i in range(5):
        state = functional.append_message(state, _msg(i))
    result = functional.compact(state, max_tokens=100_000)
    assert result == state


def test_custom_compaction_hook_summarizes():
    def summarize(messages, target_tokens, adapter):
        return [{"role": "system", "content": f"[summary of {len(messages)} messages]"}]

    manager = PromptManager(max_tokens=500, compaction_hook=summarize)
    for i in range(10):
        manager.append(_msg(i))
    history = manager.long_lived_history
    assert any("[summary of" in m["content"] for m in history)
