"""Compaction threshold, target-size, and hook tests."""

from llmbuffer import PromptConfig, PromptManager


def _msg(i, size=400):
    return {"role": "user" if i % 2 == 0 else "assistant", "content": f"m{i} " + "x" * size}


def test_no_compaction_without_limits():
    manager = PromptManager(PromptConfig())
    for i in range(50):
        manager.append(_msg(i))
    assert len(manager.long_lived_history) == 50


def test_compaction_triggers_over_threshold_and_hits_target():
    config = PromptConfig(
        max_tokens=1000,
        compaction_threshold=1000,
        post_compaction_token_threshold=500,
    )
    manager = PromptManager(config)
    for i in range(20):
        manager.append(_msg(i))  # ~100 tokens each
    # Compaction happened (history was truncated) and never exceeds the trigger
    assert len(manager.long_lived_history) < 20
    tokens = config.adapter.count_tokens(manager.long_lived_history)
    assert tokens <= 1000
    # Newest messages survive truncation
    assert manager.long_lived_history[-1]["content"].startswith("m19")

    # Immediately after a triggered compaction, size is at/under the target:
    from llmbuffer import functional

    state = manager.state
    state["long_lived"] = state["long_lived"] + [_msg(99, size=2000)]  # push over trigger
    compacted = functional.maybe_compact(state, config)
    assert config.adapter.count_tokens(compacted["long_lived"]) <= 500 or (
        len(compacted["long_lived"]) == 1
    )


def test_compaction_defaults_derive_from_max_tokens():
    config = PromptConfig(max_tokens=800)
    assert config.effective_compaction_threshold == 800
    assert config.effective_post_compaction_target == 400


def test_custom_compaction_hook_summarizes():
    def summarize(messages, target_tokens, adapter):
        return [{"role": "system", "content": f"[summary of {len(messages)} messages]"}]

    config = PromptConfig(max_tokens=500, compaction_hook=summarize)
    manager = PromptManager(config)
    for i in range(10):
        manager.append(_msg(i))
    history = manager.long_lived_history
    assert any("[summary of" in m["content"] for m in history)


def test_under_threshold_history_untouched():
    config = PromptConfig(max_tokens=100_000)
    manager = PromptManager(config)
    for i in range(5):
        manager.append(_msg(i))
    assert len(manager.long_lived_history) == 5
