"""Transition-mode tests: none, manual, agent_cycle."""

from llmbuffer import PromptConfig, PromptManager, TransitionMode, functional, new_state


def test_mode_none_goes_straight_to_long_lived():
    manager = PromptManager(PromptConfig(transition_mode=TransitionMode.NONE))
    manager.append({"role": "user", "content": "q"})
    manager.append({"role": "assistant", "content": "a"})
    assert len(manager.long_lived_history) == 2
    assert manager.short_term_history == []


def test_mode_manual_holds_until_transition():
    manager = PromptManager(PromptConfig(transition_mode="manual"))
    manager.append({"role": "user", "content": "q"})
    manager.append({"role": "assistant", "content": "a"})
    assert manager.long_lived_history == []
    assert len(manager.short_term_history) == 2
    manager.transition()
    assert len(manager.long_lived_history) == 2
    assert manager.short_term_history == []


def test_mode_agent_cycle_transitions_on_final_assistant_message():
    manager = PromptManager(PromptConfig(transition_mode="agent_cycle"))
    manager.append({"role": "user", "content": "q"})
    manager.append(
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "1", "type": "function", "function": {"name": "f", "arguments": "{}"}}],
        }
    )
    manager.append({"role": "tool", "tool_call_id": "1", "content": "result"})
    # Cycle not over: tool call pending resolution into a final answer
    assert manager.long_lived_history == []
    assert len(manager.short_term_history) == 3

    manager.append({"role": "assistant", "content": "final answer"})
    # Final assistant message (no tool_calls) ends the cycle
    assert len(manager.long_lived_history) == 4
    assert manager.short_term_history == []


def test_transition_hook_filters_messages():
    from llmbuffer import drop_tool_messages_transition_hook

    config = PromptConfig(
        transition_mode="agent_cycle",
        transition_hook=drop_tool_messages_transition_hook,
    )
    manager = PromptManager(config)
    manager.append({"role": "user", "content": "q"})
    manager.append({"role": "assistant", "tool_calls": [{"id": "1"}], "content": None})
    manager.append({"role": "tool", "tool_call_id": "1", "content": "noisy output"})
    manager.append({"role": "assistant", "content": "answer"})
    roles = [m["role"] for m in manager.long_lived_history]
    assert roles == ["user", "assistant"]
    assert manager.long_lived_history[1]["content"] == "answer"


def test_functional_append_is_pure():
    config = PromptConfig(transition_mode="manual")
    state = new_state()
    state2 = functional.append_message(state, {"role": "user", "content": "q"}, config)
    assert state["short_term"] == []
    assert len(state2["short_term"]) == 1
    # Mutating the input message afterwards must not affect stored state
    msg = {"role": "user", "content": "x"}
    state3 = functional.append_message(state2, msg, config)
    msg["content"] = "mutated"
    assert state3["short_term"][1]["content"] == "x"
