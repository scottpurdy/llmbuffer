"""Stateless functional interface.

Every function takes the current state (a plain dict) plus a
:class:`~llmbuffer.config.PromptConfig` and returns a new state, never
mutating the input. This lets stateless web apps round-trip the state
through JSON between requests::

    state = loads(row.conversation_state)
    state = append_message(state, {"role": "user", "content": text}, config)
    messages = build_messages(state, config, dynamic_system_prompt=rag_context)
    ...call the LLM...
    state = append_message(state, reply, config)
    row.conversation_state = dumps(state)
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Tuple

from .config import PromptConfig, TransitionMode
from .hooks import identity_transition_hook, truncation_compaction_hook
from .state import Message, State, validate_state


def _is_agent_cycle_end(message: Message) -> bool:
    """A final assistant message (no pending tool calls) ends an agent cycle."""
    return message.get("role") == "assistant" and not message.get("tool_calls")


def transition(state: State, config: PromptConfig) -> State:
    """Move all short-term messages into the long-lived history.

    Applies the configured transition hook (identity by default), then
    runs compaction if the long-lived history exceeds its threshold.
    """
    validate_state(state)
    hook = config.transition_hook or identity_transition_hook
    committed = hook(copy.deepcopy(state["short_term"]))
    new = {
        "version": state["version"],
        "long_lived": list(state["long_lived"]) + list(committed),
        "short_term": [],
    }
    return maybe_compact(new, config)


def append_message(state: State, message: Message, config: PromptConfig) -> State:
    """Append a message, applying the configured transition strategy."""
    validate_state(state)
    if "role" not in message:
        raise ValueError("message must have a 'role'")
    message = copy.deepcopy(message)

    if config.transition_mode is TransitionMode.NONE:
        new = {
            "version": state["version"],
            "long_lived": list(state["long_lived"]) + [message],
            "short_term": list(state["short_term"]),
        }
        return maybe_compact(new, config)

    new = {
        "version": state["version"],
        "long_lived": list(state["long_lived"]),
        "short_term": list(state["short_term"]) + [message],
    }
    if config.transition_mode is TransitionMode.AGENT_CYCLE and _is_agent_cycle_end(
        message
    ):
        return transition(new, config)
    return new


def append_messages(
    state: State, messages: List[Message], config: PromptConfig
) -> State:
    """Append several messages in order."""
    for message in messages:
        state = append_message(state, message, config)
    return state


def maybe_compact(state: State, config: PromptConfig) -> State:
    """Compact the long-lived history if it exceeds the threshold.

    Triggered when the long-lived token count exceeds
    ``compaction_threshold`` (default: ``max_tokens``); the compaction hook
    (default: oldest-first truncation) reduces it to
    ``post_compaction_token_threshold`` (default: ``max_tokens // 2``).
    """
    threshold = config.effective_compaction_threshold
    if threshold is None:
        return state
    adapter = config.adapter
    if adapter.count_tokens(state["long_lived"]) <= threshold:
        return state
    target = config.effective_post_compaction_target
    if target is None:
        target = threshold
    hook = config.compaction_hook or truncation_compaction_hook
    compacted = hook(copy.deepcopy(state["long_lived"]), target, adapter)
    return {
        "version": state["version"],
        "long_lived": list(compacted),
        "short_term": list(state["short_term"]),
    }


def build_messages(
    state: State,
    config: PromptConfig,
    dynamic_system_prompt: Optional[str] = None,
    apply_cache_markers: bool = True,
) -> List[Message]:
    """Assemble the cache-optimized message list for an LLM call.

    Strict ordering:

    1. Static system prompt
    2. Long-lived conversation history (stable cache prefix)
    3. Dynamic system prompt (if any)
    4. Short-term conversation history

    Cache markers are injected at the end of the static system prompt and
    at the end of the long-lived history (provider-dependent; a no-op for
    OpenAI-style automatic prefix caching).
    """
    validate_state(state)
    messages: List[Message] = []
    boundaries: List[int] = []

    if config.static_system_prompt:
        messages.append({"role": "system", "content": config.static_system_prompt})
        boundaries.append(0)

    long_lived = copy.deepcopy(state["long_lived"])
    if long_lived:
        messages.extend(long_lived)
        boundaries.append(len(messages) - 1)

    if dynamic_system_prompt:
        messages.append(
            {"role": config.dynamic_system_role, "content": dynamic_system_prompt}
        )

    messages.extend(copy.deepcopy(state["short_term"]))

    if apply_cache_markers:
        messages = config.adapter.apply_cache_markers(messages, boundaries)
    return messages


def cache_prefix(state: State, config: PromptConfig) -> Tuple[Message, ...]:
    """The stable prefix (static system + long-lived history) as a tuple,
    useful for asserting cache stability across turns."""
    prefix: List[Message] = []
    if config.static_system_prompt:
        prefix.append({"role": "system", "content": config.static_system_prompt})
    prefix.extend(copy.deepcopy(state["long_lived"]))
    return tuple(prefix)
