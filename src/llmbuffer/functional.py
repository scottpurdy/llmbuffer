"""Stateless functional interface.

Every function takes the current state (a plain dict) plus only the
settings it actually needs, and returns a new state — never mutating the
input. This lets stateless web apps round-trip the state through JSON
between requests::

    state = loads(row.conversation_state)
    state = append_message(state, {"role": "user", "content": text},
                           transition_mode="manual")
    messages = build_messages(state, static_system_prompt=SYSTEM,
                              dynamic_system_prompt=rag_context)
    ...call the LLM...
    state = append_message(state, reply, transition_mode="manual")
    state = compact(state, max_tokens=8_000)
    row.conversation_state = dumps(state)

Compaction is explicit: call :func:`compact` when you want the long-lived
history checked against a token budget. (The stateful
:class:`~llmbuffer.manager.PromptManager` calls it for you.)
"""

from __future__ import annotations

import copy
from typing import List, Optional, Tuple, Union

from .adapters import OpenAIAdapter, ProviderAdapter
from .config import CompactionHook, TransitionHook, TransitionMode
from .hooks import identity_transition_hook, truncation_compaction_hook
from .state import Message, State, validate_state

_DEFAULT_ADAPTER = OpenAIAdapter()


def _is_agent_cycle_end(message: Message) -> bool:
    """A final assistant message (no pending tool calls) ends an agent cycle."""
    return message.get("role") == "assistant" and not message.get("tool_calls")


def transition(
    state: State,
    transition_hook: Optional[TransitionHook] = None,
) -> State:
    """Move all short-term messages into the long-lived history.

    Applies ``transition_hook`` (identity by default) to the messages being
    committed.
    """
    validate_state(state)
    hook = transition_hook or identity_transition_hook
    committed = hook(copy.deepcopy(state["short_term"]))
    return {
        "version": state["version"],
        "long_lived": list(state["long_lived"]) + list(committed),
        "short_term": [],
    }


def append_message(
    state: State,
    message: Message,
    transition_mode: Union[TransitionMode, str] = TransitionMode.NONE,
    transition_hook: Optional[TransitionHook] = None,
) -> State:
    """Append a message, applying the given transition strategy.

    Does not compact; call :func:`compact` separately to enforce a token
    budget on the long-lived history.
    """
    validate_state(state)
    if "role" not in message:
        raise ValueError("message must have a 'role'")
    mode = TransitionMode(transition_mode)
    message = copy.deepcopy(message)

    if mode is TransitionMode.NONE:
        return {
            "version": state["version"],
            "long_lived": list(state["long_lived"]) + [message],
            "short_term": list(state["short_term"]),
        }

    new = {
        "version": state["version"],
        "long_lived": list(state["long_lived"]),
        "short_term": list(state["short_term"]) + [message],
    }
    if mode is TransitionMode.AGENT_CYCLE and _is_agent_cycle_end(message):
        return transition(new, transition_hook=transition_hook)
    return new


def append_messages(
    state: State,
    messages: List[Message],
    transition_mode: Union[TransitionMode, str] = TransitionMode.NONE,
    transition_hook: Optional[TransitionHook] = None,
) -> State:
    """Append several messages in order."""
    for message in messages:
        state = append_message(
            state,
            message,
            transition_mode=transition_mode,
            transition_hook=transition_hook,
        )
    return state


def compact(
    state: State,
    max_tokens: int,
    compaction_threshold: Optional[int] = None,
    post_compaction_token_threshold: Optional[int] = None,
    compaction_hook: Optional[CompactionHook] = None,
    adapter: Optional[ProviderAdapter] = None,
) -> State:
    """Compact the long-lived history if it exceeds the threshold.

    Triggered when the long-lived token count exceeds
    ``compaction_threshold`` (default: ``max_tokens``); ``compaction_hook``
    (default: oldest-first truncation) reduces it to
    ``post_compaction_token_threshold`` (default: ``max_tokens // 2``).
    Returns the state unchanged if under the threshold.
    """
    validate_state(state)
    adapter = adapter or _DEFAULT_ADAPTER
    threshold = compaction_threshold if compaction_threshold is not None else max_tokens
    if adapter.count_tokens(state["long_lived"]) <= threshold:
        return state
    target = (
        post_compaction_token_threshold
        if post_compaction_token_threshold is not None
        else max_tokens // 2
    )
    hook = compaction_hook or truncation_compaction_hook
    compacted = hook(copy.deepcopy(state["long_lived"]), target, adapter)
    return {
        "version": state["version"],
        "long_lived": list(compacted),
        "short_term": list(state["short_term"]),
    }


def build_messages(
    state: State,
    static_system_prompt: str = "",
    dynamic_system_prompt: Optional[str] = None,
    dynamic_system_role: str = "system",
    adapter: Optional[ProviderAdapter] = None,
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
    adapter = adapter or _DEFAULT_ADAPTER
    messages: List[Message] = []
    boundaries: List[int] = []

    if static_system_prompt:
        messages.append({"role": "system", "content": static_system_prompt})
        boundaries.append(0)

    long_lived = copy.deepcopy(state["long_lived"])
    if long_lived:
        messages.extend(long_lived)
        boundaries.append(len(messages) - 1)

    if dynamic_system_prompt:
        messages.append({"role": dynamic_system_role, "content": dynamic_system_prompt})

    messages.extend(copy.deepcopy(state["short_term"]))

    if apply_cache_markers:
        messages = adapter.apply_cache_markers(messages, boundaries)
    return messages


def cache_prefix(state: State, static_system_prompt: str = "") -> Tuple[Message, ...]:
    """The stable prefix (static system + long-lived history) as a tuple,
    useful for asserting cache stability across turns."""
    prefix: List[Message] = []
    if static_system_prompt:
        prefix.append({"role": "system", "content": static_system_prompt})
    prefix.extend(copy.deepcopy(state["long_lived"]))
    return tuple(prefix)
