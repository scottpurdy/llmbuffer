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
from .config import (
    LLMBUFFER_META_FIELD,
    CompactionHook,
    ContextConsolidationHook,
    TransitionHook,
    TransitionMode,
)
from .hooks import (
    concat_context_consolidation_hook,
    identity_transition_hook,
    truncation_compaction_hook,
)
from .state import Message, State, validate_state

_DEFAULT_ADAPTER = OpenAIAdapter()


def _context_key(message: Message) -> Optional[str]:
    meta = message.get(LLMBUFFER_META_FIELD)
    return meta.get("context_key") if isinstance(meta, dict) else None


def _strip_meta(message: Message) -> Message:
    if LLMBUFFER_META_FIELD in message:
        return {k: v for k, v in message.items() if k != LLMBUFFER_META_FIELD}
    return message


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


def append_context(
    state: State,
    content: str,
    key: str = "context",
    transition_mode: Union[TransitionMode, str] = TransitionMode.NONE,
    transition_hook: Optional[TransitionHook] = None,
) -> State:
    """Append a dynamic-context update (a keyed system message).

    Context updates are ordinary messages: they follow the same transition
    path as everything else, preserving temporal ordering — a mid-turn
    update lands at the (high-attention) end of the assembled list rather
    than buried in the prefix. The ``key`` tags the message (stored under a
    namespaced ``"_llmbuffer"`` field, stripped before sending) so that
    compaction can later consolidate the initial context and its deltas
    into a single block via a :data:`ContextConsolidationHook`.

    Contract: messages sharing a ``key`` are consolidated together at
    compaction time, in history order. Use one key per logical context
    document (e.g. ``"world-state"``, ``"user-profile"``).
    """
    return append_message(
        state,
        {
            "role": "system",
            "content": content,
            LLMBUFFER_META_FIELD: {"context_key": key},
        },
        transition_mode=transition_mode,
        transition_hook=transition_hook,
    )


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
    context_consolidation_hook: Optional[ContextConsolidationHook] = None,
    adapter: Optional[ProviderAdapter] = None,
) -> State:
    """Compact the long-lived history if it exceeds the threshold.

    Triggered when the long-lived token count exceeds
    ``compaction_threshold`` (default: ``max_tokens``). Once triggered,
    both phases always run — the prefix is being invalidated regardless,
    so compact all the way down to maximize the stable stretch before the
    next compaction:

    1. **Consolidate context**: for each context key (see
       :func:`append_context`), all keyed messages are folded into a
       single block by ``context_consolidation_hook`` (default:
       lossless concatenation). Consolidated blocks are placed at the
       front of the new history, ordered by each key's first appearance.
       Keyed messages never pass through the lossy phase.
    2. **Lossy compaction**: ``compaction_hook`` (default: oldest-first
       truncation) reduces the remaining unkeyed history so the total —
       consolidated blocks included — fits
       ``post_compaction_token_threshold`` (default: ``max_tokens // 2``).

    Returns the state unchanged if under the threshold. Context deltas
    still in short-term history are untouched; they consolidate at a
    later compaction, after they have transitioned.
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

    # Phase 1: consolidate context messages per key, preserving the order
    # in which keys first appeared.
    long_lived = copy.deepcopy(state["long_lived"])
    keyed: "dict[str, List[Message]]" = {}
    rest: List[Message] = []
    for msg in long_lived:
        key = _context_key(msg)
        if key is not None:
            keyed.setdefault(key, []).append(msg)
        else:
            rest.append(msg)

    consolidate = context_consolidation_hook or concat_context_consolidation_hook
    consolidated: List[Message] = [
        {
            "role": "system",
            "content": consolidate(key, messages),
            LLMBUFFER_META_FIELD: {"context_key": key},
        }
        for key, messages in keyed.items()
    ]

    # Phase 2: lossy-compact the unkeyed remainder into whatever budget the
    # consolidated blocks leave.
    remainder_target = max(0, target - adapter.count_tokens(consolidated))
    hook = compaction_hook or truncation_compaction_hook
    compacted_rest = hook(rest, remainder_target, adapter) if rest else []

    return {
        "version": state["version"],
        "long_lived": consolidated + list(compacted_rest),
        "short_term": list(state["short_term"]),
    }


def compact_for_request(
    state: State,
    request_budget: int,
    static_system_prompt: str = "",
    reserved_tokens: int = 0,
    compaction_threshold: Optional[int] = None,
    post_compaction_token_threshold: Optional[int] = None,
    compaction_hook: Optional[CompactionHook] = None,
    context_consolidation_hook: Optional[ContextConsolidationHook] = None,
    adapter: Optional[ProviderAdapter] = None,
) -> State:
    """Compact so the *whole request* fits a token budget.

    Derives the long-lived history budget as::

        request_budget - tokens(static_system_prompt) - reserved_tokens

    and calls :func:`compact` with it.

    ``reserved_tokens`` is a fixed headroom you declare for the dynamic
    system prompt and short-term history. It is deliberately a constant
    rather than a measurement of the current turn's content: if the budget
    tracked the (fluctuating) dynamic context, compaction could trigger on
    one turn and not the next, rewriting the long-lived prefix and
    invalidating the provider cache — exactly what this library exists to
    prevent. Reserve the worst case you expect and the derived budget stays
    deterministic across turns.

    Raises ``ValueError`` if the derived budget is not positive.
    """
    adapter = adapter or _DEFAULT_ADAPTER
    static_tokens = (
        adapter.count_tokens([{"role": "system", "content": static_system_prompt}])
        if static_system_prompt
        else 0
    )
    long_lived_budget = request_budget - static_tokens - reserved_tokens
    if long_lived_budget <= 0:
        raise ValueError(
            f"request_budget={request_budget} leaves no room for history: "
            f"static system prompt ≈{static_tokens} tokens, "
            f"reserved_tokens={reserved_tokens}"
        )
    return compact(
        state,
        max_tokens=long_lived_budget,
        compaction_threshold=compaction_threshold,
        post_compaction_token_threshold=post_compaction_token_threshold,
        compaction_hook=compaction_hook,
        context_consolidation_hook=context_consolidation_hook,
        adapter=adapter,
    )


def build_messages(
    state: State,
    static_system_prompt: str = "",
    dynamic_system_prompt: Optional[str] = None,
    dynamic_system_role: str = "system",
    adapter: Optional[ProviderAdapter] = None,
    apply_cache_markers: bool = True,
    with_metadata: bool = False,
):
    """Assemble the cache-optimized message list for an LLM call.

    Strict ordering:

    1. Static system prompt
    2. Long-lived conversation history (stable cache prefix)
    3. Dynamic system prompt (if any)
    4. Short-term conversation history

    Cache markers are injected at the end of the static system prompt and
    at the end of the long-lived history (provider-dependent; a no-op for
    OpenAI-style automatic prefix caching).

    With ``with_metadata=True``, returns ``(messages, metadata)`` where
    ``metadata`` describes the predicted cacheable prefix::

        {
            "boundaries": [0, 5],        # marker indices into messages
            "prefix_message_count": 6,   # static system + long-lived
            "prefix_tokens": 1234,       # estimated via the adapter
            "suffix_tokens": 56,         # dynamic + short-term, estimated
            "total_tokens": 1290,
        }

    These are *predictions* based on prefix stability — whether the prefix
    is actually served from cache is only knowable from the provider's
    response usage metadata.
    """
    validate_state(state)
    adapter = adapter or _DEFAULT_ADAPTER
    messages: List[Message] = []
    boundaries: List[int] = []

    if static_system_prompt:
        messages.append({"role": "system", "content": static_system_prompt})
        boundaries.append(0)

    long_lived = [_strip_meta(m) for m in copy.deepcopy(state["long_lived"])]
    if long_lived:
        messages.extend(long_lived)
        boundaries.append(len(messages) - 1)

    prefix_count = len(messages)

    if dynamic_system_prompt:
        messages.append({"role": dynamic_system_role, "content": dynamic_system_prompt})

    messages.extend(_strip_meta(m) for m in copy.deepcopy(state["short_term"]))

    metadata = None
    if with_metadata:
        prefix_tokens = adapter.count_tokens(messages[:prefix_count])
        total_tokens = adapter.count_tokens(messages)
        metadata = {
            "boundaries": list(boundaries),
            "prefix_message_count": prefix_count,
            "prefix_tokens": prefix_tokens,
            "suffix_tokens": total_tokens - prefix_tokens,
            "total_tokens": total_tokens,
        }

    if apply_cache_markers:
        messages = adapter.apply_cache_markers(messages, boundaries)

    if with_metadata:
        return messages, metadata
    return messages


def cache_prefix(state: State, static_system_prompt: str = "") -> Tuple[Message, ...]:
    """The stable prefix (static system + long-lived history) as a tuple,
    useful for asserting cache stability across turns."""
    prefix: List[Message] = []
    if static_system_prompt:
        prefix.append({"role": "system", "content": static_system_prompt})
    prefix.extend(_strip_meta(m) for m in copy.deepcopy(state["long_lived"]))
    return tuple(prefix)
