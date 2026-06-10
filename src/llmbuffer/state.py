"""Conversation state: a plain, JSON-serializable dict.

The state is intentionally a simple dict so that stateless web applications
can persist it anywhere (database row, session store, cache) between
requests. All functional-API operations treat the state as immutable and
return a new state.

Schema::

    {
        "version": 1,
        "long_lived": [ {"role": ..., "content": ...}, ... ],
        "short_term": [ {"role": ..., "content": ...}, ... ],
    }
"""

from __future__ import annotations

import copy
import json
from typing import Any, Dict, List

STATE_VERSION = 1

Message = Dict[str, Any]
State = Dict[str, Any]


def new_state(
    initial_context: "str | None" = None, context_key: str = "context"
) -> State:
    """Create a fresh conversation state.

    ``initial_context`` seeds the long-lived history with a keyed context
    message — identical to calling
    :func:`llmbuffer.functional.append_context` on an empty state in
    ``none`` mode. Doing it at creation time means the block is in the
    stable prefix from the very first request.

    Note: the state holds *all* conversation content except two things —
    the static system prompt and the per-call volatile dynamic prompt,
    both of which are passed to ``build_messages`` each call.
    """
    long_lived: List[Message] = []
    if initial_context is not None:
        long_lived.append(
            {
                "role": "system",
                "content": initial_context,
                "_llmbuffer": {"context_key": context_key},
            }
        )
    return {"version": STATE_VERSION, "long_lived": long_lived, "short_term": []}


def copy_state(state: State) -> State:
    """Deep-copy a state so callers can never mutate the cached prefix."""
    return copy.deepcopy(state)


def validate_state(state: State) -> State:
    """Validate the shape of a state dict, returning it unchanged."""
    if not isinstance(state, dict):
        raise TypeError(f"state must be a dict, got {type(state).__name__}")
    if state.get("version") != STATE_VERSION:
        raise ValueError(f"unsupported state version: {state.get('version')!r}")
    for key in ("long_lived", "short_term"):
        if not isinstance(state.get(key), list):
            raise ValueError(f"state[{key!r}] must be a list")
        for msg in state[key]:
            if not isinstance(msg, dict) or "role" not in msg:
                raise ValueError(f"invalid message in state[{key!r}]: {msg!r}")
    return state


def dumps(state: State, **json_kwargs: Any) -> str:
    """Serialize a state to a JSON string."""
    return json.dumps(validate_state(state), **json_kwargs)


def loads(payload: str) -> State:
    """Deserialize a state from a JSON string."""
    return validate_state(json.loads(payload))


def all_messages(state: State) -> List[Message]:
    """All conversation messages (long-lived followed by short-term)."""
    return list(state["long_lived"]) + list(state["short_term"])
