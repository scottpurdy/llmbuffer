"""Shared types: transition modes and hook signatures."""

from __future__ import annotations

from enum import Enum
from typing import Any, Callable, Dict, List

Message = Dict[str, Any]

# Hook signatures:
#   CompactionHook(messages, target_tokens, adapter) -> compacted messages
#   TransitionHook(messages) -> messages to commit to long-lived history.
#     Note: context-delta messages (carrying "_llmbuffer" metadata) ride
#     through transitions like any other message — custom hooks should
#     preserve them unless they specifically mean to drop context updates.
#   ContextConsolidationHook(key, messages) -> consolidated context string.
#     `messages` is every long-lived message tagged with `key`, in history
#     order; the first is the previous consolidated block (or the initial
#     context), the rest are deltas.
CompactionHook = Callable[[List[Message], int, Any], List[Message]]
TransitionHook = Callable[[List[Message]], List[Message]]
ContextConsolidationHook = Callable[[str, List[Message]], str]

# Namespaced metadata field on context messages; stripped by build_messages
# before the messages are sent to a provider.
LLMBUFFER_META_FIELD = "_llmbuffer"


class TransitionMode(str, Enum):
    """How messages move from short-term to long-lived history."""

    NONE = "none"          # append directly into long-lived history
    MANUAL = "manual"      # only on explicit transition() calls
    AGENT_CYCLE = "agent_cycle"  # after a final (non-tool-calling) assistant message
