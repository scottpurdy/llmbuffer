"""Shared types: transition modes and hook signatures."""

from __future__ import annotations

from enum import Enum
from typing import Any, Callable, Dict, List

Message = Dict[str, Any]

# Hook signatures:
#   CompactionHook(messages, target_tokens, adapter) -> compacted messages
#   TransitionHook(messages) -> messages to commit to long-lived history
CompactionHook = Callable[[List[Message], int, Any], List[Message]]
TransitionHook = Callable[[List[Message]], List[Message]]


class TransitionMode(str, Enum):
    """How messages move from short-term to long-lived history."""

    NONE = "none"          # append directly into long-lived history
    MANUAL = "manual"      # only on explicit transition() calls
    AGENT_CYCLE = "agent_cycle"  # after a final (non-tool-calling) assistant message
