"""Default hook implementations."""

from __future__ import annotations

from typing import Any, Dict, List

from .adapters import ProviderAdapter

Message = Dict[str, Any]


def truncation_compaction_hook(
    messages: List[Message], target_tokens: int, adapter: ProviderAdapter
) -> List[Message]:
    """Default compaction: drop oldest messages until under target.

    Always keeps at least the most recent message. Returns a new list;
    surviving messages are the original objects (kept identical so any
    remaining prefix stays byte-stable).
    """
    kept = list(messages)
    while len(kept) > 1 and adapter.count_tokens(kept) > target_tokens:
        kept.pop(0)
    return kept


def identity_transition_hook(messages: List[Message]) -> List[Message]:
    """Default transition: commit messages unchanged."""
    return list(messages)


def concat_context_consolidation_hook(key: str, messages: List[Message]) -> str:
    """Default context consolidation: concatenate the block and its deltas.

    Lossless and requires no LLM call — it just relocates the delta tokens
    into the consolidated block. Supply your own hook to truly rewrite the
    context (apply diffs mechanically, summarize with an LLM, etc.).
    """
    return "\n\n".join(
        m.get("content", "") for m in messages if m.get("content")
    )


def drop_tool_messages_transition_hook(messages: List[Message]) -> List[Message]:
    """Example transition hook: filter out tool calls and tool outputs
    before committing to the long-lived (cached) history."""
    result = []
    for msg in messages:
        if msg.get("role") == "tool":
            continue
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            if not msg.get("content"):
                continue
            msg = {k: v for k, v in msg.items() if k != "tool_calls"}
        result.append(msg)
    return result
