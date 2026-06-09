"""Provider adapters: token counting and cache-marker injection.

The core library is provider-agnostic. An adapter supplies:

- ``count_tokens(messages)``: estimate the token cost of a message list.
- ``apply_cache_markers(messages, boundaries)``: inject provider-specific
  cache-control hints at the static-system / long-lived-history boundaries.

``boundaries`` is a list of indices into ``messages`` marking the last
message of each stable prefix segment (e.g. end of static system prompt,
end of long-lived history).
"""

from __future__ import annotations

import copy
import json
from typing import Any, Dict, List, Sequence

Message = Dict[str, Any]


class ProviderAdapter:
    """Base adapter. Subclass to support a new provider or tokenizer."""

    name = "base"

    def count_tokens(self, messages: Sequence[Message]) -> int:
        """Rough token estimate: ~4 characters per token over JSON content.

        Deliberately dependency-free; override with a real tokenizer for
        accuracy.
        """
        total_chars = 0
        for msg in messages:
            content = msg.get("content", "")
            if not isinstance(content, str):
                content = json.dumps(content, default=str)
            total_chars += len(content)
            if msg.get("tool_calls"):
                total_chars += len(json.dumps(msg["tool_calls"], default=str))
        return total_chars // 4

    def apply_cache_markers(
        self, messages: List[Message], boundaries: Sequence[int]
    ) -> List[Message]:
        """Inject cache markers at the given boundary indices.

        Base implementation is a no-op (returns messages unchanged), which
        is correct for providers with automatic prefix caching (OpenAI).
        """
        return messages


class OpenAIAdapter(ProviderAdapter):
    """OpenAI / LiteLLM chat-completions format.

    OpenAI prefix caching is automatic and keys on the literal prefix, so
    no markers are injected — stability of the prefix is what matters.
    """

    name = "openai"


class AnthropicAdapter(ProviderAdapter):
    """Anthropic Messages API format.

    Injects ``{"cache_control": {"type": "ephemeral"}}`` on the final
    content block of each boundary message.
    """

    name = "anthropic"

    def apply_cache_markers(
        self, messages: List[Message], boundaries: Sequence[int]
    ) -> List[Message]:
        result = list(messages)
        for idx in boundaries:
            if not (0 <= idx < len(result)):
                continue
            msg = copy.deepcopy(result[idx])
            content = msg.get("content")
            if isinstance(content, str):
                msg["content"] = [
                    {
                        "type": "text",
                        "text": content,
                        "cache_control": {"type": "ephemeral"},
                    }
                ]
            elif isinstance(content, list) and content:
                content = copy.deepcopy(content)
                content[-1] = dict(content[-1])
                content[-1]["cache_control"] = {"type": "ephemeral"}
                msg["content"] = content
            result[idx] = msg
        return result


class TransformersAdapter(ProviderAdapter):
    """Hugging Face tokenizer-backed adapter for local models.

    Pass any object with an ``encode`` or ``apply_chat_template`` method
    (e.g. a ``transformers.PreTrainedTokenizer``).
    """

    name = "transformers"

    def __init__(self, tokenizer: Any):
        self.tokenizer = tokenizer

    def count_tokens(self, messages: Sequence[Message]) -> int:
        if hasattr(self.tokenizer, "apply_chat_template"):
            try:
                ids = self.tokenizer.apply_chat_template(
                    list(messages), tokenize=True, add_generation_prompt=False
                )
                return len(ids)
            except Exception:
                pass
        total = 0
        for msg in messages:
            content = msg.get("content", "")
            if not isinstance(content, str):
                content = json.dumps(content, default=str)
            total += len(self.tokenizer.encode(content))
        return total
