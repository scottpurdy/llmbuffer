"""Configuration for prompt assembly, transitions, and compaction."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Sequence

from .adapters import OpenAIAdapter, ProviderAdapter

Message = Dict[str, Any]

# Hook signatures:
#   CompactionHook(messages, target_tokens, adapter) -> compacted messages
#   TransitionHook(messages) -> messages to commit to long-lived history
CompactionHook = Callable[[List[Message], int, ProviderAdapter], List[Message]]
TransitionHook = Callable[[List[Message]], List[Message]]


class TransitionMode(str, Enum):
    """How messages move from short-term to long-lived history."""

    NONE = "none"          # append directly into long-lived history
    MANUAL = "manual"      # only on explicit transition() calls
    AGENT_CYCLE = "agent_cycle"  # after a final (non-tool-calling) assistant message


@dataclass
class PromptConfig:
    """Settings shared by the stateful and functional interfaces.

    Attributes:
        static_system_prompt: Highly stable system instructions; first in
            every assembled message list.
        transition_mode: Short-to-long transition strategy.
        adapter: Provider adapter for token counting and cache markers.
        max_tokens: Hard budget for the long-lived history. ``None``
            disables compaction.
        compaction_threshold: Token count that triggers compaction.
            Defaults to ``max_tokens`` when unset.
        post_compaction_token_threshold: Target size of the long-lived
            history after compaction. Defaults to half of ``max_tokens``.
        compaction_hook: Summarization/compaction strategy. Defaults to
            simple oldest-first truncation.
        transition_hook: Optional filter/summarizer applied to messages as
            they move into the long-lived history.
        dynamic_system_role: Role used for the dynamic system message
            (some providers reject mid-conversation "system"; use "user"
            there).
    """

    static_system_prompt: str = ""
    transition_mode: TransitionMode = TransitionMode.NONE
    adapter: ProviderAdapter = field(default_factory=OpenAIAdapter)
    max_tokens: Optional[int] = None
    compaction_threshold: Optional[int] = None
    post_compaction_token_threshold: Optional[int] = None
    compaction_hook: Optional[CompactionHook] = None
    transition_hook: Optional[TransitionHook] = None
    dynamic_system_role: str = "system"

    def __post_init__(self) -> None:
        if isinstance(self.transition_mode, str):
            self.transition_mode = TransitionMode(self.transition_mode)

    @property
    def effective_compaction_threshold(self) -> Optional[int]:
        if self.compaction_threshold is not None:
            return self.compaction_threshold
        return self.max_tokens

    @property
    def effective_post_compaction_target(self) -> Optional[int]:
        if self.post_compaction_token_threshold is not None:
            return self.post_compaction_token_threshold
        if self.max_tokens is not None:
            return self.max_tokens // 2
        return None
