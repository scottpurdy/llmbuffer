"""Stateful (OOP / Strategy) interface.

A thin wrapper over the functional core that retains conversation state
and settings in memory between calls. Unlike the functional API — where
compaction is an explicit call — the manager compacts automatically
whenever the long-lived history changes, if ``max_tokens`` is set.
"""

from __future__ import annotations

from typing import Any, List, Optional, Union

from . import functional, state as state_mod
from .adapters import OpenAIAdapter, ProviderAdapter
from .config import (
    CompactionHook,
    ContextConsolidationHook,
    TransitionHook,
    TransitionMode,
)
from .state import Message, State


class PromptManager:
    """In-memory conversation manager.

    Args:
        static_system_prompt: Highly stable system instructions; first in
            every assembled message list.
        transition_mode: Short-to-long transition strategy
            (``"none"``, ``"manual"``, or ``"agent_cycle"``).
        adapter: Provider adapter for token counting and cache markers.
        max_tokens: Token budget for the long-lived history. ``None``
            disables automatic compaction.
        compaction_threshold: Token count that triggers compaction.
            Defaults to ``max_tokens``.
        post_compaction_token_threshold: Target size after compaction.
            Defaults to ``max_tokens // 2``.
        compaction_hook: Summarization/compaction strategy. Defaults to
            simple oldest-first truncation.
        transition_hook: Optional filter/summarizer applied to messages as
            they move into the long-lived history.
        dynamic_system_role: Role used for the dynamic system message
            (some providers reject mid-conversation "system"; use "user"
            there).
        state: Initial conversation state (e.g. from a previous session).

    Example::

        manager = PromptManager(
            static_system_prompt="You are a helpful assistant.",
            transition_mode="agent_cycle",
            max_tokens=8000,
        )
        manager.append({"role": "user", "content": "Hello"})
        messages = manager.build_messages(dynamic_system_prompt=f"Time: {now}")
    """

    def __init__(
        self,
        static_system_prompt: str = "",
        transition_mode: Union[TransitionMode, str] = TransitionMode.NONE,
        adapter: Optional[ProviderAdapter] = None,
        max_tokens: Optional[int] = None,
        compaction_threshold: Optional[int] = None,
        post_compaction_token_threshold: Optional[int] = None,
        compaction_hook: Optional[CompactionHook] = None,
        transition_hook: Optional[TransitionHook] = None,
        context_consolidation_hook: Optional[ContextConsolidationHook] = None,
        dynamic_system_role: str = "system",
        initial_context: Optional[str] = None,
        context_key: str = "context",
        state: Optional[State] = None,
    ):
        self.static_system_prompt = static_system_prompt
        self.transition_mode = TransitionMode(transition_mode)
        self.adapter = adapter or OpenAIAdapter()
        self.max_tokens = max_tokens
        self.compaction_threshold = compaction_threshold
        self.post_compaction_token_threshold = post_compaction_token_threshold
        self.compaction_hook = compaction_hook
        self.transition_hook = transition_hook
        self.context_consolidation_hook = context_consolidation_hook
        self.dynamic_system_role = dynamic_system_role
        self.context_key = context_key
        if state is not None:
            if initial_context is not None:
                raise ValueError("pass either state or initial_context, not both")
            self._state = state_mod.validate_state(state)
        else:
            self._state = state_mod.new_state(
                initial_context=initial_context, context_key=context_key
            )

    # -- state access -----------------------------------------------------

    @property
    def state(self) -> State:
        """A deep copy of the current state (the internal state is never
        exposed for mutation)."""
        return state_mod.copy_state(self._state)

    @property
    def long_lived_history(self) -> List[Message]:
        return state_mod.copy_state(self._state)["long_lived"]

    @property
    def short_term_history(self) -> List[Message]:
        return state_mod.copy_state(self._state)["short_term"]

    # -- core operations ---------------------------------------------------

    def append(self, message: Message) -> "PromptManager":
        self._state = functional.append_message(
            self._state,
            message,
            transition_mode=self.transition_mode,
            transition_hook=self.transition_hook,
        )
        return self._auto_compact()

    def append_context(self, content: str, key: Optional[str] = None) -> "PromptManager":
        """Append a dynamic-context update (keyed system message). Follows
        the normal transition path; consolidated at compaction time."""
        self._state = functional.append_context(
            self._state,
            content,
            key=key if key is not None else self.context_key,
            transition_mode=self.transition_mode,
            transition_hook=self.transition_hook,
        )
        return self._auto_compact()

    def append_many(self, messages: List[Message]) -> "PromptManager":
        for message in messages:
            self.append(message)
        return self

    def transition(self) -> "PromptManager":
        """Manually move short-term messages into the long-lived history."""
        self._state = functional.transition(
            self._state, transition_hook=self.transition_hook
        )
        return self._auto_compact()

    def compact(self) -> "PromptManager":
        """Run compaction now if the long-lived history is over threshold."""
        if self.max_tokens is None:
            return self
        self._state = functional.compact(
            self._state,
            max_tokens=self.max_tokens,
            compaction_threshold=self.compaction_threshold,
            post_compaction_token_threshold=self.post_compaction_token_threshold,
            compaction_hook=self.compaction_hook,
            context_consolidation_hook=self.context_consolidation_hook,
            adapter=self.adapter,
        )
        return self

    def _auto_compact(self) -> "PromptManager":
        return self.compact()

    def compact_for_request(
        self, request_budget: int, reserved_tokens: int = 0
    ) -> "PromptManager":
        """Compact so static system + history + ``reserved_tokens`` headroom
        fits ``request_budget``. See :func:`llmbuffer.functional.compact_for_request`
        for why the headroom is a declared constant rather than measured."""
        self._state = functional.compact_for_request(
            self._state,
            request_budget,
            static_system_prompt=self.static_system_prompt,
            reserved_tokens=reserved_tokens,
            compaction_threshold=self.compaction_threshold,
            post_compaction_token_threshold=self.post_compaction_token_threshold,
            compaction_hook=self.compaction_hook,
            context_consolidation_hook=self.context_consolidation_hook,
            adapter=self.adapter,
        )
        return self

    def build_messages(
        self,
        dynamic_system_prompt: Optional[str] = None,
        apply_cache_markers: bool = True,
        with_metadata: bool = False,
    ):
        return functional.build_messages(
            self._state,
            static_system_prompt=self.static_system_prompt,
            dynamic_system_prompt=dynamic_system_prompt,
            dynamic_system_role=self.dynamic_system_role,
            adapter=self.adapter,
            apply_cache_markers=apply_cache_markers,
            with_metadata=with_metadata,
        )

    def cache_prefix(self):
        return functional.cache_prefix(
            self._state, static_system_prompt=self.static_system_prompt
        )

    # -- serialization -----------------------------------------------------

    def to_json(self, **json_kwargs: Any) -> str:
        return state_mod.dumps(self._state, **json_kwargs)

    @classmethod
    def from_json(cls, payload: str, **kwargs: Any) -> "PromptManager":
        """Restore from serialized state; settings are passed as kwargs."""
        return cls(state=state_mod.loads(payload), **kwargs)
