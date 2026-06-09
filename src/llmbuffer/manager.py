"""Stateful (OOP / Strategy) interface.

A thin wrapper over the functional core that retains conversation state
in memory between calls.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from . import functional, state as state_mod
from .config import PromptConfig
from .state import Message, State


class PromptManager:
    """In-memory conversation manager.

    Example::

        manager = PromptManager(PromptConfig(
            static_system_prompt="You are a helpful assistant.",
            transition_mode="agent_cycle",
            max_tokens=8000,
        ))
        manager.append({"role": "user", "content": "Hello"})
        messages = manager.build_messages(dynamic_system_prompt=f"Time: {now}")
    """

    def __init__(self, config: Optional[PromptConfig] = None, state: Optional[State] = None):
        self.config = config or PromptConfig()
        self._state = state_mod.validate_state(state) if state else state_mod.new_state()

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
        self._state = functional.append_message(self._state, message, self.config)
        return self

    def append_many(self, messages: List[Message]) -> "PromptManager":
        self._state = functional.append_messages(self._state, messages, self.config)
        return self

    def transition(self) -> "PromptManager":
        """Manually move short-term messages into the long-lived history."""
        self._state = functional.transition(self._state, self.config)
        return self

    def compact(self) -> "PromptManager":
        """Run compaction now if over threshold."""
        self._state = functional.maybe_compact(self._state, self.config)
        return self

    def build_messages(
        self,
        dynamic_system_prompt: Optional[str] = None,
        apply_cache_markers: bool = True,
    ) -> List[Message]:
        return functional.build_messages(
            self._state,
            self.config,
            dynamic_system_prompt=dynamic_system_prompt,
            apply_cache_markers=apply_cache_markers,
        )

    def cache_prefix(self):
        return functional.cache_prefix(self._state, self.config)

    # -- serialization -----------------------------------------------------

    def to_json(self, **json_kwargs: Any) -> str:
        return state_mod.dumps(self._state, **json_kwargs)

    @classmethod
    def from_json(cls, payload: str, config: Optional[PromptConfig] = None) -> "PromptManager":
        return cls(config=config, state=state_mod.loads(payload))
