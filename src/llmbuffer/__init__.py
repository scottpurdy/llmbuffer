"""llmbuffer — cache-optimized LLM conversation history management.

Stateful::

    from llmbuffer import PromptManager

    manager = PromptManager(
        static_system_prompt="You are a helpful assistant.",
        transition_mode="agent_cycle",
        max_tokens=8000,
    )
    manager.append({"role": "user", "content": "Hi"})
    messages = manager.build_messages(dynamic_system_prompt="Time: 12:00")

Stateless / functional::

    from llmbuffer import functional, new_state

    state = new_state()
    state = functional.append_message(state, {"role": "user", "content": "Hi"},
                                      transition_mode="manual")
    state = functional.compact(state, max_tokens=8000)
    messages = functional.build_messages(state, static_system_prompt="...")
"""

from . import functional
from .adapters import (
    AnthropicAdapter,
    OpenAIAdapter,
    ProviderAdapter,
    TransformersAdapter,
)
from .config import (
    CompactionHook,
    ContextConsolidationHook,
    TransitionHook,
    TransitionMode,
)
from .hooks import (
    concat_context_consolidation_hook,
    drop_tool_messages_transition_hook,
    identity_transition_hook,
    truncation_compaction_hook,
)
from .manager import PromptManager
from .state import dumps, loads, new_state

__version__ = "0.3.0"

__all__ = [
    "AnthropicAdapter",
    "OpenAIAdapter",
    "ProviderAdapter",
    "TransformersAdapter",
    "TransitionMode",
    "CompactionHook",
    "ContextConsolidationHook",
    "TransitionHook",
    "concat_context_consolidation_hook",
    "PromptManager",
    "functional",
    "new_state",
    "dumps",
    "loads",
    "identity_transition_hook",
    "truncation_compaction_hook",
    "drop_tool_messages_transition_hook",
    "__version__",
]
