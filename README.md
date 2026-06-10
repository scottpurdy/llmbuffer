# llmbuffer

**Cache-optimized LLM conversation history management.**

Most LLM applications naively concatenate their system prompt, conversation history, and any dynamic context into a single message list — and rebuild it from scratch every turn. This works, but it leaves significant money and latency on the table by constantly invalidating the provider's prompt cache.

`llmbuffer` assembles your messages in the order that maximises cache reuse, manages the boundary between stable and changing content, and handles compaction when history grows too long — all without you having to think about it.

```
[Static System Prompt] → [Long-Lived History] → [Dynamic Context] → [Recent Messages]
       cached ✓                  cached ✓             not cached          not cached
```

The static system prompt and committed conversation history form a **byte-stable prefix** that is never mutated or re-ordered across turns. The frequently-changing parts — RAG results, timestamps, in-flight tool calls — live at the end where they can't invalidate the prefix.

## Install

```bash
pip install llmbuffer
```

Optional extras for live benchmarking:

```bash
pip install "llmbuffer[anthropic]"    # Anthropic prompt caching
pip install "llmbuffer[openai]"       # OpenAI prefix caching
```

`llmbuffer` has **zero required dependencies** — just Python 3.9+.

## Quickstart

### Stateful (in-process)

```python
from llmbuffer import PromptManager, AnthropicAdapter

manager = PromptManager(
    static_system_prompt="You are a senior software engineering assistant...",
    transition_mode="agent_cycle",   # auto-commit turns to the stable prefix
    adapter=AnthropicAdapter(),      # inject cache_control markers
    max_tokens=8_000,                # compact long-lived history beyond this
)

# Each turn:
manager.append({"role": "user", "content": user_message})
messages = manager.build_messages(dynamic_system_prompt=rag_context)
reply = anthropic_client.messages.create(messages=messages, ...)
manager.append({"role": "assistant", "content": reply})
```

### Stateless (web app / serverless)

Pure functions over a JSON-serializable state dict — persist it anywhere between requests:

```python
from llmbuffer import functional, new_state, dumps, loads

SYSTEM = "You are a senior software engineering assistant..."

# Load state from DB / session
state = loads(row.conversation_json) if row else new_state()

# Build messages, call LLM, store updated state
state = functional.append_message(state, {"role": "user", "content": text},
                                  transition_mode="manual")
messages = functional.build_messages(state, static_system_prompt=SYSTEM,
                                     dynamic_system_prompt=rag_context)
# ... call your LLM ...
state = functional.append_message(state, reply, transition_mode="manual")
state = functional.compact(state, max_tokens=8_000)   # explicit in the functional API
row.conversation_json = dumps(state)
```

Each function takes only the settings it uses — there's no config object to thread through. Compaction is an explicit `compact()` call in the functional API (the stateful `PromptManager` runs it automatically).

## How it works

### Message ordering

`build_messages()` always emits messages in this exact order:

| Position | Content | Cache behaviour |
|----------|---------|----------------|
| 1 | **Static system prompt** | Cached — never changes |
| 2 | **Long-lived history** | Cached — stable, grows slowly |
| 3 | **Dynamic context** | Not cached — RAG results, timestamps, etc. |
| 4 | **Short-term history** | Not cached — current turn, tool calls |

### Transition modes

Control when messages graduate from short-term into the stable long-lived history:

| Mode | Behaviour |
|------|-----------|
| `none` | Every message goes straight into long-lived history |
| `manual` | Messages stay short-term until you call `transition()` |
| `agent_cycle` | Commits automatically when a non-tool-call assistant message ends the turn |

### Transition hooks

Before messages move from short-term into the long-lived (cached) history, an optional `transition_hook` can rewrite them — useful for trimming verbose tool outputs or stripping content you don't want locked into the stable prefix forever.

```python
def trim_tool_outputs(messages):
    """Keep only the last 20 lines of any tool output before it enters long-lived history."""
    result = []
    for msg in messages:
        if msg.get("role") == "tool":
            content = msg.get("content", "")
            lines = content.splitlines()
            if len(lines) > 20:
                kept = "\n".join(lines[-20:])
                msg = {**msg, "content": f"[…{len(lines) - 20} lines truncated]\n{kept}"}
        result.append(msg)
    return result

manager = PromptManager(
    transition_mode="agent_cycle",
    transition_hook=trim_tool_outputs,
)
# Functional API: pass the hook directly
# state = functional.append_message(state, msg, transition_mode="agent_cycle",
#                                   transition_hook=trim_tool_outputs)
```

The hook receives the list of short-term messages being committed and returns whatever should actually land in long-lived history. Drop messages entirely, summarise them, replace binary blobs with descriptions — the returned list is what gets cached.

### Dynamic context: two channels

Context that changes during a conversation comes in two flavours, and they want different placement:

| | Volatile context | Durable context |
|---|---|---|
| Changes | every call, significantly | rarely, in small deltas |
| Examples | timestamps, RAG results, mutable UI state | world state, user profile, session goals |
| Use | `dynamic_system_prompt=` on `build_messages()` | `append_context()` |
| Placement | end of the list, never cached | in the history stream, cached |

**Volatile context** is passed per-call and never stored — it sits after the cached prefix where it can't invalidate anything.

**Durable context** is appended as a keyed system message, riding the normal transition path so temporal ordering is preserved — a mid-turn update lands at the high-attention end of the list, not buried in the prefix:

```python
manager = PromptManager(
    static_system_prompt=SYSTEM,
    initial_context=initial_world_state,    # seeds the stable prefix at creation
    context_key="world",
    max_tokens=8_000,
)
# later, when something changes:
manager.append_context("Update: the inventory now contains 3 keys.")
```

At compaction time, the initial context and all its deltas are **consolidated**: a `ContextConsolidationHook` receives every message for a key (initial block first, deltas in order) and returns the new, fully rewritten context, which is placed at the front of the compacted history — right after the static system prompt. The default hook concatenates losslessly; supply your own to apply diffs or summarise with an LLM:

```python
def consolidate(key, messages):
    return rewrite_world_state(base=messages[0]["content"],
                               deltas=[m["content"] for m in messages[1:]])

manager = PromptManager(..., context_consolidation_hook=consolidate)
```

Keyed messages never pass through the lossy compaction hook — consolidation and lossy compaction are separate phases, and both always run once compaction triggers (the prefix is being rewritten anyway, so compact all the way down).

> **What lives outside the state:** exactly two things are *not* carried in the serialized state — the **static system prompt** and the per-call **volatile dynamic prompt**. Everything else, including durable context and its deltas, round-trips through `dumps()`/`loads()`. In the stateless pattern: rehydrated state + your constant static prompt = the complete conversation.

### Compaction

When the long-lived history exceeds `max_tokens`, a compaction hook reduces it to `max_tokens // 2` (configurable). The default hook truncates oldest-first; supply your own to summarise instead:

```python
def summarise(messages, target_tokens, adapter):
    summary = call_llm_to_summarise(messages)
    return [{"role": "system", "content": summary}]

manager = PromptManager(max_tokens=8_000, compaction_hook=summarise)
# Functional API: compaction is an explicit call
# state = functional.compact(state, max_tokens=8_000, compaction_hook=summarise)
```

### Boundary metadata

Pass `with_metadata=True` to `build_messages()` to also get the predicted cacheable-prefix layout — useful for logging, debugging, or asserting prefix stability in your own tests:

```python
messages, meta = manager.build_messages(dynamic_system_prompt=rag, with_metadata=True)
# meta == {"boundaries": [0, 12], "prefix_message_count": 13,
#          "prefix_tokens": 4203, "suffix_tokens": 310, "total_tokens": 4513}
```

These are predictions from prefix stability; actual cache hits are only reported in the provider's response usage metadata.

### Request-budget compaction

`compact()` budgets the long-lived history in isolation. When your real constraint is the whole request (static system + history + dynamic context ≤ context window), use `compact_for_request()`:

```python
state = functional.compact_for_request(
    state,
    request_budget=128_000,            # whole-request token budget
    static_system_prompt=SYSTEM,       # measured (it's stable by contract)
    reserved_tokens=8_000,             # fixed headroom for dynamic + short-term content
)
# or: manager.compact_for_request(request_budget=128_000, reserved_tokens=8_000)
```

`reserved_tokens` is deliberately a declared constant, not a measurement of the current turn: if the budget tracked the fluctuating dynamic content, compaction could trigger on one turn and not the next — rewriting the long-lived prefix and invalidating the cache. Reserve your worst case and the derived budget stays deterministic.

### Provider adapters

| Adapter | Cache markers | Token counting |
|---------|--------------|----------------|
| `OpenAIAdapter` (default) | None needed — automatic prefix caching | ~4 chars/token |
| `AnthropicAdapter` | `cache_control: {type: ephemeral}` injected at prefix boundaries | ~4 chars/token |
| `TransformersAdapter(tok)` | None | Exact via HF tokenizer |

Subclass `ProviderAdapter` to add a new provider — override `count_tokens()` and/or `apply_cache_markers()`.

## Benchmark

The benchmark suite runs a multi-turn conversation through both `llmbuffer` and a **naive** approach, and reports cache hits from the provider's own usage metadata.

The naive approach puts the static and dynamic system prompts together at the start of every message list and drops the oldest messages when the context limit is hit — this is the default pattern in most LLM applications today.

### Results (simulated, 15 turns, Anthropic pricing)

> The simulated provider models provider prefix caching exactly: a turn is a cache hit when its message list shares a prefix with a previously-seen turn. Run `--provider anthropic` or `--provider openai` for live numbers.

| Turn | Dynamic changed | llmbuffer cached | naive cached |
|------|:---------------:|:----------------:|:------------:|
| 1    | yes             | ✗ 0              | ✗ 0          |
| 2    | —               | ✓ 1,213          | ✓ 1,340      |
| 3    | —               | ✓ 1,245          | ✓ 1,368      |
| 4    | **yes**         | ✓ 1,274          | **✗ 0**      |
| 5    | —               | ✓ 1,297          | ✓ 1,416      |
| 6    | —               | ✓ 1,325          | ✓ 1,443      |
| 7    | **yes**         | ✓ 1,351          | **✗ 0**      |
| 8    | —               | ✓ 1,379          | ✓ 1,497      |
| 9    | —               | ✓ 1,403          | ✓ 1,525      |
| 10   | **yes**         | ✓ 1,430          | **✗ 0**      |
| 11   | —               | ✓ 1,458          | ✓ 1,568      |
| 12   | —               | ✓ 1,479          | ✓ 1,597      |
| 13   | **yes**         | ✓ 1,507          | **✗ 0**      |
| 14   | —               | ✓ 1,535          | ✓ 1,651      |
| 15   | —               | ✓ 1,561          | ✓ 1,677      |

| Metric | llmbuffer | naive |
|--------|----------:|------:|
| Cache hit ratio | **85.3%** | 66.1% |
| Total cached tokens | **19,457** | 15,082 |
| Est. cost (Anthropic, with caching) | **$0.016** | $0.028 |
| Est. savings vs no caching | **76.7%** | 59.5% |

Every time the dynamic context rotates (turns 4, 7, 10, 13) the naive approach suffers a **full cache miss** — the changed system prompt invalidates the entire prefix. `llmbuffer` keeps the static system and long-lived history stable, so only the new suffix is uncached regardless of what the dynamic context does.

### Run it yourself

```bash
# No API key needed:
uv run python -m llmbuffer.benchmark --provider simulated --compare --turns 15

# Live providers (needs API key):
uv run python -m llmbuffer.benchmark --provider anthropic --compare --turns 15
uv run python -m llmbuffer.benchmark --provider openai --compare --turns 15
uv run python -m llmbuffer.benchmark --provider gemini --compare --turns 15

# Ollama (local, needs server log access):
uv run python -m llmbuffer.benchmark --provider ollama \
    --ollama-log ~/.ollama/logs/server.log --compare

# JSON output:
uv run python -m llmbuffer.benchmark --provider anthropic --compare --format json
```

## Development

```bash
# Clone and set up:
git clone https://github.com/scottpurdy/llmbuffer
cd llmbuffer
uv sync

# Run tests:
uv run pytest

# Run benchmark (simulated, no API key needed):
uv run python -m llmbuffer.benchmark --provider simulated --compare
```

The test suite includes explicit **cache-stability tests** asserting that the static system prompt and long-lived history are byte-identical across turns — verifying the cache prefix is never accidentally mutated.

## License

MIT
