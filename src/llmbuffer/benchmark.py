"""Caching benchmark suite.

Benchmarks real prompt-cache hits reported by the provider's own API and
optionally compares the llmbuffer approach against a naive implementation.

Providers with native cache-hit reporting:
  gemini     — Google AI Studio (GEMINI_API_KEY or GOOGLE_API_KEY).
               Implicit context caching; reports cachedContentTokenCount.
               Default model: gemini-2.0-flash
  anthropic  — Anthropic API (ANTHROPIC_API_KEY); prompt caching.
               Default model: claude-haiku-4-5-20251001
  openai     — OpenAI API (OPENAI_API_KEY); prefix caching.
               Default model: gpt-4o-mini

Ollama (no cache field in the API response):
  Pass --ollama-log PATH to the llama.cpp server log for exact KV-cache stats.
  Without --ollama-log the command fails with a clear explanation.

All other providers will fail with a note about the missing cache stats.

Comparison mode (--compare):
  Runs the same conversation through both a naive chat implementation and
  the llmbuffer-managed implementation. In the naive version the static and
  dynamic system prompts are concatenated at the front; when the message list
  exceeds the token budget the oldest messages are dropped — exactly the
  worst case for cache stability. llmbuffer keeps the stable prefix fixed and
  injects the dynamic context after it.

Run:
  uv run python -m llmbuffer.benchmark                         # simulated
  uv run python -m llmbuffer.benchmark --provider gemini --compare --turns 18
  uv run python -m llmbuffer.benchmark --provider anthropic --compare
  uv run python -m llmbuffer.benchmark --provider ollama \\
      --ollama-log ~/.ollama/logs/server.log --compare
  uv run python -m llmbuffer.benchmark --format json --output report.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .config import TransitionMode
from .manager import PromptManager

Message = Dict[str, Any]

# ---------------------------------------------------------------------------
# Conversation script
# ---------------------------------------------------------------------------

# A realistic, sizeable system prompt. The length matters: providers like
# Gemini require >4096 tokens in the stable prefix before implicit caching
# kicks in. The padding lines are intentionally verbose policy text.
STATIC_SYSTEM = (
    "You are an expert software engineering assistant with deep knowledge of "
    "system design, algorithms, data structures, and engineering best practices "
    "across backend services, APIs, databases, infrastructure, and developer tooling. "
    "Your answers must be concise (2-4 sentences unless a longer answer is clearly "
    "warranted), technically accurate, and immediately actionable. "
    "When writing code, prefer Python 3.11+ unless the question specifies another language. "
    "Always consider trade-offs between performance, maintainability, and operational "
    "complexity. Acknowledge when multiple valid approaches exist and recommend the "
    "simplest one that satisfies the stated requirements without over-engineering. "
    "Never use filler phrases such as 'Great question!', 'Certainly!', or 'Of course!'. "
    "\n\n"
    "CODE STYLE GUIDE\n"
    "- Follow PEP 8 with a 100-character line limit.\n"
    "- Use type annotations throughout; prefer `from __future__ import annotations`.\n"
    "- Prefer dataclasses or Pydantic models over plain dicts for structured data.\n"
    "- Use pathlib.Path over os.path. Use contextlib.suppress over bare except:pass.\n"
    "- Avoid mutable default arguments; use `field(default_factory=...)` in dataclasses.\n"
    "- Name booleans with is_/has_/can_ prefixes. Name coroutines with async_ prefix only "
    "when the sync counterpart also exists and disambiguation is needed.\n"
    "- Prefer composition over inheritance. Favour immutable data structures when "
    "state management is a concern. Keep functions under 40 lines; split at logical seams.\n"
    "\n"
    "ARCHITECTURE PRINCIPLES\n"
    "- Apply SOLID principles thoughtfully, not dogmatically.\n"
    "- Treat observability (structured logging, metrics, distributed tracing) as a "
    "first-class requirement, not an afterthought.\n"
    "- Design for failure: assume external calls can fail; use retries with exponential "
    "backoff and jitter, circuit breakers for repeated failures, and timeouts everywhere.\n"
    "- Prefer proven libraries over hand-rolled solutions for auth, crypto, serialisation, "
    "and rate limiting. Evaluate dependencies by maintenance activity and CVE history.\n"
    "- Every public API must be versioned from day one. Use semantic versioning. "
    "Breaking changes require a new major version and a deprecation window of ≥6 months.\n"
    "\n"
    "SECURITY REQUIREMENTS\n"
    "- Validate all input at system boundaries; never trust client-supplied data.\n"
    "- Use parameterised queries or an ORM; never format SQL strings manually.\n"
    "- Hash passwords with bcrypt (cost ≥12) or argon2id; never store plaintext or MD5.\n"
    "- Store secrets in environment variables or a secrets manager (Vault, AWS Secrets "
    "Manager); never in source code, config files, or logs.\n"
    "- Set Content-Security-Policy, X-Frame-Options, and HSTS headers on all HTTP responses.\n"
    "- Rotate all credentials on a schedule; rotate immediately on suspected compromise.\n"
    "\n"
    "DATABASE CONVENTIONS\n"
    "- Name tables in snake_case plural (users, order_items). Name foreign keys "
    "<table_singular>_id (user_id, order_id).\n"
    "- Add created_at (NOT NULL DEFAULT now()) and updated_at to every table.\n"
    "- Index every foreign key column and every column used in WHERE/ORDER BY clauses.\n"
    "- Prefer partial indexes over full-table indexes when the query filters by a "
    "low-cardinality boolean or status column.\n"
    "- Use connection pooling (PgBouncer in transaction mode for PostgreSQL). "
    "Size the pool as (CPU cores × 2) + 1 as a starting heuristic.\n"
    "- Run EXPLAIN ANALYZE before shipping any query that touches >10k rows.\n"
    "\n"
    "TESTING STANDARDS\n"
    "- Unit tests must run in <100ms each with no I/O; use fakes or in-process stubs.\n"
    "- Integration tests may use real databases (PostgreSQL, Redis) via Docker Compose; "
    "never mock the database layer in integration tests.\n"
    "- Every public function must have at least one happy-path and one error-path test.\n"
    "- Aim for 80%+ line coverage; prioritise coverage of error-handling and edge cases "
    "over happy paths.\n"
    "- Load tests must be run before every major release against a production-like dataset.\n"
    + (
        "- Document all non-obvious invariants with a single-line comment starting with "
        "'Invariant:'. Do not document what the code does; document why a constraint exists.\n"
    ) * 6
)

# Simulates RAG results or other session-specific context that changes periodically.
DYNAMIC_CONTENTS = [
    (
        "[SESSION CONTEXT — retrieved at query time]\n"
        "Relevant documentation excerpts:\n"
        "• PEP 703 (no-GIL CPython): experimental in 3.12, stabilised in 3.14. "
        "Thread-safety now required for C extensions. asyncio unaffected.\n"
        "• SQLAlchemy 2.0: Session.execute() replaces Query API. "
        "Use select() constructs; lazy loading is opt-in via selectinload().\n"
        "Active sprint: backend performance hardening. Deadline: end of week."
    ),
    (
        "[SESSION CONTEXT — retrieved at query time]\n"
        "Relevant documentation excerpts:\n"
        "• FastAPI 0.115: Pydantic v2 is the only supported validator. "
        "Use model_config = ConfigDict(strict=True) for runtime type safety.\n"
        "• Redis 8.0: native HNSW vector search, JSON path filtering, "
        "40% throughput improvement on pub/sub under high fan-out.\n"
        "Active sprint: API redesign and caching layer. Deadline: Thursday."
    ),
    (
        "[SESSION CONTEXT — retrieved at query time]\n"
        "Relevant documentation excerpts:\n"
        "• Kubernetes 1.32: sidecar containers are GA — prefer them over "
        "init containers for log shippers and service-mesh proxies.\n"
        "• OpenTelemetry Python SDK 1.28: zero-code instrumentation via "
        "OTEL_PYTHON_CONFIGURATOR. Baggage propagation is now automatic.\n"
        "Active sprint: observability and infrastructure hardening. On-call this week."
    ),
    (
        "[SESSION CONTEXT — retrieved at query time]\n"
        "Relevant documentation excerpts:\n"
        "• PostgreSQL 17: MERGE with RETURNING clause, incremental sort "
        "improvements (~2× on partially sorted data), COPY FROM with ON ERROR.\n"
        "• Ruff 0.8: full isort replacement, Jupyter notebook linting, "
        "noqa comment parsing parity with flake8.\n"
        "Active sprint: data layer optimisation. Major release cut next Monday."
    ),
    (
        "[SESSION CONTEXT — retrieved at query time]\n"
        "Relevant documentation excerpts:\n"
        "• React 19 stable: Server Components and Actions are production-ready. "
        "useOptimistic and useFormStatus simplify async mutation patterns.\n"
        "• Vite 6.0: Environment API replaces custom SSR plugins; "
        "Rolldown integration cuts cold build times by ~60%.\n"
        "Active sprint: frontend modernisation. Design review on Wednesday."
    ),
    (
        "[SESSION CONTEXT — retrieved at query time]\n"
        "Relevant documentation excerpts:\n"
        "• Python 3.13 t-builds: free-threaded mode now passes the full CPython "
        "test suite. GIL can be disabled at runtime with PYTHON_GIL=0.\n"
        "• uv 0.5: workspace support, global tool installs, full pip-compat resolver. "
        "Drop-in replacement for pip + venv + pip-tools in CI pipelines.\n"
        "Active sprint: developer experience and toolchain modernisation."
    ),
]

DYNAMIC_INTERVAL = 3  # rotate dynamic context every N turns

QUESTIONS = [
    "What's the most efficient way to deduplicate a list of dicts in Python while preserving insertion order?",
    "How should I structure database migrations in a team environment to prevent conflicts?",
    "Explain optimistic vs pessimistic locking and when to choose each.",
    "What's the recommended pattern for async retries with exponential backoff in Python?",
    "How do I design an API that's easy to version without breaking existing clients?",
    "What are the key differences between Redis Streams and a message queue like RabbitMQ?",
    "Walk me through profiling a slow Python web endpoint step by step.",
    "What's the right way to manage secrets in a containerised app across environments?",
    "Explain connection pooling — when does it help, when does it hurt, how do I size it?",
    "What pattern should I use for background tasks in FastAPI?",
    "How do I implement a rate limiter that works correctly across multiple API instances?",
    "What's the best strategy for caching DB query results, including cache invalidation?",
    "How should I structure error handling in a service-oriented architecture?",
    "What are the trade-offs between event sourcing and a traditional CRUD model?",
    "How do I keep API documentation in sync with the actual implementation?",
    "What's the most pragmatic approach to testing external API integrations?",
    "How do I handle schema changes that must deploy with zero downtime?",
    "What's the trade-off between offset-based and cursor-based pagination?",
    "How should I design a multi-tenant app — row-level security vs separate schemas?",
    "What's the recommended approach for distributed tracing in microservices?",
]

# Token budget for compaction. The naive approach trims its combined message
# list to this limit; llmbuffer uses this as its long-lived history budget.
MAX_TOKENS = 3_500

# USD per million input tokens: (uncached price, cached price)
PRICING: Dict[str, Dict[str, Any]] = {
    "gemini": {"model": "gemini-2.0-flash", "uncached": 0.10, "cached": 0.025},
    "anthropic": {"model": "claude-haiku-4-5-20251001", "uncached": 0.80, "cached": 0.08},
    "openai": {"model": "gpt-4o-mini", "uncached": 0.15, "cached": 0.075},
    "ollama": {"model": "qwen3.6:latest", "uncached": 0.0, "cached": 0.0},
    "simulated": {"model": "simulated", "uncached": 3.00, "cached": 0.30},
}

# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class TurnResult:
    turn: int
    input_tokens: int
    cached_tokens: int
    cache_hit: bool
    dynamic_changed: bool = False
    approach: str = "llmbuffer"


@dataclass
class BenchmarkReport:
    provider: str
    model: str
    approach: str = "llmbuffer"
    turns: List[TurnResult] = field(default_factory=list)

    @property
    def total_input_tokens(self) -> int:
        return sum(t.input_tokens for t in self.turns)

    @property
    def total_cached_tokens(self) -> int:
        return sum(t.cached_tokens for t in self.turns)

    @property
    def cache_hit_ratio(self) -> float:
        if not self.total_input_tokens:
            return 0.0
        return self.total_cached_tokens / self.total_input_tokens

    def cost_estimate(self) -> Dict[str, float]:
        pricing = PRICING.get(self.provider, {"uncached": 0.0, "cached": 0.0})
        uncached_tokens = self.total_input_tokens - self.total_cached_tokens
        cost_with = (
            uncached_tokens * pricing["uncached"]
            + self.total_cached_tokens * pricing["cached"]
        ) / 1_000_000
        cost_without = self.total_input_tokens * pricing["uncached"] / 1_000_000
        return {
            "cost_with_caching_usd": round(cost_with, 6),
            "cost_without_caching_usd": round(cost_without, 6),
            "savings_usd": round(cost_without - cost_with, 6),
            "savings_pct": round(
                100 * (1 - cost_with / cost_without) if cost_without else 0.0, 2
            ),
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "approach": self.approach,
            "turns": [asdict(t) for t in self.turns],
            "total_input_tokens": self.total_input_tokens,
            "total_cached_tokens": self.total_cached_tokens,
            "cache_hit_ratio": round(self.cache_hit_ratio, 4),
            **self.cost_estimate(),
        }

    def to_markdown(self, *, heading: bool = True) -> str:
        d = self.to_dict()
        lines: List[str] = []
        if heading:
            lines += [
                f"# llmbuffer benchmark — {self.provider} / {self.model}",
                "",
            ]
        lines += [
            f"### {self.approach}",
            "",
            "| Turn | Dynamic changed | Input tokens | Cached tokens | Cache hit |",
            "|------|:---------------:|-------------:|--------------:|:---------:|",
        ]
        for t in self.turns:
            dyn = "yes" if t.dynamic_changed else "—"
            hit = "✓" if t.cache_hit else "✗"
            lines.append(
                f"| {t.turn:>4} | {dyn:^15} | {t.input_tokens:>12,} | "
                f"{t.cached_tokens:>13,} | {hit:^9} |"
            )
        lines += [
            "",
            f"**Total input tokens:** {d['total_input_tokens']:,}  ",
            f"**Total cached tokens:** {d['total_cached_tokens']:,}  ",
            f"**Cache hit ratio:** {d['cache_hit_ratio']:.1%}",
        ]
        if d["cost_without_caching_usd"] > 0:
            lines += [
                f"**Cost with caching:** ${d['cost_with_caching_usd']}  ",
                f"**Cost without caching:** ${d['cost_without_caching_usd']}  ",
                f"**Savings:** ${d['savings_usd']} ({d['savings_pct']}%)",
            ]
        return "\n".join(lines)


def _comparison_markdown(llmbuf: BenchmarkReport, naive: BenchmarkReport) -> str:
    n = len(llmbuf.turns)
    lc = llmbuf.cost_estimate()
    nc = naive.cost_estimate()
    summary = [
        "## Summary",
        "",
        "| Metric                     |   llmbuffer |       naive |",
        "|:---------------------------|------------:|------------:|",
        f"| Total input tokens         | {llmbuf.total_input_tokens:>11,} | {naive.total_input_tokens:>11,} |",
        f"| Total cached tokens        | {llmbuf.total_cached_tokens:>11,} | {naive.total_cached_tokens:>11,} |",
        f"| Cache hit ratio            | {llmbuf.cache_hit_ratio:>10.1%} | {naive.cache_hit_ratio:>10.1%} |",
    ]
    if lc["cost_without_caching_usd"] > 0:
        summary += [
            f"| Est. cost (with caching)   | ${lc['cost_with_caching_usd']:>10} | ${nc['cost_with_caching_usd']:>10} |",
            f"| Est. savings               | ${lc['savings_usd']:>10} ({lc['savings_pct']}%) | ${nc['savings_usd']:>10} ({nc['savings_pct']}%) |",
        ]
    return "\n".join([
        f"# llmbuffer benchmark — {llmbuf.provider} / {llmbuf.model}",
        f"> {n}-turn comparison: **llmbuffer** vs **naive** approach",
        "",
        llmbuf.to_markdown(heading=False),
        "",
        naive.to_markdown(heading=False),
        "",
        *summary,
    ])


# ---------------------------------------------------------------------------
# Provider error
# ---------------------------------------------------------------------------


class _ProviderError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# Provider classes
# ---------------------------------------------------------------------------


class _GeminiProvider:
    """Google AI Studio — reports cachedContentTokenCount in usageMetadata."""

    def __init__(self, model: str, api_key: str) -> None:
        self.model = model
        self._api_key = api_key

    def send(self, messages: List[Message]) -> Tuple[str, int, int]:
        # Convert OpenAI-format messages to Gemini contents + systemInstruction.
        system_parts: List[Dict[str, Any]] = []
        contents: List[Dict[str, Any]] = []
        for msg in messages:
            role = msg["role"]
            raw = msg.get("content") or ""
            text = raw if isinstance(raw, str) else json.dumps(raw, default=str)
            if role == "system":
                system_parts.append({"text": text})
            elif role == "user":
                # Merge consecutive user messages (e.g. dynamic context + question).
                if contents and contents[-1]["role"] == "user":
                    contents[-1]["parts"].append({"text": "\n\n" + text})
                else:
                    contents.append({"role": "user", "parts": [{"text": text}]})
            else:
                contents.append({"role": "model", "parts": [{"text": text}]})

        body: Dict[str, Any] = {
            "contents": contents,
            "generationConfig": {"maxOutputTokens": 300, "temperature": 0.0},
        }
        if system_parts:
            body["systemInstruction"] = {"parts": system_parts}

        url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model}:generateContent?key={self._api_key}"
        )
        req = urllib.request.Request(
            url,
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            raise _ProviderError(
                f"Gemini API error {exc.code}: {exc.read().decode(errors='replace')}"
            ) from exc

        usage = data.get("usageMetadata", {})
        # Field name varies between model generations.
        cached = int(
            usage.get("cachedContentTokenCount")
            or usage.get("cachedInputTokenCount")
            or 0
        )
        total = int(usage.get("promptTokenCount") or 0)
        reply = data["candidates"][0]["content"]["parts"][0]["text"]
        return reply, total, cached


class _AnthropicProvider:
    """Anthropic Messages API — reports cache_read_input_tokens."""

    def __init__(self, model: str) -> None:
        try:
            import anthropic as _ant
            self._client = _ant.Anthropic()
            self._not_given = _ant.NOT_GIVEN
        except ImportError:
            raise _ProviderError(
                "The 'anthropic' package is required: "
                "pip install 'llmbuffer[anthropic]'"
            )
        self.model = model

    def send(self, messages: List[Message]) -> Tuple[str, int, int]:
        system_msgs = [m for m in messages if m["role"] == "system"]
        convo = [m for m in messages if m["role"] != "system"]
        # Inject cache_control so Anthropic actually caches the system prompt.
        if system_msgs:
            system_content = [
                {"type": "text", "text": m["content"], "cache_control": {"type": "ephemeral"}}
                for m in system_msgs
            ]
        else:
            system_content = self._not_given

        resp = self._client.messages.create(
            model=self.model,
            max_tokens=300,
            system=system_content,
            messages=convo,
        )
        usage = resp.usage
        cached = int(getattr(usage, "cache_read_input_tokens", 0) or 0)
        created = int(getattr(usage, "cache_creation_input_tokens", 0) or 0)
        total = usage.input_tokens + cached + created
        text = "".join(b.text for b in resp.content if b.type == "text")
        return text, total, cached


class _OpenAIProvider:
    """OpenAI Chat Completions — reports prompt_tokens_details.cached_tokens."""

    def __init__(self, model: str) -> None:
        try:
            import openai as _oai
            self._client = _oai.OpenAI()
        except ImportError:
            raise _ProviderError(
                "The 'openai' package is required: pip install 'llmbuffer[openai]'"
            )
        self.model = model

    def send(self, messages: List[Message]) -> Tuple[str, int, int]:
        resp = self._client.chat.completions.create(
            model=self.model, messages=messages, max_tokens=300, temperature=0
        )
        usage = resp.usage
        details = getattr(usage, "prompt_tokens_details", None)
        cached = int((getattr(details, "cached_tokens", 0) or 0) if details else 0)
        return resp.choices[0].message.content or "", usage.prompt_tokens, cached


class _OllamaProvider:
    """Ollama — cache stats read from the llama.cpp server log."""

    def __init__(self, model: str, base_url: str, log_path: str) -> None:
        self.model = model
        self._base_url = base_url.rstrip("/")
        self._log_path = log_path

    def _log_end(self) -> int:
        try:
            with open(self._log_path, "rb") as f:
                f.seek(0, 2)
                return f.tell()
        except OSError:
            return 0

    def _cached_tokens_since(self, pos: int) -> int:
        """Read new log lines and return tokens restored from the KV cache."""
        try:
            with open(self._log_path) as f:
                f.seek(pos)
                for line in f:
                    # "restored context checkpoint (... n_tokens = N ...)"
                    m = re.search(r"restored context checkpoint.*n_tokens\s*=\s*(\d+)", line)
                    if m:
                        return int(m.group(1))
        except OSError:
            pass
        return 0

    def send(self, messages: List[Message]) -> Tuple[str, int, int]:
        log_pos = self._log_end()
        body: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"num_predict": 300, "temperature": 0},
        }
        req = urllib.request.Request(
            f"{self._base_url}/api/chat",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                data = json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            raise _ProviderError(
                f"Ollama error {exc.code}: {exc.read().decode(errors='replace')}"
            ) from exc

        prompt_tokens = int(data.get("prompt_eval_count", 0) or 0)
        cached = self._cached_tokens_since(log_pos)
        text = data.get("message", {}).get("content", "")
        return text, prompt_tokens, cached


def _make_provider(
    name: str,
    model: str,
    ollama_log: Optional[str] = None,
    base_url: str = "http://localhost:11434",
) -> Any:
    if name == "gemini":
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise _ProviderError(
                "Set GEMINI_API_KEY (or GOOGLE_API_KEY) to use the Gemini provider."
            )
        return _GeminiProvider(model, api_key)
    if name == "anthropic":
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise _ProviderError("Set ANTHROPIC_API_KEY to use the Anthropic provider.")
        return _AnthropicProvider(model)
    if name == "openai":
        if not os.environ.get("OPENAI_API_KEY"):
            raise _ProviderError("Set OPENAI_API_KEY to use the OpenAI provider.")
        return _OpenAIProvider(model)
    if name == "ollama":
        if not ollama_log:
            raise _ProviderError(
                "Ollama does not report cache hits in its API responses.\n"
                "Pass --ollama-log PATH (e.g. --ollama-log ~/.ollama/logs/server.log)\n"
                "to read exact KV-cache stats from the llama.cpp server log instead."
            )
        return _OllamaProvider(model, base_url, ollama_log)
    raise _ProviderError(
        f"Provider '{name}' does not expose cache-hit statistics via its API.\n"
        "Supported providers: gemini, anthropic, openai, ollama (with --ollama-log)."
    )


# ---------------------------------------------------------------------------
# Conversation helpers
# ---------------------------------------------------------------------------


def _dynamic_context(turn: int) -> str:
    idx = ((turn - 1) // DYNAMIC_INTERVAL) % len(DYNAMIC_CONTENTS)
    return DYNAMIC_CONTENTS[idx]


def _dynamic_changed(turn: int) -> bool:
    return turn == 1 or (turn - 1) % DYNAMIC_INTERVAL == 0


def _make_llmbuffer_manager() -> PromptManager:
    return PromptManager(
        static_system_prompt=STATIC_SYSTEM,
        transition_mode=TransitionMode.AGENT_CYCLE,
        max_tokens=MAX_TOKENS,
        dynamic_system_role="user",
    )


class _NaiveChat:
    """Naive chat: static + dynamic system combined at the front, oldest-first trim.

    This is the worst case for cache stability: every time the dynamic context
    rotates the entire cache prefix is invalidated, and every drop of old messages
    also changes the prefix seen by the provider.
    """

    def __init__(self, adapter: Any) -> None:
        self._adapter = adapter
        self._history: List[Message] = []

    def build_and_trim(self, turn: int, question: str) -> List[Message]:
        combined_system = STATIC_SYSTEM + "\n\n" + _dynamic_context(turn)
        messages: List[Message] = [{"role": "system", "content": combined_system}]
        messages.extend(self._history)
        messages.append({"role": "user", "content": question})
        # Drop the oldest conversation messages (never the system) until under budget.
        while self._adapter.count_tokens(messages) > MAX_TOKENS and len(messages) > 2:
            messages.pop(1)
        return messages

    def commit(self, messages: List[Message], reply: str) -> None:
        # Persist everything that was actually sent (minus the system message).
        self._history = messages[1:]
        self._history.append({"role": "assistant", "content": reply})


# ---------------------------------------------------------------------------
# Simulated benchmark (no network; used by tests)
# ---------------------------------------------------------------------------


class _SimulatedCache:
    """Exact prefix-match cache model: a turn is a cache hit when the longest
    prefix of its message list matches a prefix seen in a prior turn."""

    def __init__(self) -> None:
        self._seen: List[str] = []

    def query(self, messages: List[Message], adapter: Any) -> Tuple[int, int]:
        total = adapter.count_tokens(messages)
        cached = 0
        for cut in range(len(messages), 0, -1):
            key = json.dumps(messages[:cut], sort_keys=True)
            if key in self._seen:
                cached = adapter.count_tokens(messages[:cut])
                break
        for cut in range(1, len(messages) + 1):
            key = json.dumps(messages[:cut], sort_keys=True)
            if key not in self._seen:
                self._seen.append(key)
        return total, cached


def run_simulated(
    n_turns: int = 6, compare: bool = False
) -> Tuple[BenchmarkReport, Optional[BenchmarkReport]]:
    """No-network benchmark using a deterministic simulated cache.

    Returns ``(llmbuffer_report, naive_report)``. ``naive_report`` is ``None``
    when ``compare=False``.
    """
    manager = _make_llmbuffer_manager()
    adapter = manager.adapter
    sim_llmbuf = _SimulatedCache()
    sim_naive = _SimulatedCache() if compare else None

    llmbuf_report = BenchmarkReport(provider="simulated", model="simulated", approach="llmbuffer")
    naive_report = (
        BenchmarkReport(provider="simulated", model="simulated", approach="naive")
        if compare
        else None
    )
    naive_chat = _NaiveChat(adapter) if compare else None
    reply_template = "Short answer to question {i}."

    for i, question in enumerate(QUESTIONS[:n_turns], start=1):
        dynamic = _dynamic_context(i)
        changed = _dynamic_changed(i)
        reply = reply_template.format(i=i)

        # llmbuffer turn
        manager.append({"role": "user", "content": question})
        llmbuf_msgs = manager.build_messages(
            dynamic_system_prompt=dynamic, apply_cache_markers=False
        )
        total, cached = sim_llmbuf.query(llmbuf_msgs, adapter)
        llmbuf_report.turns.append(
            TurnResult(
                turn=i,
                input_tokens=total,
                cached_tokens=cached,
                cache_hit=cached > 0,
                dynamic_changed=changed,
            )
        )
        manager.append({"role": "assistant", "content": reply})

        # naive turn
        if compare and naive_chat and sim_naive and naive_report:
            naive_msgs = naive_chat.build_and_trim(i, question)
            n_total, n_cached = sim_naive.query(naive_msgs, adapter)
            naive_report.turns.append(
                TurnResult(
                    turn=i,
                    input_tokens=n_total,
                    cached_tokens=n_cached,
                    cache_hit=n_cached > 0,
                    dynamic_changed=changed,
                    approach="naive",
                )
            )
            naive_chat.commit(naive_msgs, reply)

    return llmbuf_report, naive_report


# ---------------------------------------------------------------------------
# Live benchmark
# ---------------------------------------------------------------------------


def run_live(
    provider_name: str,
    model: str,
    n_turns: int = 15,
    compare: bool = False,
    ollama_log: Optional[str] = None,
    base_url: str = "http://localhost:11434",
) -> Tuple[BenchmarkReport, Optional[BenchmarkReport]]:
    """Run the benchmark against a live provider.

    Returns ``(llmbuffer_report, naive_report)``. ``naive_report`` is ``None``
    when ``compare=False``.

    Note: comparison mode sends two requests per turn (one per approach).
    """
    provider = _make_provider(provider_name, model, ollama_log, base_url)

    llmbuf_mgr = _make_llmbuffer_manager()
    llmbuf_report = BenchmarkReport(
        provider=provider_name, model=provider.model, approach="llmbuffer"
    )
    naive_chat = _NaiveChat(llmbuf_mgr.adapter) if compare else None
    naive_report = (
        BenchmarkReport(provider=provider_name, model=provider.model, approach="naive")
        if compare
        else None
    )

    for i, question in enumerate(QUESTIONS[:n_turns], start=1):
        dynamic = _dynamic_context(i)
        changed = _dynamic_changed(i)

        # --- llmbuffer turn ---
        llmbuf_mgr.append({"role": "user", "content": question})
        llmbuf_messages = llmbuf_mgr.build_messages(
            dynamic_system_prompt=dynamic, apply_cache_markers=True
        )
        reply, total, cached = provider.send(llmbuf_messages)
        llmbuf_report.turns.append(
            TurnResult(
                turn=i,
                input_tokens=total,
                cached_tokens=cached,
                cache_hit=cached > 0,
                dynamic_changed=changed,
            )
        )
        llmbuf_mgr.append({"role": "assistant", "content": reply})

        # --- naive turn ---
        if compare and naive_chat is not None and naive_report is not None:
            naive_messages = naive_chat.build_and_trim(i, question)
            _, n_total, n_cached = provider.send(naive_messages)
            naive_report.turns.append(
                TurnResult(
                    turn=i,
                    input_tokens=n_total,
                    cached_tokens=n_cached,
                    cache_hit=n_cached > 0,
                    dynamic_changed=changed,
                    approach="naive",
                )
            )
            # Use the same reply for both approaches so conversation content is identical.
            naive_chat.commit(naive_messages, reply)

    return llmbuf_report, naive_report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="llmbuffer caching benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  uv run python -m llmbuffer.benchmark\n"
            "  uv run python -m llmbuffer.benchmark --provider gemini --compare --turns 18\n"
            "  uv run python -m llmbuffer.benchmark --provider anthropic --compare\n"
            "  uv run python -m llmbuffer.benchmark --provider ollama \\\n"
            "      --ollama-log ~/.ollama/logs/server.log --compare\n"
        ),
    )
    parser.add_argument(
        "--provider",
        choices=["simulated", "gemini", "anthropic", "openai", "ollama"],
        default="gemini",
        help="provider to benchmark (default: gemini)",
    )
    parser.add_argument("--model", help="model override")
    parser.add_argument("--turns", type=int, default=15)
    parser.add_argument(
        "--compare",
        action="store_true",
        help="also run the naive approach and show a side-by-side comparison",
    )
    parser.add_argument(
        "--ollama-log",
        metavar="PATH",
        help="path to Ollama server log (required for --provider ollama)",
    )
    parser.add_argument(
        "--base-url",
        default="http://localhost:11434",
        help="Ollama server base URL",
    )
    parser.add_argument("--format", choices=["json", "markdown"], default="markdown")
    parser.add_argument("--output", metavar="PATH", help="write report to file")
    args = parser.parse_args(argv)

    model = args.model or PRICING.get(args.provider, {}).get("model", "")

    try:
        if args.provider == "simulated":
            llmbuf_report, naive_report = run_simulated(args.turns, compare=args.compare)
        else:
            llmbuf_report, naive_report = run_live(
                args.provider,
                model,
                n_turns=args.turns,
                compare=args.compare,
                ollama_log=args.ollama_log,
                base_url=args.base_url,
            )
    except _ProviderError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    comparing = naive_report is not None and args.compare
    if args.format == "json":
        if comparing:
            out = json.dumps(
                {"llmbuffer": llmbuf_report.to_dict(), "naive": naive_report.to_dict()},
                indent=2,
            )
        else:
            out = json.dumps(llmbuf_report.to_dict(), indent=2)
    else:
        if comparing:
            out = _comparison_markdown(llmbuf_report, naive_report)
        else:
            out = llmbuf_report.to_markdown()

    if args.output:
        with open(args.output, "w") as f:
            f.write(out + "\n")
        print(f"report written to {args.output}")
    else:
        print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
