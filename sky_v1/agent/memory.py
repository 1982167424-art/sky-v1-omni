"""sky_v1.agent.memory: Short-term rolling context + long-term RAG-style memory."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from sky_v1.utils.logging import get_logger

log = get_logger("agent.memory")

_VALID_ROLES = {"user", "assistant", "system", "tool"}


class _InMemoryStore:
    """Tiny fallback vector-less store for LongTermMemory when nothing is provided.

    Stores memories as (id, text, metadata) and recalls via simple substring /
    case-insensitive keyword scoring — deterministic and zero dependency.
    """

    def __init__(self) -> None:
        self._items: list[tuple[str, str, dict]] = []

    def add(self, memory_id: str, text: str, metadata: dict) -> None:
        if not isinstance(memory_id, str):
            raise TypeError(
                f"memory_id must be str, got {type(memory_id).__name__}"
            )
        if not isinstance(text, str):
            raise TypeError(f"text must be str, got {type(text).__name__}")
        if not isinstance(metadata, dict):
            raise TypeError(
                f"metadata must be dict, got {type(metadata).__name__}"
            )
        self._items.append((memory_id, text, dict(metadata)))

    def query(self, text: str, top_k: int = 3) -> list[dict]:
        if not isinstance(text, str):
            raise TypeError(f"text must be str, got {type(text).__name__}")
        if not isinstance(top_k, int) or isinstance(top_k, bool):
            raise TypeError(f"top_k must be int, got {type(top_k).__name__}")
        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}")
        q_tokens = {t for t in text.lower().split() if t}
        scored: list[tuple[int, dict]] = []
        for mid, body, meta in self._items:
            body_low = body.lower()
            overlap = sum(1 for t in q_tokens if t and t in body_low)
            if q_tokens:
                score = overlap
            else:
                score = 0
            entry = {
                "id": mid,
                "text": body,
                "metadata": dict(meta),
                "score": float(score),
            }
            scored.append((score, entry))
        scored.sort(key=lambda kv: kv[0], reverse=True)
        return [entry for _, entry in scored[:top_k]]


class ShortTermMemory:
    """Rolling chat-context buffer (list of messages) bounded by max_turns."""

    def __init__(self, max_turns: int = 20) -> None:
        if not isinstance(max_turns, int) or isinstance(max_turns, bool):
            raise TypeError(
                f"max_turns must be int, got {type(max_turns).__name__}"
            )
        if max_turns < 1:
            raise ValueError(f"max_turns must be >= 1, got {max_turns}")
        self.max_turns: int = max_turns
        self._messages: list[dict] = []

    def add(
        self, role: str, content: str, extra: dict | None = None
    ) -> None:
        if not isinstance(role, str):
            raise TypeError(f"role must be str, got {type(role).__name__}")
        if role not in _VALID_ROLES:
            raise ValueError(
                f"role must be one of {sorted(_VALID_ROLES)}, got {role!r}"
            )
        if not isinstance(content, str):
            raise TypeError(
                f"content must be str, got {type(content).__name__}"
            )
        if extra is None:
            extra = {}
        if not isinstance(extra, dict):
            raise TypeError(f"extra must be dict or None, got {type(extra).__name__}")
        ts = datetime.now(timezone.utc).isoformat()
        msg = {
            "role": role,
            "content": content,
            "ts": ts,
            "extra": dict(extra),
        }
        self._messages.append(msg)
        while len(self._messages) > self.max_turns:
            self._messages.pop(0)

    def get(self, last_n: int | None = None) -> list[dict]:
        if last_n is None:
            return [dict(m) for m in self._messages]
        if not isinstance(last_n, int) or isinstance(last_n, bool):
            raise TypeError(f"last_n must be int or None, got {type(last_n).__name__}")
        if last_n < 1:
            raise ValueError(f"last_n must be >= 1 when provided, got {last_n}")
        return [dict(m) for m in self._messages[-last_n:]]

    def clear(self) -> None:
        self._messages.clear()

    def __len__(self) -> int:
        return len(self._messages)


class LongTermMemory:
    """Long-term memory wrapping a store + optional embedder.

    If `store` is None we default to a deterministic _InMemoryStore so the
    subsystem remains usable without any external vector DB dependency.
    """

    def __init__(
        self,
        store: Any | None = None,
        embedder: Any | None = None,
    ) -> None:
        self.embedder = embedder
        if store is None:
            self.store: Any = _InMemoryStore()
        else:
            self.store = store

    def remember(self, text: str, metadata: dict | None = None) -> str:
        if not isinstance(text, str):
            raise TypeError(f"text must be str, got {type(text).__name__}")
        if metadata is None:
            metadata = {}
        if not isinstance(metadata, dict):
            raise TypeError(
                f"metadata must be dict or None, got {type(metadata).__name__}"
            )
        memory_id = str(uuid.uuid4())
        try:
            if hasattr(self.store, "add") and callable(self.store.add):
                self.store.add(memory_id, text, dict(metadata))
            else:
                log.warning(
                    "LongTermMemory store has no .add; skipping persistence",
                    store_type=type(self.store).__name__,
                )
        except Exception as e:
            log.error("LongTermMemory.remember failed", error=str(e))
        return memory_id

    def recall(self, query: str, top_k: int = 3) -> list[dict]:
        if not isinstance(query, str):
            raise TypeError(f"query must be str, got {type(query).__name__}")
        if not isinstance(top_k, int) or isinstance(top_k, bool):
            raise TypeError(f"top_k must be int, got {type(top_k).__name__}")
        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}")
        try:
            embedded_query: Any = query
            if (
                self.embedder is not None
                and hasattr(self.embedder, "embed")
                and callable(self.embedder.embed)
            ):
                try:
                    embedded_query = self.embedder.embed(query)
                except Exception as e:
                    log.warning("embedder.embed failed, using raw query", error=str(e))
                    embedded_query = query
            if hasattr(self.store, "query") and callable(self.store.query):
                try:
                    sig_arg_count = getattr(self.store.query, "__code__", None)
                    if sig_arg_count is not None and hasattr(sig_arg_count, "co_argcount"):
                        argcount = sig_arg_count.co_argcount
                        if argcount >= 3:
                            raw = self.store.query(embedded_query, top_k=top_k)
                        else:
                            raw = self.store.query(embedded_query)
                    else:
                        raw = self.store.query(embedded_query, top_k=top_k)
                except TypeError:
                    raw = self.store.query(embedded_query)
            else:
                raw = []
        except Exception as e:
            log.error("LongTermMemory.recall failed", error=str(e))
            raw = []

        normalized: list[dict] = []
        if isinstance(raw, list):
            for item in raw:
                if isinstance(item, dict):
                    normalized.append(dict(item))
                else:
                    normalized.append({"text": str(item), "metadata": {}})
        elif raw is None:
            normalized = []
        else:
            normalized = [{"text": str(raw), "metadata": {}}]
        return normalized
