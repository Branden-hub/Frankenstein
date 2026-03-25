"""
Living AI System — Working Memory (Tier 1)
Active context window — current reasoning state,
recent conversation, in-progress task context.
High bandwidth, small capacity, highly volatile.
"""

import hashlib
import json
import time
from collections import deque
from pathlib import Path

import structlog

log = structlog.get_logger(__name__)

MAX_TOKENS_PER_SESSION = 32000
AVERAGE_CHARS_PER_TOKEN = 4


class WorkingMemoryEntry:
    def __init__(self, role: str, content: str, trace_id: str, timestamp: float):
        self.role = role
        self.content = content
        self.trace_id = trace_id
        self.timestamp = timestamp
        self.token_estimate = len(content) // AVERAGE_CHARS_PER_TOKEN


class WorkingMemory:
    """
    Tier 1 memory — the active context window.
    Maintains conversation history per session with token-budget enforcement.
    When budget is exceeded, oldest messages are evicted (rolling FIFO).
    All state is held in process — no database required for working memory.
    """

    def __init__(self):
        self._sessions: dict[str, deque] = {}
        self._session_token_counts: dict[str, int] = {}
        self._initialised = False

    async def initialise(self) -> None:
        self._initialised = True
        log.info("working_memory.initialised")

    async def add(
        self,
        session_id: str,
        role: str,
        content: str,
        trace_id: str,
    ) -> None:
        """Add a new entry to the session's working memory."""
        if session_id not in self._sessions:
            self._sessions[session_id] = deque()
            self._session_token_counts[session_id] = 0

        entry = WorkingMemoryEntry(
            role=role,
            content=content,
            trace_id=trace_id,
            timestamp=time.time(),
        )

        self._sessions[session_id].append(entry)
        self._session_token_counts[session_id] += entry.token_estimate

        # Enforce token budget — evict oldest messages
        while (
            self._session_token_counts[session_id] > MAX_TOKENS_PER_SESSION
            and self._sessions[session_id]
        ):
            evicted = self._sessions[session_id].popleft()
            self._session_token_counts[session_id] -= evicted.token_estimate
            log.debug(
                "working_memory.evicted",
                session_id=session_id,
                role=evicted.role,
                tokens_freed=evicted.token_estimate,
            )

    async def get_context(self, session_id: str) -> list[dict]:
        """Return the current context window for a session as a list of dicts."""
        if session_id not in self._sessions:
            return []
        return [
            {
                "role": entry.role,
                "content": entry.content,
                "timestamp": entry.timestamp,
                "trace_id": entry.trace_id,
            }
            for entry in self._sessions[session_id]
        ]

    async def get_token_count(self, session_id: str) -> int:
        """Return current token estimate for a session."""
        return self._session_token_counts.get(session_id, 0)

    async def clear_session(self, session_id: str) -> None:
        """Clear all working memory for a session."""
        self._sessions.pop(session_id, None)
        self._session_token_counts.pop(session_id, None)

    async def compute_consistency_hash(self) -> str:
        """
        Compute a hash of all current working memory state.
        Used by the CEE for cross-tier consistency verification.
        """
        state = {
            sid: [
                {"role": e.role, "content": e.content}
                for e in entries
            ]
            for sid, entries in self._sessions.items()
        }
        serialised = json.dumps(state, sort_keys=True)
        return hashlib.sha256(serialised.encode()).hexdigest()

    def get_status(self) -> dict:
        """Return status summary for health endpoint."""
        return {
            "active_sessions": len(self._sessions),
            "total_tokens": sum(self._session_token_counts.values()),
            "max_tokens_per_session": MAX_TOKENS_PER_SESSION,
        }
