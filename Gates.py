"""
Living AI System — Capability Gates
Every capability is gated. Default state is disabled.
All invocations are logged to the immutable audit trail.
"""

import hashlib
import json
import sqlite3
import time
import uuid
from pathlib import Path

import structlog

log = structlog.get_logger(__name__)

AUDIT_DB_PATH = Path("data/audit_log.db")

CAPABILITIES: dict[str, dict] = {
    "vision":          {"default": False, "risk": "low"},
    "audio_input":     {"default": False, "risk": "low"},
    "speech_output":   {"default": False, "risk": "low"},
    "web_search":      {"default": False, "risk": "medium"},
    "code_execution":  {"default": False, "risk": "high"},
    "file_write":      {"default": False, "risk": "high"},
    "api_calls":       {"default": False, "risk": "medium"},
    "memory_write":    {"default": True,  "risk": "medium"},
    "agent_spawn":     {"default": False, "risk": "high"},
    "browser_control": {"default": False, "risk": "high"},
}


class CapabilityGate:
    """
    Every capability module is governed by a binary flag.
    Default state is disabled for all high-risk capabilities.
    Checks are synchronous and must pass before any capability invocation.
    All checks are recorded in the immutable audit log.
    """

    def __init__(self):
        self._enabled: dict[str, bool] = {
            name: info["default"]
            for name, info in CAPABILITIES.items()
        }
        self._audit_db: sqlite3.Connection | None = None
        self._prev_hash: str = "genesis"
        self._init_audit_db()

    def _init_audit_db(self) -> None:
        """Initialise the append-only hash-chained audit log."""
        AUDIT_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        self._audit_db = sqlite3.connect(str(AUDIT_DB_PATH), check_same_thread=False)
        self._audit_db.execute("""
            CREATE TABLE IF NOT EXISTS audit_log (
                id TEXT PRIMARY KEY,
                timestamp REAL NOT NULL,
                trace_id TEXT,
                action_type TEXT NOT NULL,
                capability TEXT NOT NULL,
                session_id TEXT,
                outcome TEXT NOT NULL,
                payload_hash TEXT NOT NULL,
                prev_hash TEXT NOT NULL,
                chain_hash TEXT NOT NULL
            )
        """)
        self._audit_db.commit()

        # Load the last chain hash
        cursor = self._audit_db.execute(
            "SELECT chain_hash FROM audit_log ORDER BY timestamp DESC LIMIT 1"
        )
        row = cursor.fetchone()
        if row:
            self._prev_hash = row[0]

    def enable(self, capability: str) -> None:
        """Enable a capability."""
        if capability not in CAPABILITIES:
            raise ValueError(f"Unknown capability: {capability}")
        self._enabled[capability] = True
        log.info("capability_gate.enabled", capability=capability)

    def disable(self, capability: str) -> None:
        """Disable a capability."""
        if capability not in CAPABILITIES:
            raise ValueError(f"Unknown capability: {capability}")
        self._enabled[capability] = False
        log.info("capability_gate.disabled", capability=capability)

    def check(self, capability: str, session_id: str = "", trace_id: str = "") -> bool:
        """
        Check whether a capability is enabled.
        All checks — pass or fail — are recorded in the audit log.
        """
        if capability not in CAPABILITIES:
            log.warning("capability_gate.unknown_capability", capability=capability)
            self._log_audit(
                action_type="check",
                capability=capability,
                session_id=session_id,
                trace_id=trace_id,
                outcome="denied_unknown",
            )
            return False

        granted = self._enabled.get(capability, False)
        self._log_audit(
            action_type="check",
            capability=capability,
            session_id=session_id,
            trace_id=trace_id,
            outcome="granted" if granted else "denied",
        )

        if not granted:
            log.warning(
                "capability_gate.denied",
                capability=capability,
                session_id=session_id,
            )

        return granted

    def _log_audit(
        self,
        action_type: str,
        capability: str,
        session_id: str,
        trace_id: str,
        outcome: str,
    ) -> None:
        """
        Write an immutable hash-chained audit record.
        Each record stores SHA256(prev_hash + action_json).
        """
        if not self._audit_db:
            return

        record_id = str(uuid.uuid4())
        timestamp = time.time()

        payload = json.dumps({
            "action_type": action_type,
            "capability": capability,
            "session_id": session_id,
            "outcome": outcome,
        }, sort_keys=True)

        payload_hash = hashlib.sha256(payload.encode()).hexdigest()
        chain_input = self._prev_hash + payload_hash
        chain_hash = hashlib.sha256(chain_input.encode()).hexdigest()

        try:
            self._audit_db.execute(
                """INSERT INTO audit_log
                   (id, timestamp, trace_id, action_type, capability,
                    session_id, outcome, payload_hash, prev_hash, chain_hash)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    record_id, timestamp, trace_id, action_type,
                    capability, session_id, outcome,
                    payload_hash, self._prev_hash, chain_hash,
                ),
            )
            self._audit_db.commit()
            self._prev_hash = chain_hash
        except Exception as exc:
            log.error("capability_gate.audit_error", error=str(exc))

    def get_status(self) -> dict:
        """Return current gate states for health endpoint."""
        return {
            "capabilities": {
                name: {
                    "enabled": self._enabled.get(name, False),
                    "risk": info["risk"],
                }
                for name, info in CAPABILITIES.items()
            }
        }
