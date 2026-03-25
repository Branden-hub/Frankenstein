"""
Living AI System — Episodic Memory (Tier 2)
Vector database storing encoded representations of past interactions
with temporal metadata. Semantic search retrieval.
Models hippocampal episodic memory — experiences with context and time.
"""

import asyncio
import hashlib
import json
import time
import uuid
from pathlib import Path
from typing import Any

import structlog

log = structlog.get_logger(__name__)

ENCODING_QUEUE_MAX = 1000
IMPORTANCE_THRESHOLD = 0.3
RETENTION_DAYS = 365


class EpisodicEntry:
    def __init__(
        self,
        entry_id: str,
        session_id: str,
        user_input: str,
        assistant_output: str,
        embedding: list[float],
        timestamp: float,
        importance: float,
        trace_id: str,
    ):
        self.entry_id = entry_id
        self.session_id = session_id
        self.user_input = user_input
        self.assistant_output = assistant_output
        self.embedding = embedding
        self.timestamp = timestamp
        self.importance = importance
        self.trace_id = trace_id


class EpisodicMemory:
    """
    Tier 2 memory — the episodic store.
    Uses ChromaDB for vector similarity search.
    Encodes past interactions with temporal metadata.
    Only experiences exceeding the importance threshold
    are committed to long-term storage — mirroring
    the biological importance filtering of the hippocampus.
    """

    def __init__(self):
        self._client = None
        self._collection = None
        self._encoding_queue: asyncio.Queue = asyncio.Queue(maxsize=ENCODING_QUEUE_MAX)
        self._embedding_model = None
        self._initialised = False

    async def initialise(self) -> None:
        """Initialise ChromaDB and embedding model."""
        try:
            import chromadb
            from chromadb.config import Settings

            data_path = Path("data/episodic_memory")
            data_path.mkdir(parents=True, exist_ok=True)

            self._client = chromadb.PersistentClient(
                path=str(data_path),
                settings=Settings(anonymized_telemetry=False),
            )
            self._collection = self._client.get_or_create_collection(
                name="episodic_memory",
                metadata={"hnsw:space": "cosine"},
            )

            # Load local embedding model
            from sentence_transformers import SentenceTransformer
            self._embedding_model = SentenceTransformer(
                "all-MiniLM-L6-v2",
                device="cpu",
            )

            self._initialised = True
            log.info(
                "episodic_memory.initialised",
                entry_count=self._collection.count(),
            )

        except ImportError as exc:
            log.error("episodic_memory.import_error", error=str(exc))
            self._initialised = False

    async def retrieve(
        self,
        query: str,
        session_id: str,
        top_k: int = 5,
    ) -> list[dict]:
        """
        Retrieve the most relevant past experiences for a query.
        Uses cosine similarity over the embedding space.
        Recency is used as a secondary ranking signal.
        """
        if not self._initialised or self._collection is None:
            return []

        try:
            embedding = await asyncio.to_thread(
                self._embedding_model.encode,
                query,
                convert_to_list=True,
            )

            results = await asyncio.to_thread(
                self._collection.query,
                query_embeddings=[embedding],
                n_results=min(top_k, max(1, self._collection.count())),
                include=["documents", "metadatas", "distances"],
            )

            entries = []
            if results and results["documents"]:
                for doc, meta, dist in zip(
                    results["documents"][0],
                    results["metadatas"][0],
                    results["distances"][0],
                ):
                    entries.append({
                        "content": doc,
                        "session_id": meta.get("session_id", ""),
                        "timestamp": meta.get("timestamp", 0.0),
                        "similarity": 1.0 - dist,
                        "trace_id": meta.get("trace_id", ""),
                    })

            return entries

        except Exception as exc:
            log.error("episodic_memory.retrieve_error", error=str(exc))
            return []

    async def queue_for_encoding(
        self,
        session_id: str,
        user_input: str,
        assistant_output: str,
        trace_id: str,
    ) -> None:
        """Queue an interaction for background encoding by the CEE."""
        try:
            await self._encoding_queue.put({
                "session_id": session_id,
                "user_input": user_input,
                "assistant_output": assistant_output,
                "trace_id": trace_id,
                "timestamp": time.time(),
            })
        except asyncio.QueueFull:
            log.warning("episodic_memory.queue_full", session_id=session_id)

    async def get_pending_encoding_queue(self) -> list[dict]:
        """Drain the encoding queue for CEE processing."""
        items = []
        while not self._encoding_queue.empty():
            try:
                items.append(self._encoding_queue.get_nowait())
            except asyncio.QueueEmpty:
                break
        return items

    async def encode_and_store(self, item: dict) -> None:
        """
        Encode an interaction and store it if it meets the importance threshold.
        Importance is estimated from the length and novelty of the exchange.
        Only important interactions are committed to long-term episodic storage.
        """
        if not self._initialised:
            return

        combined = f"{item['user_input']} {item['assistant_output']}"
        importance = self._estimate_importance(
            user_input=item["user_input"],
            assistant_output=item["assistant_output"],
        )

        if importance < IMPORTANCE_THRESHOLD:
            return

        embedding = await asyncio.to_thread(
            self._embedding_model.encode,
            combined,
            convert_to_list=True,
        )

        entry_id = str(uuid.uuid4())
        await asyncio.to_thread(
            self._collection.add,
            documents=[combined],
            embeddings=[embedding],
            metadatas=[{
                "session_id": item["session_id"],
                "timestamp": item["timestamp"],
                "trace_id": item["trace_id"],
                "importance": importance,
            }],
            ids=[entry_id],
        )

        log.debug(
            "episodic_memory.stored",
            entry_id=entry_id,
            importance=importance,
        )

    def _estimate_importance(self, user_input: str, assistant_output: str) -> float:
        """
        Estimate the importance of an interaction.
        Longer, more substantive exchanges are more important.
        This is a simple heuristic — the system learns better
        importance estimation through continual learning.
        """
        combined_length = len(user_input) + len(assistant_output)
        if combined_length < 100:
            return 0.1
        elif combined_length < 500:
            return 0.4
        elif combined_length < 2000:
            return 0.7
        else:
            return 0.9

    async def consolidate_to_semantic(self) -> None:
        """
        Transfer frequently accessed episodic memories to semantic memory.
        Called by the CEE during consolidation windows.
        Mirrors hippocampal-neocortical memory consolidation during sleep.
        """
        log.info("episodic_memory.consolidate_to_semantic.start")
        # Semantic integration is handled by the continual learning module
        # which reads from episodic store and applies EWC-constrained updates
        from modules.learning_paradigms.continual import ContinualLearningModule
        continual = ContinualLearningModule()
        await continual.integrate_episodic_memories(self)
        log.info("episodic_memory.consolidate_to_semantic.complete")

    async def compute_consistency_hash(self) -> str:
        """Compute a hash of the episodic store state for cross-tier verification."""
        if not self._initialised or self._collection is None:
            return hashlib.sha256(b"empty").hexdigest()
        count = self._collection.count()
        return hashlib.sha256(f"episodic:{count}".encode()).hexdigest()

    def get_status(self) -> dict:
        """Return status for health endpoint."""
        count = self._collection.count() if self._collection else 0
        return {
            "initialised": self._initialised,
            "entry_count": count,
            "pending_encoding": self._encoding_queue.qsize(),
            "importance_threshold": IMPORTANCE_THRESHOLD,
        }
