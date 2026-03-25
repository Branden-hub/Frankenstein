"""
Living AI System — Knowledge Base (Tier 5)
Connected databases, document stores, knowledge graphs —
infinite external memory with retrieval-augmented access.
Updatable without weight changes.
Implements hybrid retrieval: dense + sparse + graph.
"""

import asyncio
import hashlib
import json
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any

import structlog

log = structlog.get_logger(__name__)

DB_PATH = Path("data/knowledge_base.db")
GRAPH_PATH = Path("data/knowledge_graph.json")


class KnowledgeBase:
    """
    Tier 5 memory — the external knowledge store.
    SQLite with FTS5 full-text search for structured facts.
    NetworkX knowledge graph for relational reasoning.
    Hybrid RAG pipeline for retrieval.
    """

    def __init__(self):
        self._db_conn: sqlite3.Connection | None = None
        self._graph = None
        self._embedding_model = None
        self._chroma_client = None
        self._kb_collection = None
        self._initialised = False

    async def initialise(self) -> None:
        """Initialise all knowledge store components."""
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        GRAPH_PATH.parent.mkdir(parents=True, exist_ok=True)

        await self._init_sqlite()
        await self._init_graph()
        await self._init_vector_store()

        self._initialised = True
        log.info("knowledge_base.initialised")

    async def _init_sqlite(self) -> None:
        """Initialise SQLite with FTS5 full-text search."""
        self._db_conn = await asyncio.to_thread(
            sqlite3.connect, str(DB_PATH), check_same_thread=False
        )
        cursor = self._db_conn.cursor()

        # Facts table with FTS5
        cursor.executescript("""
            CREATE TABLE IF NOT EXISTS facts (
                id TEXT PRIMARY KEY,
                domain TEXT NOT NULL,
                subject TEXT NOT NULL,
                predicate TEXT NOT NULL,
                object TEXT NOT NULL,
                confidence REAL DEFAULT 1.0,
                source TEXT,
                created_at REAL,
                valid_from REAL,
                valid_to REAL
            );

            CREATE VIRTUAL TABLE IF NOT EXISTS facts_fts USING fts5(
                subject,
                predicate,
                object,
                domain,
                content='facts',
                content_rowid='rowid'
            );

            CREATE TABLE IF NOT EXISTS entities (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                entity_type TEXT NOT NULL,
                attributes TEXT,
                confidence REAL DEFAULT 1.0,
                created_at REAL
            );

            CREATE TABLE IF NOT EXISTS relations (
                id TEXT PRIMARY KEY,
                source_entity_id TEXT NOT NULL,
                relation_type TEXT NOT NULL,
                target_entity_id TEXT NOT NULL,
                confidence REAL DEFAULT 1.0,
                timestamp REAL,
                valid_from REAL,
                valid_to REAL
            );

            CREATE TABLE IF NOT EXISTS spaced_repetition (
                id TEXT PRIMARY KEY,
                fact_id TEXT NOT NULL,
                next_review REAL NOT NULL,
                interval_days REAL DEFAULT 1.0,
                ease_factor REAL DEFAULT 2.5,
                review_count INTEGER DEFAULT 0
            );
        """)
        self._db_conn.commit()

    async def _init_graph(self) -> None:
        """Initialise the knowledge graph."""
        try:
            import networkx as nx

            if GRAPH_PATH.exists():
                with open(GRAPH_PATH) as f:
                    data = json.load(f)
                self._graph = nx.node_link_graph(data)
            else:
                self._graph = nx.DiGraph()

            log.info(
                "knowledge_graph.initialised",
                nodes=self._graph.number_of_nodes(),
                edges=self._graph.number_of_edges(),
            )
        except ImportError:
            log.warning("knowledge_base.networkx_not_available")
            self._graph = None

    async def _init_vector_store(self) -> None:
        """Initialise ChromaDB collection for knowledge base vectors."""
        try:
            import chromadb
            from chromadb.config import Settings
            from sentence_transformers import SentenceTransformer

            kb_path = Path("data/knowledge_vectors")
            kb_path.mkdir(parents=True, exist_ok=True)

            self._chroma_client = chromadb.PersistentClient(
                path=str(kb_path),
                settings=Settings(anonymized_telemetry=False),
            )
            self._kb_collection = self._chroma_client.get_or_create_collection(
                name="knowledge_base",
                metadata={"hnsw:space": "cosine"},
            )
            self._embedding_model = SentenceTransformer(
                "all-MiniLM-L6-v2",
                device="cpu",
            )
        except ImportError as exc:
            log.warning("knowledge_base.vector_store_unavailable", error=str(exc))

    async def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        """
        Hybrid retrieval pipeline:
        1. Dense vector similarity search
        2. BM25 sparse keyword search (FTS5)
        3. Knowledge graph multi-hop traversal
        4. Deduplicate and rerank
        Returns top_k most relevant results.
        """
        results = []

        # Dense retrieval
        dense_results = await self._dense_retrieve(query, top_k=top_k * 2)
        results.extend(dense_results)

        # Sparse FTS5 retrieval
        sparse_results = await self._sparse_retrieve(query, top_k=top_k * 2)
        results.extend(sparse_results)

        # Graph traversal
        graph_results = await self._graph_retrieve(query, hops=2)
        results.extend(graph_results)

        # Deduplicate by content
        seen_content = set()
        unique_results = []
        for r in results:
            content_key = r.get("content", "")[:100]
            if content_key not in seen_content:
                seen_content.add(content_key)
                unique_results.append(r)

        # Simple relevance reranking — sort by score descending
        unique_results.sort(key=lambda x: x.get("score", 0.0), reverse=True)

        return unique_results[:top_k]

    async def _dense_retrieve(self, query: str, top_k: int) -> list[dict]:
        """Dense vector similarity retrieval from ChromaDB."""
        if not self._kb_collection or not self._embedding_model:
            return []
        if self._kb_collection.count() == 0:
            return []
        try:
            embedding = await asyncio.to_thread(
                self._embedding_model.encode, query, convert_to_list=True
            )
            results = await asyncio.to_thread(
                self._kb_collection.query,
                query_embeddings=[embedding],
                n_results=min(top_k, self._kb_collection.count()),
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
                        "source": "dense_retrieval",
                        "score": 1.0 - dist,
                        "metadata": meta,
                    })
            return entries
        except Exception as exc:
            log.error("knowledge_base.dense_retrieve_error", error=str(exc))
            return []

    async def _sparse_retrieve(self, query: str, top_k: int) -> list[dict]:
        """Sparse FTS5 keyword retrieval from SQLite."""
        if not self._db_conn:
            return []
        try:
            cursor = self._db_conn.cursor()
            results = await asyncio.to_thread(
                cursor.execute,
                """
                SELECT f.subject, f.predicate, f.object, f.domain, f.confidence
                FROM facts_fts fts
                JOIN facts f ON fts.rowid = f.rowid
                WHERE facts_fts MATCH ?
                ORDER BY rank
                LIMIT ?
                """,
                (query, top_k),
            )
            rows = cursor.fetchall()
            return [
                {
                    "content": f"{row[0]} {row[1]} {row[2]}",
                    "source": "fts5_retrieval",
                    "score": float(row[4]),
                    "metadata": {"domain": row[3]},
                }
                for row in rows
            ]
        except Exception as exc:
            log.error("knowledge_base.sparse_retrieve_error", error=str(exc))
            return []

    async def _graph_retrieve(self, query: str, hops: int = 2) -> list[dict]:
        """Multi-hop knowledge graph traversal."""
        if not self._graph:
            return []
        try:
            import networkx as nx

            # Find nodes whose labels match query terms
            query_terms = set(query.lower().split())
            matching_nodes = [
                n for n in self._graph.nodes
                if any(term in str(n).lower() for term in query_terms)
            ]

            results = []
            for node in matching_nodes[:5]:
                # Traverse up to `hops` edges from each matching node
                paths = nx.single_source_shortest_path(
                    self._graph, node, cutoff=hops
                )
                for target, path in paths.items():
                    if target != node and len(path) > 1:
                        results.append({
                            "content": f"{node} -> {' -> '.join(str(p) for p in path[1:])}",
                            "source": "graph_traversal",
                            "score": 1.0 / len(path),
                            "metadata": {"path_length": len(path)},
                        })
            return results[:10]
        except Exception as exc:
            log.error("knowledge_base.graph_retrieve_error", error=str(exc))
            return []

    async def add_fact(
        self,
        domain: str,
        subject: str,
        predicate: str,
        obj: str,
        confidence: float = 1.0,
        source: str = "system",
    ) -> str:
        """Add a new fact to the knowledge base."""
        if not self._db_conn:
            raise RuntimeError("Knowledge base not initialised")

        fact_id = str(uuid.uuid4())
        now = time.time()

        cursor = self._db_conn.cursor()
        await asyncio.to_thread(
            cursor.execute,
            """
            INSERT INTO facts (id, domain, subject, predicate, object,
                               confidence, source, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (fact_id, domain, subject, predicate, obj, confidence, source, now),
        )
        self._db_conn.commit()

        # Also add to FTS5 index
        await asyncio.to_thread(
            cursor.execute,
            """
            INSERT INTO facts_fts (rowid, subject, predicate, object, domain)
            SELECT rowid, subject, predicate, object, domain
            FROM facts WHERE id = ?
            """,
            (fact_id,),
        )
        self._db_conn.commit()

        # Add to knowledge graph
        if self._graph is not None:
            self._graph.add_edge(
                subject,
                obj,
                relation=predicate,
                confidence=confidence,
                domain=domain,
            )
            await self._save_graph()

        # Add to vector store
        if self._kb_collection and self._embedding_model:
            combined = f"{subject} {predicate} {obj}"
            embedding = await asyncio.to_thread(
                self._embedding_model.encode, combined, convert_to_list=True
            )
            await asyncio.to_thread(
                self._kb_collection.add,
                documents=[combined],
                embeddings=[embedding],
                metadatas=[{"domain": domain, "fact_id": fact_id}],
                ids=[fact_id],
            )

        return fact_id

    async def _save_graph(self) -> None:
        """Persist the knowledge graph to disk."""
        if self._graph is None:
            return
        try:
            import networkx as nx
            data = nx.node_link_data(self._graph)
            with open(GRAPH_PATH, "w") as f:
                json.dump(data, f)
        except Exception as exc:
            log.error("knowledge_base.graph_save_error", error=str(exc))

    async def run_spaced_repetition(self) -> None:
        """Rehearse facts due for review according to spaced repetition schedule."""
        if not self._db_conn:
            return
        now = time.time()
        cursor = self._db_conn.cursor()
        cursor.execute(
            "SELECT fact_id, interval_days, ease_factor FROM spaced_repetition WHERE next_review <= ?",
            (now,),
        )
        due = cursor.fetchall()
        for fact_id, interval, ease in due:
            new_interval = interval * ease
            new_next = now + new_interval * 86400
            cursor.execute(
                """UPDATE spaced_repetition
                   SET next_review = ?, interval_days = ?, review_count = review_count + 1
                   WHERE fact_id = ?""",
                (new_next, new_interval, fact_id),
            )
        self._db_conn.commit()
        if due:
            log.info("knowledge_base.spaced_repetition", reviewed=len(due))

    async def distil(self) -> None:
        """Compress redundant representations — called during consolidation windows."""
        log.info("knowledge_base.distil.start")
        # Future: identify semantically duplicate facts and merge them
        log.info("knowledge_base.distil.complete")

    def get_status(self) -> dict:
        """Return status for health endpoint."""
        fact_count = 0
        if self._db_conn:
            cursor = self._db_conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM facts")
            fact_count = cursor.fetchone()[0]

        graph_nodes = self._graph.number_of_nodes() if self._graph else 0
        graph_edges = self._graph.number_of_edges() if self._graph else 0
        kb_vectors = self._kb_collection.count() if self._kb_collection else 0

        return {
            "initialised": self._initialised,
            "fact_count": fact_count,
            "graph_nodes": graph_nodes,
            "graph_edges": graph_edges,
            "kb_vectors": kb_vectors,
        }
