from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any

from voicechatai.domain.ports.vector_store_port import VectorSearchMatch, VectorStorePort
from voicechatai.infrastructure.vector_store._store_utils import _build_payload

_SAFE_IDENTIFIER_RE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")


def _validate_identifier(name: str) -> str:
    if not _SAFE_IDENTIFIER_RE.match(name):
        raise ValueError(f"Unsafe SQL identifier: {name!r}")
    return name


def _try_register_vector(conn: Any) -> None:
    """Register the pgvector type on *conn*; silently skip if the package is not installed."""
    try:
        from pgvector.psycopg import register_vector

        register_vector(conn)
    except ImportError:
        pass


@dataclass(slots=True)
class PgVectorStore(VectorStorePort):
    """PostgreSQL + pgvector adapter implementing the vector store port.

    Requires the ``pgvector`` extension to be available in the target database.
    Set PGVECTOR_CONN to a psycopg-compatible connection string, e.g.::

        postgresql://user:pass@localhost:5432/mydb
    """

    table_name: str = os.getenv("PGVECTOR_TABLE", "vectors")
    conn_string: str = os.getenv("PGVECTOR_CONN", "")
    dimensions: int = int(os.getenv("PGVECTOR_DIMENSIONS", "384"))
    conn: Any | None = None

    def __post_init__(self) -> None:
        _validate_identifier(self.table_name)

        if self.conn is None:
            try:
                import psycopg
                from pgvector.psycopg import register_vector
            except ImportError as exc:
                raise RuntimeError(
                    "psycopg[binary] and pgvector packages are required for PgVectorStore."
                ) from exc

            if not self.conn_string:
                raise RuntimeError(
                    "PGVECTOR_CONN environment variable is required for PgVectorStore."
                )

            self.conn = psycopg.connect(self.conn_string)
            register_vector(self.conn)
        else:
            _try_register_vector(self.conn)

        self._ensure_schema()

    def _ensure_schema(self) -> None:
        with self.conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self.table_name} (
                    id        TEXT PRIMARY KEY,
                    document  TEXT NOT NULL DEFAULT '',
                    metadata  JSONB NOT NULL DEFAULT '{{}}',
                    embedding vector({self.dimensions})
                );
                """
            )
        self.conn.commit()

    def query(self, vector: list[float], top_k: int) -> list[VectorSearchMatch]:
        if not vector or top_k <= 0:
            return []

        with self.conn.cursor() as cur:
            cur.execute(
                f"""
                WITH ranked AS (
                    SELECT id, document, metadata,
                           embedding <=> %s::vector AS dist
                    FROM {self.table_name}
                    ORDER BY dist
                    LIMIT %s
                )
                SELECT id, document, metadata, 1.0 - dist AS score FROM ranked;
                """,
                (vector, top_k),
            )
            rows = cur.fetchall()

        matches: list[VectorSearchMatch] = []
        for item_id, document, metadata_raw, score in rows:
            if isinstance(metadata_raw, str):
                metadata_raw = json.loads(metadata_raw)
            payload = _build_payload(metadata_raw or {}, document or "")
            matches.append(
                VectorSearchMatch(item_id=str(item_id), score=float(score), payload=payload)
            )

        return matches

    def upsert(
        self,
        ids: list[str],
        documents: list[str],
        metadatas: list[dict[str, str]],
        embeddings: list[list[float]],
    ) -> int:
        if not ids:
            return 0

        if not (len(ids) == len(documents) == len(metadatas) == len(embeddings)):
            raise ValueError("ids, documents, metadatas, and embeddings must have equal length.")

        rows = [
            (item_id, doc, json.dumps(meta), emb)
            for item_id, doc, meta, emb in zip(ids, documents, metadatas, embeddings)
        ]
        with self.conn.cursor() as cur:
            cur.executemany(
                f"""
                INSERT INTO {self.table_name} (id, document, metadata, embedding)
                VALUES (%s, %s, %s::jsonb, %s)
                ON CONFLICT (id) DO UPDATE
                SET document  = EXCLUDED.document,
                    metadata  = EXCLUDED.metadata,
                    embedding = EXCLUDED.embedding;
                """,
                rows,
            )
        self.conn.commit()
        return len(ids)

    def close(self) -> None:
        """Close the underlying database connection."""
        if self.conn is not None:
            self.conn.close()
