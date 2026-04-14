from __future__ import annotations

import json
import unittest


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------

class StubCursor:
    """Minimal cursor stub supporting context manager and recording calls."""

    def __init__(self, rows: list[tuple] | None = None) -> None:
        self.executed: list[tuple] = []
        self._rows: list[tuple] = rows or []

    def execute(self, sql: str, params=None) -> None:
        self.executed.append((sql, params))

    def executemany(self, sql: str, params_seq) -> None:
        for params in params_seq:
            self.executed.append((sql, params))

    def fetchall(self) -> list[tuple]:
        return list(self._rows)

    def __enter__(self) -> "StubCursor":
        return self

    def __exit__(self, *args) -> None:
        pass


class StubConn:
    """Minimal connection stub that always returns the same StubCursor instance."""

    def __init__(self, rows: list[tuple] | None = None) -> None:
        self._rows = rows or []
        self.cursor_instance = StubCursor(rows=self._rows)
        self.committed = False

    def cursor(self) -> StubCursor:
        return self.cursor_instance

    def commit(self) -> None:
        self.committed = True


def _make_store(rows: list[tuple] | None = None, **kwargs):
    """Build a PgVectorStore with a stub connection, bypassing real DB."""
    from voicechatai.infrastructure.vector_store.pgvector_store import PgVectorStore

    stub = StubConn(rows=rows)
    return PgVectorStore(conn=stub, **kwargs), stub


# ---------------------------------------------------------------------------
# Tests: initialisation
# ---------------------------------------------------------------------------

class PgVectorStoreInitTests(unittest.TestCase):
    def test_init_creates_extension_and_table(self) -> None:
        store, stub = _make_store()

        executed_sql = [sql for sql, _ in stub.cursor_instance.executed]
        self.assertTrue(any("CREATE EXTENSION IF NOT EXISTS vector" in s for s in executed_sql))
        self.assertTrue(any("CREATE TABLE IF NOT EXISTS" in s for s in executed_sql))

    def test_init_commits_schema(self) -> None:
        _, stub = _make_store()
        self.assertTrue(stub.committed)

    def test_init_rejects_unsafe_table_name(self) -> None:
        from voicechatai.infrastructure.vector_store.pgvector_store import PgVectorStore

        with self.assertRaises(ValueError):
            PgVectorStore(table_name="bad; DROP TABLE users; --", conn=StubConn())

    def test_init_raises_without_conn_string_when_no_conn(self) -> None:
        from voicechatai.infrastructure.vector_store.pgvector_store import PgVectorStore

        # conn=None triggers real connection path; without psycopg installed in test env
        # it raises RuntimeError for missing package OR missing conn string.
        with self.assertRaises(RuntimeError):
            PgVectorStore(conn=None, conn_string="")

    def test_default_table_name_is_vectors(self) -> None:
        store, _ = _make_store()
        self.assertEqual(store.table_name, "vectors")

    def test_custom_table_name_respected(self) -> None:
        store, _ = _make_store(table_name="my_store")
        self.assertEqual(store.table_name, "my_store")

    def test_schema_ddl_includes_custom_table_name(self) -> None:
        _, stub = _make_store(table_name="faq_vecs")
        create_stmts = [sql for sql, _ in stub.cursor_instance.executed if "CREATE TABLE" in sql]
        self.assertTrue(any("faq_vecs" in s for s in create_stmts))


# ---------------------------------------------------------------------------
# Tests: query
# ---------------------------------------------------------------------------

class PgVectorStoreQueryTests(unittest.TestCase):
    def _store_with_rows(self, rows: list[tuple]):
        """Each call to cursor() must return a fresh cursor with correct rows."""
        store, stub = _make_store(rows=rows)
        # Replace cursor instance to serve query rows (after init consumed schema DDL).
        stub.cursor_instance = StubCursor(rows=rows)
        return store, stub

    def test_query_returns_empty_for_empty_vector(self) -> None:
        store, _ = _make_store()
        result = store.query([], top_k=5)
        self.assertEqual(result, [])

    def test_query_returns_empty_for_zero_top_k(self) -> None:
        store, _ = _make_store()
        result = store.query([0.1, 0.2], top_k=0)
        self.assertEqual(result, [])

    def test_query_returns_empty_for_negative_top_k(self) -> None:
        store, _ = _make_store()
        result = store.query([0.1, 0.2], top_k=-1)
        self.assertEqual(result, [])

    def test_query_maps_rows_to_matches(self) -> None:
        rows = [
            ("faq-1", "Answer one", json.dumps({"topic": "greet"}), 0.92),
            ("faq-2", "Answer two", json.dumps({"topic": "bye"}), 0.75),
        ]
        store, stub = self._store_with_rows(rows)

        results = store.query([0.1, 0.2, 0.3], top_k=2)

        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].item_id, "faq-1")
        self.assertAlmostEqual(results[0].score, 0.92)
        self.assertEqual(results[0].payload["topic"], "greet")

    def test_query_falls_back_to_document_as_answer_when_no_answer_key(self) -> None:
        rows = [("id1", "Doc text here", json.dumps({}), 0.80)]
        store, stub = self._store_with_rows(rows)

        results = store.query([0.1], top_k=1)

        self.assertEqual(results[0].payload.get("answer"), "Doc text here")

    def test_query_preserves_explicit_answer_in_metadata(self) -> None:
        rows = [("id1", "Doc text", json.dumps({"answer": "explicit answer"}), 0.70)]
        store, stub = self._store_with_rows(rows)

        results = store.query([0.1], top_k=1)

        self.assertEqual(results[0].payload["answer"], "explicit answer")

    def test_query_handles_metadata_already_as_dict(self) -> None:
        rows = [("id1", "Doc", {"key": "val"}, 0.60)]
        store, stub = self._store_with_rows(rows)

        results = store.query([0.1], top_k=1)

        self.assertEqual(results[0].payload["key"], "val")

    def test_query_passes_vector_and_top_k_to_cursor(self) -> None:
        store, stub = _make_store()
        stub.cursor_instance = StubCursor(rows=[])

        store.query([0.5, 0.6], top_k=3)

        executed_sqls = [sql for sql, _ in stub.cursor_instance.executed]
        self.assertTrue(any("SELECT" in s for s in executed_sqls))
        params = next(p for _, p in stub.cursor_instance.executed if p and 3 in p)
        self.assertEqual(params[1], 3)  # CTE passes (vector, top_k) — top_k is at index 1


# ---------------------------------------------------------------------------
# Tests: upsert
# ---------------------------------------------------------------------------

class PgVectorStoreUpsertTests(unittest.TestCase):
    def test_upsert_returns_zero_for_empty_ids(self) -> None:
        store, _ = _make_store()
        result = store.upsert([], [], [], [])
        self.assertEqual(result, 0)

    def test_upsert_returns_count_of_ids(self) -> None:
        store, _ = _make_store()
        result = store.upsert(
            ids=["a", "b"],
            documents=["doc a", "doc b"],
            metadatas=[{"k": "v"}, {}],
            embeddings=[[0.1, 0.2], [0.3, 0.4]],
        )
        self.assertEqual(result, 2)

    def test_upsert_commits_after_inserts(self) -> None:
        store, stub = _make_store()
        stub.committed = False  # reset after init commit
        store.upsert(["x"], ["doc"], [{}], [[0.1]])
        self.assertTrue(stub.committed)

    def test_upsert_executes_insert_on_conflict_for_each_id(self) -> None:
        store, stub = _make_store()
        stub.cursor_instance = StubCursor()

        store.upsert(
            ids=["id1", "id2"],
            documents=["d1", "d2"],
            metadatas=[{}, {}],
            embeddings=[[0.1], [0.2]],
        )

        insert_calls = [
            sql for sql, _ in stub.cursor_instance.executed
            if "INSERT INTO" in sql and "ON CONFLICT" in sql
        ]
        self.assertEqual(len(insert_calls), 2)

    def test_upsert_serialises_metadata_as_json(self) -> None:
        store, stub = _make_store()
        stub.cursor_instance = StubCursor()

        store.upsert(
            ids=["z"],
            documents=["doc"],
            metadatas=[{"answer": "hello"}],
            embeddings=[[0.9]],
        )

        params_list = [p for _, p in stub.cursor_instance.executed if p]
        # Last param tuple is ('z', 'doc', <json>, [0.9])
        _, _, meta_json, _ = params_list[-1]
        self.assertEqual(json.loads(meta_json), {"answer": "hello"})

    def test_upsert_rejects_ragged_input_lists(self) -> None:
        store, _ = _make_store()
        with self.assertRaises(ValueError):
            store.upsert(ids=["a", "b"], documents=["only one"], metadatas=[{}, {}], embeddings=[[], []])


if __name__ == "__main__":
    unittest.main()
