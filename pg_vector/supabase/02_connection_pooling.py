"""
Demonstrates proper connection management for production (Supabase + Ollama)
"""

import os
from contextlib import contextmanager

import ollama
import psycopg2
from psycopg2 import pool
from dotenv import load_dotenv

load_dotenv()

EMBED_MODEL = "nomic-embed-text"
EMBED_DIM = 768                 # nomic-embed-text produces 768 numbers
TABLE = "documents_ollama"      # separate table, so it can't clash with older 1536-dim tables


def normalize_dsn(url):
    """psycopg2 only understands postgresql://, not SQLAlchemy's postgresql+psycopg://"""
    if not url:
        return url
    return url.replace("postgresql+psycopg2://", "postgresql://").replace(
        "postgresql+psycopg://", "postgresql://"
    )


DATABASE_URL = normalize_dsn(
    os.getenv("SUPABASE_DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/postgres")
)
SUPABASE_POOLER_URL = normalize_dsn(os.getenv("SUPABASE_DATABASE_URL"))


class DatabasePool:
    """Connection pool manager for pgvector"""

    _instance = None
    _pool = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def initialize(self, min_conn: int = 1, max_conn: int = 10):
        """Initialize the connection pool"""
        if self._pool is None:
            self._pool = psycopg2.pool.ThreadedConnectionPool(
                minconn=min_conn,
                maxconn=max_conn,
                dsn=SUPABASE_POOLER_URL or DATABASE_URL,
            )
            print(f"✅ Pool initialized: {min_conn}-{max_conn} connections")

    @contextmanager
    def get_connection(self):
        """Borrow a connection; commit on success, roll back on error, always return it"""
        if self._pool is None:
            self.initialize()

        conn = self._pool.getconn()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            self._pool.putconn(conn)

    def close_all(self):
        """Close all connections in the pool"""
        if self._pool:
            self._pool.closeall()
            self._pool = None
            print("✅ All connections closed")


# Global pool instance
db_pool = DatabasePool()


# ---------- helpers ----------

def embed(text: str) -> list:
    """Turn text into a 768-number vector using local Ollama"""
    response = ollama.embed(model=EMBED_MODEL, input=text)
    return response["embeddings"][0]


def to_pgvector(vector: list) -> str:
    """Format a Python list as pgvector text, e.g. '[0.1,0.2,0.3]'"""
    return "[" + ",".join(str(x) for x in vector) + "]"


SAMPLE_DOCS = [
    "PostgreSQL is an open-source relational database with strong SQL support.",
    "pgvector adds vector similarity search to PostgreSQL.",
    "Connection pooling reuses database connections instead of opening new ones.",
    "Supabase is a hosted Postgres platform with built-in connection poolers.",
    "Ollama runs language and embedding models locally on your machine.",
]


def setup_table():
    """Create the table if missing and seed it with sample documents"""
    with db_pool.get_connection() as conn:
        cur = conn.cursor()
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {TABLE} (
                id bigserial PRIMARY KEY,
                content text NOT NULL,
                embedding vector({EMBED_DIM})
            );
            """
        )
        cur.execute(f"SELECT COUNT(*) FROM {TABLE};")
        if cur.fetchone()[0] == 0:
            for text in SAMPLE_DOCS:
                cur.execute(
                    f"INSERT INTO {TABLE} (content, embedding) VALUES (%s, %s::vector)",
                    (text, to_pgvector(embed(text))),
                )
            print(f"✅ Seeded {len(SAMPLE_DOCS)} sample documents")
        cur.close()


def search_with_pooling(query_embedding: list, limit: int = 5):
    """Execute a similarity search using a pooled connection"""
    vec = to_pgvector(query_embedding)
    with db_pool.get_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            f"""
            SELECT content, 1 - (embedding <=> %s::vector) AS similarity
            FROM {TABLE}
            ORDER BY embedding <=> %s::vector
            LIMIT %s
            """,
            (vec, vec, limit),
        )
        results = cur.fetchall()
        cur.close()
        return results


def demonstrate_pooling():
    """Show connection pooling in action"""
    print("\n🔄 Demonstrating connection pooling...")

    db_pool.initialize(min_conn=2, max_conn=5)
    setup_table()

    # Run multiple queries: the same backend PIDs repeat, proving reuse
    for i in range(5):
        with db_pool.get_connection() as conn:
            cur = conn.cursor()
            cur.execute(f"SELECT COUNT(*) FROM {TABLE};")
            count = cur.fetchone()[0]
            cur.close()
            print(
                f"   Query {i+1}: {count} documents "
                f"(connection PID {conn.get_backend_pid()})"
            )

    # Real search with a pooled connection
    query = "How do I reuse database connections?"
    print(f"\n🔎 Search: {query}")
    for content, score in search_with_pooling(embed(query), limit=3):
        print(f"   {score:.3f}  {content}")

    print("\n✅ All queries used pooled connections!")


def main():
    print("=" * 60)
    print("Connection Pooling for Production")
    print("=" * 60)

    print("\n📝 Why Connection Pooling?")
    print("   - PostgreSQL connections are expensive to create")
    print("   - Each connection uses memory on the server")
    print("   - Connection limits exist on every database plan")
    print("   - Pooling reuses connections efficiently")

    print("\n📝 Supabase connection options:")
    print("   - Direct:            db.<ref>.supabase.co:5432 (IPv6 only on free plan)")
    print("   - Session pooler:    <pooler-host>:5432 (IPv4, works everywhere)")
    print("   - Transaction pooler:<pooler-host>:6543 (IPv4, many short-lived clients)")

    if SUPABASE_POOLER_URL:
        print("\n✅ Using Supabase pooler")
    else:
        print("\n⚠️  Using local database (set SUPABASE_POOLER_URL for Supabase)")

    try:
        demonstrate_pooling()
    except Exception as e:
        print(f"\n⚠️  Could not demonstrate: {e}")
    finally:
        db_pool.close_all()


if __name__ == "__main__":
    main()