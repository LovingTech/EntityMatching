import psycopg2 as pg
from typing import List, Tuple

class DB:
    def __init__(self, dsn: str, embedding_table_name = "embed_v2"):
        self.conn = pg.connect(dsn)
        self.embedding_table_name = embedding_table_name

    def search(self, embedding: List[float], top_k: int = 50) -> List[Tuple[str, float]]:
        with self.conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT n.name, emb.embedding <=> %s::vector AS distance
                FROM {self.embedding_table_name} emb
                JOIN names n ON n.id = emb.id
                ORDER BY emb.embedding <=> %s::vector DESC
                LIMIT %s;
                """,
                (embedding, embedding, top_k)
            )
            return cur.fetchall()
