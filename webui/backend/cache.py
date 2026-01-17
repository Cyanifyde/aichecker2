import hashlib
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class CachePolicy:
    mode: str = "unbounded"
    max_items: int = 10000
    ttl_days: int = 30


class CacheStore:
    def __init__(self, path: Path, policy: CachePolicy):
        self.path = path
        self.policy = policy
        self.conn = sqlite3.connect(self.path)
        self._init_db()

    def _init_db(self) -> None:
        cur = self.conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS cache (
                image_hash TEXT PRIMARY KEY,
                image_bytes BLOB,
                prob_ai REAL,
                decision TEXT,
                confidence REAL,
                ood_score REAL,
                model_version TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS feedback (
                image_hash TEXT PRIMARY KEY,
                label TEXT,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        self.conn.commit()

    @staticmethod
    def hash_bytes(data: bytes) -> str:
        return hashlib.sha256(data).hexdigest()

    def get(self, image_hash: str) -> Optional[dict]:
        cur = self.conn.cursor()
        cur.execute("SELECT * FROM cache WHERE image_hash = ?", (image_hash,))
        row = cur.fetchone()
        if not row:
            return None
        keys = [d[0] for d in cur.description]
        return dict(zip(keys, row))

    def set(self, record: dict) -> None:
        cur = self.conn.cursor()
        cur.execute(
            """
            INSERT OR REPLACE INTO cache
            (image_hash, image_bytes, prob_ai, decision, confidence, ood_score, model_version)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record["image_hash"],
                record["image_bytes"],
                record["prob_ai"],
                record["decision"],
                record["confidence"],
                record["ood_score"],
                record["model_version"],
            ),
        )
        self.conn.commit()

    def set_feedback(self, image_hash: str, label: str) -> None:
        cur = self.conn.cursor()
        cur.execute(
            """
            INSERT OR REPLACE INTO feedback (image_hash, label, updated_at)
            VALUES (?, ?, CURRENT_TIMESTAMP)
            """,
            (image_hash, label),
        )
        self.conn.commit()

    def list_history(self) -> list[dict]:
        cur = self.conn.cursor()
        cur.execute(
            "SELECT image_hash, prob_ai, decision, confidence, ood_score, created_at FROM cache ORDER BY created_at DESC"
        )
        rows = cur.fetchall()
        keys = [d[0] for d in cur.description]
        return [dict(zip(keys, row)) for row in rows]

    def list_feedback(self) -> list[dict]:
        cur = self.conn.cursor()
        cur.execute("SELECT image_hash, label, updated_at FROM feedback")
        rows = cur.fetchall()
        keys = [d[0] for d in cur.description]
        return [dict(zip(keys, row)) for row in rows]
