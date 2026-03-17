import hashlib
import json
import logging
import os
import sqlite3
import time
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


class DiskCache:
    """
    Tiny disk cache (SQLite) with TTL.
    Stores JSON payloads as TEXT.
    """

    def __init__(self, path: Optional[str] = None, default_ttl_s: int = 24 * 3600):
        self.path = path or os.getenv("DISK_CACHE_PATH") or os.path.join(os.getcwd(), "cache.sqlite")
        self.default_ttl_s = int(default_ttl_s)
        self._init_db()

    def _init_db(self) -> None:
        os.makedirs(os.path.dirname(self.path), exist_ok=True) if os.path.dirname(self.path) else None
        with sqlite3.connect(self.path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS cache (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    expires_at INTEGER NOT NULL,
                    created_at INTEGER NOT NULL
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_cache_expires_at ON cache(expires_at)")
            conn.commit()

    def _now(self) -> int:
        return int(time.time())

    def get(self, key: str) -> Optional[Any]:
        now = self._now()
        with sqlite3.connect(self.path) as conn:
            row = conn.execute("SELECT value, expires_at FROM cache WHERE key = ?", (key,)).fetchone()
            if not row:
                return None
            value_s, expires_at = row
            if int(expires_at) <= now:
                try:
                    conn.execute("DELETE FROM cache WHERE key = ?", (key,))
                    conn.commit()
                except Exception:
                    pass
                return None
            try:
                return json.loads(value_s)
            except Exception:
                return None

    def set(self, key: str, value: Any, ttl_s: Optional[int] = None) -> None:
        ttl = int(ttl_s if ttl_s is not None else self.default_ttl_s)
        now = self._now()
        expires_at = now + ttl
        try:
            value_s = json.dumps(value, ensure_ascii=False)
        except Exception:
            # best-effort fallback
            value_s = json.dumps({"_cache_error": "non_json_value"}, ensure_ascii=False)
        with sqlite3.connect(self.path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO cache(key, value, expires_at, created_at) VALUES (?, ?, ?, ?)",
                (key, value_s, expires_at, now),
            )
            conn.commit()

    def purge_expired(self) -> int:
        now = self._now()
        with sqlite3.connect(self.path) as conn:
            cur = conn.execute("DELETE FROM cache WHERE expires_at <= ?", (now,))
            conn.commit()
            return int(cur.rowcount or 0)

    @staticmethod
    def make_key(namespace: str, payload: Any) -> str:
        s = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
        h = hashlib.sha256(s.encode("utf-8")).hexdigest()[:24]
        return f"{namespace}:{h}"


disk_cache = DiskCache()


def get_or_set_json(namespace: str, payload: Any, builder: Callable[[], Any], ttl_s: Optional[int] = None) -> Any:
    key = disk_cache.make_key(namespace, payload)
    cached = disk_cache.get(key)
    if cached is not None:
        return cached
    val = builder()
    disk_cache.set(key, val, ttl_s=ttl_s)
    return val

