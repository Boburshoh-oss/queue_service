"""
SQLite-based event queue for offline-resilient data upload.
Manages event lifecycle: PENDING → SENDING → SENT → CLEANED
"""

import os
import sqlite3
import threading
import uuid
from datetime import datetime, timedelta, timezone


tz = timezone(timedelta(hours=5))  # Asia/Tashkent UTC+5


class QueueDB:
    """Thread-safe SQLite queue manager for visitor data upload events."""

    def __init__(self, db_path: str = None):
        if db_path is None:
            db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "queue.db")
        self.db_path = db_path
        self._lock = threading.Lock()
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
        return conn

    def _init_db(self):
        with self._lock:
            conn = self._get_conn()
            try:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS event_queue (
                        event_id TEXT PRIMARY KEY,
                        date TEXT NOT NULL,
                        deepstream_json_path TEXT,
                        face_analyzer_json_path TEXT,
                        status TEXT NOT NULL DEFAULT 'PENDING',
                        retry_count INTEGER NOT NULL DEFAULT 0,
                        next_retry_time TEXT,
                        created_at TEXT NOT NULL,
                        sent_at TEXT
                    )
                """)
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_status
                    ON event_queue(status)
                """)
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_date
                    ON event_queue(date)
                """)
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS employee_approvals (
                        approval_id TEXT PRIMARY KEY,
                        visitor_face_id TEXT NOT NULL,
                        photo_path TEXT,
                        visit_count INTEGER NOT NULL DEFAULT 0,
                        visit_history TEXT,
                        age TEXT,
                        gender TEXT,
                        embedding_json TEXT,
                        status TEXT NOT NULL DEFAULT 'PENDING',
                        retry_count INTEGER NOT NULL DEFAULT 0,
                        next_retry_time TEXT,
                        created_at TEXT NOT NULL,
                        sent_at TEXT,
                        resolved_at TEXT
                    )
                """)
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_approval_status
                    ON employee_approvals(status)
                """)
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_approval_visitor
                    ON employee_approvals(visitor_face_id)
                """)
                conn.commit()
            finally:
                conn.close()

    def enqueue(self, date: str, deepstream_json_path: str = None,
                face_analyzer_json_path: str = None) -> str:
        """Add a new event to the queue. Returns event_id."""
        event_id = str(uuid.uuid4())
        now = datetime.now(tz).isoformat()

        with self._lock:
            conn = self._get_conn()
            try:
                conn.execute(
                    """INSERT INTO event_queue
                       (event_id, date, deepstream_json_path, face_analyzer_json_path,
                        status, retry_count, next_retry_time, created_at)
                       VALUES (?, ?, ?, ?, 'PENDING', 0, ?, ?)""",
                    (event_id, date, deepstream_json_path, face_analyzer_json_path,
                     now, now)
                )
                conn.commit()
            finally:
                conn.close()
        return event_id

    def get_pending_events(self) -> list[dict]:
        """Get events ready to be processed (PENDING or retryable SENDING)."""
        now = datetime.now(tz).isoformat()

        with self._lock:
            conn = self._get_conn()
            try:
                rows = conn.execute(
                    """SELECT * FROM event_queue
                       WHERE status = 'PENDING'
                          OR (status = 'SENDING' AND next_retry_time <= ?)
                       ORDER BY created_at ASC""",
                    (now,)
                ).fetchall()
                return [dict(row) for row in rows]
            finally:
                conn.close()

    def mark_sending(self, event_id: str):
        """Mark event as currently being sent."""
        with self._lock:
            conn = self._get_conn()
            try:
                conn.execute(
                    "UPDATE event_queue SET status = 'SENDING' WHERE event_id = ?",
                    (event_id,)
                )
                conn.commit()
            finally:
                conn.close()

    def mark_sent(self, event_id: str):
        """Mark event as successfully sent."""
        now = datetime.now(tz).isoformat()

        with self._lock:
            conn = self._get_conn()
            try:
                conn.execute(
                    """UPDATE event_queue
                       SET status = 'SENT', sent_at = ?
                       WHERE event_id = ?""",
                    (now, event_id)
                )
                conn.commit()
            finally:
                conn.close()

    def mark_retry(self, event_id: str, retry_count: int, next_retry_time: str):
        """Mark event for retry with new retry time."""
        with self._lock:
            conn = self._get_conn()
            try:
                conn.execute(
                    """UPDATE event_queue
                       SET status = 'SENDING', retry_count = ?, next_retry_time = ?
                       WHERE event_id = ?""",
                    (retry_count, next_retry_time, event_id)
                )
                conn.commit()
            finally:
                conn.close()

    def mark_cleaned(self, event_id: str):
        """Mark event as cleaned (data files removed)."""
        with self._lock:
            conn = self._get_conn()
            try:
                conn.execute(
                    "UPDATE event_queue SET status = 'CLEANED' WHERE event_id = ?",
                    (event_id,)
                )
                conn.commit()
            finally:
                conn.close()

    def get_cleanable_events(self, retention_days: int = 30) -> list[dict]:
        """Get SENT events older than retention period."""
        cutoff = (datetime.now(tz) - timedelta(days=retention_days)).isoformat()

        with self._lock:
            conn = self._get_conn()
            try:
                rows = conn.execute(
                    """SELECT * FROM event_queue
                       WHERE status = 'SENT' AND sent_at <= ?
                       ORDER BY sent_at ASC""",
                    (cutoff,)
                ).fetchall()
                return [dict(row) for row in rows]
            finally:
                conn.close()

    def delete_by_date(self, date: str) -> int:
        """Delete all non-CLEANED events for a given date. Returns count deleted."""
        with self._lock:
            conn = self._get_conn()
            try:
                cursor = conn.execute(
                    "DELETE FROM event_queue WHERE date = ? AND status != 'CLEANED'",
                    (date,)
                )
                conn.commit()
                return cursor.rowcount
            finally:
                conn.close()

    def event_exists_for_date(self, date: str) -> bool:
        """Check if an event already exists for a given date (not CLEANED)."""
        with self._lock:
            conn = self._get_conn()
            try:
                row = conn.execute(
                    """SELECT 1 FROM event_queue
                       WHERE date = ? AND status != 'CLEANED'
                       LIMIT 1""",
                    (date,)
                ).fetchone()
                return row is not None
            finally:
                conn.close()

    def get_all_events(self) -> list[dict]:
        """Get all events (for diagnostics)."""
        with self._lock:
            conn = self._get_conn()
            try:
                rows = conn.execute(
                    "SELECT * FROM event_queue ORDER BY created_at DESC"
                ).fetchall()
                return [dict(row) for row in rows]
            finally:
                conn.close()

    def get_stats(self) -> dict:
        """Get queue statistics by status."""
        with self._lock:
            conn = self._get_conn()
            try:
                rows = conn.execute(
                    """SELECT status, COUNT(*) as count
                       FROM event_queue GROUP BY status"""
                ).fetchall()
                stats = {row['status']: row['count'] for row in rows}
                # Add approval stats
                rows2 = conn.execute(
                    """SELECT status, COUNT(*) as count
                       FROM employee_approvals GROUP BY status"""
                ).fetchall()
                for row in rows2:
                    stats[f"approval_{row['status']}"] = row['count']
                return stats
            finally:
                conn.close()

    # ── employee approval methods ─────────────────────────────────

    def enqueue_approval(self, visitor_face_id: str, photo_path: str = None,
                         visit_count: int = 0, visit_history: str = None,
                         age: str = None, gender: str = None,
                         embedding_json: str = None) -> str:
        """Add a new employee approval request. Returns approval_id."""
        approval_id = str(uuid.uuid4())
        now = datetime.now(tz).isoformat()

        with self._lock:
            conn = self._get_conn()
            try:
                conn.execute(
                    """INSERT INTO employee_approvals
                       (approval_id, visitor_face_id, photo_path, visit_count,
                        visit_history, age, gender, embedding_json,
                        status, retry_count, next_retry_time, created_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'PENDING', 0, ?, ?)""",
                    (approval_id, visitor_face_id, photo_path, visit_count,
                     visit_history, age, gender, embedding_json,
                     now, now)
                )
                conn.commit()
            finally:
                conn.close()
        return approval_id

    def get_pending_approvals(self) -> list[dict]:
        """Get approval requests ready to be sent."""
        now = datetime.now(tz).isoformat()

        with self._lock:
            conn = self._get_conn()
            try:
                rows = conn.execute(
                    """SELECT * FROM employee_approvals
                       WHERE status = 'PENDING'
                          OR (status = 'SENDING' AND next_retry_time <= ?)
                       ORDER BY created_at ASC""",
                    (now,)
                ).fetchall()
                return [dict(row) for row in rows]
            finally:
                conn.close()

    def get_sent_approvals(self) -> list[dict]:
        """Get approvals that have been sent but not yet resolved."""
        with self._lock:
            conn = self._get_conn()
            try:
                rows = conn.execute(
                    """SELECT * FROM employee_approvals
                       WHERE status = 'SENT'
                       ORDER BY sent_at ASC"""
                ).fetchall()
                return [dict(row) for row in rows]
            finally:
                conn.close()

    def mark_approval_sending(self, approval_id: str):
        with self._lock:
            conn = self._get_conn()
            try:
                conn.execute(
                    "UPDATE employee_approvals SET status = 'SENDING' WHERE approval_id = ?",
                    (approval_id,)
                )
                conn.commit()
            finally:
                conn.close()

    def mark_approval_sent(self, approval_id: str):
        now = datetime.now(tz).isoformat()
        with self._lock:
            conn = self._get_conn()
            try:
                conn.execute(
                    """UPDATE employee_approvals
                       SET status = 'SENT', sent_at = ?
                       WHERE approval_id = ?""",
                    (now, approval_id)
                )
                conn.commit()
            finally:
                conn.close()

    def mark_approval_retry(self, approval_id: str, retry_count: int, next_retry_time: str):
        with self._lock:
            conn = self._get_conn()
            try:
                conn.execute(
                    """UPDATE employee_approvals
                       SET status = 'SENDING', retry_count = ?, next_retry_time = ?
                       WHERE approval_id = ?""",
                    (retry_count, next_retry_time, approval_id)
                )
                conn.commit()
            finally:
                conn.close()

    def mark_approval_resolved(self, approval_id: str, new_status: str):
        """Mark approval as APPROVED or REJECTED."""
        now = datetime.now(tz).isoformat()
        with self._lock:
            conn = self._get_conn()
            try:
                conn.execute(
                    """UPDATE employee_approvals
                       SET status = ?, resolved_at = ?
                       WHERE approval_id = ?""",
                    (new_status, now, approval_id)
                )
                conn.commit()
            finally:
                conn.close()

    def approval_exists_for_visitor(self, visitor_face_id: str) -> bool:
        """Check if an active approval already exists for this visitor."""
        with self._lock:
            conn = self._get_conn()
            try:
                row = conn.execute(
                    """SELECT 1 FROM employee_approvals
                       WHERE visitor_face_id = ?
                         AND status NOT IN ('REJECTED')
                       LIMIT 1""",
                    (visitor_face_id,)
                ).fetchone()
                return row is not None
            finally:
                conn.close()
