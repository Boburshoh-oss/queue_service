#!/usr/bin/env python3
"""
Queue Worker Service — polls SQLite queue and uploads visitor data to API.
Handles internet outages with exponential backoff retry.
Also processes employee approval requests (send, poll, promote).
Runs as a systemd daemon.
"""

import base64
import json
import logging
import os
import signal
import sys
import time
from datetime import datetime, timedelta, timezone
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path

import importlib

import requests

from queue_db import QueueDB

# Access face_analyze for promotion
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "face_analyze"))
_db_module = os.environ.get("FACEDB_BACKEND", "database_pgvector")
FaceDB = importlib.import_module(_db_module).FaceDB


tz = timezone(timedelta(hours=5))  # Asia/Tashkent


class QueueWorker:
    """Daemon that processes queued events and uploads data to the API."""

    def __init__(self, config_path: str = None):
        if config_path is None:
            config_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "config.json"
            )
        self.config = self._load_config(config_path)
        self.db = QueueDB()
        self.logger = self._setup_logging()
        self._running = True
        self._last_cleanup = datetime.min.replace(tzinfo=tz)

        # Config shortcuts
        api = self.config["api"]
        self.upload_url = api["base_url"].rstrip("/") + api["endpoint"]
        self.device_index = api["device_index"]
        self.api_timeout = api.get("timeout_seconds", 30)

        retry = self.config["retry"]
        self.initial_delay = retry.get("initial_delay_seconds", 60)
        self.max_delay = retry.get("max_delay_seconds", 1800)
        self.backoff_multiplier = retry.get("backoff_multiplier", 2)

        worker = self.config["worker"]
        self.poll_interval = worker.get("poll_interval_seconds", 30)
        self.conn_timeout = worker.get("connectivity_timeout_seconds", 5)

        cleanup = self.config["cleanup"]
        self.retention_days = cleanup.get("retention_days", 30)
        self.cleanup_interval_hours = cleanup.get("cleanup_check_interval_hours", 6)

        # Approval config
        approval = self.config.get("approval", {})
        self.approval_url = api["base_url"].rstrip("/") + approval.get(
            "endpoint", "/api/v1/visitors/employees/request-approval/"
        )
        self.approval_check_url = api["base_url"].rstrip("/") + approval.get(
            "check_endpoint", "/api/v1/visitors/employees/check-approval/"
        )
        self.approval_check_interval = approval.get("check_interval_seconds", 300)
        self.auto_approve_days = approval.get("auto_approve_days", 3)
        self._last_approval_check = datetime.min.replace(tzinfo=tz)

        # FaceDB for promotion (lazy-loaded)
        self._face_db = None
        face_analyze_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "face_analyze"
        )
        self._face_analyze_config_path = os.path.join(face_analyze_dir, "config.json")
        self._face_db_path = os.path.join(face_analyze_dir, "faces_vdb")

    # ── config & logging ──────────────────────────────────────────

    @staticmethod
    def _load_config(path: str) -> dict:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _setup_logging(self) -> logging.Logger:
        logger = logging.getLogger("queue_worker")
        logger.setLevel(logging.INFO)

        # Avoid duplicate handlers on reload
        if logger.handlers:
            return logger

        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # Console (systemd journal picks this up)
        console = logging.StreamHandler(sys.stdout)
        console.setFormatter(formatter)
        logger.addHandler(console)

        # File — rotates daily, keeps 14 days
        log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, "worker.log")

        file_handler = TimedRotatingFileHandler(
            log_file, when="midnight", backupCount=14, encoding="utf-8"
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        return logger

    # ── connectivity ──────────────────────────────────────────────

    def check_connectivity(self) -> bool:
        """Quick HEAD request to API server to verify internet."""
        try:
            resp = requests.head(
                self.config["api"]["base_url"],
                timeout=self.conn_timeout,
            )
            return resp.status_code < 500
        except requests.RequestException:
            return False

    # ── retry calculation ─────────────────────────────────────────

    def _next_retry_time(self, retry_count: int) -> str:
        delay = min(
            self.initial_delay * (self.backoff_multiplier ** retry_count),
            self.max_delay,
        )
        return (datetime.now(tz) + timedelta(seconds=delay)).isoformat()

    # ── event processing ──────────────────────────────────────────

    def _process_event(self, event: dict) -> bool:
        """Upload one event to the API. Returns True on success."""
        event_id = event["event_id"]
        face_json_path = event.get("face_analyzer_json_path")

        if not face_json_path or not os.path.isfile(face_json_path):
            self.logger.error(
                "Event %s: face_analyzer JSON not found at %s — skipping",
                event_id, face_json_path,
            )
            # Still mark retry so it can be picked up later if file appears
            retry_count = event["retry_count"] + 1
            self.db.mark_retry(event_id, retry_count, self._next_retry_time(retry_count))
            return False

        try:
            with open(face_json_path, "r", encoding="utf-8") as f:
                daily_info = json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            self.logger.error("Event %s: failed to read JSON — %s", event_id, exc)
            retry_count = event["retry_count"] + 1
            self.db.mark_retry(event_id, retry_count, self._next_retry_time(retry_count))
            return False

        payload = {
            "device_index": self.device_index,
            "data": daily_info,
        }

        try:
            resp = requests.post(
                self.upload_url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=self.api_timeout,
            )

            if resp.status_code == 201:
                self.db.mark_sent(event_id)
                self.logger.info(
                    "✅ Event %s (date=%s) uploaded successfully",
                    event_id, event["date"],
                )
                return True

            self.logger.warning(
                "Event %s: API returned %s — %s",
                event_id, resp.status_code, resp.text[:200],
            )
        except requests.RequestException as exc:
            self.logger.warning("Event %s: network error — %s", event_id, exc)

        # Schedule retry
        retry_count = event["retry_count"] + 1
        next_time = self._next_retry_time(retry_count)
        self.db.mark_retry(event_id, retry_count, next_time)
        self.logger.info(
            "Event %s: retry #%d scheduled at %s",
            event_id, retry_count, next_time,
        )
        return False

    # ── queue processing (single pass) ────────────────────────────

    def process_queue(self):
        """Process all pending events in one pass."""
        events = self.db.get_pending_events()
        if not events:
            return

        self.logger.info("Found %d pending event(s)", len(events))

        if not self.check_connectivity():
            self.logger.warning("No connectivity — deferring %d event(s)", len(events))
            return

        for event in events:
            self.db.mark_sending(event["event_id"])
            self._process_event(event)

    # ── approval processing ───────────────────────────────────────

    def _get_face_db(self) -> FaceDB:
        """Lazy-load FaceDB for employee promotion."""
        if self._face_db is None:
            fa_config = None
            if os.path.isfile(self._face_analyze_config_path):
                with open(self._face_analyze_config_path, "r", encoding="utf-8") as f:
                    fa_config = json.load(f)
            self._face_db = FaceDB(db_path=self._face_db_path, config=fa_config)
        return self._face_db

    def _process_approval(self, approval: dict) -> bool:
        """Send one approval request to the dashboard API. Returns True on success."""
        approval_id = approval["approval_id"]
        photo_path = approval.get("photo_path")

        # Encode photo as base64
        photo_b64 = None
        if photo_path and os.path.isfile(photo_path):
            with open(photo_path, "rb") as f:
                photo_b64 = base64.b64encode(f.read()).decode("utf-8")

        payload = {
            "device_index": self.device_index,
            "visitor_face_id": approval["visitor_face_id"],
            "photo": photo_b64,
            "visit_count": approval.get("visit_count", 0),
            "visit_history": [
                f"{v['date']} {v.get('time', '')}" if isinstance(v, dict) else str(v)
                for v in json.loads(approval.get("visit_history", "[]") or "[]")
            ],
            "age": approval.get("age", ""),
            "gender": approval.get("gender", ""),
        }

        try:
            resp = requests.post(
                self.approval_url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=self.api_timeout,
            )

            if resp.status_code in (200, 201):
                self.db.mark_approval_sent(approval_id)
                self.logger.info(
                    "✅ Approval %s (visitor=%s) sent to dashboard",
                    approval_id, approval["visitor_face_id"],
                )
                return True

            self.logger.warning(
                "Approval %s: API returned %s — %s",
                approval_id, resp.status_code, resp.text[:200],
            )
        except requests.RequestException as exc:
            self.logger.warning("Approval %s: network error — %s", approval_id, exc)

        # Schedule retry
        retry_count = approval["retry_count"] + 1
        next_time = self._next_retry_time(retry_count)
        self.db.mark_approval_retry(approval_id, retry_count, next_time)
        self.logger.info(
            "Approval %s: retry #%d scheduled at %s",
            approval_id, retry_count, next_time,
        )
        return False

    def process_approvals(self):
        """Send all pending approval requests in one pass."""
        approvals = self.db.get_pending_approvals()
        if not approvals:
            return

        self.logger.info("Found %d pending approval(s)", len(approvals))

        if not self.check_connectivity():
            self.logger.warning("No connectivity — deferring %d approval(s)", len(approvals))
            return

        for approval in approvals:
            self.db.mark_approval_sending(approval["approval_id"])
            self._process_approval(approval)

    def check_approval_status(self):
        """Poll dashboard for resolved approvals and promote/reject accordingly."""
        now = datetime.now(tz)
        if (now - self._last_approval_check) < timedelta(seconds=self.approval_check_interval):
            return

        self._last_approval_check = now

        sent_approvals = self.db.get_sent_approvals()
        if not sent_approvals:
            return

        if not self.check_connectivity():
            return

        self.logger.info("Checking status of %d sent approval(s)", len(sent_approvals))

        for approval in sent_approvals:
            approval_id = approval["approval_id"]
            visitor_face_id = approval["visitor_face_id"]

            try:
                resp = requests.get(
                    self.approval_check_url,
                    params={
                        "device_index": self.device_index,
                        "visitor_face_id": visitor_face_id,
                    },
                    timeout=self.api_timeout,
                )

                if resp.status_code != 200:
                    continue

                data = resp.json()
                status = data.get("status", "").upper()

                if status == "APPROVED":
                    self.db.mark_approval_resolved(approval_id, "APPROVED")
                    employee_id = data.get("employee_id", visitor_face_id)
                    face_db = self._get_face_db()
                    if face_db.promote_to_employee(visitor_face_id, employee_id):
                        self.logger.info(
                            "✅ Visitor %s approved and promoted to employee %s",
                            visitor_face_id, employee_id,
                        )
                    else:
                        self.logger.warning(
                            "Visitor %s approved but promotion failed (not found in visitor DB)",
                            visitor_face_id,
                        )

                elif status == "REJECTED":
                    self.db.mark_approval_resolved(approval_id, "REJECTED")
                    self.logger.info(
                        "❌ Visitor %s approval rejected", visitor_face_id,
                    )

                # status == "PENDING" → still waiting, do nothing

            except requests.RequestException as exc:
                self.logger.warning(
                    "Approval %s: status check failed — %s", approval_id, exc,
                )
            except (json.JSONDecodeError, KeyError) as exc:
                self.logger.warning(
                    "Approval %s: invalid status response — %s", approval_id, exc,
                )

        # Auto-promote any approvals that have been waiting too long
        self._auto_promote_expired_approvals()

    def _auto_promote_expired_approvals(self):
        """Auto-promote visitors whose approval has been SENT but unresolved for too long."""
        expired = self.db.get_expired_sent_approvals(self.auto_approve_days)
        if not expired:
            return

        self.logger.info(
            "Found %d approval(s) expired after %d days — auto-promoting",
            len(expired), self.auto_approve_days,
        )

        for approval in expired:
            approval_id = approval["approval_id"]
            visitor_face_id = approval["visitor_face_id"]
            try:
                face_db = self._get_face_db()
                if face_db.promote_to_employee(visitor_face_id, visitor_face_id):
                    self.db.mark_approval_resolved(approval_id, "AUTO_APPROVED")
                    self.logger.info(
                        "✅ Visitor %s auto-promoted to employee (no response after %d days)",
                        visitor_face_id, self.auto_approve_days,
                    )
                else:
                    self.logger.warning(
                        "Visitor %s: auto-promote failed (not found in visitor DB)",
                        visitor_face_id,
                    )
            except Exception as exc:
                self.logger.error(
                    "Approval %s: auto-promote error — %s", approval_id, exc,
                )

    # ── cleanup ───────────────────────────────────────────────────

    def cleanup_old_events(self):
        """Mark old SENT events as CLEANED."""
        now = datetime.now(tz)
        if (now - self._last_cleanup) < timedelta(hours=self.cleanup_interval_hours):
            return

        self._last_cleanup = now
        events = self.db.get_cleanable_events(self.retention_days)
        if not events:
            return

        for event in events:
            self.db.mark_cleaned(event["event_id"])
            self.logger.info(
                "Cleaned event %s (date=%s, sent_at=%s)",
                event["event_id"], event["date"], event["sent_at"],
            )

        self.logger.info("Cleaned %d old event(s)", len(events))

    # ── main loop ─────────────────────────────────────────────────

    def _handle_signal(self, signum, _frame):
        sig_name = signal.Signals(signum).name
        self.logger.info("Received %s — shutting down gracefully", sig_name)
        self._running = False

    def run(self):
        """Main daemon loop."""
        signal.signal(signal.SIGTERM, self._handle_signal)
        signal.signal(signal.SIGINT, self._handle_signal)

        self.logger.info("Queue worker started (poll every %ds)", self.poll_interval)
        self.logger.info("Upload URL: %s", self.upload_url)
        self.logger.info("Device index: %s", self.device_index)

        stats = self.db.get_stats()
        if stats:
            self.logger.info("Current queue stats: %s", stats)

        while self._running:
            try:
                self.process_queue()
                self.process_approvals()
                self.check_approval_status()
                self.cleanup_old_events()
            except Exception:
                self.logger.exception("Unexpected error in main loop")

            # Interruptible sleep
            for _ in range(self.poll_interval):
                if not self._running:
                    break
                time.sleep(1)

        self.logger.info("Queue worker stopped")


# ── entry point ───────────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Queue Worker Service")
    parser.add_argument(
        "--config",
        default=None,
        help="Path to config.json (default: queue_service/config.json)",
    )
    args = parser.parse_args()

    worker = QueueWorker(config_path=args.config)
    worker.run()


if __name__ == "__main__":
    main()
