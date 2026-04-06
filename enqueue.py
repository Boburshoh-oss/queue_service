#!/usr/bin/env python3
"""
Enqueue module — adds events to the upload queue.
Can be imported as a library or used from the command line.
"""

import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from queue_db import QueueDB

logger = logging.getLogger(__name__)


def enqueue_event(
    date: str,
    deepstream_json_path: str = None,
    face_analyzer_json_path: str = None,
    db_path: str = None,
    force: bool = False,
) -> str | None:
    """
    Add an event to the upload queue.

    Args:
        date: Date string (YYYY-MM-DD)
        deepstream_json_path: Path to entry_log JSON
        face_analyzer_json_path: Path to daily_info JSON
        db_path: Optional custom path to queue.db
        force: If True, delete existing event for this date before enqueueing

    Returns:
        event_id on success, None on duplicate/error
    """
    db = QueueDB(db_path=db_path)

    # Check for existing (non-CLEANED) event for this date
    if db.event_exists_for_date(date):
        if force:
            deleted = db.delete_by_date(date)
            logger.info("Deleted %d existing event(s) for date %s", deleted, date)
        else:
            logger.warning("Event already exists for date %s — skipping (use --force to replace)", date)
            return None

    # Validate that face_analyzer JSON exists
    if face_analyzer_json_path and not os.path.isfile(face_analyzer_json_path):
        logger.warning(
            "face_analyzer JSON not found at %s — enqueueing anyway",
            face_analyzer_json_path,
        )

    # Validate that deepstream JSON exists
    if deepstream_json_path and not os.path.isfile(deepstream_json_path):
        logger.warning(
            "deepstream JSON not found at %s — enqueueing anyway",
            deepstream_json_path,
        )

    event_id = db.enqueue(date, deepstream_json_path, face_analyzer_json_path)
    logger.info(
        "Enqueued event %s for date %s (face_json=%s)",
        event_id, date, face_analyzer_json_path,
    )
    return event_id


def enqueue_approval(
    visitor_face_id: str,
    photo_path: str = None,
    visit_count: int = 0,
    visit_history: list = None,
    age: str = None,
    gender: str = None,
    embedding: list = None,
    db_path: str = None,
) -> str | None:
    """
    Add an employee approval request to the queue.

    Args:
        visitor_face_id: The visitor's face ID in the vector DB
        photo_path: Path to best face crop image
        visit_count: Total number of visits
        visit_history: List of visit dicts [{date, time, head_id}, ...]
        age: Predicted age range
        gender: Predicted gender
        embedding: Face embedding as list of floats
        db_path: Optional custom path to queue.db

    Returns:
        approval_id on success, None on duplicate/error
    """
    import json as _json

    db = QueueDB(db_path=db_path)

    # Check if approval already exists for this visitor
    if db.approval_exists_for_visitor(visitor_face_id):
        logger.info("Approval already exists for visitor %s — skipping", visitor_face_id)
        return None

    visit_history_json = _json.dumps(visit_history) if visit_history else "[]"
    embedding_json = _json.dumps(embedding) if embedding else None

    approval_id = db.enqueue_approval(
        visitor_face_id=visitor_face_id,
        photo_path=photo_path,
        visit_count=visit_count,
        visit_history=visit_history_json,
        age=age,
        gender=gender,
        embedding_json=embedding_json,
    )
    logger.info(
        "Enqueued approval %s for visitor %s (visits=%d)",
        approval_id, visitor_face_id, visit_count,
    )
    return approval_id


# ── CLI ───────────────────────────────────────────────────────────

def main():
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    parser = argparse.ArgumentParser(description="Enqueue an upload event")
    parser.add_argument("--date", required=True, help="Date (YYYY-MM-DD)")
    parser.add_argument("--deepstream-json", default=None, help="Path to entry_log JSON")
    parser.add_argument("--face-json", required=True, help="Path to daily_info JSON")
    parser.add_argument("--db-path", default=None, help="Custom path to queue.db")
    parser.add_argument("--force", action="store_true", help="Delete existing event for this date and re-enqueue")
    args = parser.parse_args()

    event_id = enqueue_event(
        date=args.date,
        deepstream_json_path=args.deepstream_json,
        face_analyzer_json_path=args.face_json,
        db_path=args.db_path,
        force=args.force,
    )

    if event_id:
        print(f"✅ Event enqueued: {event_id}")
    else:
        print("⚠️  Event not enqueued (duplicate or error)")
        sys.exit(1)


if __name__ == "__main__":
    main()
