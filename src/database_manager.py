import os
import sqlite3
from pathlib import Path
from typing import List, Tuple

DB_FILENAME = "auravision_memory.db"
TABLE_NAME = "spatial_memory"


def _get_db_path() -> str:
    project_root = Path(__file__).resolve().parent.parent
    return str(project_root / DB_FILENAME)


def _get_connection() -> sqlite3.Connection:
    db_path = _get_db_path()
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def _initialize_database() -> None:
    db_path = _get_db_path()
    os.makedirs(os.path.dirname(db_path), exist_ok=True)

    with _get_connection() as conn:
        conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                item_detected TEXT,
                spatial_location TEXT,
                confidence_score TEXT
            )
            """
        )
        conn.commit()


def insert_memory(item: str, location: str, confidence: str) -> None:
    """Insert a new spatial memory record into the local SQLite database."""
    _initialize_database()
    with _get_connection() as conn:
        conn.execute(
            f"INSERT INTO {TABLE_NAME} (item_detected, spatial_location, confidence_score) VALUES (?, ?, ?)",
            (item, location, confidence),
        )
        conn.commit()


def get_recent_memories(limit: int = 5) -> List[Tuple[int, str, str, str, str]]:
    """Retrieve the most recent spatial memory records from the local SQLite database."""
    _initialize_database()
    with _get_connection() as conn:
        cursor = conn.execute(
            f"SELECT id, timestamp, item_detected, spatial_location, confidence_score "
            f"FROM {TABLE_NAME} ORDER BY timestamp DESC LIMIT ?",
            (limit,),
        )
        rows = cursor.fetchall()

    return [
        (row["id"], row["timestamp"], row["item_detected"], row["spatial_location"], row["confidence_score"])
        for row in rows
    ]


if __name__ == "__main__":
    insert_memory("monitor", "on desk", "high")
    memories = get_recent_memories(limit=5)
    print("Recent memories:")
    for memory in memories:
        print(memory)
