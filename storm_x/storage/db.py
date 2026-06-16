"""SQLite schema definition and query helpers for STORM-X.

All tables are created on first use. Connection is obtained per-call (thread-safe via
check_same_thread=False + WAL mode). No ORM — plain sqlite3 from stdlib.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from loguru import logger

from storm_x.config import settings

_DB_PATH = Path(__file__).parent.parent.parent / settings.storage.db_path


def _connect() -> sqlite3.Connection:
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(_DB_PATH), check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Create all tables if they don't exist. Safe to call on every startup."""
    with _connect() as conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS bias_history (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            city            TEXT    NOT NULL,
            calendar_month  INTEGER NOT NULL,
            forecast_date   TEXT    NOT NULL,
            forecast_value  REAL    NOT NULL,
            observed_value  REAL    NOT NULL,
            error           REAL    NOT NULL,
            recorded_at     TEXT    NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_bias_city_month
            ON bias_history (city, calendar_month, forecast_date);

        CREATE TABLE IF NOT EXISTS calibration_history (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            bet_id          TEXT    NOT NULL,
            model_prob      REAL    NOT NULL,
            calibrated_prob REAL    NOT NULL,
            outcome         INTEGER,          -- 0 or 1; NULL until resolved
            resolved_at     TEXT
        );

        CREATE TABLE IF NOT EXISTS bet_history (
            bet_id              TEXT    PRIMARY KEY,
            city                TEXT    NOT NULL,
            market_token_id     TEXT    NOT NULL,
            bracket_description TEXT    NOT NULL,
            side                TEXT    NOT NULL,    -- YES or NO
            entry_price         REAL    NOT NULL,
            size                REAL    NOT NULL,
            edge_at_entry       REAL    NOT NULL,
            kelly_fraction      REAL    NOT NULL,
            regime_score        REAL    NOT NULL,
            model_prob          REAL    NOT NULL,
            calibrated_prob     REAL    NOT NULL,
            status              TEXT    NOT NULL DEFAULT 'open',  -- open | closed
            resolution_outcome  INTEGER,             -- 0 or 1; NULL until resolved
            pnl                 REAL,
            created_at          TEXT    NOT NULL,
            resolved_at         TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_bet_status ON bet_history (status);

        CREATE TABLE IF NOT EXISTS market_state (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            city                TEXT    NOT NULL,
            fetch_time          TEXT    NOT NULL,
            members_json        TEXT    NOT NULL,    -- JSON array of floats
            member_count        INTEGER NOT NULL,
            model_names         TEXT    NOT NULL,    -- JSON list of model names
            generation_hash     TEXT    NOT NULL     -- SHA1 of members_json for change detection
        );
        CREATE INDEX IF NOT EXISTS idx_market_city_time
            ON market_state (city, fetch_time);

        CREATE TABLE IF NOT EXISTS climatology (
            id      INTEGER PRIMARY KEY AUTOINCREMENT,
            city    TEXT    NOT NULL,
            month   INTEGER NOT NULL,
            day     INTEGER NOT NULL,
            mean    REAL    NOT NULL,
            std     REAL    NOT NULL,
            UNIQUE(city, month, day)
        );

        CREATE TABLE IF NOT EXISTS ensemble_cache (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            city        TEXT    NOT NULL,
            hour_key    TEXT    NOT NULL,   -- YYYY-MM-DDTHH (UTC, truncated to hour)
            model_name  TEXT    NOT NULL,
            members_json TEXT   NOT NULL,
            fetched_at  TEXT    NOT NULL,
            UNIQUE(city, hour_key, model_name)
        );
        """)
    logger.info("DB initialised at {}", _DB_PATH)


# ── Ensemble cache helpers ─────────────────────────────────────────────────────

def cache_get_ensemble(city: str, hour_key: str, model_name: str) -> list[float] | None:
    """Return cached member array if it exists and is within TTL, else None."""
    from datetime import timedelta
    ttl = settings.ensemble.cache_ttl_minutes
    with _connect() as conn:
        row = conn.execute(
            "SELECT members_json, fetched_at FROM ensemble_cache "
            "WHERE city=? AND hour_key=? AND model_name=?",
            (city, hour_key, model_name),
        ).fetchone()
    if not row:
        return None
    fetched = datetime.fromisoformat(row["fetched_at"])
    if (datetime.now(timezone.utc) - fetched).total_seconds() > ttl * 60:
        return None
    return json.loads(row["members_json"])


def cache_set_ensemble(city: str, hour_key: str, model_name: str, members: list[float]) -> None:
    with _connect() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO ensemble_cache "
            "(city, hour_key, model_name, members_json, fetched_at) VALUES (?,?,?,?,?)",
            (city, hour_key, model_name, json.dumps(members),
             datetime.now(timezone.utc).isoformat()),
        )


# ── Bias history helpers ───────────────────────────────────────────────────────

def insert_bias_row(city: str, month: int, forecast_date: str,
                    forecast_val: float, observed_val: float) -> None:
    with _connect() as conn:
        conn.execute(
            "INSERT INTO bias_history "
            "(city, calendar_month, forecast_date, forecast_value, observed_value, error, recorded_at) "
            "VALUES (?,?,?,?,?,?,?)",
            (city, month, forecast_date, forecast_val, observed_val,
             observed_val - forecast_val, datetime.now(timezone.utc).isoformat()),
        )


def get_bias_rows(city: str, month: int, window_days: int = 90) -> list[sqlite3.Row]:
    cutoff = (datetime.now(timezone.utc).date().replace(day=1)).__str__()  # rough cutoff
    with _connect() as conn:
        return conn.execute(
            "SELECT * FROM bias_history WHERE city=? AND calendar_month BETWEEN ? AND ? "
            "ORDER BY forecast_date DESC LIMIT 200",
            (city, max(1, month - 1), min(12, month + 1)),
        ).fetchall()


# ── Bet history helpers ────────────────────────────────────────────────────────

def insert_bet(row: dict[str, Any]) -> None:
    cols = ", ".join(row.keys())
    placeholders = ", ".join("?" for _ in row)
    with _connect() as conn:
        conn.execute(
            f"INSERT OR IGNORE INTO bet_history ({cols}) VALUES ({placeholders})",
            list(row.values()),
        )


def resolve_bet(bet_id: str, outcome: int, pnl: float) -> None:
    with _connect() as conn:
        conn.execute(
            "UPDATE bet_history SET status='closed', resolution_outcome=?, pnl=?, resolved_at=? WHERE bet_id=?",
            (outcome, pnl, datetime.now(timezone.utc).isoformat(), bet_id),
        )


def get_open_bets(city: str | None = None) -> list[sqlite3.Row]:
    with _connect() as conn:
        if city:
            return conn.execute(
                "SELECT * FROM bet_history WHERE status='open' AND city=?", (city,)
            ).fetchall()
        return conn.execute("SELECT * FROM bet_history WHERE status='open'").fetchall()


def get_resolved_bets(limit: int = 60) -> list[sqlite3.Row]:
    with _connect() as conn:
        return conn.execute(
            "SELECT * FROM bet_history WHERE status='closed' "
            "ORDER BY resolved_at DESC LIMIT ?", (limit,)
        ).fetchall()


# ── Climatology helpers ────────────────────────────────────────────────────────

def upsert_climatology(city: str, month: int, day: int, mean: float, std: float) -> None:
    with _connect() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO climatology (city, month, day, mean, std) VALUES (?,?,?,?,?)",
            (city, month, day, mean, std),
        )


def get_climatology(city: str, month: int, day: int) -> tuple[float, float] | None:
    with _connect() as conn:
        row = conn.execute(
            "SELECT mean, std FROM climatology WHERE city=? AND month=? AND day=?",
            (city, month, day),
        ).fetchone()
    return (row["mean"], row["std"]) if row else None


def climatology_loaded(city: str) -> bool:
    with _connect() as conn:
        count = conn.execute(
            "SELECT COUNT(*) FROM climatology WHERE city=?", (city,)
        ).fetchone()[0]
    return count > 300  # at least ~300 month-day combos means data is loaded
