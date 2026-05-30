"""
SQLAlchemy database setup for the Data Center Digital Twin Platform.
Uses SQLite with WAL mode for concurrent read/write access from the
background simulation thread and the API request handlers.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Generator

from dotenv import load_dotenv
from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    Index,
    Integer,
    String,
    Text,
    create_engine,
    event,
    text,
)
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

load_dotenv()

DB_PATH = os.getenv("DB_PATH", "./data_center.db")
DATABASE_URL = f"sqlite:///{DB_PATH}"

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},
    pool_size=5,
    max_overflow=10,
)


@event.listens_for(engine, "connect")
def _set_sqlite_pragmas(dbapi_connection, connection_record):
    """Enable WAL mode and foreign keys on every new connection."""
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA synchronous=NORMAL")
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()


SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


# ─────────────────────────────────────────
# ORM Table Definitions
# ─────────────────────────────────────────

class AssetDB(Base):
    """Static asset inventory — created once at startup."""
    __tablename__ = "assets"

    id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    asset_type = Column(String, nullable=False)
    room = Column(String, nullable=True)
    row = Column(String, nullable=True)
    position = Column(Integer, nullable=True)
    rated_kw = Column(Float, nullable=False, default=0.0)
    extra_json = Column(Text, nullable=True)


class TelemetryDB(Base):
    """Historical telemetry — one row per asset per simulation cycle."""
    __tablename__ = "telemetry"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    cycle_id = Column(Integer, nullable=False, index=True)
    asset_id = Column(String, nullable=False, index=True)
    asset_type = Column(String, nullable=False)
    data_json = Column(Text, nullable=False)


class LatestTelemetryDB(Base):
    """Fast-lookup table: always holds the single most recent reading per asset."""
    __tablename__ = "latest_telemetry"

    asset_id = Column(String, primary_key=True)
    timestamp = Column(DateTime, nullable=False)
    cycle_id = Column(Integer, nullable=False)
    asset_type = Column(String, nullable=False)
    data_json = Column(Text, nullable=False)


class PUEHistoryDB(Base):
    """PUE time series used for linear-regression forecasting."""
    __tablename__ = "pue_history"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    cycle_id = Column(Integer, nullable=False)
    pue = Column(Float, nullable=False)
    it_load_kw = Column(Float, nullable=False)
    cooling_load_kw = Column(Float, nullable=False)
    total_facility_kw = Column(Float, nullable=False)
    outside_air_temp_c = Column(Float, nullable=False, default=15.0)


class AlertDB(Base):
    """Persisted alert records — written when alert fires or updates."""
    __tablename__ = "alerts"

    id = Column(String, primary_key=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    last_seen = Column(DateTime, nullable=False)
    asset_id = Column(String, nullable=False, index=True)
    metric = Column(String, nullable=False)
    severity = Column(String, nullable=False)
    message = Column(Text, nullable=False)
    confidence = Column(Integer, nullable=False, default=50)
    explanation = Column(Text, nullable=False)
    duration_cycles = Column(Integer, nullable=False, default=1)
    deviation = Column(Float, nullable=False, default=0.0)
    baseline_value = Column(Float, nullable=False, default=0.0)
    current_value = Column(Float, nullable=False, default=0.0)
    is_active = Column(Boolean, nullable=False, default=True)


class RecommendationDB(Base):
    """Generated optimization recommendations."""
    __tablename__ = "recommendations"

    id = Column(String, primary_key=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    recommendation_type = Column(String, nullable=False)
    asset_id = Column(String, nullable=True)
    priority = Column(String, nullable=False)
    action_text = Column(Text, nullable=False)
    expected_savings_kw = Column(Float, nullable=False, default=0.0)
    pue_impact = Column(Float, nullable=False, default=0.0)
    confidence = Column(Integer, nullable=False, default=80)
    rationale = Column(Text, nullable=False)
    is_active = Column(Boolean, nullable=False, default=True)


class EventDB(Base):
    """Operations audit trail — every alert and recommendation generates an event."""
    __tablename__ = "events"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    event_type = Column(String, nullable=False)
    asset_id = Column(String, nullable=True, index=True)
    severity = Column(String, nullable=True)
    message = Column(Text, nullable=False)
    confidence = Column(Integer, nullable=True)


class PlacementDecisionDB(Base):
    """Audit log of workload placement recommendations."""
    __tablename__ = "placement_decisions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    workload_kw = Column(Float, nullable=False)
    recommended_rack_id = Column(String, nullable=True)
    thermal_risk_score = Column(Float, nullable=False, default=0.0)
    pue_impact = Column(Float, nullable=False, default=0.0)
    overall_score = Column(Float, nullable=False, default=0.0)
    was_accepted = Column(Boolean, nullable=True)


class EnergyScheduleDB(Base):
    """Deferrable load schedule entries."""
    __tablename__ = "energy_schedule"

    id = Column(Integer, primary_key=True, autoincrement=True)
    load_name = Column(String, nullable=False)
    load_type = Column(String, nullable=False)
    scheduled_start = Column(DateTime, nullable=False)
    scheduled_end = Column(DateTime, nullable=False)
    power_kw = Column(Float, nullable=False)
    cost_saving = Column(Float, nullable=False, default=0.0)
    status = Column(String, nullable=False, default="pending")
    created_at = Column(DateTime, nullable=False)


class DemandPeakDB(Base):
    """15-minute demand peak readings for ratchet tracking."""
    __tablename__ = "demand_peaks"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    demand_kw = Column(Float, nullable=False)
    period_type = Column(String, nullable=False)
    is_monthly_peak = Column(Boolean, nullable=False, default=False)
    month_year = Column(String, nullable=False)


class MonthlyBillDB(Base):
    """Monthly electricity bill records."""
    __tablename__ = "monthly_bills"

    id = Column(Integer, primary_key=True, autoincrement=True)
    month_year = Column(String, nullable=False, unique=True)
    energy_cost = Column(Float, nullable=False)
    demand_charge = Column(Float, nullable=False)
    coincident_charge = Column(Float, nullable=False)
    facilities_charge = Column(Float, nullable=False)
    ratchet_adjustment = Column(Float, nullable=False, default=0.0)
    total = Column(Float, nullable=False)
    created_at = Column(DateTime, nullable=False)


class OptimizationConflictDB(Base):
    """Audit log of detected and resolved optimization layer conflicts."""
    __tablename__ = "optimization_conflicts"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    conflict_type = Column(String, nullable=False)
    layer_a = Column(String, nullable=False)
    layer_b = Column(String, nullable=False)
    description = Column(Text, nullable=False)
    resolution = Column(Text, nullable=False)
    winner = Column(String, nullable=False)
    resolved_at = Column(DateTime, nullable=True)


# Additional composite indexes for common query patterns
Index("ix_telemetry_asset_cycle", TelemetryDB.asset_id, TelemetryDB.cycle_id)
Index("ix_alerts_active", AlertDB.is_active, AlertDB.severity)
Index("ix_events_ts_sev", EventDB.timestamp, EventDB.severity)


# ─────────────────────────────────────────
# Database Lifecycle
# ─────────────────────────────────────────

def init_db() -> None:
    """Create all tables if they do not exist."""
    Base.metadata.create_all(bind=engine)


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency that provides a per-request SQLAlchemy session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_session() -> Session:
    """Return a raw session for use outside of FastAPI dependency injection."""
    return SessionLocal()


# ─────────────────────────────────────────
# Helper Utilities
# ─────────────────────────────────────────

def upsert_latest_telemetry(
    db: Session,
    asset_id: str,
    asset_type: str,
    cycle_id: int,
    timestamp: datetime,
    data: dict,
) -> None:
    """Insert or replace the latest telemetry row for an asset."""
    existing = db.get(LatestTelemetryDB, asset_id)
    if existing:
        existing.timestamp = timestamp
        existing.cycle_id = cycle_id
        existing.asset_type = asset_type
        existing.data_json = json.dumps(data)
    else:
        db.add(
            LatestTelemetryDB(
                asset_id=asset_id,
                timestamp=timestamp,
                cycle_id=cycle_id,
                asset_type=asset_type,
                data_json=json.dumps(data),
            )
        )


def purge_old_telemetry(db: Session, keep_cycles: int = 200) -> None:
    """Delete telemetry rows older than keep_cycles to prevent unbounded growth."""
    subq = (
        db.query(TelemetryDB.cycle_id)
        .distinct()
        .order_by(TelemetryDB.cycle_id.desc())
        .limit(keep_cycles)
        .subquery()
    )
    db.query(TelemetryDB).filter(TelemetryDB.cycle_id.notin_(subq)).delete(
        synchronize_session=False
    )
    db.commit()


def get_simulation_cycle_count(db: Session) -> int:
    """Return the highest cycle_id in the telemetry table."""
    result = db.execute(text("SELECT MAX(cycle_id) FROM telemetry")).scalar()
    return int(result) if result is not None else 0
