"""
Realistic data-center telemetry simulator for the Digital Twin Platform.

Generates time-varying readings for 80 racks, 8 CRACs, 2 chillers, 2 UPS
units, and 8 PDUs using sinusoidal diurnal patterns plus Gaussian noise.
Anomaly injection fires periodically to exercise the alert and
recommendation pipelines during demos.

The module is importable standalone — it has no hard dependency on FastAPI
or the running API server.
"""

from __future__ import annotations

import json
import logging
import math
import random
import time
import threading
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from backend.database import (
    AlertDB,
    AssetDB,
    EventDB,
    PUEHistoryDB,
    RecommendationDB,
    SessionLocal,
    TelemetryDB,
    get_simulation_cycle_count,
    init_db,
    purge_old_telemetry,
    upsert_latest_telemetry,
)
from backend.models import (
    AshraeClass,
    AssetInventory,
    ChillerAsset,
    ChillerTelemetry,
    CRACAsset,
    CRACTelemetry,
    PDUAsset,
    PDUTelemetry,
    RackAsset,
    RackTelemetry,
    TelemetrySnapshot,
    UPSAsset,
    UPSTelemetry,
)

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────
# Asset definitions (static)
# ─────────────────────────────────────────

ROOMS = ["A", "B"]
ROWS_PER_ROOM = 4
RACKS_PER_ROW = 10
CRACS_PER_ROOM = 4
NUM_CHILLERS = 2
NUM_UPS = 2
PDUS_PER_ROOM = 4

# Rated capacities (kW)
RACK_RATED_KW = 20.0
CRAC_RATED_COOLING_KW = 100.0
CHILLER_RATED_KW = 500.0
UPS_RATED_KW = 200.0
PDU_RATED_KW = 100.0


def _make_rack_id(room: str, row: int, position: int) -> str:
    return f"{room}-R{row}-{position:03d}"


def _make_crac_id(room: str, num: int) -> str:
    return f"{room}-CRAC-{num}"


def _make_chiller_id(num: int) -> str:
    return f"CH-{num}"


def _make_ups_id(num: int) -> str:
    return f"UPS-{num}"


def _make_pdu_id(room: str, num: int) -> str:
    return f"{room}-PDU-{num}"


def build_asset_inventory() -> AssetInventory:
    """Build the complete static asset inventory from parametric definitions."""
    racks: List[RackAsset] = []
    cracs: List[CRACAsset] = []
    chillers: List[ChillerAsset] = []
    upses: List[UPSAsset] = []
    pdus: List[PDUAsset] = []

    for room in ROOMS:
        for row in range(1, ROWS_PER_ROOM + 1):
            for pos in range(1, RACKS_PER_ROW + 1):
                rid = _make_rack_id(room, row, pos)
                racks.append(
                    RackAsset(
                        id=rid,
                        name=f"Room {room} Row {row} Rack {pos:03d}",
                        room=f"Room {room}",
                        row=f"Row {row}",
                        position=pos,
                        rated_kw=RACK_RATED_KW,
                    )
                )
        for n in range(1, CRACS_PER_ROOM + 1):
            cid = _make_crac_id(room, n)
            cracs.append(
                CRACAsset(
                    id=cid,
                    name=f"Room {room} CRAC Unit {n}",
                    room=f"Room {room}",
                    rated_cooling_kw=CRAC_RATED_COOLING_KW,
                    setpoint_c=18.0,
                )
            )
        for n in range(1, PDUS_PER_ROOM + 1):
            pid = _make_pdu_id(room, n)
            pdus.append(
                PDUAsset(
                    id=pid,
                    name=f"Room {room} PDU {n}",
                    room=f"Room {room}",
                    rated_kw=PDU_RATED_KW,
                )
            )

    for n in range(1, NUM_CHILLERS + 1):
        chillers.append(
            ChillerAsset(
                id=_make_chiller_id(n),
                name=f"Chiller {n}",
                rated_cooling_kw=CHILLER_RATED_KW,
                cop_design=4.5,
            )
        )
    for n in range(1, NUM_UPS + 1):
        upses.append(
            UPSAsset(
                id=_make_ups_id(n),
                name=f"UPS {n}",
                rated_kw=UPS_RATED_KW,
            )
        )

    return AssetInventory(
        racks=racks,
        cracs=cracs,
        chillers=chillers,
        upses=upses,
        pdus=pdus,
        total_rack_count=len(racks),
        total_rated_it_kw=len(racks) * RACK_RATED_KW,
    )


# ─────────────────────────────────────────
# Diurnal helper
# ─────────────────────────────────────────

def _time_load_factor(hour: Optional[float] = None) -> float:
    """
    Return 0.0–1.0 reflecting IT load intensity for a given hour.
    Uses a cosine curve peaking at 14:00 and troughing near 02:00.
    """
    if hour is None:
        now = datetime.utcnow()
        hour = now.hour + now.minute / 60.0
    # cos(0) = 1 at peak; cos(π) = -1 at trough → normalise to [0,1]
    raw = math.cos(2 * math.pi * (hour - 14) / 24)
    return (raw + 1) / 2  # range [0, 1]


def _outside_air_temp(load_factor: float) -> float:
    """Simulate outside air temperature correlated with time of day (°C)."""
    # 8°C trough at night, 22°C peak in afternoon; add small noise
    return 8 + 14 * load_factor + random.gauss(0, 0.8)


def _ashrae_class(inlet_temp_c: float) -> AshraeClass:
    """Return the ASHRAE thermal class based on inlet temperature."""
    if inlet_temp_c <= 32:
        return AshraeClass.A1
    elif inlet_temp_c <= 35:
        return AshraeClass.A2
    elif inlet_temp_c <= 40:
        return AshraeClass.A3
    else:
        return AshraeClass.ABOVE_A2


# ─────────────────────────────────────────
# Core simulator class
# ─────────────────────────────────────────

class DataCenterSimulator:
    """
    Stateful simulator that generates realistic data-center telemetry.

    Maintains anomaly state between cycles so multi-cycle alert conditions
    (e.g. PUE > 1.8 for 3+ cycles) fire naturally during a demo.
    """

    def __init__(self) -> None:
        self.inventory = build_asset_inventory()
        self._cycle_counter: int = 0
        self._start_time: float = time.time()

        # Anomaly state: maps asset_id → remaining anomaly cycles
        self._anomaly_map: Dict[str, int] = {}
        # Consecutive alert counter: maps (asset_id, metric) → count
        self._alert_streak: Dict[Tuple[str, str], int] = {}
        # Previous power readings for spike detection
        self._prev_power: Dict[str, float] = {}
        # CW setpoint (°C) — can be adjusted by recommendations
        self._chw_setpoint: float = 8.0
        # CRAC supply setpoint (°C)
        self._crac_setpoint: float = 18.0
        # Thread lock for state mutations
        self._lock = threading.Lock()

    # ── Anomaly injection ──────────────────

    def _maybe_inject_anomaly(self, rack_ids: List[str]) -> None:
        """5 % chance per cycle to start a new 3–8 cycle anomaly on a random rack."""
        if random.random() < 0.05 and rack_ids:
            target = random.choice(rack_ids)
            duration = random.randint(3, 8)
            with self._lock:
                self._anomaly_map[target] = duration

    def _tick_anomalies(self) -> None:
        """Decrement anomaly counters; remove expired ones."""
        with self._lock:
            expired = [k for k, v in self._anomaly_map.items() if v <= 1]
            for k in expired:
                del self._anomaly_map[k]
            for k in self._anomaly_map:
                self._anomaly_map[k] -= 1

    def _is_anomaly(self, asset_id: str) -> bool:
        with self._lock:
            return asset_id in self._anomaly_map

    # ── Rack telemetry generation ──────────

    def _gen_rack(self, rack: RackAsset, load_factor: float) -> RackTelemetry:
        """Generate one rack telemetry reading."""
        anomaly = self._is_anomaly(rack.id)

        # Base power: 3–18 kW sinusoid + per-rack noise
        base_kw = 3 + 15 * load_factor
        power_kw = base_kw + random.gauss(0, 1.2)
        if anomaly:
            power_kw *= random.uniform(1.2, 1.5)  # power surge
        power_kw = max(1.0, min(RACK_RATED_KW, power_kw))

        # Inlet temperature: rises with ambient + load; hotter in higher rows
        row_num = int(rack.row.split()[-1]) if rack.row else 1
        row_bias = (row_num - 1) * 0.5  # hot air rises; upper rows warmer
        inlet_temp = (
            18.0
            + 4 * load_factor
            + row_bias
            + random.gauss(0, 0.6)
        )
        if anomaly:
            inlet_temp += random.uniform(6, 12)

        inlet_temp = max(16.0, min(45.0, inlet_temp))

        # Outlet = inlet + delta proportional to power density
        temp_rise = (power_kw / RACK_RATED_KW) * 14 + random.gauss(0, 0.4)
        outlet_temp = inlet_temp + temp_rise

        humidity = 50 + random.gauss(0, 3)
        humidity = max(30.0, min(70.0, humidity))

        return RackTelemetry(
            asset_id=rack.id,
            timestamp=datetime.utcnow(),
            inlet_temp_c=round(inlet_temp, 2),
            outlet_temp_c=round(outlet_temp, 2),
            power_kw=round(power_kw, 3),
            humidity_pct=round(humidity, 1),
            temp_rise_c=round(outlet_temp - inlet_temp, 2),
            ashrae_class=_ashrae_class(inlet_temp),
            room=rack.room,
            row=rack.row,
            position=rack.position,
        )

    # ── CRAC telemetry generation ──────────

    def _gen_crac(
        self,
        crac: CRACAsset,
        avg_rack_outlet: float,
        load_factor: float,
        failed: bool = False,
    ) -> CRACTelemetry:
        """Generate CRAC/CRAH telemetry."""
        if failed:
            # Failed unit: minimal cooling, high return air
            supply = self._crac_setpoint + random.gauss(0, 0.5)
            return_air = avg_rack_outlet + random.uniform(2, 5)
            delta_t = return_air - supply
            cooling_kw = max(5.0, CRAC_RATED_COOLING_KW * 0.05)
        else:
            supply = self._crac_setpoint + random.gauss(0, 0.3)
            # Normal delta-T is 8–14°C; low delta-T syndrome if containment poor
            target_delta = 10 + 2 * load_factor
            delta_t = target_delta + random.gauss(0, 1.2)
            delta_t = max(3.0, delta_t)
            return_air = supply + delta_t
            # Cooling load approximated from airflow × density × delta-T
            # For simplicity: scaled from load factor
            cooling_kw = (
                CRAC_RATED_COOLING_KW * (0.5 + 0.4 * load_factor)
                + random.gauss(0, 3)
            )
            cooling_kw = max(10.0, min(CRAC_RATED_COOLING_KW, cooling_kw))

        return CRACTelemetry(
            asset_id=crac.id,
            timestamp=datetime.utcnow(),
            supply_air_temp_c=round(supply, 2),
            return_air_temp_c=round(return_air, 2),
            delta_t_c=round(delta_t, 2),
            cooling_load_kw=round(cooling_kw, 2),
            setpoint_c=round(self._crac_setpoint, 1),
            room=crac.room,
        )

    # ── Chiller telemetry generation ───────

    def _gen_chiller(self, chiller: ChillerAsset, total_cooling_kw: float) -> ChillerTelemetry:
        """Generate chiller telemetry based on total cooling demand."""
        chw_supply = self._chw_setpoint + random.gauss(0, 0.3)
        delta_chw = 4 + 4 * (total_cooling_kw / (NUM_CHILLERS * CHILLER_RATED_KW))
        chw_return = chw_supply + delta_chw + random.gauss(0, 0.2)

        load_pct = (
            total_cooling_kw / (NUM_CHILLERS * CHILLER_RATED_KW) * 100
            + random.gauss(0, 2)
        )
        load_pct = max(20.0, min(98.0, load_pct))

        cop = 4.5 - (load_pct / 100) * 1.5 + random.gauss(0, 0.1)
        cop = max(2.0, min(6.0, cop))

        power_kw = (total_cooling_kw / NUM_CHILLERS) / cop + random.gauss(0, 1)
        power_kw = max(5.0, power_kw)

        return ChillerTelemetry(
            asset_id=chiller.id,
            timestamp=datetime.utcnow(),
            chw_supply_temp_c=round(chw_supply, 2),
            chw_return_temp_c=round(chw_return, 2),
            load_pct=round(load_pct, 1),
            power_kw=round(power_kw, 2),
            cop=round(cop, 2),
        )

    # ── UPS telemetry generation ───────────

    def _gen_ups(self, ups: UPSAsset, total_it_kw: float) -> UPSTelemetry:
        """Generate UPS telemetry."""
        it_per_ups = total_it_kw / NUM_UPS
        load_pct = (it_per_ups / ups.rated_kw * 100) + random.gauss(0, 2)
        load_pct = max(10.0, min(98.0, load_pct))

        output_kw = ups.rated_kw * load_pct / 100
        input_kw = output_kw * (1 + 0.03 + random.gauss(0, 0.005))  # ~3% loss

        battery_pct = 98 + random.gauss(0, 1)
        battery_pct = max(80.0, min(100.0, battery_pct))

        return UPSTelemetry(
            asset_id=ups.id,
            timestamp=datetime.utcnow(),
            load_pct=round(load_pct, 1),
            input_kw=round(input_kw, 2),
            output_kw=round(output_kw, 2),
            battery_pct=round(battery_pct, 1),
        )

    # ── PDU telemetry generation ───────────

    def _gen_pdu(self, pdu: PDUAsset, room_it_kw: float, pdu_count: int) -> PDUTelemetry:
        """Generate PDU telemetry."""
        base_load = room_it_kw / pdu_count
        # Distribute load across 3 phases with slight imbalance
        phase_a = base_load / 3 * (1 + random.gauss(0, 0.04))
        phase_b = base_load / 3 * (1 + random.gauss(0, 0.04))
        phase_c = base_load - phase_a - phase_b + random.gauss(0, 0.2)
        total = phase_a + phase_b + phase_c
        total = max(0.5, total)
        load_pct = (total / pdu.rated_kw) * 100

        return PDUTelemetry(
            asset_id=pdu.id,
            timestamp=datetime.utcnow(),
            phase_a_kw=round(max(0, phase_a), 3),
            phase_b_kw=round(max(0, phase_b), 3),
            phase_c_kw=round(max(0, phase_c), 3),
            total_kw=round(total, 3),
            load_pct=round(load_pct, 1),
            room=pdu.room,
        )

    # ── Full snapshot generation ───────────

    def generate_snapshot(self) -> TelemetrySnapshot:
        """
        Generate a complete telemetry snapshot for the current simulated instant.
        Injects anomalies probabilistically and ticks existing ones.
        """
        with self._lock:
            self._cycle_counter += 1
            cycle = self._cycle_counter

        load_factor = _time_load_factor()
        outside_temp = _outside_air_temp(load_factor)

        rack_ids = [r.id for r in self.inventory.racks]
        self._maybe_inject_anomaly(rack_ids)

        # ── Racks ──
        rack_telemetry: List[RackTelemetry] = []
        for rack in self.inventory.racks:
            rt = self._gen_rack(rack, load_factor)
            rack_telemetry.append(rt)
            self._prev_power[rack.id] = rt.power_kw

        total_it_kw = sum(r.power_kw for r in rack_telemetry)
        avg_outlet = sum(r.outlet_temp_c for r in rack_telemetry) / max(len(rack_telemetry), 1)

        # ── CRACs ──
        crac_telemetry: List[CRACTelemetry] = []
        for crac in self.inventory.cracs:
            ct = self._gen_crac(crac, avg_outlet, load_factor)
            crac_telemetry.append(ct)

        total_cooling_kw = sum(c.cooling_load_kw for c in crac_telemetry)

        # ── Chillers ──
        chiller_telemetry: List[ChillerTelemetry] = [
            self._gen_chiller(ch, total_cooling_kw)
            for ch in self.inventory.chillers
        ]

        # ── UPS ──
        ups_telemetry: List[UPSTelemetry] = [
            self._gen_ups(u, total_it_kw)
            for u in self.inventory.upses
        ]

        # ── PDUs ──
        pdu_telemetry: List[PDUTelemetry] = []
        for room in ROOMS:
            room_label = f"Room {room}"
            room_racks = [r for r in rack_telemetry if r.room == room_label]
            room_it_kw = sum(r.power_kw for r in room_racks)
            room_pdus = [p for p in self.inventory.pdus if p.room == room_label]
            for pdu in room_pdus:
                pdu_telemetry.append(self._gen_pdu(pdu, room_it_kw, len(room_pdus)))

        self._tick_anomalies()

        return TelemetrySnapshot(
            timestamp=datetime.utcnow(),
            cycle_id=cycle,
            racks=rack_telemetry,
            cracs=crac_telemetry,
            chillers=chiller_telemetry,
            upses=ups_telemetry,
            pdus=pdu_telemetry,
            outside_air_temp_c=round(outside_temp, 2),
        )

    @property
    def cycle_count(self) -> int:
        return self._cycle_counter


# ─────────────────────────────────────────
# Database persistence helpers
# ─────────────────────────────────────────

def _persist_snapshot(snapshot: TelemetrySnapshot) -> None:
    """Write a complete snapshot to SQLite in a single transaction."""
    from backend.alerts import detect_all_alerts
    from backend.metrics import (
        calculate_carbon,
        calculate_pue,
        calculate_stranded_capacity,
    )
    from backend.recommendations import generate_all_recommendations

    db = SessionLocal()
    try:
        ts = snapshot.timestamp
        cycle = snapshot.cycle_id

        def _write(asset_id: str, asset_type: str, data: dict) -> None:
            db.add(
                TelemetryDB(
                    timestamp=ts,
                    cycle_id=cycle,
                    asset_id=asset_id,
                    asset_type=asset_type,
                    data_json=json.dumps(data),
                )
            )
            upsert_latest_telemetry(db, asset_id, asset_type, cycle, ts, data)

        for r in snapshot.racks:
            _write(r.asset_id, "rack", r.model_dump(mode="json"))
        for c in snapshot.cracs:
            _write(c.asset_id, "crac", c.model_dump(mode="json"))
        for ch in snapshot.chillers:
            _write(ch.asset_id, "chiller", ch.model_dump(mode="json"))
        for u in snapshot.upses:
            _write(u.asset_id, "ups", u.model_dump(mode="json"))
        for p in snapshot.pdus:
            _write(p.asset_id, "pdu", p.model_dump(mode="json"))

        # Compute facility metrics
        it_load = sum(r.power_kw for r in snapshot.racks)
        cooling_load = sum(c.cooling_load_kw for c in snapshot.cracs)
        pue = calculate_pue(it_load, cooling_load)
        total_fac = it_load * pue

        db.add(
            PUEHistoryDB(
                timestamp=ts,
                cycle_id=cycle,
                pue=pue,
                it_load_kw=it_load,
                cooling_load_kw=cooling_load,
                total_facility_kw=total_fac,
                outside_air_temp_c=snapshot.outside_air_temp_c,
            )
        )

        db.commit()

        # Alerts
        alerts = detect_all_alerts(snapshot)
        _ingest_alerts(db, alerts, ts)

        # Recommendations
        recs = generate_all_recommendations(snapshot)
        _ingest_recommendations(db, recs, ts)

        # Periodic cleanup
        if cycle % 50 == 0:
            purge_old_telemetry(db, keep_cycles=200)

        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def _ingest_alerts(db, alerts: list, ts: datetime) -> None:
    """Upsert alerts into the DB and log them to the events table."""
    # Mark previously active alerts that are no longer firing as resolved
    active_ids = {a["id"] for a in alerts}
    stale = (
        db.query(AlertDB)
        .filter(AlertDB.is_active == True)
        .filter(AlertDB.id.notin_(active_ids))
        .all()
    )
    for row in stale:
        row.is_active = False
        db.add(
            EventDB(
                timestamp=ts,
                event_type="alert_cleared",
                asset_id=row.asset_id,
                severity=row.severity,
                message=f"Cleared: {row.message}",
                confidence=None,
            )
        )

    for a in alerts:
        existing = db.get(AlertDB, a["id"])
        if existing:
            existing.last_seen = ts
            existing.confidence = a["confidence"]
            existing.duration_cycles = a["duration_cycles"]
            existing.current_value = a["current_value"]
            existing.deviation = a["deviation"]
            existing.is_active = True
        else:
            db.add(
                AlertDB(
                    id=a["id"],
                    timestamp=ts,
                    last_seen=ts,
                    asset_id=a["asset_id"],
                    metric=a["metric"],
                    severity=a["severity"],
                    message=a["message"],
                    confidence=a["confidence"],
                    explanation=a["explanation"],
                    duration_cycles=a["duration_cycles"],
                    deviation=a["deviation"],
                    baseline_value=a["baseline_value"],
                    current_value=a["current_value"],
                    is_active=True,
                )
            )
            db.add(
                EventDB(
                    timestamp=ts,
                    event_type="alert_fired",
                    asset_id=a["asset_id"],
                    severity=a["severity"],
                    message=a["message"],
                    confidence=a["confidence"],
                )
            )


def _ingest_recommendations(db, recs: list, ts: datetime) -> None:
    """Upsert recommendations into the DB and log new ones."""
    active_ids = {r["id"] for r in recs}

    # Deactivate recs that no longer apply
    db.query(RecommendationDB).filter(
        RecommendationDB.is_active == True,
        RecommendationDB.id.notin_(active_ids),
    ).update({"is_active": False}, synchronize_session=False)

    for r in recs:
        existing = db.get(RecommendationDB, r["id"])
        if existing:
            # Update mutable fields in-place
            existing.timestamp = ts
            existing.action_text = r["action_text"]
            existing.expected_savings_kw = r.get("expected_savings_kw", 0.0)
            existing.pue_impact = r.get("pue_impact", 0.0)
            existing.confidence = r.get("confidence", 80)
            existing.is_active = True
        else:
            db.add(
                RecommendationDB(
                    id=r["id"],
                    timestamp=ts,
                    recommendation_type=r["recommendation_type"],
                    asset_id=r.get("asset_id"),
                    priority=r["priority"],
                    action_text=r["action_text"],
                    expected_savings_kw=r.get("expected_savings_kw", 0.0),
                    pue_impact=r.get("pue_impact", 0.0),
                    confidence=r.get("confidence", 80),
                    rationale=r["rationale"],
                    is_active=True,
                )
            )
            db.add(
                EventDB(
                    timestamp=ts,
                    event_type="recommendation_generated",
                    asset_id=r.get("asset_id"),
                    severity=r["priority"],
                    message=r["action_text"],
                    confidence=r.get("confidence"),
                )
            )


# ─────────────────────────────────────────
# Asset initialisation in SQLite
# ─────────────────────────────────────────

def seed_assets(inventory: AssetInventory) -> None:
    """Write asset definitions to the database if not already present."""
    db = SessionLocal()
    try:
        existing = db.query(AssetDB).count()
        if existing > 0:
            return  # already seeded

        rows: List[AssetDB] = []
        for r in inventory.racks:
            rows.append(
                AssetDB(
                    id=r.id,
                    name=r.name,
                    asset_type="rack",
                    room=r.room,
                    row=r.row,
                    position=r.position,
                    rated_kw=r.rated_kw,
                )
            )
        for c in inventory.cracs:
            rows.append(
                AssetDB(
                    id=c.id,
                    name=c.name,
                    asset_type="crac",
                    room=c.room,
                    rated_kw=c.rated_cooling_kw,
                )
            )
        for ch in inventory.chillers:
            rows.append(
                AssetDB(
                    id=ch.id,
                    name=ch.name,
                    asset_type="chiller",
                    rated_kw=ch.rated_cooling_kw,
                )
            )
        for u in inventory.upses:
            rows.append(
                AssetDB(id=u.id, name=u.name, asset_type="ups", rated_kw=u.rated_kw)
            )
        for p in inventory.pdus:
            rows.append(
                AssetDB(
                    id=p.id,
                    name=p.name,
                    asset_type="pdu",
                    room=p.room,
                    rated_kw=p.rated_kw,
                )
            )

        db.bulk_save_objects(rows)
        db.commit()
        logger.info("Seeded %d assets into database.", len(rows))
    finally:
        db.close()


# ─────────────────────────────────────────
# Background simulation thread
# ─────────────────────────────────────────

_simulator_instance: Optional[DataCenterSimulator] = None
_simulation_thread: Optional[threading.Thread] = None
_last_snapshot: Optional[TelemetrySnapshot] = None
_last_snapshot_lock = threading.Lock()


def get_simulator() -> DataCenterSimulator:
    """Return the global simulator instance, creating it if necessary."""
    global _simulator_instance
    if _simulator_instance is None:
        _simulator_instance = DataCenterSimulator()
    return _simulator_instance


def get_last_snapshot() -> Optional[TelemetrySnapshot]:
    """Return the most recently generated snapshot (thread-safe)."""
    with _last_snapshot_lock:
        return _last_snapshot


def _simulation_loop(interval: int) -> None:
    """Target function for the background simulation thread."""
    global _last_snapshot
    sim = get_simulator()
    logger.info("Simulation thread started (interval=%ds).", interval)
    while True:
        try:
            snapshot = sim.generate_snapshot()
            with _last_snapshot_lock:
                _last_snapshot = snapshot
            _persist_snapshot(snapshot)
            logger.debug("Cycle %d complete — IT load %.1f kW", snapshot.cycle_id, sum(r.power_kw for r in snapshot.racks))
        except Exception as exc:
            logger.error("Simulation cycle error: %s", exc, exc_info=True)
        time.sleep(interval)


def start_simulation(interval: int = 30) -> threading.Thread:
    """
    Initialise the database, seed assets, and start the background simulation.
    Returns the thread handle.
    """
    global _simulation_thread

    init_db()
    inventory = build_asset_inventory()
    seed_assets(inventory)

    # Run one immediate cycle so the API has data right away
    sim = get_simulator()
    try:
        snapshot = sim.generate_snapshot()
        global _last_snapshot
        with _last_snapshot_lock:
            _last_snapshot = snapshot
        _persist_snapshot(snapshot)
    except Exception as exc:
        logger.warning("Initial simulation cycle failed: %s", exc)

    thread = threading.Thread(
        target=_simulation_loop,
        args=(interval,),
        daemon=True,
        name="dc-twin-simulator",
    )
    thread.start()
    _simulation_thread = thread
    return thread
