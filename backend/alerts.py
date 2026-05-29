"""
Alert detection engine for the Data Center Digital Twin Platform.

Each detector returns a dict or None — plain data structures so they can
be called from tests without any database or API context.

Confidence scoring model
─────────────────────────
  Base score (0–60):   scaled from how far the reading deviates from its
                       threshold, capped at the threshold × 2.
  Duration bonus (30): +5 per consecutive cycle active, max 30.
  Rate-of-change (10): +10 if the value is still moving in the bad direction.
  Final score is clamped to [0, 100].
"""

from __future__ import annotations

import hashlib
import math
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from backend.models import AlertSeverity, TelemetrySnapshot

# ─────────────────────────────────────────
# Thresholds (engineering constants)
# ─────────────────────────────────────────

RACK_OVERLOAD_WARNING_KW = 18.0
RACK_OVERLOAD_CRITICAL_KW = 20.0

PDU_WARNING_PCT = 80.0
PDU_CRITICAL_PCT = 90.0
PDU_RATED_KW = 100.0

UPS_WARNING_PCT = 80.0
UPS_CRITICAL_PCT = 90.0

POWER_SPIKE_PCT = 15.0      # % increase in one cycle that triggers spike alert
PUE_HIGH_THRESHOLD = 1.8
PUE_HIGH_CONSECUTIVE = 3    # cycles before PUE alert fires
PDU_PHASE_IMBALANCE_PCT = 10.0  # max allowed deviation of any phase from mean


# ─────────────────────────────────────────
# Confidence score calculation
# ─────────────────────────────────────────

def calculate_confidence_score(
    deviation: float,
    threshold: float,
    duration_cycles: int = 1,
    is_worsening: bool = False,
) -> int:
    """
    Compute an alert confidence score in the range [0, 100].

    Args:
        deviation:       Absolute difference between current value and threshold.
        threshold:       The threshold value that was crossed.
        duration_cycles: How many consecutive cycles the condition has been true.
        is_worsening:    True if the value is still moving away from safe range.

    Returns:
        Integer confidence score 0–100.
    """
    # Base: proportion of deviation relative to the threshold magnitude
    # Full base score (60) when deviation equals or exceeds the threshold itself
    if threshold == 0:
        base = 60.0
    else:
        base = min(60.0, 60.0 * (deviation / abs(threshold)))

    duration_bonus = min(30.0, (duration_cycles - 1) * 5.0)
    rate_bonus = 10.0 if is_worsening else 0.0

    score = base + duration_bonus + rate_bonus
    # Clamp strictly to [0, 100] — large deviations should reach 100
    return int(max(0, min(100, math.ceil(score))))


# ─────────────────────────────────────────
# Alert ID helper
# ─────────────────────────────────────────

def _alert_id(asset_id: str, metric: str) -> str:
    """Deterministic ID from asset + metric — so upserts work correctly."""
    key = f"{asset_id}::{metric}"
    return hashlib.md5(key.encode()).hexdigest()[:16]


# ─────────────────────────────────────────
# Individual detectors
# ─────────────────────────────────────────

def detect_rack_overload(
    rack_id: str,
    power_kw: float,
    prev_power_kw: Optional[float] = None,
    duration_cycles: int = 1,
) -> Optional[Dict]:
    """
    Detect rack power overload (warning at 18 kW, critical at 20 kW).

    Args:
        rack_id:        Rack asset identifier.
        power_kw:       Current rack power draw in kW.
        prev_power_kw:  Previous cycle power for rate-of-change detection.
        duration_cycles: Consecutive cycles this condition has been true.

    Returns:
        Alert dict or None if no condition.
    """
    if power_kw >= RACK_OVERLOAD_CRITICAL_KW:
        threshold = RACK_OVERLOAD_CRITICAL_KW
        severity = AlertSeverity.CRITICAL
        message = f"Rack {rack_id} critical overload: {power_kw:.2f} kW (limit {threshold:.0f} kW)"
    elif power_kw >= RACK_OVERLOAD_WARNING_KW:
        threshold = RACK_OVERLOAD_WARNING_KW
        severity = AlertSeverity.WARNING
        message = f"Rack {rack_id} overload warning: {power_kw:.2f} kW (limit {threshold:.0f} kW)"
    else:
        return None

    deviation = power_kw - threshold
    is_worsening = (prev_power_kw is not None and power_kw > prev_power_kw)
    confidence = calculate_confidence_score(deviation, threshold, duration_cycles, is_worsening)

    return {
        "id": _alert_id(rack_id, "rack_overload"),
        "asset_id": rack_id,
        "metric": "power_kw",
        "severity": severity.value,
        "message": message,
        "confidence": confidence,
        "explanation": (
            f"Rack power {power_kw:.2f} kW exceeds {'critical' if severity == AlertSeverity.CRITICAL else 'warning'} "
            f"threshold of {threshold:.0f} kW.  "
            f"Sustained for {duration_cycles} cycle(s)."
        ),
        "duration_cycles": duration_cycles,
        "deviation": round(deviation, 3),
        "baseline_value": threshold,
        "current_value": round(power_kw, 3),
    }


def detect_pdu_overload(
    pdu_id: str,
    load_kw: float,
    rated_kw: float = PDU_RATED_KW,
    duration_cycles: int = 1,
) -> Optional[Dict]:
    """
    Detect PDU overload: warning at 80 % rated, critical at 90 % rated.
    """
    load_pct = (load_kw / rated_kw) * 100 if rated_kw > 0 else 0

    if load_pct >= PDU_CRITICAL_PCT:
        threshold_pct = PDU_CRITICAL_PCT
        severity = AlertSeverity.CRITICAL
    elif load_pct >= PDU_WARNING_PCT:
        threshold_pct = PDU_WARNING_PCT
        severity = AlertSeverity.WARNING
    else:
        return None

    threshold_kw = rated_kw * threshold_pct / 100
    deviation = load_kw - threshold_kw
    confidence = calculate_confidence_score(deviation, threshold_kw, duration_cycles)

    return {
        "id": _alert_id(pdu_id, "pdu_overload"),
        "asset_id": pdu_id,
        "metric": "load_pct",
        "severity": severity.value,
        "message": f"PDU {pdu_id} load {load_pct:.1f} % of rated capacity",
        "confidence": confidence,
        "explanation": (
            f"PDU total load {load_kw:.2f} kW = {load_pct:.1f} % of rated {rated_kw:.0f} kW.  "
            f"Threshold: {threshold_pct:.0f} % ({threshold_kw:.0f} kW)."
        ),
        "duration_cycles": duration_cycles,
        "deviation": round(deviation, 3),
        "baseline_value": round(threshold_kw, 1),
        "current_value": round(load_kw, 3),
    }


def detect_ups_high_load(
    ups_id: str,
    load_pct: float,
    duration_cycles: int = 1,
) -> Optional[Dict]:
    """
    Detect UPS high load: warning at 80 %, critical at 90 %.
    """
    if load_pct >= UPS_CRITICAL_PCT:
        threshold = UPS_CRITICAL_PCT
        severity = AlertSeverity.CRITICAL
    elif load_pct >= UPS_WARNING_PCT:
        threshold = UPS_WARNING_PCT
        severity = AlertSeverity.WARNING
    else:
        return None

    deviation = load_pct - threshold
    confidence = calculate_confidence_score(deviation, threshold, duration_cycles)

    return {
        "id": _alert_id(ups_id, "ups_high_load"),
        "asset_id": ups_id,
        "metric": "load_pct",
        "severity": severity.value,
        "message": f"UPS {ups_id} load at {load_pct:.1f} % of rated capacity",
        "confidence": confidence,
        "explanation": (
            f"UPS load {load_pct:.1f} % exceeds {'critical' if severity == AlertSeverity.CRITICAL else 'warning'} "
            f"threshold of {threshold:.0f} %.  Risk of UPS transfer to bypass if overloaded further."
        ),
        "duration_cycles": duration_cycles,
        "deviation": round(deviation, 2),
        "baseline_value": threshold,
        "current_value": round(load_pct, 2),
    }


def detect_power_spike(
    asset_id: str,
    current_kw: float,
    prev_kw: float,
    spike_pct: float = POWER_SPIKE_PCT,
    duration_cycles: int = 1,
) -> Optional[Dict]:
    """
    Detect abnormal power spike: >spike_pct% increase in one cycle.
    """
    if prev_kw <= 0:
        return None
    increase_pct = (current_kw - prev_kw) / prev_kw * 100
    if increase_pct <= spike_pct:
        return None

    deviation = current_kw - prev_kw
    confidence = calculate_confidence_score(deviation, prev_kw * spike_pct / 100, duration_cycles, True)

    return {
        "id": _alert_id(asset_id, "power_spike"),
        "asset_id": asset_id,
        "metric": "power_kw",
        "severity": AlertSeverity.WARNING.value,
        "message": f"{asset_id} power spike: +{increase_pct:.1f} % in one cycle",
        "confidence": confidence,
        "explanation": (
            f"Power draw rose from {prev_kw:.2f} kW to {current_kw:.2f} kW "
            f"(+{increase_pct:.1f} %) in a single simulation cycle.  "
            f"Threshold: {spike_pct:.0f} %."
        ),
        "duration_cycles": duration_cycles,
        "deviation": round(deviation, 3),
        "baseline_value": round(prev_kw, 3),
        "current_value": round(current_kw, 3),
    }


def detect_pue_trend(
    pue_value: float,
    consecutive_high: int,
    threshold: float = PUE_HIGH_THRESHOLD,
    required_consecutive: int = PUE_HIGH_CONSECUTIVE,
) -> Optional[Dict]:
    """
    Fire alert when PUE exceeds threshold for required_consecutive cycles.
    """
    if pue_value < threshold or consecutive_high < required_consecutive:
        return None

    deviation = pue_value - threshold
    confidence = calculate_confidence_score(deviation, threshold, consecutive_high)

    return {
        "id": _alert_id("facility", "pue_high"),
        "asset_id": "facility",
        "metric": "pue",
        "severity": AlertSeverity.WARNING.value,
        "message": f"PUE {pue_value:.3f} above {threshold:.1f} for {consecutive_high} consecutive cycles",
        "confidence": confidence,
        "explanation": (
            f"Facility PUE {pue_value:.3f} has exceeded the {threshold:.1f} threshold "
            f"for {consecutive_high} consecutive readings.  "
            f"Investigate cooling plant efficiency and airflow containment."
        ),
        "duration_cycles": consecutive_high,
        "deviation": round(deviation, 4),
        "baseline_value": threshold,
        "current_value": round(pue_value, 4),
    }


def detect_phase_imbalance(
    pdu_id: str,
    phase_a_kw: float,
    phase_b_kw: float,
    phase_c_kw: float,
    threshold_pct: float = PDU_PHASE_IMBALANCE_PCT,
    duration_cycles: int = 1,
) -> Optional[Dict]:
    """
    Flag phase imbalance when any PDU phase deviates more than threshold_pct
    from the three-phase mean.
    """
    phases = [phase_a_kw, phase_b_kw, phase_c_kw]
    mean_kw = sum(phases) / 3
    if mean_kw <= 0:
        return None

    max_deviation_pct = max(abs(p - mean_kw) / mean_kw * 100 for p in phases)
    if max_deviation_pct < threshold_pct:
        return None

    worst_phase = max(phases, key=lambda p: abs(p - mean_kw))
    deviation = abs(worst_phase - mean_kw)
    confidence = calculate_confidence_score(
        max_deviation_pct, threshold_pct, duration_cycles
    )

    return {
        "id": _alert_id(pdu_id, "phase_imbalance"),
        "asset_id": pdu_id,
        "metric": "phase_imbalance_pct",
        "severity": AlertSeverity.WARNING.value,
        "message": (
            f"PDU {pdu_id} phase imbalance: {max_deviation_pct:.1f} % "
            f"(threshold {threshold_pct:.0f} %)"
        ),
        "confidence": confidence,
        "explanation": (
            f"Phase loads (A={phase_a_kw:.2f}, B={phase_b_kw:.2f}, C={phase_c_kw:.2f} kW).  "
            f"Max deviation {max_deviation_pct:.1f} % from mean {mean_kw:.2f} kW.  "
            f"Rebalance PDU circuits to reduce neutral current."
        ),
        "duration_cycles": duration_cycles,
        "deviation": round(deviation, 3),
        "baseline_value": round(mean_kw, 3),
        "current_value": round(worst_phase, 3),
    }


# ─────────────────────────────────────────
# Full alert scan across a snapshot
# ─────────────────────────────────────────

# Module-level state for consecutive-cycle tracking (in-process only)
_pue_high_count: int = 0
_prev_powers: Dict[str, float] = {}
_alert_durations: Dict[str, int] = {}


def detect_all_alerts(snapshot: TelemetrySnapshot) -> List[Dict]:
    """
    Run all detectors against a complete telemetry snapshot and return
    a list of active alert dicts.

    Side-effects: updates module-level PUE streak counter and prev-power map.
    """
    global _pue_high_count, _prev_powers, _alert_durations

    results: List[Dict] = []

    # ── Rack overload and power spike ─────
    for rack in snapshot.racks:
        prev = _prev_powers.get(rack.asset_id)
        dur_key = _alert_id(rack.asset_id, "rack_overload")
        dur = _alert_durations.get(dur_key, 0)

        a = detect_rack_overload(rack.asset_id, rack.power_kw, prev, dur + 1)
        if a:
            _alert_durations[dur_key] = dur + 1
            results.append(a)
        else:
            _alert_durations.pop(dur_key, None)

        if prev is not None:
            spike_key = _alert_id(rack.asset_id, "power_spike")
            s = detect_power_spike(rack.asset_id, rack.power_kw, prev)
            if s:
                results.append(s)

        _prev_powers[rack.asset_id] = rack.power_kw

    # ── PDU overload and phase imbalance ──
    for pdu in snapshot.pdus:
        dur_key = _alert_id(pdu.asset_id, "pdu_overload")
        dur = _alert_durations.get(dur_key, 0)

        a = detect_pdu_overload(pdu.asset_id, pdu.total_kw, PDU_RATED_KW, dur + 1)
        if a:
            _alert_durations[dur_key] = dur + 1
            results.append(a)
        else:
            _alert_durations.pop(dur_key, None)

        pi = detect_phase_imbalance(
            pdu.asset_id, pdu.phase_a_kw, pdu.phase_b_kw, pdu.phase_c_kw
        )
        if pi:
            results.append(pi)

    # ── UPS overload ──────────────────────
    for ups in snapshot.upses:
        dur_key = _alert_id(ups.asset_id, "ups_high_load")
        dur = _alert_durations.get(dur_key, 0)

        a = detect_ups_high_load(ups.asset_id, ups.load_pct, dur + 1)
        if a:
            _alert_durations[dur_key] = dur + 1
            results.append(a)
        else:
            _alert_durations.pop(dur_key, None)

    # ── PUE trend ─────────────────────────
    from backend.metrics import calculate_pue
    it_load = sum(r.power_kw for r in snapshot.racks)
    cooling_load = sum(c.cooling_load_kw for c in snapshot.cracs)
    pue = calculate_pue(it_load, cooling_load)

    if pue >= PUE_HIGH_THRESHOLD:
        _pue_high_count += 1
    else:
        _pue_high_count = 0

    a = detect_pue_trend(pue, _pue_high_count)
    if a:
        results.append(a)

    return results
