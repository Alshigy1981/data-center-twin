"""
US commercial tariff simulation (PG&E E-19 / ConEd SC-9 style).

Engineering assumptions:
- Tariff based on PG&E E-19 and ConEd SC-9 commercial schedules (2024)
- Thermal mass time constant 45 min is typical for raised floor DC
- Pre-cooling effectiveness assumes 30% of facility load is HVAC
- Demand charges represent 30-50% of typical US commercial DC electricity bill
- Ratchet clause is standard in US commercial tariffs
"""

from __future__ import annotations

import calendar
import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List

# ─── Tariff constants ─────────────────────────────────────────────────────────

PEAK_DEMAND_CHARGE = 18.50         # $/kW/month
COINCIDENT_PEAK_CHARGE = 12.00    # $/kW/month
FACILITIES_CHARGE = 850.00        # $/month
RATCHET_FACTOR = 0.85             # 85% of 12-month rolling peak

# Energy rates ($/kWh)
_RATES = {
    "off_peak": 0.06,
    "shoulder_am": 0.12,   # 08:00–12:00
    "on_peak": 0.28,
    "shoulder_pm": 0.14,   # 18:00–21:00
}

# Carbon intensity (kgCO2/kWh) by hour range
_CARBON_SCHEDULE = [
    (0, 6, 0.18),    # 00:00–06:00 renewables
    (6, 10, 0.28),   # 06:00–10:00 ramping gas
    (10, 14, 0.22),  # 10:00–14:00 solar peak
    (14, 20, 0.31),  # 14:00–20:00 peak demand
    (20, 24, 0.25),  # 20:00–24:00 declining
]

# Grid peak events: 10 per month, weekdays 14:00–17:00
_GRID_PEAK_HOUR_START = 14
_GRID_PEAK_HOUR_END = 17
_GRID_PEAK_EVENTS_PER_MONTH = 10


@dataclass
class USCommercialTariff:
    """
    US commercial electricity tariff simulator.

    Models PG&E E-19 / ConEd SC-9 style commercial tariffs including:
    - Time-of-use energy rates
    - Monthly demand charges with ratchet clause
    - Grid peak event simulation
    - Carbon intensity by hour
    """

    def get_period_type(self, dt: datetime) -> str:
        """Return period type: on_peak / shoulder / off_peak."""
        if dt.weekday() >= 5:  # Saturday=5, Sunday=6
            return "off_peak"
        h = dt.hour
        if 0 <= h < 8 or 21 <= h < 24:
            return "off_peak"
        elif 8 <= h < 12:
            return "shoulder"
        elif 12 <= h < 18:
            return "on_peak"
        elif 18 <= h < 21:
            return "shoulder"
        return "off_peak"

    def get_current_rate(self, dt: datetime) -> float:
        """Return energy rate in $/kWh for the given datetime."""
        if dt.weekday() >= 5:
            return _RATES["off_peak"]
        h = dt.hour
        if 0 <= h < 8 or 21 <= h < 24:
            return _RATES["off_peak"]
        elif 8 <= h < 12:
            return _RATES["shoulder_am"]
        elif 12 <= h < 18:
            return _RATES["on_peak"]
        elif 18 <= h < 21:
            return _RATES["shoulder_pm"]
        return _RATES["off_peak"]

    def get_carbon_intensity(self, dt: datetime) -> float:
        """Return grid carbon intensity in kgCO2/kWh for the given datetime."""
        h = dt.hour
        for start, end, intensity in _CARBON_SCHEDULE:
            if start <= h < end:
                return intensity
        return 0.25  # fallback

    def is_grid_peak_event(self, dt: datetime) -> bool:
        """
        Determine if the given datetime falls within a grid peak event.

        Uses a deterministic approach: hashlib selects which 10 weekdays
        in the month are peak event days, making results reproducible.
        """
        if dt.weekday() >= 5:
            return False
        if not (_GRID_PEAK_HOUR_START <= dt.hour < _GRID_PEAK_HOUR_END):
            return False
        return dt.day in self._get_peak_event_days(dt.year, dt.month)

    def _get_peak_event_days(self, year: int, month: int) -> set:
        """Return the set of day-of-month numbers that are grid peak event days."""
        # Enumerate all weekdays in the month
        num_days = calendar.monthrange(year, month)[1]
        weekdays = [d for d in range(1, num_days + 1)
                    if datetime(year, month, d).weekday() < 5]
        if len(weekdays) <= _GRID_PEAK_EVENTS_PER_MONTH:
            return set(weekdays)

        # Use hashlib for deterministic selection
        seed_str = f"{year}-{month:02d}-grid-peak"
        seed_bytes = hashlib.sha256(seed_str.encode()).digest()
        # Use seed to deterministically shuffle and pick
        import random
        rng = random.Random(int.from_bytes(seed_bytes[:8], "big"))
        selected = rng.sample(weekdays, _GRID_PEAK_EVENTS_PER_MONTH)
        return set(selected)

    def get_rate_forecast_24h(self, from_dt: datetime) -> List[dict]:
        """
        Return 24 hourly rate dicts starting from from_dt (rounded to the hour).
        Keys: hour, dt_iso, rate_per_kwh, period_type, carbon_intensity, is_grid_peak
        """
        # Round to current hour
        start = from_dt.replace(minute=0, second=0, microsecond=0)
        result = []
        for i in range(24):
            dt = start + timedelta(hours=i)
            result.append({
                "hour": i,
                "dt_iso": dt.isoformat(),
                "rate_per_kwh": self.get_current_rate(dt),
                "period_type": self.get_period_type(dt),
                "carbon_intensity": self.get_carbon_intensity(dt),
                "is_grid_peak": self.is_grid_peak_event(dt),
            })
        return result

    def calculate_energy_cost(
        self,
        kw: float,
        start_dt: datetime,
        duration_hrs: float,
    ) -> float:
        """
        Calculate energy cost for running kw load from start_dt for duration_hrs.
        Sums cost hour by hour using the appropriate rate for each hour.
        """
        total = 0.0
        dt = start_dt.replace(minute=0, second=0, microsecond=0)
        hours_remaining = duration_hrs
        while hours_remaining > 0:
            fraction = min(1.0, hours_remaining)
            rate = self.get_current_rate(dt)
            total += kw * rate * fraction
            dt += timedelta(hours=1)
            hours_remaining -= 1.0
        return round(total, 4)

    def calculate_demand_charge(
        self,
        peak_kw: float,
        peak_12mo_kw: float = 0.0,
    ) -> dict:
        """
        Calculate demand charges including ratchet clause.

        Returns dict with:
          demand_charge, ratchet_demand_kw, ratchet_adjustment, billed_demand_kw
        """
        ratchet_demand_kw = RATCHET_FACTOR * peak_12mo_kw
        billed_demand_kw = max(peak_kw, ratchet_demand_kw)
        demand_charge = billed_demand_kw * PEAK_DEMAND_CHARGE
        ratchet_adjustment = max(0.0, (billed_demand_kw - peak_kw) * PEAK_DEMAND_CHARGE)
        return {
            "demand_charge": round(demand_charge, 2),
            "ratchet_demand_kw": round(ratchet_demand_kw, 2),
            "ratchet_adjustment": round(ratchet_adjustment, 2),
            "billed_demand_kw": round(billed_demand_kw, 2),
        }

    def calculate_monthly_bill(
        self,
        energy_kwh: float,
        peak_kw: float,
        month_str: str,
        peak_12mo_kw: float = 0.0,
    ) -> dict:
        """
        Calculate full monthly electricity bill.

        Args:
            energy_kwh: total energy consumed in kWh
            peak_kw: monthly peak demand in kW
            month_str: string like "2024-06"
            peak_12mo_kw: rolling 12-month peak for ratchet calculation

        Returns dict with all bill components and total.
        """
        # Use a representative mix of hours for energy cost
        # Approximate: weekdays have on-peak, shoulder, off-peak distribution
        # Simplified: use weighted average rate
        avg_rate = (
            8 * _RATES["off_peak"]       # 00–08 off-peak
            + 4 * _RATES["shoulder_am"]  # 08–12 shoulder
            + 6 * _RATES["on_peak"]      # 12–18 on-peak
            + 3 * _RATES["shoulder_pm"]  # 18–21 shoulder
            + 3 * _RATES["off_peak"]     # 21–24 off-peak
        ) / 24
        energy_cost = round(energy_kwh * avg_rate, 2)

        dc = self.calculate_demand_charge(peak_kw, peak_12mo_kw)
        demand_charge = dc["demand_charge"]
        ratchet_adjustment = dc["ratchet_adjustment"]
        billed_demand_kw = dc["billed_demand_kw"]

        coincident_peak_charge = round(billed_demand_kw * COINCIDENT_PEAK_CHARGE, 2)
        facilities_charge = FACILITIES_CHARGE

        total = round(
            energy_cost
            + demand_charge
            + coincident_peak_charge
            + facilities_charge,
            2,
        )

        return {
            "energy_cost": energy_cost,
            "demand_charge": demand_charge,
            "coincident_peak_charge": coincident_peak_charge,
            "facilities_charge": facilities_charge,
            "ratchet_adjustment": ratchet_adjustment,
            "total": total,
            "month": month_str,
        }

    def get_next_offpeak_window(self, from_dt: datetime) -> datetime:
        """Return the next datetime that is in an off-peak period."""
        dt = from_dt + timedelta(hours=1)
        dt = dt.replace(minute=0, second=0, microsecond=0)
        for _ in range(48):  # search within 48 hours
            if self.get_period_type(dt) == "off_peak":
                return dt
            dt += timedelta(hours=1)
        return dt

    def get_cheapest_n_hours(
        self,
        n: int,
        from_dt: datetime,
        within_hours: int = 24,
    ) -> List[datetime]:
        """
        Return the n cheapest hours within the next within_hours hours.
        Returns list of datetime objects (start of each hour).
        """
        start = from_dt.replace(minute=0, second=0, microsecond=0)
        candidates = []
        for i in range(within_hours):
            dt = start + timedelta(hours=i)
            candidates.append((self.get_current_rate(dt), dt))
        candidates.sort(key=lambda x: x[0])
        return [dt for _, dt in candidates[:n]]


# ─── Module-level singleton ───────────────────────────────────────────────────

tariff = USCommercialTariff()
