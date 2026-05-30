"""
Tests for WorkloadPlacementOptimizer (backend/workload_placement.py).

All tests use MagicMock snapshots simulating 80 racks.
"""

from __future__ import annotations

import unittest
from datetime import datetime
from unittest.mock import MagicMock

from backend.workload_placement import WorkloadPlacementOptimizer


# ─── Mock helpers ─────────────────────────────────────────────────────────────


def _make_rack(
    asset_id: str,
    inlet_temp_c: float = 20.0,
    outlet_temp_c: float = 30.0,
    power_kw: float = 10.0,
    room: str = "A",
    row: str = "Row1",
) -> MagicMock:
    rack = MagicMock()
    rack.asset_id = asset_id
    rack.inlet_temp_c = inlet_temp_c
    rack.outlet_temp_c = outlet_temp_c
    rack.power_kw = power_kw
    rack.room = room
    rack.row = row
    return rack


def _make_pdu(
    asset_id: str = "PDU-1",
    phase_a_kw: float = 20.0,
    phase_b_kw: float = 20.0,
    phase_c_kw: float = 20.0,
    total_kw: float = 60.0,
    room: str = "A",
) -> MagicMock:
    pdu = MagicMock()
    pdu.asset_id = asset_id
    pdu.phase_a_kw = phase_a_kw
    pdu.phase_b_kw = phase_b_kw
    pdu.phase_c_kw = phase_c_kw
    pdu.total_kw = total_kw
    pdu.room = room
    return pdu


def _make_ups(
    asset_id: str = "UPS-1",
    output_kw: float = 100.0,
) -> MagicMock:
    ups = MagicMock()
    ups.asset_id = asset_id
    ups.output_kw = output_kw
    return ups


def _make_snapshot_80(
    inlet_temp_c: float = 20.0,
    power_kw: float = 10.0,
) -> MagicMock:
    """Create a snapshot with 80 racks (2 rooms x 4 rows x 10 racks)."""
    racks = []
    rooms = ["A", "B"]
    for room in rooms:
        for row_num in range(1, 5):
            for pos in range(1, 11):
                rack_id = f"{room}-R{row_num}-{pos:03d}"
                racks.append(_make_rack(
                    rack_id,
                    inlet_temp_c=inlet_temp_c,
                    outlet_temp_c=inlet_temp_c + 12.0,
                    power_kw=power_kw,
                    room=room,
                ))

    pdus = [_make_pdu(f"{room}-PDU-{i}", room=room) for room in rooms for i in range(1, 5)]
    upses = [_make_ups(f"UPS-{i}") for i in range(1, 3)]

    snap = MagicMock()
    snap.racks = racks
    snap.pdus = pdus
    snap.upses = upses
    snap.cracs = []
    return snap


# ─── Test cases ───────────────────────────────────────────────────────────────


class TestWorkloadPlacementOptimizer(unittest.TestCase):

    def setUp(self):
        self.optimizer = WorkloadPlacementOptimizer()

    # ── Basic placement tests ─────────────────────────────────────────────────

    def test_placement_recommends_valid_rack(self):
        """recommend_placement returns a rack_id that exists in the 80 racks."""
        snap = _make_snapshot_80(inlet_temp_c=20.0, power_kw=10.0)
        result = self.optimizer.recommend_placement(5.0, snap)
        rack_ids = {r.asset_id for r in snap.racks}
        self.assertIsNotNone(result["recommended_rack_id"])
        self.assertIn(result["recommended_rack_id"], rack_ids)

    def test_placement_respects_capacity_constraint(self):
        """All racks at 19 kW + 2 kW workload = 21 kW > 20 kW limit → no placement."""
        snap = _make_snapshot_80(inlet_temp_c=20.0, power_kw=19.0)
        result = self.optimizer.recommend_placement(2.0, snap)
        self.assertIsNone(result["recommended_rack_id"])

    def test_placement_respects_ashrae_limit(self):
        """Rack at 34.5°C inlet + 4 kW * 0.15 = 35.1°C > 35°C limit → infeasible."""
        snap = _make_snapshot_80(inlet_temp_c=34.5, power_kw=5.0)
        result = self.optimizer.recommend_placement(4.0, snap)
        # projected_inlet = 34.5 + 4 * 0.15 = 35.1 > 35.0 → all infeasible
        self.assertIsNone(result["recommended_rack_id"])

    def test_placement_prefers_cool_racks(self):
        """Optimizer should prefer racks with lower inlet temps."""
        snap = MagicMock()
        snap.pdus = []
        snap.upses = []
        snap.cracs = []

        # Mix of cool (18°C) and hot (28°C) racks
        cool_racks = [_make_rack(f"COOL-{i}", inlet_temp_c=18.0, power_kw=8.0) for i in range(40)]
        hot_racks = [_make_rack(f"HOT-{i}", inlet_temp_c=28.0, power_kw=8.0) for i in range(40)]
        snap.racks = cool_racks + hot_racks

        result = self.optimizer.recommend_placement(3.0, snap)
        self.assertIsNotNone(result["recommended_rack_id"])
        # The recommended rack should be a cool rack
        self.assertTrue(result["recommended_rack_id"].startswith("COOL-"))

    # ── Score function tests ──────────────────────────────────────────────────

    def test_thermal_risk_score_safe_range(self):
        """Inlet 20°C + 5 kW * 0.15 = 20.75°C <= 24°C → risk = 0."""
        rack = _make_rack("R1", inlet_temp_c=20.0)
        score = self.optimizer._thermal_risk_score(rack, 5.0)
        self.assertEqual(score, 0.0)

    def test_thermal_risk_score_ashrae_breach(self):
        """Inlet 30°C + 20 kW * 0.15 = 33°C > 32°C → risk > 1.0."""
        rack = _make_rack("R1", inlet_temp_c=30.0)
        score = self.optimizer._thermal_risk_score(rack, 20.0)
        self.assertGreater(score, 1.0)

    def test_thermal_risk_score_boundary_24(self):
        """Inlet 24°C + 0 kW → projected = 24°C → risk = 0."""
        rack = _make_rack("R1", inlet_temp_c=24.0)
        score = self.optimizer._thermal_risk_score(rack, 0.0)
        self.assertEqual(score, 0.0)

    def test_thermal_risk_score_in_warning_range(self):
        """Inlet 24°C + 4 kW * 0.15 = 24.6°C → 0 < risk < 1."""
        rack = _make_rack("R1", inlet_temp_c=24.0)
        score = self.optimizer._thermal_risk_score(rack, 4.0)
        # projected = 24.6, in (24, 32] range → (24.6-24)/8 = 0.075
        self.assertAlmostEqual(score, (24.6 - 24.0) / 8.0, places=5)
        self.assertGreater(score, 0.0)
        self.assertLess(score, 1.0)

    def test_phase_imbalance_calculation(self):
        """Phase imbalance score is non-negative."""
        pdu = _make_pdu(phase_a_kw=30.0, phase_b_kw=20.0, phase_c_kw=25.0, total_kw=75.0)
        score = self.optimizer._phase_imbalance_score(0, 5.0, [pdu])
        self.assertGreaterEqual(score, 0.0)

    # ── Migration tests ───────────────────────────────────────────────────────

    def test_migration_finds_candidates(self):
        """Hot rack snapshot returns non-empty migration candidates."""
        snap = MagicMock()
        snap.pdus = []
        snap.upses = []
        snap.cracs = []
        # Mix: 10 hot racks (28°C) + 70 cool racks (20°C)
        hot_racks = [_make_rack(f"HOT-{i}", inlet_temp_c=28.0, power_kw=14.0) for i in range(10)]
        cool_racks = [_make_rack(f"COOL-{i}", inlet_temp_c=18.0, power_kw=5.0) for i in range(70)]
        snap.racks = hot_racks + cool_racks

        candidates = self.optimizer.find_migration_candidates(snap)
        self.assertIsInstance(candidates, list)
        self.assertGreater(len(candidates), 0)

    def test_migration_benefit_score_positive(self):
        """Migration benefit_score should be >= 0."""
        snap = MagicMock()
        snap.pdus = []
        snap.upses = []
        snap.cracs = []
        hot_racks = [_make_rack(f"HOT-{i}", inlet_temp_c=28.0, power_kw=12.0) for i in range(5)]
        cool_racks = [_make_rack(f"COOL-{i}", inlet_temp_c=18.0, power_kw=4.0) for i in range(75)]
        snap.racks = hot_racks + cool_racks

        candidates = self.optimizer.find_migration_candidates(snap)
        for c in candidates:
            self.assertGreaterEqual(c["benefit_score"], 0.0)

    def test_migration_risk_levels(self):
        """Migration risk levels are LOW/MEDIUM/HIGH based on power."""
        snap = MagicMock()
        snap.pdus = []
        snap.upses = []
        snap.cracs = []
        # 8 kW rack → LOW, 12 kW → MEDIUM, 17 kW → HIGH
        racks = (
            [_make_rack(f"LOW-{i}", inlet_temp_c=28.0, power_kw=8.0) for i in range(3)]
            + [_make_rack(f"COOL-{i}", inlet_temp_c=18.0, power_kw=4.0) for i in range(77)]
        )
        snap.racks = racks

        candidates = self.optimizer.find_migration_candidates(snap)
        for c in candidates:
            self.assertIn(c["migration_risk"], ("LOW", "MEDIUM", "HIGH"))

    # ── Infeasibility tests ───────────────────────────────────────────────────

    def test_infeasible_when_all_racks_full(self):
        """All racks at 19.5 kW + 2 kW = 21.5 kW > 20 kW → no valid placement."""
        snap = _make_snapshot_80(inlet_temp_c=20.0, power_kw=19.5)
        result = self.optimizer.recommend_placement(2.0, snap)
        self.assertIsNone(result["recommended_rack_id"])

    def test_placement_feasibility_result_structure(self):
        """Result always has required keys."""
        snap = _make_snapshot_80()
        result = self.optimizer.recommend_placement(5.0, snap)
        for key in ["recommended_rack_id", "workload_kw", "thermal_risk_score",
                    "projected_inlet_temp", "overall_score", "rationale", "alternatives"]:
            self.assertIn(key, result)

    # ── Weight tests ──────────────────────────────────────────────────────────

    def test_placement_weights_sum_to_one(self):
        """set_weights should reject weights that don't sum to 1.0."""
        with self.assertRaises(ValueError):
            self.optimizer.set_weights(0.5, 0.5, 0.5, 0.5)

    def test_placement_weights_accept_valid(self):
        """set_weights should accept weights that sum to 1.0."""
        self.optimizer.set_weights(0.4, 0.3, 0.2, 0.1)
        self.assertAlmostEqual(self.optimizer.w1, 0.4)
        self.assertAlmostEqual(self.optimizer.w2, 0.3)
        self.assertAlmostEqual(self.optimizer.w3, 0.2)
        self.assertAlmostEqual(self.optimizer.w4, 0.1)

    # ── score_all_racks tests ─────────────────────────────────────────────────

    def test_score_all_racks_returns_80(self):
        """score_all_racks returns exactly 80 entries for an 80-rack snapshot."""
        snap = _make_snapshot_80()
        scores = self.optimizer.score_all_racks(5.0, snap)
        self.assertEqual(len(scores), 80)

    def test_score_all_racks_has_required_fields(self):
        """Each score dict has all required fields."""
        snap = _make_snapshot_80()
        scores = self.optimizer.score_all_racks(5.0, snap)
        required = ["rack_id", "overall_score", "thermal_risk", "pue_impact",
                    "stranded_loss", "phase_impact", "feasible", "infeasibility_reason"]
        for score in scores[:5]:  # check first 5
            for field in required:
                self.assertIn(field, score)

    def test_score_feasible_racks_have_none_infeasibility_reason(self):
        """Feasible racks have infeasibility_reason=None."""
        snap = _make_snapshot_80(inlet_temp_c=20.0, power_kw=5.0)
        scores = self.optimizer.score_all_racks(3.0, snap)
        feasible_scores = [s for s in scores if s["feasible"]]
        self.assertGreater(len(feasible_scores), 0)
        for s in feasible_scores:
            self.assertIsNone(s["infeasibility_reason"])

    # ── Thermal preview tests ──────────────────────────────────────────────────

    def test_thermal_preview_returns_dict(self):
        """get_thermal_preview returns a dict for a valid rack_id."""
        snap = _make_snapshot_80(inlet_temp_c=20.0, power_kw=8.0)
        rack_id = snap.racks[0].asset_id
        preview = self.optimizer.get_thermal_preview(rack_id, 3.0, snap)
        self.assertIsNotNone(preview)
        self.assertIsInstance(preview, dict)

    def test_thermal_preview_returns_none_for_invalid_rack(self):
        """get_thermal_preview returns None for an unknown rack_id."""
        snap = _make_snapshot_80()
        preview = self.optimizer.get_thermal_preview("NONEXISTENT-RACK", 3.0, snap)
        self.assertIsNone(preview)

    def test_thermal_preview_has_required_fields(self):
        """Thermal preview contains all required fields."""
        snap = _make_snapshot_80(inlet_temp_c=22.0, power_kw=8.0)
        rack_id = snap.racks[5].asset_id
        preview = self.optimizer.get_thermal_preview(rack_id, 4.0, snap)
        required = ["rack_id", "workload_kw", "before_inlet_c", "before_outlet_c",
                    "before_ashrae_class", "after_inlet_c", "after_outlet_c",
                    "after_ashrae_class", "thermal_risk_score", "risk_level"]
        for field in required:
            self.assertIn(field, preview)

    def test_thermal_preview_shows_temperature_increase(self):
        """After adding load, inlet and outlet temperatures should increase."""
        snap = _make_snapshot_80(inlet_temp_c=20.0, power_kw=8.0)
        rack_id = snap.racks[0].asset_id
        preview = self.optimizer.get_thermal_preview(rack_id, 5.0, snap)
        self.assertGreater(preview["after_inlet_c"], preview["before_inlet_c"])
        self.assertGreater(preview["after_outlet_c"], preview["before_outlet_c"])

    def test_thermal_preview_risk_levels(self):
        """risk_level is one of safe/warning/critical."""
        snap = _make_snapshot_80(inlet_temp_c=20.0, power_kw=5.0)
        rack_id = snap.racks[0].asset_id
        preview = self.optimizer.get_thermal_preview(rack_id, 2.0, snap)
        self.assertIn(preview["risk_level"], ("safe", "warning", "critical"))

    def test_thermal_preview_safe_for_low_load(self):
        """Low inlet (18°C) + small workload should be safe."""
        snap = _make_snapshot_80(inlet_temp_c=18.0, power_kw=5.0)
        rack_id = snap.racks[0].asset_id
        preview = self.optimizer.get_thermal_preview(rack_id, 1.0, snap)
        self.assertEqual(preview["risk_level"], "safe")

    # ── Alternatives tests ─────────────────────────────────────────────────────

    def test_alternatives_are_feasible(self):
        """All returned alternatives should be feasible racks."""
        snap = _make_snapshot_80(inlet_temp_c=20.0, power_kw=8.0)
        result = self.optimizer.recommend_placement(3.0, snap)
        for alt in result.get("alternatives", []):
            self.assertTrue(alt["feasible"])

    def test_alternatives_max_5(self):
        """At most 5 alternatives returned."""
        snap = _make_snapshot_80(inlet_temp_c=20.0, power_kw=8.0)
        result = self.optimizer.recommend_placement(3.0, snap)
        self.assertLessEqual(len(result.get("alternatives", [])), 5)

    # ── Singleton test ─────────────────────────────────────────────────────────

    def test_module_singleton(self):
        """Module-level placement_optimizer singleton should be accessible."""
        from backend.workload_placement import placement_optimizer
        self.assertIsInstance(placement_optimizer, WorkloadPlacementOptimizer)


if __name__ == "__main__":
    unittest.main()
