"""
Unit tests for backend/rl_agent.py

Tests the Gymnasium environment and MaintenanceAgent wrapper.
No database or API required.
"""

import numpy as np
import pytest

from backend.rl_agent import (
    N_ASSETS, N_CRACS, N_CHILLERS, N_UPS,
    DataCenterMaintenanceEnv,
    MaintenanceAgent,
    _ASSET_IDS,
    _ASSET_TYPES,
)


# ─── Environment ──────────────────────────────────────────────────────────────

class TestDataCenterMaintenanceEnv:
    def setup_method(self):
        self.env = DataCenterMaintenanceEnv()

    def test_observation_space_shape(self):
        assert self.env.observation_space.shape == (N_ASSETS * 4,)

    def test_action_space_size(self):
        assert self.env.action_space.n == N_ASSETS

    def test_reset_returns_valid_obs(self):
        obs, info = self.env.reset(seed=42)
        assert obs.shape == (N_ASSETS * 4,)
        assert obs.dtype == np.float32
        assert (obs >= 0.0).all() and (obs <= 1.0).all()

    def test_step_returns_correct_shapes(self):
        self.env.reset(seed=0)
        action = self.env.action_space.sample()
        obs, reward, done, truncated, info = self.env.step(action)
        assert obs.shape == (N_ASSETS * 4,)
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert not truncated
        assert "failures" in info and "maintained" in info and "day" in info

    def test_episode_terminates_after_30_days(self):
        self.env.reset(seed=1)
        action = np.zeros(N_ASSETS, dtype=np.int8)
        done = False
        steps = 0
        while not done:
            _, _, done, _, _ = self.env.step(action)
            steps += 1
            assert steps <= 31, "Episode should terminate within 30 steps"
        assert steps == 30

    def test_maintenance_resets_health(self):
        self.env.reset(seed=5)
        # Drive health down manually
        for i in range(N_ASSETS):
            self.env._state[i, 0] = 0.1
        action = np.ones(N_ASSETS, dtype=np.int8)
        self.env.step(action)
        # After maintenance, health should be 1.0 (before degradation this step is applied first)
        # The env degrades then applies maintenance, so we check after step
        # Health at end = 1.0 (maintained) unless failures happened
        for i in range(N_ASSETS):
            assert self.env._state[i, 0] >= 0.7, f"Asset {i} health should be ≥ 0.7 post-maintenance"

    def test_no_maintenance_degrades_health(self):
        self.env.reset(seed=7)
        initial_health = self.env._state[:, 0].copy()
        action = np.zeros(N_ASSETS, dtype=np.int8)
        self.env.step(action)
        avg_after = self.env._state[:, 0].mean()
        avg_before = initial_health.mean()
        assert avg_after < avg_before, "Health should degrade without maintenance"

    def test_reward_penalises_failures(self):
        """Leaving very unhealthy assets unmaintained should produce a worse reward."""
        env_sick = DataCenterMaintenanceEnv()
        env_sick.reset(seed=99)
        for i in range(N_ASSETS):
            env_sick._state[i, 0] = 0.01  # near-certain failure

        action_none = np.zeros(N_ASSETS, dtype=np.int8)
        action_all = np.ones(N_ASSETS, dtype=np.int8)

        # Run several steps; maintained env should accumulate better reward
        reward_none = sum(env_sick.step(action_none)[1] for _ in range(3))

        env_cared = DataCenterMaintenanceEnv()
        env_cared.reset(seed=99)
        for i in range(N_ASSETS):
            env_cared._state[i, 0] = 0.01
        reward_all = sum(env_cared.step(action_all)[1] for _ in range(3))

        assert reward_all > reward_none, "Maintaining sick assets should yield better cumulative reward"

    def test_risk_summary_length(self):
        self.env.reset()
        summary = self.env.risk_summary()
        assert len(summary) == N_ASSETS

    def test_risk_summary_fields(self):
        self.env.reset()
        for entry in self.env.risk_summary():
            for key in ["asset_id", "asset_type", "health_score", "failure_probability_30d",
                        "maintenance_recommended", "anomaly_score"]:
                assert key in entry, f"Missing field: {key}"
            assert 0.0 <= entry["health_score"] <= 1.0
            assert 0.0 <= entry["failure_probability_30d"] <= 1.0

    def test_asset_ids_match_constants(self):
        summary = self.env.risk_summary()
        ids = [e["asset_id"] for e in summary]
        assert ids == _ASSET_IDS

    def test_asset_types_match_constants(self):
        summary = self.env.risk_summary()
        types = [e["asset_type"] for e in summary]
        assert types == _ASSET_TYPES

    def test_n_assets_constant(self):
        assert N_ASSETS == N_CRACS + N_CHILLERS + N_UPS
        assert N_ASSETS == 12


# ─── MaintenanceAgent ─────────────────────────────────────────────────────────

class TestMaintenanceAgent:
    def setup_method(self):
        self.agent = MaintenanceAgent()

    def test_initial_policy_is_rule_based(self):
        """Without training, agent must fall back to rule-based scheduling."""
        if not self.agent.is_trained:
            result = self.agent.get_schedule()
            assert result["policy"] == "rule_based"

    def test_get_schedule_returns_expected_fields(self):
        result = self.agent.get_schedule()
        for key in ["assets_scheduled", "n_scheduled", "risk_summary", "policy", "training_steps"]:
            assert key in result, f"Missing field: {key}"

    def test_n_scheduled_matches_list(self):
        result = self.agent.get_schedule()
        assert result["n_scheduled"] == len(result["assets_scheduled"])

    def test_scheduled_assets_are_valid_ids(self):
        result = self.agent.get_schedule()
        for aid in result["assets_scheduled"]:
            assert aid in _ASSET_IDS, f"Unknown asset id: {aid}"

    def test_risk_summary_has_all_assets(self):
        result = self.agent.get_schedule()
        assert len(result["risk_summary"]) == N_ASSETS

    def test_train_increments_steps(self):
        before = self.agent._training_steps
        self.agent.train(total_timesteps=1_000)
        assert self.agent._training_steps == before + 1_000

    def test_trained_flag_after_training(self):
        self.agent.train(total_timesteps=1_000)
        assert self.agent.is_trained

    def test_policy_becomes_ppo_after_training(self):
        self.agent.train(total_timesteps=1_000)
        result = self.agent.get_schedule()
        assert result["policy"] == "ppo"

    def test_train_saves_model(self):
        from backend.rl_agent import _MODEL_PATH
        self.agent.train(total_timesteps=1_000)
        assert _MODEL_PATH.exists(), "Model file should be saved after training"
