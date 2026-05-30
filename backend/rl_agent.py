"""
PPO-based maintenance scheduling agent for the Data Center Digital Twin.

Environment (DataCenterMaintenanceEnv):
  - Episode length : 30 simulated days
  - Assets managed : 8 CRACs + 2 chillers + 2 UPS = 12 critical assets
  - Observation    : float32 vector of shape (48,) — [health, age_norm, load, anomaly] × 12
  - Action         : MultiBinary(12) — 1 = schedule maintenance this day
  - Reward         : uptime bonus − failure costs − unnecessary maintenance costs

The PPO policy (stable-baselines3) learns to proactively maintain assets
before they fail. If no trained model exists, a rule-based fallback triggers
maintenance on assets with health < 0.4 or high anomaly scores.
"""

from __future__ import annotations

import logging
import math
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO

logger = logging.getLogger(__name__)

# ─── Asset configuration ──────────────────────────────────────────────────────

N_CRACS = 8
N_CHILLERS = 2
N_UPS = 2
N_ASSETS = N_CRACS + N_CHILLERS + N_UPS  # 12

_ASSET_IDS: List[str] = (
    [f"A-CRAC-{i+1}" for i in range(4)]
    + [f"B-CRAC-{i+1}" for i in range(4)]
    + [f"CH-{i+1}" for i in range(N_CHILLERS)]
    + [f"UPS-{i+1}" for i in range(N_UPS)]
)
_ASSET_TYPES: List[str] = (
    ["CRAC"] * N_CRACS + ["Chiller"] * N_CHILLERS + ["UPS"] * N_UPS
)

# ─── Reward coefficients ──────────────────────────────────────────────────────

_FAILURE_COST = 5.0        # penalty per unexpected failure
_MAINTENANCE_COST = 0.4    # cost per scheduled maintenance action
_UPTIME_BONUS = 0.1        # daily reward when no failures occur

# ─── Paths ────────────────────────────────────────────────────────────────────

_MODEL_PATH = Path(__file__).parent.parent / "assets" / "ppo_maintenance.zip"
_TB_LOG_DIR = str(Path(__file__).parent.parent / "assets" / "tb_logs")

EPISODE_DAYS = 30


# ─── Gymnasium environment ────────────────────────────────────────────────────

class DataCenterMaintenanceEnv(gym.Env):
    """
    Simulates 30-day asset health degradation.

    State vector per asset (4 floats, all in [0, 1]):
      [health_score, age_normalized, load_pct, anomaly_score]

    Action: MultiBinary(N_ASSETS) — schedule maintenance on that asset today.
    """

    metadata = {"render_modes": []}

    def __init__(self) -> None:
        super().__init__()
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(N_ASSETS * 4,), dtype=np.float32
        )
        self.action_space = spaces.MultiBinary(N_ASSETS)
        self._state: np.ndarray = np.zeros((N_ASSETS, 4), dtype=np.float32)
        self._day: int = 0
        self._rng = np.random.default_rng()

    # ── Internal dynamics ────────────────────────────────────────────────────

    def _init_state(self) -> np.ndarray:
        state = np.zeros((N_ASSETS, 4), dtype=np.float32)
        for i in range(N_ASSETS):
            state[i, 0] = self._rng.uniform(0.6, 1.0)   # health
            state[i, 1] = self._rng.uniform(0.0, 0.5)   # age (normalised to episode)
            state[i, 2] = self._rng.uniform(0.3, 0.9)   # load pct
            state[i, 3] = 0.0                             # anomaly score
        return state

    def _degrade(self) -> None:
        """Age all assets by one day; anomalies grow as health falls."""
        for i in range(N_ASSETS):
            load = float(self._state[i, 2])
            delta = 0.02 + 0.03 * load + float(self._rng.exponential(0.005))
            self._state[i, 0] = max(0.0, float(self._state[i, 0]) - delta)
            self._state[i, 1] = min(1.0, float(self._state[i, 1]) + 1.0 / EPISODE_DAYS)
            p_anomaly = max(0.0, 1.0 - float(self._state[i, 0])) * 0.5
            if self._rng.random() < p_anomaly:
                self._state[i, 3] = min(1.0, float(self._state[i, 3]) + 0.3)
            else:
                self._state[i, 3] = max(0.0, float(self._state[i, 3]) - 0.1)

    def _failure_probability(self, i: int) -> float:
        """Sigmoid failure probability; near-zero above health=0.3, rises sharply below."""
        health = float(self._state[i, 0])
        return 1.0 / (1.0 + math.exp(10.0 * (health - 0.2)))

    def _check_failures(self) -> List[int]:
        return [i for i in range(N_ASSETS) if self._rng.random() < self._failure_probability(i)]

    def _apply_maintenance(self, action: np.ndarray) -> int:
        count = 0
        for i in range(N_ASSETS):
            if int(action[i]) == 1:
                self._state[i, 0] = 1.0
                self._state[i, 1] = 0.0
                self._state[i, 3] = 0.0
                count += 1
        return count

    # ── Gymnasium interface ───────────────────────────────────────────────────

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._state = self._init_state()
        self._day = 0
        return self._state.flatten(), {}

    def step(self, action: np.ndarray):
        self._degrade()
        n_maintained = self._apply_maintenance(action)
        failures = self._check_failures()
        for i in failures:
            self._state[i, 0] = 0.7  # emergency repair brings partial health back

        reward = float(
            _UPTIME_BONUS
            - _FAILURE_COST * len(failures)
            - _MAINTENANCE_COST * n_maintained
        )

        self._day += 1
        done = self._day >= EPISODE_DAYS
        info = {"day": self._day, "failures": len(failures), "maintained": n_maintained}
        return self._state.flatten(), reward, done, False, info

    # ── Inspection ───────────────────────────────────────────────────────────

    def risk_summary(self) -> List[Dict[str, Any]]:
        """Current per-asset health and risk metrics for API consumption."""
        out = []
        for i in range(N_ASSETS):
            health = float(self._state[i, 0])
            p30 = min(1.0, self._failure_probability(i) * EPISODE_DAYS)
            out.append({
                "asset_id": _ASSET_IDS[i],
                "asset_type": _ASSET_TYPES[i],
                "health_score": round(health, 3),
                "failure_probability_30d": round(p30, 3),
                "maintenance_recommended": bool(health < 0.4 or float(self._state[i, 3]) > 0.5),
                "anomaly_score": round(float(self._state[i, 3]), 3),
            })
        return out


# ─── Agent wrapper ────────────────────────────────────────────────────────────

class MaintenanceAgent:
    """Thread-safe wrapper around a stable-baselines3 PPO model."""

    def __init__(self) -> None:
        self._env = DataCenterMaintenanceEnv()
        self._env.reset()
        self._model: Optional[PPO] = None
        self._lock = threading.Lock()
        self._training_steps = 0
        self._load_if_exists()

    def _load_if_exists(self) -> None:
        if _MODEL_PATH.exists():
            try:
                self._model = PPO.load(str(_MODEL_PATH), env=self._env)
                logger.info("Loaded PPO maintenance model from %s", _MODEL_PATH)
            except Exception as exc:
                logger.warning("Could not load PPO model: %s", exc)

    def train(self, total_timesteps: int = 50_000) -> Dict[str, Any]:
        """Train (or continue training) the PPO policy. Saves model after training."""
        with self._lock:
            if self._model is None:
                self._model = PPO(
                    "MlpPolicy",
                    self._env,
                    verbose=0,
                    learning_rate=3e-4,
                    n_steps=512,
                    batch_size=64,
                    n_epochs=10,
                    gamma=0.99,
                    tensorboard_log=_TB_LOG_DIR,
                )
            self._model.learn(total_timesteps=total_timesteps, reset_num_timesteps=False)
            self._training_steps += total_timesteps
            _MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
            self._model.save(str(_MODEL_PATH))
            logger.info("PPO model saved — total steps: %d", self._training_steps)

        return {
            "status": "trained",
            "total_timesteps": self._training_steps,
            "model_path": str(_MODEL_PATH),
        }

    def get_schedule(self) -> Dict[str, Any]:
        """
        Produce a maintenance schedule via policy inference (or rule-based fallback).
        Updates the environment state by stepping one day forward.
        """
        with self._lock:
            obs = self._env._state.flatten()

            if self._model is not None:
                action, _ = self._model.predict(obs, deterministic=True)
                policy = "ppo"
            else:
                # Rule-based: maintain if health < 0.4 or anomaly > 0.5
                action = np.array(
                    [1 if (self._env._state[i, 0] < 0.4 or self._env._state[i, 3] > 0.5) else 0
                     for i in range(N_ASSETS)],
                    dtype=np.int8,
                )
                policy = "rule_based"

            summary = self._env.risk_summary()
            scheduled = [_ASSET_IDS[i] for i in range(N_ASSETS) if int(action[i]) == 1]

            # Advance the simulation by one step
            self._env.step(action)

        return {
            "assets_scheduled": scheduled,
            "n_scheduled": len(scheduled),
            "risk_summary": summary,
            "policy": policy,
            "training_steps": self._training_steps,
        }

    @property
    def is_trained(self) -> bool:
        return self._model is not None


# ─── Singleton ────────────────────────────────────────────────────────────────

_agent: Optional[MaintenanceAgent] = None
_agent_lock = threading.Lock()


def get_agent() -> MaintenanceAgent:
    global _agent
    if _agent is None:
        with _agent_lock:
            if _agent is None:
                _agent = MaintenanceAgent()
    return _agent
