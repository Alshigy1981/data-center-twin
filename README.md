# Data Center Digital Twin Platform

A production-style MVP for real-time data center monitoring, cooling optimisation,
power management, carbon reporting, multi-objective setpoint optimization,
reinforcement-learning-based maintenance scheduling, workload placement, energy
cost scheduling, and LSTM-based predictive world modelling — built with FastAPI +
Streamlit + SQLite + stable-baselines3 (PPO) + PyTorch + scipy + PuLP.

## Architecture

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                        Data Center Digital Twin                              │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────────┐      HTTP/REST     ┌──────────────────────────────┐   │
│  │   Streamlit      │ ◄─────────────────►│   FastAPI Backend  :8000     │   │
│  │   Dashboard      │      JSON API      └───────────────┬──────────────┘   │
│  │   :8501          │                                    │                  │
│  └──────────────────┘                                    │                  │
│                                                          │                  │
│  ┌────────────────────────────────────────┐              │                  │
│  │           SQLite Database (WAL mode)   │ ◄────────────┤                  │
│  │  assets | telemetry | alerts | events  │              │                  │
│  │  recommendations | pue_history         │              │                  │
│  │  placement_decisions | energy_schedule │              │                  │
│  │  demand_peaks | monthly_bills          │              │                  │
│  └────────────────────────────────────────┘              │                  │
│                                                          │                  │
│  ┌──────────────────────────────────────────────────┐    │                  │
│  │           Background Simulation Thread           │ ◄──┤                  │
│  │  80 racks | 8 CRACs | 2 chillers | 2 UPS | 8 PDUs│    │                  │
│  │  Sinusoidal diurnal + noise + anomaly injection  │    │                  │
│  └──────────────────────────────────────────────────┘    │                  │
│                                                          │                  │
│  ┌──────────────────────┐  ┌─────────────────────────┐   │                  │
│  │  Multi-Obj Optimizer │  │  PPO Maintenance Agent  │   │                  │
│  │  scipy L-BFGS-B      │  │  stable-baselines3      │   │                  │
│  │  CHW + CRAC setpoints│  │  12 critical assets     │   │                  │
│  │  Pareto front        │  │  30-day episode         │   │                  │
│  └──────────────────────┘  └─────────────────────────┘   │                  │
│                                                          │                  │
│  ┌──────────────────────┐  ┌─────────────────────────┐   │                  │
│  │  MILP Workload       │  │  LSTM World Model        │   │                  │
│  │  Placement (PuLP/CBC)│  │  PyTorch 2-layer LSTM   │   │                  │
│  │  4-objective rack    │  │  48h→24h telemetry      │   │                  │
│  │  assignment          │  │  forecast + PPO augment │   │                  │
│  └──────────────────────┘  └─────────────────────────┘   │                  │
│                                                          │                  │
│  ┌────────────────────────────────────────────────────┐   │                  │
│  │  Energy Cost Scheduler (greedy DP)                 │ ◄─┘                  │
│  │  TOU rates | demand charges | pre-cooling strategy │                     │
│  └────────────────────────────────────────────────────┘                     │
└──────────────────────────────────────────────────────────────────────────────┘
```

## Features

| Feature | Description |
|---|---|
| Real-time telemetry | 80 racks, 8 CRACs, 2 chillers, 2 UPS, 8 PDUs — simulated every 30s |
| PUE monitoring | Live, rolling average, linear-regression 24h forecast |
| Thermal heatmap | Interactive floor-plan heatmap with ASHRAE class overlays |
| 3D visualization | Plotly 3D + Three.js WebGL room layout with live rack temperature coloring |
| Cooling recommendations | Automatic chilled-water setpoint and airflow recommendations |
| Power monitoring | Rack/PDU/UPS overload detection, phase imbalance, spike detection |
| Anomaly confidence | 0–100 confidence score with deviation, duration, and rate-of-change |
| Stranded capacity | Cooling-limited vs. power-limited constraint analysis |
| Carbon reporting | Live and annualised CO₂ emissions with configurable grid factor |
| What-if simulation | 5 scenario types: GPU rack addition, CRAC failure, temperature delta, etc. |
| Audit trail | Paginated, filterable event log of all alerts and recommendations |
| **Multi-objective optimizer** | scipy L-BFGS-B finds optimal CHW + CRAC setpoints minimising weighted PUE, carbon, and thermal risk; Pareto front for trade-off visualization |
| **PPO maintenance agent** | stable-baselines3 PPO policy schedules proactive maintenance across 12 critical assets over a 30-day Gymnasium episode; rule-based fallback before training |
| **MILP workload placement** | PuLP/CBC rack assignment over 4 objectives: thermal risk, PUE impact, stranded capacity, phase imbalance; ASHRAE A2 hard constraint; migration recommender |
| **Energy cost scheduler** | Greedy DP load scheduler against PG&E E-19/ConEd SC-9 TOU tariffs; pre-cooling strategy; demand charge tracking; ratchet clause detection |
| **LSTM world model** | PyTorch 2-layer LSTM forecasts 24h of telemetry for all 12 critical assets from 48h history; forecast embeddings augment PPO observations for forward-looking decisions |

## Quick Start

> **Python version:** PyTorch requires Python ≤ 3.12. If your system Python is 3.13+,
> create a Python 3.11 virtual environment first:
> ```bash
> python3.11 -m venv .venv && source .venv/bin/activate
> ```

### Docker Compose (recommended)

```bash
git clone <repo-url>
cd data_center_twin
docker compose up --build
```

- Dashboard: http://localhost:8501
- API docs:  http://localhost:8000/docs

### Manual (local development)

```bash
cd data_center_twin

# Create Python 3.11 venv (required for PyTorch on macOS/systems with Python 3.13+)
python3.11 -m venv .venv && source .venv/bin/activate

# macOS — PyTorch from PyPI (no CPU index needed)
pip install torch
# Linux — PyTorch CPU-only build
# pip install torch --index-url https://download.pytorch.org/whl/cpu

pip install -r requirements.txt

cp .env.example .env          # or edit .env with your settings

# Terminal 1 — start API (LSTM pre-training runs in background ~105s on first start)
uvicorn backend.main:app --reload --port 8000

# Terminal 2 — start dashboard
streamlit run dashboard/app.py --server.port 8501

# Terminal 3 — run tests (264 tests)
pytest tests/ -v
```

## Starting and Stopping the App

The API and dashboard are independent processes — **both must be running** for the dashboard to display live data.

```bash
cd ~/data_center_twin
source .venv/bin/activate

# Terminal 1 — FastAPI backend (required first)
uvicorn backend.main:app --port 8000

# Terminal 2 — Streamlit dashboard
streamlit run dashboard/app.py --server.port 8501
```

| Service | URL |
|---|---|
| Dashboard | http://localhost:8501 |
| API (Swagger docs) | http://localhost:8000/docs |
| API (ReDoc) | http://localhost:8000/redoc |

To stop either server: press `Ctrl-C` in its terminal, or run `kill $(lsof -ti:8000 -ti:8501)`.

> **LSTM pre-training:** On first startup, the LSTM world model trains on 4,000 simulator steps in a background thread (~105 seconds on M-series Mac). The server is immediately available; `/lstm/status` shows `"training_status": "warming_up"` until complete, then `"complete"`. The trained model is cached at `models/lstm_world_model.pt` and reloaded on subsequent starts.

> **If the dashboard shows "This site can't be reached":** the servers are not running. Run the two commands above.

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `DB_PATH` | `./data_center.db` | SQLite database file path |
| `SIMULATION_INTERVAL` | `30` | Seconds between simulation cycles |
| `CARBON_INTENSITY_KG_PER_KWH` | `0.233` | Grid carbon factor (UK average) |
| `API_BASE_URL` | `http://localhost:8000` | URL dashboard uses to reach the API |
| `APP_VERSION` | `1.0.0` | Application version string |
| `LOG_LEVEL` | `INFO` | Python logging level |
| `LSTM_ENABLED` | `true` | Set `false` to disable LSTM entirely (no torch required) |
| `LSTM_SEQUENCE_LENGTH` | `48` | Hours of history required before forecasting |
| `LSTM_FORECAST_HORIZON` | `24` | Hours ahead to forecast |
| `LSTM_HIDDEN_SIZE` | `128` | LSTM hidden units per layer |
| `LSTM_NUM_LAYERS` | `2` | Number of stacked LSTM layers |
| `LSTM_PRETRAIN_STEPS` | `4000` | Simulator steps for initial pre-training |
| `LSTM_RETRAIN_INTERVAL_HOURS` | `24` | Hours between automatic retraining runs |

## API Endpoints

### Core

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Service health, uptime, version |
| `GET` | `/assets` | Full asset inventory with hierarchy |
| `GET` | `/telemetry/live` | Latest snapshot for all 100 assets |
| `GET` | `/telemetry/history` | Historical readings, `?limit=N&asset_id=X` |
| `GET` | `/metrics/pue` | Live PUE, rolling avg, trend, 24h forecast |
| `GET` | `/metrics/stranded-capacity` | Available IT capacity before constraint |
| `GET` | `/metrics/carbon` | Live and annualised CO₂ emissions |
| `GET` | `/alerts` | Active alerts with confidence scores |
| `GET` | `/recommendations` | Prioritised optimisation actions |
| `POST` | `/simulate/what-if` | Run what-if scenario, get projection |
| `GET` | `/events/log` | Paginated audit trail |

### Multi-Objective Optimizer

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/optimize/cooling` | Optimal CHW + CRAC setpoints with Pareto front. Query params: `w_pue`, `w_carbon`, `w_risk` (weights, auto-normalised) |
| `GET` | `/optimize/pareto` | Pareto-efficient setpoint pairs for scatter-plot visualization |

### PPO Maintenance Agent

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/rl/maintenance-schedule` | Per-asset health scores, 30-day failure probabilities, scheduled assets |
| `POST` | `/rl/train` | Train the PPO policy. Body: `{"total_timesteps": 50000}` |

### Workload Placement (MILP)

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/optimizer/placement/recommend` | Best rack for a new workload. `?workload_kw=N` |
| `GET` | `/optimizer/placement/scores` | Scores for all candidate racks |
| `GET` | `/optimizer/placement/migrations` | Hot rack migration recommendations |
| `GET` | `/optimizer/placement/preview/{rack_id}` | Thermal/power impact preview |
| `POST` | `/optimizer/placement/weights` | Update objective weights |

### Energy Cost Scheduler

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/optimizer/energy/rate/current` | Current TOU rate and period |
| `GET` | `/optimizer/energy/rate/forecast` | 24-hour rate forecast |
| `GET` | `/optimizer/energy/schedule` | Optimised load schedule for today |
| `GET` | `/optimizer/energy/bill/current-month` | MTD bill with demand charge breakdown |
| `GET` | `/optimizer/energy/demand/current` | Current demand peak tracking |
| `GET` | `/optimizer/energy/precooling` | Pre-cooling recommendation for next on-peak window |

### Unified Optimization

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/optimizer/unified` | Combined recommendations from all 4 optimization layers |
| `GET` | `/optimizer/pareto` | 200-point 4-objective Pareto front |
| `GET` | `/optimizer/conflicts` | Detected conflicts between optimization layers |
| `GET` | `/optimizer/savings/summary` | Estimated monthly savings across all layers |

### LSTM World Model

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/lstm/status` | Training status, history length, model accuracy |
| `GET` | `/lstm/forecast` | 24h telemetry forecasts for all 12 critical assets |
| `GET` | `/lstm/forecast/{asset_id}` | Single-asset forecast with attention weights |
| `GET` | `/lstm/accuracy` | Per-feature MAE/RMSE and per-horizon accuracy (1h/6h/12h/24h) |
| `GET` | `/lstm/risk-timeline` | Assets sorted by predicted failure urgency (CRITICAL → LOW) |
| `GET` | `/lstm/compare-decisions` | PPO maintenance decisions with and without LSTM context |
| `POST` | `/lstm/retrain` | Trigger background retraining from fresh simulator data |

**Example — optimize with carbon-heavy weights:**
```bash
curl "http://localhost:8000/optimize/cooling?w_pue=0.3&w_carbon=0.6&w_risk=0.1"
```

**Example — trigger PPO training:**
```bash
curl -X POST http://localhost:8000/rl/train \
     -H "Content-Type: application/json" \
     -d '{"total_timesteps": 100000}'
```

**Example — check LSTM status:**
```bash
curl http://localhost:8000/lstm/status
```

**Example — get 24h forecast for one asset:**
```bash
curl http://localhost:8000/lstm/forecast/A-CRAC-1
```

## Module Overview

### `backend/optimizer.py` — Multi-Objective Cooling Optimizer

Uses scipy `L-BFGS-B` to minimise a weighted sum of three objectives over two continuous decision variables:

| Decision variable | Range | Effect |
|---|---|---|
| `chw_setpoint_c` | 6–14 °C | Higher → better chiller COP, constrained by CRAC setpoint via coil approach |
| `crac_setpoint_c` | 16–24 °C | Higher → more fan power (smaller ΔT), unlocks higher CHW setpoint |

**Coil approach coupling:** A chilled-water coil can only achieve a supply air temperature ≥ 5 °C above the entering CHW temperature. Raising the CRAC setpoint unlocks a higher effective CHW temperature, improving chiller COP. This creates a genuine Pareto trade-off: PUE vs thermal risk.

### `backend/rl_agent.py` — PPO Maintenance Scheduling Agent

A [Gymnasium](https://gymnasium.farama.org/) environment simulates 30 days of asset health degradation across **12 critical assets** (8 CRACs + 2 chillers + 2 UPS):

| Concept | Detail |
|---|---|
| **Observation** | Float32 vector `(48,)` — `[health, age_norm, load, anomaly]` × 12; extended to `(1584,)` when LSTM world model is active (+1536 forecast embedding) |
| **Action** | `MultiBinary(12)` — schedule maintenance on that asset today |
| **Health degradation** | 2–5% per day, accelerated by load; anomaly score accumulates as health falls |
| **Failure probability** | Sigmoid: near-zero above health 0.3, rises sharply below |
| **Reward** | `+0.1` uptime bonus − `5.0` per failure − `0.4` per maintenance action |
| **Policy** | PPO (MLP, `n_steps=512`, `batch_size=64`, `gamma=0.99`) |
| **Fallback** | Rule-based: maintain assets with health < 0.4 or anomaly > 0.5 |

### `backend/workload_placement.py` — MILP Rack Assignment

PuLP/CBC solver assigns incoming workloads to racks with 4 objectives and hard ASHRAE A2 inlet temperature constraint:

| Objective | Weight | Description |
|---|---|---|
| Thermal risk | 0.35 | Predicted inlet temp delta above A2 threshold |
| PUE impact | 0.25 | Incremental cooling overhead (thermal resistance 0.15 °C/kW) |
| Stranded capacity | 0.25 | Cooling headroom consumed |
| Phase imbalance | 0.15 | PDU phase deviation (rack_index % 3 assignment) |

Solver timeout: 5 seconds. Falls back to greedy rank if infeasible.

### `backend/energy_scheduler.py` — Energy Cost Scheduler

Greedy DP scheduler defers deferrable loads (UPS test, batch compute, firmware updates, etc.) to off-peak TOU windows:

- **Tariffs:** PG&E E-19 and ConEd SC-9 with on-peak ($0.28/kWh), mid-peak ($0.18/kWh), off-peak ($0.06/kWh)
- **Demand charges:** $18.50/kW/month on 15-minute peak
- **Ratchet clause:** 85% of prior 11-month peak sets minimum billable demand
- **Pre-cooling:** Drops CHW setpoint −1.5 °C in the 2-hour window before on-peak to reduce cooling load during expensive hours

### `backend/lstm/` — LSTM World Model

PyTorch-based predictive model that forecasts 24 hours of telemetry for all 12 critical assets from a 48-hour rolling history window:

| Component | Description |
|---|---|
| `feature_engineer.py` | Extracts 8 normalized features per asset per timestep: `inlet_temp`, `outlet_temp`, `power_kw`, `humidity`, `delta_t`, `wear_score`, `hour_sin`, `hour_cos`. Asset-type-specific normalization. Sinusoidal time encoding mapped to [0, 1]. |
| `model.py` | `DataCenterLSTM`: 2-layer LSTM (hidden=128) → self-attention over sequence → context vector → FC head. Input `(batch, 48, 8)` → output `(batch, 24, 8)` + attention weights `(batch, 48)`. Xavier/orthogonal weight init. |
| `trainer.py` | Per-feature weighted MSE loss (`wear_score=3×`, `inlet_temp=2×`). Adam optimizer, gradient clipping 1.0, early stopping patience=10. Pre-trains in ~105s on M-series Mac with 4,000 simulator steps + batch size 512. |
| `forecaster.py` | `LSTMForecaster`: rolling `deque(maxlen=48)` of telemetry snapshots. `get_forecast_embedding()` returns a `(1536,)` vector (12 assets × 128 hidden) for PPO obs augmentation. Gracefully returns empty forecasts when history < 48 readings. |
| `world_model.py` | `WorldModel`: coordinates forecaster + PPO agent. `get_enhanced_observation()` appends 1536-dim LSTM embedding to standard 48-dim PPO observation. `compare_decisions()` shows maintenance decisions with and without LSTM context. |

**Wear score** is the most important predicted feature (3× loss weight) because it directly drives PPO maintenance scheduling. The model predicts failure hours by finding when wear crosses 0.9 in the 24-hour forecast horizon, and classifies risk as CRITICAL (<12h), HIGH (<48h), MEDIUM (<168h), or LOW.

**Pre-trained model** is saved to `models/lstm_world_model.pt` (≈1.1 MB) after the first startup. Subsequent starts load the cached weights and skip pre-training.

## Engineering Assumptions

| Parameter | Value | Source |
|---|---|---|
| ASHRAE A1 inlet range | 15–32 °C | ASHRAE TC 9.9 (2021) |
| ASHRAE A2 inlet range | 10–35 °C | ASHRAE TC 9.9 (2021) |
| Overhead factor | 5% of IT load | ASHRAE best practice |
| CRAC rated capacity | 100 kW/unit | Industry typical |
| Chiller design COP | 4.5 | ASHRAE 90.1 baseline |
| UK grid carbon factor | 0.233 kgCO₂/kWh | BEIS 2023 |
| PUE alert threshold | 1.8 | Green Grid tier guidance |
| Rack overload warning | 18 kW | Typical per-rack budget |
| Rack overload critical | 20 kW | Rated maximum |
| PDU warning | 80% rated | NEC 80% continuous rule |
| PDU critical | 90% rated | Engineering safe limit |
| Healthy CRAC delta-T | ≥ 8 °C | ASHRAE containment guideline |
| Chiller Carnot efficiency | 55% | Typical centrifugal chiller |
| Cooling tower condenser temp | 30 °C | Standard design point |
| CHW–CRAC coil approach | 5 °C | Good HEX design practice |
| Thermal risk onset margin | 6 °C below A1 limit | Conservative operating buffer |
| RL episode length | 30 days | Typical maintenance planning horizon |
| Asset failure cost | 5× maintenance cost | Emergency repair premium |
| Thermal resistance factor | 0.15 °C/kW | Hot-aisle containment assumption |
| LSTM sequence length | 48 hours | Captures 2 full diurnal cycles |
| LSTM forecast horizon | 24 hours | Standard operational planning window |
| Wear score failure threshold | 0.9 | Conservative failure onset |

## Dashboard Sections

1. **KPI Row** — IT load, PUE (live + projected), stranded capacity, carbon, alert count
2. **Thermal Heatmap** — 8×10 grid with blue→green→amber→red temperature colorscale
3. **3D Visualization** — Plotly 3D floor plan + Three.js WebGL live view with rack temperature coloring and alert pulsing
4. **Temperature Trends** — PUE history, ASHRAE bands, IT load time series
5. **Power Monitoring** — UPS gauges, PDU load bars, facility power pie chart
6. **Carbon Trend** — CO₂ kg/hr over time with aspirational target line
7. **Alerts Panel** — Severity badges, confidence bars, explanation tooltips
8. **Recommendations** — Priority badges (HIGH/MEDIUM/LOW), savings estimates
9. **Event Log** — Filterable audit trail table with severity color-coding
10. **What-If Results** — Scenario output with projected PUE and stranded capacity
11. **Optimization Dashboard** — Unified 4-layer recommendations, Pareto front, conflict detection, savings summary
12. **Workload Placement** — MILP rack scores, migration recommendations, impact preview
13. **Energy Scheduling** — TOU rate forecast, optimised load schedule, MTD bill breakdown
14. **LSTM World Model Status** — Training status badge, history fill, model accuracy score
15. **24h Asset Forecast Grid** — 4×3 mini-charts of predicted wear and temperature per asset
16. **Predicted Failure Risk Timeline** — Gantt-style chart sorted by urgency (CRITICAL → LOW)
17. **LSTM Attention Heatmap** — Per-hour attention weights showing which past readings drive predictions
18. **PPO + LSTM Decision Comparison** — Side-by-side maintenance decisions with and without forecast context
19. **Forecast Accuracy Tracker** — Per-horizon MAE bars (1h/6h/12h/24h) and risk distribution pie

## Extending to Real Infrastructure

### Building Management System (BMS)
Replace `simulator.py` with a BMS connector reading from:
- **Modbus TCP/RTU** — `pymodbus` library for CRAC/chiller controllers
- **BACnet** — `BAC0` library for building automation protocols
- **MQTT** — `paho-mqtt` for IoT sensor feeds
- **REST/OPC-UA** — direct API calls to modern DCIM platforms

### Real-time Data Pipeline
```
BMS/SCADA → MQTT Broker → Telemetry Adapter → FastAPI → SQLite/TimescaleDB
```

### Scaling the Database
Swap `sqlite:///` for `postgresql+asyncpg://` and migrate tables to
TimescaleDB for native time-series compression and continuous aggregates.

### Extending the RL Agent
- Replace the synthetic health model with real CMMS (Computerised Maintenance Management System) data
- Add more asset types (PDUs, network switches) to the action space
- Use `VecEnv` and multiple parallel environments for faster PPO training
- Export TensorBoard logs from `assets/tb_logs/` to monitor training convergence
- Feed real-time LSTM forecast embeddings to PPO for fully model-based RL (Dreamer/MuZero style)

### Extending the LSTM World Model
- Replace synthetic wear accumulator with real sensor wear readings from CMMS
- Add GPU/accelerator support by setting `LSTM_ENABLED=true` and installing `torch` with CUDA
- Increase `LSTM_PRETRAIN_STEPS` to 10,000+ for better accuracy (requires disabling early stopping or raising patience)
- Retrain periodically against live telemetry via `POST /lstm/retrain`

### Authentication
Add `python-jose` + `passlib` for JWT bearer tokens on all API endpoints.

## License

MIT License — Copyright 2024 DC Twin Project
