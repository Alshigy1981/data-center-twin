# Data Center Digital Twin Platform

A production-style MVP for real-time data center monitoring, cooling optimisation,
power management, carbon reporting, multi-objective setpoint optimization, and
reinforcement-learning-based maintenance scheduling — built with FastAPI + Streamlit +
SQLite + stable-baselines3 (PPO) + scipy.

## Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                      Data Center Digital Twin                        │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────────┐     HTTP/REST      ┌──────────────────────┐   │
│  │   Streamlit      │ ◄────────────────► │   FastAPI Backend    │   │
│  │   Dashboard      │    JSON API        │   :8000              │   │
│  │   :8501          │                    └───────────┬──────────┘   │
│  └──────────────────┘                                │              │
│                                                      │              │
│  ┌──────────────────────────────────────┐            │              │
│  │           SQLite Database            │ ◄──────────┤              │
│  │  assets | telemetry | alerts         │            │              │
│  │  recommendations | events | pue_hist │            │              │
│  └──────────────────────────────────────┘            │              │
│                                                      │              │
│  ┌────────────────────────────────────────────────┐  │              │
│  │           Background Simulation Thread         │ ◄┤              │
│  │  80 racks | 8 CRACs | 2 chillers | 2 UPS       │  │              │
│  │  Sinusoidal diurnal + noise + anomaly injection │  │              │
│  └────────────────────────────────────────────────┘  │              │
│                                                      │              │
│  ┌─────────────────────────────────┐  ┌─────────────┴────────────┐  │
│  │   Multi-Objective Optimizer     │  │   PPO Maintenance Agent  │  │
│  │   scipy L-BFGS-B               │  │   stable-baselines3      │  │
│  │   CHW + CRAC setpoints         │  │   12 critical assets     │  │
│  │   Pareto front (PUE vs risk)   │  │   30-day episode         │  │
│  └─────────────────────────────────┘  └──────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────┘
```

## Features

| Feature | Description |
|---|---|
| Real-time telemetry | 80 racks, 8 CRACs, 2 chillers, 2 UPS, 8 PDUs — simulated every 30s |
| PUE monitoring | Live, rolling average, linear-regression 24h forecast |
| Thermal heatmap | Interactive floor-plan heatmap with ASHRAE class overlays |
| Cooling recommendations | Automatic chilled-water setpoint and airflow recommendations |
| Power monitoring | Rack/PDU/UPS overload detection, phase imbalance, spike detection |
| Anomaly confidence | 0–100 confidence score with deviation, duration, and rate-of-change |
| Stranded capacity | Cooling-limited vs. power-limited constraint analysis |
| Carbon reporting | Live and annualised CO₂ emissions with configurable grid factor |
| What-if simulation | 5 scenario types: GPU rack addition, CRAC failure, temperature delta, etc. |
| Audit trail | Paginated, filterable event log of all alerts and recommendations |
| **Multi-objective optimizer** | scipy L-BFGS-B finds optimal CHW + CRAC setpoints minimising weighted PUE, carbon, and thermal risk; includes Pareto front approximation for trade-off visualization |
| **PPO maintenance agent** | stable-baselines3 PPO policy schedules proactive maintenance across 12 critical assets over a 30-day Gymnasium episode; rule-based fallback before training |

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

# Terminal 1 — start API
uvicorn backend.main:app --reload --port 8000

# Terminal 2 — start dashboard
streamlit run dashboard/app.py --server.port 8501

# Terminal 3 — run tests (233 tests)
pytest tests/ -v
```

## Starting and Stopping the App

The API and dashboard are independent processes — **both must be running** for the dashboard to display live data. They do not auto-restart; you need to start them each session.

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
| `GET` | `/optimize/cooling` | Optimal CHW + CRAC setpoints. Query params: `w_pue`, `w_carbon`, `w_risk` (weights, default 0.5/0.3/0.2, auto-normalised). Returns expected PUE, carbon reduction, COP, thermal risk, and full Pareto front. |
| `GET` | `/optimize/pareto` | Pareto-efficient setpoint pairs (PUE vs thermal risk) for scatter-plot visualization. |

### PPO Maintenance Agent

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/rl/maintenance-schedule` | Current maintenance schedule. Returns per-asset health scores, 30-day failure probabilities, and scheduled assets. Uses PPO policy if trained, rule-based fallback otherwise. |
| `POST` | `/rl/train` | Train (or continue training) the PPO policy. Body: `{"total_timesteps": 50000}` (1k–500k). Synchronous; model saved to `assets/ppo_maintenance.zip`. |

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

## Module Overview

### `backend/optimizer.py` — Multi-Objective Cooling Optimizer

Uses scipy `L-BFGS-B` to minimise a weighted sum of three objectives over two continuous decision variables:

| Decision variable | Range | Effect |
|---|---|---|
| `chw_setpoint_c` | 6–14 °C | Higher → better chiller COP, but constrained by CRAC setpoint via coil approach |
| `crac_setpoint_c` | 16–24 °C | Higher → more fan power (smaller ΔT), unlocks higher CHW setpoint |

**Coil approach coupling:** A chilled-water coil can only achieve a supply air temperature that is ≥ 5 °C above the entering CHW temperature. So raising the CRAC setpoint unlocks a higher effective CHW temperature, improving chiller COP. This creates a genuine Pareto trade-off: PUE vs thermal risk.

**Pareto front:** A 10×10 uniform grid over the setpoint space is filtered to retain only non-dominated points, returned for dashboard scatter-plot visualization.

### `backend/rl_agent.py` — PPO Maintenance Scheduling Agent

A [Gymnasium](https://gymnasium.farama.org/) environment simulates 30 days of asset health degradation across **12 critical assets** (8 CRACs + 2 chillers + 2 UPS):

| Concept | Detail |
|---|---|
| **Observation** | Float32 vector of shape `(48,)` — `[health, age_norm, load, anomaly]` × 12 |
| **Action** | `MultiBinary(12)` — schedule maintenance on that asset today |
| **Health degradation** | 2–5% per day, accelerated by load; anomaly score accumulates as health falls |
| **Failure probability** | Sigmoid: near-zero above health 0.3, rises sharply below |
| **Reward** | `+0.1` uptime bonus − `5.0` per failure − `0.4` per maintenance action |
| **Policy** | PPO (MLP, `n_steps=512`, `batch_size=64`, `gamma=0.99`) |
| **Fallback** | Rule-based: maintain assets with health < 0.4 or anomaly > 0.5 |

Training is incremental — call `/rl/train` multiple times to continue from the saved checkpoint. Practical training for convergence: ~200k–500k timesteps.

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

## Dashboard Sections

1. **KPI Row** — IT load, PUE (live + projected), stranded capacity, carbon, alert count
2. **Thermal Heatmap** — 8×10 grid with blue→green→amber→red temperature colorscale
3. **Temperature Trends** — PUE history, ASHRAE bands, IT load time series
4. **Power Monitoring** — UPS gauges, PDU load bars, facility power pie chart
5. **Carbon Trend** — CO₂ kg/hr over time with aspirational target line
6. **Alerts Panel** — Severity badges, confidence bars, explanation tooltips
7. **Recommendations** — Priority badges (HIGH/MEDIUM/LOW), savings estimates
8. **Event Log** — Filterable audit trail table with severity color-coding
9. **What-If Results** — Scenario output with projected PUE and stranded capacity

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

### Authentication
Add `python-jose` + `passlib` for JWT bearer tokens on all API endpoints.

## License

MIT License — Copyright 2024 DC Twin Project
