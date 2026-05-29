# Data Center Digital Twin Platform

A production-style MVP for real-time data center monitoring, cooling optimisation,
power management, and carbon reporting — built with FastAPI + Streamlit + SQLite.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      Data Center Digital Twin                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────────┐     HTTP/REST      ┌──────────────────┐  │
│  │   Streamlit      │ ◄────────────────► │   FastAPI        │  │
│  │   Dashboard      │    JSON API        │   Backend        │  │
│  │  :8501           │                    │  :8000           │  │
│  └──────────────────┘                    └────────┬─────────┘  │
│                                                   │            │
│  ┌──────────────────────────────────────┐         │            │
│  │            SQLite Database           │ ◄───────┘            │
│  │  assets | telemetry | alerts         │                      │
│  │  recommendations | events | pue_hist │                      │
│  └──────────────────────────────────────┘                      │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │               Background Simulation Thread               │  │
│  │  80 racks | 8 CRACs | 2 chillers | 2 UPS | 8 PDUs       │  │
│  │  Sinusoidal diurnal pattern + noise + anomaly injection  │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## Features

| Feature | Description |
|---|---|
| Real-time telemetry | 80 racks, 8 CRACs, 2 chillers, 2 UPS, 8 PDUs — simulated every 30s |
| PUE monitoring | Live, rolling average, linear-regression 24h forecast |
| Thermal heatmap | Interactive floor-plan heatmap with ASHRAE class overlays |
| Cooling optimisation | Automatic chilled-water setpoint and airflow recommendations |
| Power monitoring | Rack/PDU/UPS overload detection, phase imbalance, spike detection |
| Anomaly confidence | 0-100 confidence score with deviation, duration, and rate-of-change |
| Stranded capacity | Cooling-limited vs. power-limited constraint analysis |
| Carbon reporting | Live and annualised CO₂ emissions with configurable grid factor |
| What-if simulation | 5 scenario types: GPU rack addition, CRAC failure, temperature delta, etc. |
| Audit trail | Paginated, filterable event log of all alerts and recommendations |

## Quick Start

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
pip install -r requirements.txt
cp .env.example .env          # or edit .env with your settings

# Terminal 1 — start API
uvicorn backend.main:app --reload --port 8000

# Terminal 2 — start dashboard
streamlit run dashboard/app.py --server.port 8501

# Terminal 3 — run tests
pytest tests/ -v
```

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

## Engineering Assumptions

| Parameter | Value | Source |
|---|---|---|
| ASHRAE A1 inlet range | 15–32°C | ASHRAE TC 9.9 (2021) |
| ASHRAE A2 inlet range | 10–35°C | ASHRAE TC 9.9 (2021) |
| Overhead factor | 5% of IT load | ASHRAE best practice |
| CRAC rated capacity | 100 kW/unit | Industry typical |
| Chiller design COP | 4.5 | ASHRAE 90.1 baseline |
| UK grid carbon factor | 0.233 kgCO₂/kWh | BEIS 2023 |
| PUE alert threshold | 1.8 | Green Grid tier guidance |
| Rack overload warning | 18 kW | Typical per-rack budget |
| Rack overload critical | 20 kW | Rated maximum |
| PDU warning | 80% rated | NEC 80% continuous rule |
| PDU critical | 90% rated | Engineering safe limit |
| Healthy CRAC delta-T | ≥ 8°C | ASHRAE containment guideline |

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

### Authentication
Add `python-jose` + `passlib` for JWT bearer tokens on all API endpoints.

## License

MIT License — Copyright 2024 DC Twin Project
