# Sat Aim — Satellite Pointing Geometry Calculator

A Streamlit application that computes satellite pointing geometry for corner reflector (CR) sites. Given a TLE, a ground site, and a scene center time, it calculates azimuth/elevation, a ±X° pointing window, and ancillary geometry for radar calibration.

## Features

| Tab | Description |
|---|---|
| **Pointing Card** | Azimuth, elevation, slant range, magnetic azimuth, sunlit flag, window duration. Export as CSV or PDF. |
| **Geometry Plot** | Dual-axis Plotly chart of azimuth and elevation over the pointing window. |
| **Sky Plot** | Polar plot showing the satellite track from the observer's perspective. |
| **Map** | Folium map with site marker and azimuth direction line. |
| **Raw Table** | Per-second geometry data (az, el, slant, sunlit, ascending) for the full window. |
| **Methodology** | Equations, solver diagram, and assumptions behind the calculations. |

## Quick Start

```bash
# Create and activate conda environment
conda create -n sat_aim python=3.12 -y
conda activate sat_aim

# Install dependencies
pip install streamlit skyfield scipy pygeomag fpdf2 folium streamlit-folium plotly pandas

# Run the app
streamlit run app.py
```

The app opens at `http://localhost:8501`.

## Usage

1. **TLE Source** — Select a SAR constellation from Celestrak or paste TLE lines directly.
2. **Site** — Enter the corner reflector's latitude, longitude, and height.
3. **Scene Center** — Set the UTC date and time of the pass.
4. **Window** — Choose a criterion (off-boresight, azimuth, or elevation) and half-width (1–45°).
5. Click **Compute** to generate results across all six tabs.

## Module Structure

```
Sat_Aim/
├── app.py                  # Streamlit UI (6 tabs)
├── core/
│   ├── tle.py              # TLE fetch/parse/validate (Celestrak)
│   ├── magnetic.py         # WMM declination via pygeomag
│   ├── propagator.py       # SatAim class, state_at, solve_window
│   └── export.py           # CSV + PDF pointing card export
├── tests/
│   └── test_propagator.py  # Solver correctness and edge cases
├── docs/
│   └── superpowers/
│       ├── specs/          # Design specification
│       └── plans/          # Implementation plan
├── output/                 # Generated files (gitignored)
└── README.md
```

## Core API

```python
from core.propagator import SatAim

sa = SatAim(
    tle_lines=("1 ...", "2 ..."),
    name="ICEYE-X12",
    lat=1.29, lon=103.85, height_m=20,
)

# Single-instant geometry
state = sa.state_at(t_utc)
print(state.az_true_deg, state.el_deg, state.slant_km, state.sunlit)

# Pointing window
window = sa.solve_window(t_center_utc, criterion="off_boresight", half_width_deg=5.0)
print(window.t_start_utc, window.t_stop_utc, window.duration_s)
```

## Window Solver

The solver finds times where the selected criterion reaches ±X° from the scene center value:

1. **Off-boresight** (default) — angular separation of unit LOS vectors via `arccos(L̂_ref · L̂_t)`. Most physically meaningful for corner reflector RCS.
2. **Azimuth** — wrapped azimuth difference (handles 0°/360° discontinuity).
3. **Elevation** — absolute elevation difference.

Algorithm: coarse scan at 1 s steps outward from scene center, bracket first crossing of f(t) = X, then `scipy.optimize.brentq` refinement to ±0.01 s. Falls back to ±300 s if no crossing within ±120 s.

## Testing

```bash
conda activate sat_aim
python -m pytest tests/ -v
```

## Dependencies

| Package | Purpose |
|---|---|
| `streamlit` | Web UI |
| `skyfield` | SGP4 propagation, topocentric geometry |
| `scipy` | Brentq root-finding for window solver |
| `pygeomag` | World Magnetic Model declination |
| `fpdf2` | PDF pointing card export |
| `folium` + `streamlit-folium` | Interactive map |
| `plotly` | Geometry plots, sky plot, solver diagram |
| `pandas` | Raw data table |

## License

Internal use only.
