# Sat_Aim — Design Specification

**Date:** 2026-07-06
**Version:** Rev 2 (scene-center + pointing window)

---

## 1. Purpose

Given a TLE, a site (corner reflector), and a **scene center time** (the moment the satellite is at boresight), compute:

- Azimuth and elevation (grazing angle) at scene center — the pointing target.
- Start/stop times of the window during which the satellite's angular offset from the scene-center line-of-sight is within **±X°**.
- Ancillary geometry: magnetic azimuth, slant range, sunlit flag, direction of motion.

No pass-search, no elevation mask, no ascending/descending filtering.

---

## 2. Tech Stack

| Layer                | Choice                                                   |
| -------------------- | -------------------------------------------------------- |
| UI                   | Streamlit ≥ 1.30                                         |
| Orbit propagation    | `skyfield` ≥ 1.48                                        |
| TLE fetch            | `requests` → Celestrak (Space-Track deferred to V2)     |
| Time                 | `datetime` + `zoneinfo`                                  |
| Magnetic declination | `pygeomag` (WMM)                                         |
| Plots                | `plotly`                                                 |
| Map                  | `folium` + `streamlit-folium`                            |
| Export               | `pandas` (CSV), `fpdf2` (PDF card)                       |
| Root-finding         | `scipy.optimize.brentq` for window endpoints             |
| Environment          | Conda env `sat_aim`, Python 3.13                         |

---

## 3. Module Structure

```
sat_aim/
├── app.py                  # Streamlit UI — inputs, caching, rendering
├── core/
│   ├── __init__.py
│   ├── tle.py              # fetch from Celestrak, parse, validate, age check
│   ├── propagator.py       # SatAim class — Skyfield wrappers, geometry, window solver
│   ├── magnetic.py         # WMM declination via pygeomag
│   └── export.py           # CSV + PDF card generation
├── tests/
│   ├── __init__.py
│   └── test_propagator.py  # solver sanity + edge cases
├── requirements.txt
└── environment.yml         # conda env spec
```

### 3.1 `core/propagator.py` — SatAim Class

```python
from dataclasses import dataclass
from datetime import datetime
from typing import Literal
import numpy as np

@dataclass
class LosState:
    t_utc: datetime
    az_true_deg: float        # 0–360, clockwise from true north
    el_deg: float             # elevation / grazing angle
    slant_km: float
    sunlit: bool
    ascending: bool           # radial velocity > 0 (moving away from Earth center)
    los_enu_unit: np.ndarray  # unit LOS vector in ENU frame
    az_mag_deg: float         # magnetic azimuth (WMM)

@dataclass
class Window:
    t_start_utc: datetime
    t_stop_utc: datetime
    duration_s: float
    minus_s: float            # t_center - t_start
    plus_s: float             # t_stop - t_center
    clamped_by_horizon: bool
    criterion: str
    half_width_deg: float

class SatAim:
    def __init__(self, tle_lines: tuple[str, str], name: str | None,
                 lat: float, lon: float, height_m: float):
        """Initialize with TLE and site coordinates.
        Caches Skyfield timescale, satellite, and site objects."""
        ...

    def state_at(self, t_utc: datetime) -> LosState:
        """Compute az/el/slant/sunlit/ascending/mag-az at a single instant."""
        ...

    def off_boresight_deg(self, los_ref: np.ndarray, los_t: np.ndarray) -> float:
        """Angular separation between two unit LOS vectors."""
        ...

    def solve_window(
        self,
        t_center_utc: datetime,
        criterion: Literal["off_boresight", "azimuth", "elevation"],
        half_width_deg: float,
        max_search_s: float = 120.0,
        tol_s: float = 0.01,
    ) -> Window:
        """Find t_start, t_stop where f(t) = half_width_deg.
        Uses coarse scan + brentq bracketing."""
        ...
```

### 3.2 `core/tle.py`

```python
def fetch_tle_celestrak(norad_id: int) -> tuple[str, str, str]:
    """Fetch TLE from Celestrak by NORAD ID. Returns (line1, line2, name)."""

def parse_tle(text: str) -> tuple[str, str, str | None]:
    """Parse pasted TLE text. Returns (line1, line2, name_or_none)."""

def parse_tle_file(text: str) -> list[tuple[str, str, str | None]]:
    """Parse multi-satellite .tle file."""

def validate_tle(line1: str, line2: str) -> tuple[datetime, float]:
    """Verify checksums, extract epoch, return (epoch_dt, age_days)."""

# SAR satellite dropdown — constellation names for user selection
# Implementation: use Celestrak group query or populate from known NORAD IDs
# User can also enter a custom NORAD ID for any satellite
SAR_CONSTELLATIONS = [
    "ICEYE", "Capella", "Umbra", "SAOCOM", "COSMO-SkyMed",
    "TerraSAR-X", "TanDEM-X", "RADARSAT",
]
```

### 3.3 `core/magnetic.py`

```python
def magnetic_declination(lat: float, lon: float, height_m: float,
                         year: int) -> float:
    """WMM declination in degrees (positive = east)."""
```

### 3.4 `core/export.py`

```python
def export_csv_card(window: Window, state: LosState, site_info: dict) -> bytes:
    """Single-row CSV with pointing card values."""

def export_csv_raw(samples: list[dict]) -> bytes:
    """Multi-row CSV: t_utc, az_true, el, off_boresight, slant_km."""

def export_pdf_card(window: Window, state: LosState, site_info: dict,
                    sat_name: str, tle_epoch: datetime, tle_age_days: float) -> bytes:
    """PDF pointing card — monospaced layout, single page."""
```

---

## 4. Window Solver Algorithm

### 4.1 Criterion Functions

Given scene center time `t_c` and reference LOS `los_ref = state_at(t_c).los_enu_unit`:

| Criterion        | f(t)                                                      |
| ---------------- | --------------------------------------------------------- |
| Off-boresight    | `arccos(dot(los_ref, state_at(t).los_enu_unit))`         |
| Azimuth only     | Wrapped azimuth difference `\|A(t) - A(t_c)\|` (0–180°) |
| Elevation only   | `\|E(t) - E(t_c)\|`                                     |

### 4.2 Solver Procedure

1. Compute `f(t_c) = 0`. If satellite below horizon at `t_c`, refuse with elevation value.
2. Coarse scan `f(t)` at 1 s steps out to ±120 s (both sides), until `f > X` is first crossed.
3. Bracket each crossing, then `scipy.optimize.brentq(f - X, t_lo, t_hi, xtol=tol_s)`.
4. Emit `t_start`, `t_stop`, and the two half-durations separately.

### 4.3 Edge Cases

| Case                                      | Handling                                                            |
| ----------------------------------------- | ------------------------------------------------------------------- |
| No crossing within ±120 s                 | Extend to ±300 s, then report "window exceeds search horizon"       |
| Satellite drops below horizon before X    | Clamp to horizon crossing, set `clamped_by_horizon = True`          |
| Azimuth-only near zenith (multi-valued)   | Warn user; recommend off-boresight                                  |
| Scene center at TCA (symmetric window)    | Works naturally — no special handling needed                        |
| Azimuth wrap across 360°/0°              | Use wrapped difference function, not raw subtraction                |

---

## 5. UI Layout

### 5.1 Sidebar — Inputs

**TLE source** (radio):
- Paste TLE (Line 1 + Line 2 + optional name)
- Fetch from Celestrak (SAR-sat dropdown + custom NORAD ID)
- Upload `.tle` file

**Site**:
- Lat, Lon, Height (m, ellipsoidal)
- Preset dropdown ("Singapore 1.29, 103.85, 20") + session-state saved presets

**Scene center time**:
- Date + time picker (single instant)
- Timezone toggle (UTC / local; default `Asia/Singapore` for display, UTC internally)

**Window definition**:
- Window criterion (radio): Off-boresight angle (default) / Azimuth only / Elevation only
- Half-width X (°, numeric input, default 5.0, range 0.1–45)

**Advanced** (collapsed expander):
- Apply refraction correction (checkbox, default off; uses Skyfield's built-in atmospheric refraction model for optical observations — note: negligible effect for radar LOS geometry, included for completeness)
- Magnetic model epoch (year, default = scene-center year)
- TLE age warning threshold (days, default 3)
- Solver time resolution (s, default 0.01)

**Action**: `▶ Compute Pointing` button

### 5.2 Main Pane — Tabs

**Tab 1 — Pointing Card** (primary deliverable):
```
Sat_Aim — Pointing Card
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Satellite : ICEYE-X12  (NORAD 12345)
TLE epoch : 2026-07-05 03:12 UTC (age 1.5 d)  ✓
Site      : 1.2900° N, 103.8500° E, 20 m
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SCENE CENTER
  Time (UTC)   : 2026-07-07 14:23:11.000
  Time (local) : 2026-07-07 22:23:11.000 SGT
  Azimuth      : 087.4° true   (086.9° magnetic)
  Elevation    : 62.1°
  Slant range  : 612.8 km
  Direction    : Ascending (Vz > 0)
  Sunlit       : ☀ Yes
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
POINTING WINDOW  (criterion: off-boresight ≤ 5.0°)
  Start (UTC)  : 2026-07-07 14:23:09.412
  Stop  (UTC)  : 2026-07-07 14:23:12.587
  Duration     : 3.175 s
  Asymmetry    : −0.412 s / +1.587 s about center
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```
`⬇ Download PDF` / `⬇ Download CSV` buttons.

**Tab 2 — Geometry vs Time**:
- Plotly line chart, x = seconds from scene center
- Multi-axis: Azimuth (°), Elevation (°), Off-boresight angle (°)
- Horizontal line at ±X threshold, vertical markers at window start/stop
- Range: max(±3× window half-duration, ±30 s)

**Tab 3 — Sky Plot**:
- Plotly polar scatter (azimuth as θ, elevation as r, inverted)
- Satellite arc in ±60 s neighborhood
- Scene-center point highlighted, window segment thickened

**Tab 4 — Map**:
- Folium map centered on site
- Site marker, satellite ground track (±60 s), sub-satellite point at scene center
- Window segment on ground track highlighted in distinct color

**Tab 5 — Raw Table**:
- Dense sample (100 Hz) across window ±20%
- Columns: `t_utc, az_true, el, off_boresight, slant_km`
- Downloadable CSV

**Tab 6 — Methodology**:
- Text description of how the calculations work, for transparency and auditability
- **Scene Center Geometry**: Skyfield topocentric position → ENU frame → azimuth (atan2 of E,N), elevation (arctan of Up/horizontal), slant range (Euclidean distance)
- **Off-boresight Angle**: `arccos(L̂(t_c) · L̂(t))` — angular separation between two unit LOS vectors in ENU. Physically corresponds to RCS loss off boresight for a corner reflector.
- **Window Solver Diagram**: Plotly figure showing:
  - f(t) curve (criterion function) vs time from scene center
  - Horizontal line at ±X threshold
  - Coarse scan steps (markers at 1 s intervals)
  - Bracket region (shaded)
  - brentq root (vertical line at converged t_start/t_stop)
- **Magnetic Declination**: WMM (World Magnetic Model) applied to true azimuth; formula: `az_mag = az_true + declination`
- **Assumptions & Limitations**: No atmospheric refraction by default (negligible for radar LOS), cylindrical shadow approximation for sunlit check, 0.01 s default solver resolution, TLE propagator accuracy degrades beyond ~3 days

---

## 6. Caching Strategy

- `@st.cache_resource` on `SatAim.__init__` — Skyfield objects are expensive to recreate
- `@st.cache_data` on TLE fetch — avoid re-fetching on every Streamlit rerun
- Computation results stored in `st.session_state` keyed by `(t_center, criterion, X)` so tab switching doesn't recompute

---

## 7. Validation Guards

| Guard                                            | Behavior                                  |
| ------------------------------------------------ | ----------------------------------------- |
| TLE age > warning threshold (default 3 d)        | Yellow warning banner                     |
| TLE age > 14 d (configurable)                    | Block computation, show error             |
| Satellite below horizon at scene center           | Refuse with "elevation = −X.X°" message   |
| X ≤ 0 or X > 45°                               | Refuse with range error                   |
| Site coords out of range (lat ±90, lon ±180)    | Refuse with validation error              |
| TLE checksum failure                             | Refuse with "invalid TLE" message         |

---

## 8. Testing (`tests/test_propagator.py`)

| Test                               | Expected Result                                            |
| ---------------------------------- | ---------------------------------------------------------- |
| Analytic sphere (TCA, equatorial)  | Window symmetric, duration = `2X / azimuth_rate`           |
| Azimuth wrap near due north        | Correct duration, no 360°/0° artifact                     |
| Off-boresight vs azimuth agreement | Windows agree within 10% for low-elevation pass            |
| Below-horizon rejection            | Clear error message with elevation value                   |
| TLE age validation                 | Warn > 3 d, block > 14 d                                 |

Use real TLE data (known ICEYE satellite) with fixed scene center for deterministic results.

---

## 9. Conda Environment

```yaml
name: sat_aim
channels:
  - conda-forge
  - defaults
dependencies:
  - python=3.13
  - pip
  - pip:
    - streamlit>=1.30
    - skyfield>=1.48
    - requests
    - pygeomag
    - fpdf2
    - folium
    - streamlit-folium
    - plotly
    - pandas
    - scipy
    - numpy
```

---

## 10. Acceptance Criteria

- [ ] User enters TLE + site + scene center + X → gets pointing card in < 2 s
- [ ] Window start/stop reproducible to ±0.01 s across runs
- [ ] Azimuth/elevation at scene center match Heavens-Above for validation TLE to ≤ 0.1°
- [ ] Off-boresight is the default criterion; azimuth-only available
- [ ] PDF pointing card renders start/stop with millisecond precision
- [ ] Below-horizon scene center is refused with a clear diagnostic
