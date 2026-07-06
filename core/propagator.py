"""Satellite propagation, geometry, and window solving.

Uses Skyfield for SGP4 propagation and topocentric geometry, and the World
Magnetic Model (via core.magnetic) for true-to-magnetic azimuth conversion.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Literal

import numpy as np
from scipy.optimize import brentq
from skyfield.api import EarthSatellite, load, wgs84

from core.magnetic import magnetic_declination


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class LosState:
    """Line-of-sight state at a single instant."""

    t_utc: datetime
    az_true_deg: float        # 0-360, clockwise from true north
    el_deg: float             # elevation / grazing angle
    slant_km: float
    sunlit: bool
    ascending: bool           # radial velocity > 0
    los_enu_unit: np.ndarray  # unit LOS vector in ENU frame (E, N, U)
    az_mag_deg: float         # magnetic azimuth


@dataclass
class Window:
    """Visibility / aim window around a centre time."""

    t_start_utc: datetime
    t_stop_utc: datetime
    duration_s: float
    minus_s: float            # t_center - t_start
    plus_s: float             # t_stop - t_center
    clamped_by_horizon: bool
    criterion: str
    half_width_deg: float


@dataclass
class PassInfo:
    """Rise / peak / set times and geometry for a single satellite pass."""

    rise_utc: datetime
    peak_utc: datetime
    set_utc: datetime
    duration_s: float
    max_el_deg: float
    rise_az_deg: float        # azimuth at rise
    set_az_deg: float         # azimuth at set


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _wrap_az_diff(a1: float, a2: float) -> float:
    """Wrapped azimuth difference in [0, 180] degrees.

    Uses modulo arithmetic so the result is always the smaller angle between
    the two azimuths, regardless of wrap-around at 360.
    """
    diff = abs(a1 - a2) % 360.0
    if diff > 180.0:
        diff = 360.0 - diff
    return diff


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class SatAim:
    """Satellite line-of-sight calculator and window solver.

    Parameters
    ----------
    tle_lines : tuple of str
        Two TLE lines (line 1 and line 2).
    name : str or None
        Optional satellite name.  Falls back to the TLE designator.
    lat : float
        Observer geodetic latitude in degrees.
    lon : float
        Observer geodetic longitude in degrees.
    height_m : float
        Observer height above WGS-84 ellipsoid in metres.
    """

    def __init__(
        self,
        tle_lines: tuple[str, str],
        name: str | None,
        lat: float,
        lon: float,
        height_m: float,
    ) -> None:
        self.ts = load.timescale()
        self.sat = EarthSatellite(tle_lines[0], tle_lines[1], name, self.ts)
        self.name = name or self.sat.name
        self.site = wgs84.latlon(lat, lon, height_m)
        self.lat = lat
        self.lon = lon
        self.height_m = height_m

    # ------------------------------------------------------------------
    # State at a single time
    # ------------------------------------------------------------------

    def state_at(
        self,
        t_utc: datetime,
        mag_year: int | None = None,
    ) -> LosState:
        """Return the satellite line-of-sight state at *t_utc*.

        Parameters
        ----------
        t_utc : datetime
            Observation time (must be timezone-aware UTC, or naive datetimes
            are treated as UTC).
        mag_year : int or None
            Decimal year for the magnetic model.  Defaults to the year of
            *t_utc*.

        Returns
        -------
        LosState
        """
        if t_utc.tzinfo is None:
            t_utc = t_utc.replace(tzinfo=timezone.utc)

        t_sf = self.ts.utc(
            t_utc.year, t_utc.month, t_utc.day,
            t_utc.hour, t_utc.minute, t_utc.second + t_utc.microsecond / 1e6,
        )

        # Topocentric alt / az / distance
        diff = self.sat - self.site
        topo = diff.at(t_sf)
        alt, az, distance = topo.altaz()

        az_deg = float(az.degrees)
        el_deg = float(alt.degrees)
        slant_km = float(distance.km)

        # ENU unit vector  (E, N, U)
        az_rad = math.radians(az_deg)
        el_rad = math.radians(el_deg)
        los_enu = np.array([
            math.sin(az_rad) * math.cos(el_rad),
            math.cos(az_rad) * math.cos(el_rad),
            math.sin(el_rad),
        ])
        los_enu_unit = los_enu / np.linalg.norm(los_enu)

        # Ascending: geocentric radial velocity > 0
        sat_geo = self.sat.at(t_sf)
        pos_geo = sat_geo.position.km
        vel_geo = sat_geo.velocity.km_per_s
        radial_vel = float(np.dot(vel_geo, pos_geo) / np.linalg.norm(pos_geo))
        ascending = radial_vel > 0.0

        # Sunlit: dot-product shadow test
        eph = load("de421.bsp")
        sat_pos = sat_geo.position.km
        sun_pos = eph["earth"].at(t_sf).observe(eph["sun"]).position.km
        sat_to_sun = sun_pos - sat_pos
        sat_to_earth = -sat_pos
        is_sunlit = bool(np.dot(sat_to_sun, sat_to_earth) < 0.0)

        # Magnetic azimuth
        year = mag_year if mag_year is not None else t_utc.year
        decl = magnetic_declination(self.lat, self.lon, self.height_m, year)
        az_mag_deg = (az_deg + decl) % 360.0

        return LosState(
            t_utc=t_utc,
            az_true_deg=az_deg,
            el_deg=el_deg,
            slant_km=slant_km,
            sunlit=is_sunlit,
            ascending=ascending,
            los_enu_unit=los_enu_unit,
            az_mag_deg=az_mag_deg,
        )

    # ------------------------------------------------------------------
    # Off-boresight angle
    # ------------------------------------------------------------------

    @staticmethod
    def off_boresight_deg(los_ref: np.ndarray, los_t: np.ndarray) -> float:
        """Angular separation between two unit LOS vectors, in degrees."""
        dot = float(np.clip(np.dot(los_ref, los_t), -1.0, 1.0))
        return math.degrees(math.acos(dot))

    # ------------------------------------------------------------------
    # Window solver
    # ------------------------------------------------------------------

    def solve_window(
        self,
        t_center_utc: datetime,
        criterion: Literal["off_boresight", "azimuth", "elevation"],
        half_width_deg: float,
        max_search_s: float = 120.0,
        tol_s: float = 0.01,
        mag_year: int | None = None,
    ) -> Window:
        """Solve for the time window around *t_center_utc*.

        The window extends symmetrically (in the chosen metric) until the
        half-width is reached or the satellite drops below the horizon.

        Parameters
        ----------
        t_center_utc : datetime
            Centre of the search window.
        criterion : str
            ``"off_boresight"`` -- angle between LOS at *t* and at center.
            ``"azimuth"`` -- wrapped azimuth difference.
            ``"elevation"`` -- absolute elevation difference.
        half_width_deg : float
            Half-width of the window in degrees.  Must be in (0, 45].
        max_search_s : float
            Maximum seconds to search in each direction from center.
        tol_s : float
            Tolerance for the Brent root-finder, in seconds.
        mag_year : int or None
            Decimal year for the magnetic model.

        Returns
        -------
        Window

        Raises
        ------
        ValueError
            If *half_width_deg* is out of range or the satellite is below
            the horizon at *t_center_utc*.
        """
        if not (0.0 < half_width_deg <= 45.0):
            raise ValueError(
                f"half_width_deg must be in (0, 45], got {half_width_deg}"
            )

        ref = self.state_at(t_center_utc, mag_year=mag_year)
        if ref.el_deg <= 0.0:
            raise ValueError(
                "Satellite is below the horizon at t_center_utc "
                f"(el={ref.el_deg:.2f} deg)"
            )

        # -- objective function ------------------------------------------------
        def f(dt_offset_s: float) -> float:
            t = t_center_utc + timedelta(seconds=dt_offset_s)
            st = self.state_at(t, mag_year=mag_year)
            if criterion == "off_boresight":
                return self.off_boresight_deg(ref.los_enu_unit, st.los_enu_unit)
            elif criterion == "azimuth":
                return _wrap_az_diff(ref.az_true_deg, st.az_true_deg)
            elif criterion == "elevation":
                return abs(st.el_deg - ref.el_deg)
            else:
                raise ValueError(f"Unknown criterion: {criterion!r}")

        # -- crossing finder ---------------------------------------------------
        def find_crossing(direction: int) -> tuple[datetime, bool]:
            """Scan outward from center and bracket the crossing.

            Returns (boundary_time, clamped_by_horizon).
            """
            prev_t = t_center_utc
            prev_f = 0.0

            n_steps = int(max_search_s)
            for i in range(1, n_steps + 1):
                dt_s = direction * float(i)
                t_i = t_center_utc + timedelta(seconds=dt_s)
                st_i = self.state_at(t_i, mag_year=mag_year)

                if st_i.el_deg <= 0.0:
                    # Horizon reached before criterion crossing.
                    if i == 1:
                        return t_center_utc, True
                    return prev_t, True

                f_i = f(dt_s)
                if f_i >= half_width_deg:
                    # Bracket: [prev_t .. t_i]
                    lo = direction * float(i - 1)
                    hi = direction * float(i)
                    try:
                        root = brentq(
                            lambda s: f(s) - half_width_deg,
                            lo, hi, xtol=tol_s,
                        )
                    except ValueError:
                        root = lo
                    return t_center_utc + timedelta(seconds=root), False

                prev_t = t_i
                prev_f = f_i

            # Reached max_search_s without crossing.
            return t_center_utc + timedelta(seconds=direction * max_search_s), False

        # -- search both directions -------------------------------------------
        t_fwd, clamped_fwd = find_crossing(+1)
        t_bwd, clamped_bwd = find_crossing(-1)

        clamped_by_horizon = clamped_fwd or clamped_bwd

        t_start = min(t_bwd, t_fwd)
        t_stop = max(t_bwd, t_fwd)

        # Half-width actually used (may differ slightly at horizon boundary).
        if criterion == "off_boresight":
            hw = self.off_boresight_deg(
                ref.los_enu_unit,
                self.state_at(t_stop, mag_year=mag_year).los_enu_unit,
            )
        elif criterion == "azimuth":
            hw = _wrap_az_diff(
                ref.az_true_deg,
                self.state_at(t_stop, mag_year=mag_year).az_true_deg,
            )
        else:
            hw = abs(
                self.state_at(t_stop, mag_year=mag_year).el_deg - ref.el_deg
            )

        return Window(
            t_start_utc=t_start,
            t_stop_utc=t_stop,
            duration_s=(t_stop - t_start).total_seconds(),
            minus_s=(t_center_utc - t_start).total_seconds(),
            plus_s=(t_stop - t_center_utc).total_seconds(),
            clamped_by_horizon=clamped_by_horizon,
            criterion=criterion,
            half_width_deg=hw,
        )

    # ------------------------------------------------------------------
    # Next pass finder
    # ------------------------------------------------------------------

    def next_pass(
        self,
        t_utc: datetime,
        max_search_h: float = 24.0,
        step_s: float = 30.0,
        tol_s: float = 0.1,
        mag_year: int | None = None,
    ) -> PassInfo:
        """Find the next satellite pass after *t_utc*.

        Scans forward from *t_utc* in coarse steps until the satellite
        rises above the horizon, then refines rise/peak/set times with
        Brent's method.

        Parameters
        ----------
        t_utc : datetime
            Start of the search (timezone-aware UTC, or naive treated as UTC).
        max_search_h : float
            Maximum hours to search forward before giving up.
        step_s : float
            Coarse scan step in seconds.
        tol_s : float
            Tolerance for Brent root-finder, in seconds.
        mag_year : int or None
            Decimal year for the magnetic model.

        Returns
        -------
        PassInfo

        Raises
        ------
        ValueError
            If no pass is found within *max_search_h* hours.
        """
        if t_utc.tzinfo is None:
            t_utc = t_utc.replace(tzinfo=timezone.utc)

        max_search_s = max_search_h * 3600.0

        # -- elevation function -----------------------------------------------
        def el_at(dt_s: float) -> float:
            t = t_utc + timedelta(seconds=dt_s)
            return self.state_at(t, mag_year=mag_year).el_deg

        # -- coarse scan: find first dt where el > 0 --------------------------
        n_steps = int(max_search_s / step_s)
        prev_dt = 0.0
        prev_el = el_at(0.0)

        rise_dt = None
        for i in range(1, n_steps + 1):
            cur_dt = float(i) * step_s
            cur_el = el_at(cur_dt)
            if prev_el <= 0.0 and cur_el > 0.0:
                # Bracket found: rise is between prev_dt and cur_dt.
                rise_dt = brentq(lambda s: el_at(s), prev_dt, cur_dt, xtol=tol_s)
                break
            prev_dt = cur_dt
            prev_el = cur_el

        if rise_dt is None:
            raise ValueError(
                f"No pass found within {max_search_h:.0f} h of {t_utc.isoformat()}"
            )

        # -- find peak: scan forward from rise until el starts decreasing -----
        t_rise = t_utc + timedelta(seconds=rise_dt)
        peak_dt = rise_dt
        peak_el = el_at(rise_dt)

        # Scan in step_s increments past rise to find the peak region.
        scan_start = rise_dt
        scan_end = rise_dt + max_search_s  # generous upper bound
        prev_dt = rise_dt
        prev_el = peak_el

        peak_found = False
        for i in range(1, int((scan_end - scan_start) / step_s) + 1):
            cur_dt = scan_start + float(i) * step_s
            cur_el = el_at(cur_dt)
            if cur_el <= 0.0:
                # Set happened between prev_dt and cur_dt.  Peak was the
                # maximum we saw.
                peak_found = True
                break
            if cur_el > peak_el:
                peak_el = cur_el
                peak_dt = cur_dt
            prev_dt = cur_dt
            prev_el = cur_el

        if not peak_found:
            raise ValueError("Could not determine pass peak (search exhausted).")

        # Refine peak with golden-section search on [-step, +step] around peak_dt
        # Simple approach: scan finer grid around peak_dt
        fine_step = max(step_s / 10.0, tol_s * 2)
        best_dt = peak_dt
        best_el = peak_el
        for delta in np.arange(-step_s, step_s + fine_step, fine_step):
            cand_dt = peak_dt + delta
            if cand_dt < rise_dt:
                continue
            cand_el = el_at(cand_dt)
            if cand_el > best_el:
                best_el = cand_el
                best_dt = cand_dt
        peak_dt = best_dt

        # -- find set: bracket el crossing 0 after peak -----------------------
        prev_dt = peak_dt
        prev_el = el_at(peak_dt)
        set_dt = None
        for i in range(1, int(max_search_s / step_s) + 1):
            cur_dt = peak_dt + float(i) * step_s
            cur_el = el_at(cur_dt)
            if prev_el > 0.0 and cur_el <= 0.0:
                set_dt = brentq(lambda s: el_at(s), prev_dt, cur_dt, xtol=tol_s)
                break
            prev_dt = cur_dt
            prev_el = cur_el

        if set_dt is None:
            raise ValueError("Could not determine pass set time.")

        # -- build result ------------------------------------------------------
        t_set = t_utc + timedelta(seconds=set_dt)
        rise_state = self.state_at(t_rise, mag_year=mag_year)
        set_state = self.state_at(t_set, mag_year=mag_year)

        return PassInfo(
            rise_utc=t_rise,
            peak_utc=t_utc + timedelta(seconds=peak_dt),
            set_utc=t_set,
            duration_s=set_dt - rise_dt,
            max_el_deg=best_el,
            rise_az_deg=rise_state.az_true_deg,
            set_az_deg=set_state.az_true_deg,
        )
