"""Sat Aim -- Satellite pointing geometry calculator.

Streamlit application with six tabs:
  1. Pointing Card
  2. Geometry Plot
  3. Sky Plot
  4. Map
  5. Raw Table
  6. Methodology
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.export import export_csv_card, export_csv_raw, export_pdf_card
from core.magnetic import magnetic_declination
from core.propagator import LosState, PassInfo, SatAim, Window
from core.tle import SAR_SATELLITES, fetch_tle_celestrak, parse_tle, validate_tle

# Folium imports -- guard so the app still loads if the package is missing.
try:
    import folium
    from streamlit_folium import st_folium

    HAS_FOLIUM = True
except ImportError:
    HAS_FOLIUM = False


# ---------------------------------------------------------------------------
# 1. Page config
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Sat Aim", layout="wide")

st.title("Sat Aim -- Satellite Pointing Geometry")


# ---------------------------------------------------------------------------
# 2. Sidebar inputs
# ---------------------------------------------------------------------------

st.sidebar.header("TLE Source")

tle_mode = st.sidebar.selectbox("TLE input mode", ["Celestrak", "Paste TLE"])

tle_lines: tuple[str, str] | None = None
sat_name: str | None = None

if tle_mode == "Celestrak":
    sat_filter = st.sidebar.text_input("Satellite name filter (substring)")
    try:
        tle_list = fetch_tle_celestrak(
            sat_filter if sat_filter else None
        )
        sat_options = [name for _, name in tle_list]
        chosen = st.sidebar.selectbox("Satellite", sat_options) if sat_options else None
        if chosen:
            for (l1, l2), name in tle_list:
                if name == chosen:
                    tle_lines = (l1, l2)
                    sat_name = name
                    break
    except Exception as exc:
        st.sidebar.error(f"CelesTrak fetch failed: {exc}")
else:
    tle_text = st.sidebar.text_area(
        "Paste TLE (name line optional, then line 1 and line 2)",
        height=120,
    )
    if tle_text.strip():
        try:
            (l1, l2), name = parse_tle(tle_text)
            if validate_tle(l1, l2):
                tle_lines = (l1, l2)
                sat_name = name
            else:
                st.sidebar.error("TLE validation failed.")
        except ValueError as exc:
            st.sidebar.error(f"TLE parse error: {exc}")

st.sidebar.divider()
st.sidebar.header("Observer Site")

lat = st.sidebar.number_input("Latitude (deg)", value=1.29, step=0.01, format="%.2f")
lon = st.sidebar.number_input("Longitude (deg)", value=103.85, step=0.01, format="%.2f")
height_m = st.sidebar.slider("Site height (m)", min_value=0, max_value=5000, value=20)

st.sidebar.divider()
st.sidebar.header("Scene Center Time (UTC)")

# Pre-fill from session state if user jumped to a pass time.
_jump: datetime | None = st.session_state.pop("_jump_time", None)
_default_dt = _jump if _jump is not None else datetime(2026, 7, 7, 14, 23, 11)

scene_date = st.sidebar.date_input("Date", value=_default_dt.date())
scene_time = st.sidebar.time_input("Time", value=_default_dt.time(), step=60)

st.sidebar.divider()
st.sidebar.header("Window Parameters")

criterion = st.sidebar.selectbox(
    "Criterion",
    ["off_boresight", "azimuth", "elevation"],
    index=0,
)
half_width = st.sidebar.slider("Half-width (deg)", min_value=1, max_value=45, value=5)

compute_btn = st.sidebar.button("Compute", type="primary", use_container_width=True)


# ---------------------------------------------------------------------------
# 3. Session state caching
# ---------------------------------------------------------------------------

def _make_key(t_center: datetime, criterion: str, half_width: float) -> tuple:
    """Build a hashable cache key from the computation parameters."""
    return (t_center.isoformat(), criterion, half_width)


_RESULTS_KEY = "_sat_aim_results"


def run_computation(
    tle_lines: tuple[str, str],
    sat_name: str | None,
    lat: float,
    lon: float,
    height_m: float,
    t_center: datetime,
    criterion: str,
    half_width: float,
) -> dict:
    """Run the SatAim computation and return a results dict.

    Cached in st.session_state under _RESULTS_KEY.
    """
    key = _make_key(t_center, criterion, half_width)
    if _RESULTS_KEY in st.session_state and st.session_state[_RESULTS_KEY].get("_key") == key:
        return st.session_state[_RESULTS_KEY]

    sat_aim = SatAim(tle_lines, sat_name, lat, lon, height_m)
    ref_state = sat_aim.state_at(t_center)
    window = sat_aim.solve_window(t_center, criterion, half_width)

    # Build the 1-second table over the window.
    n_steps = int(window.duration_s) + 1
    times = [window.t_start_utc + timedelta(seconds=i) for i in range(n_steps)]
    raw_states = [sat_aim.state_at(t) for t in times]
    rows = []
    for st_row in raw_states:
        rows.append(
            {
                "Time": st_row.t_utc.strftime("%Y-%m-%d %H:%M:%S"),
                "Az True": round(st_row.az_true_deg, 2),
                "Az Mag": round(st_row.az_mag_deg, 2),
                "El": round(st_row.el_deg, 2),
                "Slant": round(st_row.slant_km, 2),
                "Sunlit": st_row.sunlit,
                "Ascending": st_row.ascending,
            }
        )
    df_raw = pd.DataFrame(rows)

    # Geometry plot arrays (1-second resolution).
    times_geo = times
    az_geo = [sat_aim.state_at(t).az_true_deg for t in times_geo]
    el_geo = [sat_aim.state_at(t).el_deg for t in times_geo]

    # Magnetic declination value.
    mag_year = t_center.year
    mag_dec = magnetic_declination(lat, lon, height_m, mag_year)

    results = {
        "_key": key,
        "sat_aim": sat_aim,
        "ref_state": ref_state,
        "window": window,
        "df_raw": df_raw,
        "raw_states": raw_states,
        "times_geo": times_geo,
        "az_geo": az_geo,
        "el_geo": el_geo,
        "mag_dec": mag_dec,
        "t_center": t_center,
        "criterion": criterion,
        "half_width": half_width,
    }
    st.session_state[_RESULTS_KEY] = results
    return results


# Trigger computation when the button is pressed.
if compute_btn:
    if tle_lines is None:
        st.error("Please provide a valid TLE first.")
    else:
        t_center = datetime.combine(scene_date, scene_time).replace(tzinfo=timezone.utc)
        try:
            run_computation(
                tle_lines, sat_name, lat, lon, height_m, t_center, criterion, half_width
            )
        except ValueError as exc:
            msg = str(exc)
            if "below the horizon" in msg:
                # Satellite is below horizon — find the next pass.
                sat_aim = SatAim(tle_lines, sat_name, lat, lon, height_m)
                try:
                    pass_info = sat_aim.next_pass(t_center, max_search_h=24)
                    st.warning(
                        f"Satellite is **below the horizon** at the requested time "
                        f"(el = {sat_aim.state_at(t_center).el_deg:.1f}°).\n\n"
                        f"**Next pass:**  "
                        f"rise {pass_info.rise_utc.strftime('%H:%M:%S')} UTC "
                        f"(az {pass_info.rise_az_deg:.0f}°) → "
                        f"peak {pass_info.peak_utc.strftime('%H:%M:%S')} UTC "
                        f"(el {pass_info.max_el_deg:.1f}°) → "
                        f"set {pass_info.set_utc.strftime('%H:%M:%S')} UTC "
                        f"(az {pass_info.set_az_deg:.0f}°)  "
                        f"— duration {pass_info.duration_s:.0f} s"
                    )
                    # Offer a button to jump to the pass peak time.
                    if st.button(
                        f"Use peak time ({pass_info.peak_utc.strftime('%Y-%m-%d %H:%M:%S')} UTC)",
                        type="primary",
                    ):
                        st.session_state["_jump_time"] = pass_info.peak_utc
                        st.rerun()
                except ValueError as exc2:
                    st.error(f"Satellite is below horizon and no pass found in 24 h: {exc2}")
            else:
                st.error(f"Computation failed: {exc}")
        except Exception as exc:
            st.error(f"Computation failed: {exc}")


# ---------------------------------------------------------------------------
# Retrieve the latest cached results (if any).
# ---------------------------------------------------------------------------

_results: dict | None = st.session_state.get(_RESULTS_KEY)


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    ["Pointing Card", "Geometry Plot", "Sky Plot", "Map", "Raw Table", "Methodology"]
)


# ===== Tab 1 -- Pointing Card ==============================================

with tab1:
    if _results is None:
        st.info("Press **Compute** in the sidebar to generate results.")
    else:
        ref: LosState = _results["ref_state"]
        win: Window = _results["window"]

        st.subheader("Pointing Card")

        # Metric widgets.
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Azimuth (true)", f"{ref.az_true_deg:.2f} deg")
        m2.metric("Elevation", f"{ref.el_deg:.2f} deg")
        m3.metric("Slant Range", f"{ref.slant_km:.1f} km")
        m4.metric("Window Duration", f"{win.duration_s:.1f} s")

        st.divider()

        # Info table.
        info_data = {
            "Parameter": [
                "Azimuth (magnetic)",
                "Sunlit",
                "Ascending",
                "Window start (UTC)",
                "Window stop (UTC)",
                "Window minus",
                "Window plus",
                "Clamped by horizon",
                "Criterion",
                "Half-width used",
            ],
            "Value": [
                f"{ref.az_mag_deg:.2f} deg",
                "Yes" if ref.sunlit else "No",
                "Yes" if ref.ascending else "No",
                win.t_start_utc.strftime("%Y-%m-%d %H:%M:%S"),
                win.t_stop_utc.strftime("%Y-%m-%d %H:%M:%S"),
                f"{win.minus_s:.2f} s",
                f"{win.plus_s:.2f} s",
                "Yes" if win.clamped_by_horizon else "No",
                win.criterion,
                f"{win.half_width_deg:.2f} deg",
            ],
        }
        st.table(pd.DataFrame(info_data))

        # Download buttons.
        st.divider()
        dl1, dl2 = st.columns(2)
        sa: SatAim = _results["sat_aim"]
        csv_card_bytes = export_csv_card(win, ref, sa.name, sa.lat, sa.lon, sa.height_m)
        dl1.download_button(
            "Download CSV Card",
            data=csv_card_bytes,
            file_name="pointing_card.csv",
            mime="text/csv",
        )
        pdf_card_bytes = export_pdf_card(win, ref, sa.name, sa.lat, sa.lon, sa.height_m)
        dl2.download_button(
            "Download PDF Card",
            data=pdf_card_bytes,
            file_name="pointing_card.pdf",
            mime="application/pdf",
        )


# ===== Tab 2 -- Geometry Plot ==============================================

with tab2:
    if _results is None:
        st.info("Press **Compute** in the sidebar to generate results.")
    else:
        times_geo = _results["times_geo"]
        az_geo = _results["az_geo"]
        el_geo = _results["el_geo"]
        t_center = _results["t_center"]
        win: Window = _results["window"]
        hw = _results["half_width"]

        fig = go.Figure()

        # Azimuth trace (left y-axis).
        fig.add_trace(
            go.Scatter(
                x=times_geo,
                y=az_geo,
                mode="lines",
                name="Azimuth (deg)",
                line=dict(color="royalblue"),
            )
        )

        # Elevation trace (right y-axis).
        fig.add_trace(
            go.Scatter(
                x=times_geo,
                y=el_geo,
                mode="lines",
                name="Elevation (deg)",
                line=dict(color="firebrick"),
                yaxis="y2",
            )
        )

        # Vertical line at scene center.
        fig.add_vline(
            x=t_center, line_dash="dash", line_color="green",
            annotation_text="Scene center",
        )

        # Horizontal lines at +/- half-width on the relevant axis.
        ref_state: LosState = _results["ref_state"]
        crit = _results["criterion"]
        if crit == "off_boresight":
            # Off-boresight is not directly plotted as a y-value, so skip hlines.
            pass
        elif crit == "azimuth":
            fig.add_hline(y=ref_state.az_true_deg + hw, line_dash="dot", line_color="orange")
            fig.add_hline(y=ref_state.az_true_deg - hw, line_dash="dot", line_color="orange")
        elif crit == "elevation":
            fig.add_hline(y=ref_state.el_deg + hw, line_dash="dot", line_color="orange", yaxis="y2")
            fig.add_hline(y=ref_state.el_deg - hw, line_dash="dot", line_color="orange", yaxis="y2")

        # Shaded window region.
        fig.add_vrect(
            x0=win.t_start_utc,
            x1=win.t_stop_utc,
            fillcolor="LightGreen",
            opacity=0.2,
            line_width=0,
            annotation_text="Window",
        )

        fig.update_layout(
            title="Azimuth and Elevation vs Time",
            xaxis_title="Time (UTC)",
            yaxis=dict(title="Azimuth (deg)", side="left"),
            yaxis2=dict(title="Elevation (deg)", overlaying="y", side="right"),
            legend=dict(x=0.01, y=0.99),
            height=500,
        )

        st.plotly_chart(fig, width="stretch")


# ===== Tab 3 -- Sky Plot ===================================================

with tab3:
    if _results is None:
        st.info("Press **Compute** in the sidebar to generate results.")
    else:
        sat_aim: SatAim = _results["sat_aim"]
        ref_state: LosState = _results["ref_state"]
        win: Window = _results["window"]
        t_center = _results["t_center"]

        # Sample track points.
        n_track = int(win.duration_s) + 1
        track_times = [win.t_start_utc + timedelta(seconds=i) for i in range(n_track)]
        track_states = [sat_aim.state_at(t) for t in track_times]

        track_az = [s.az_true_deg for s in track_states]
        track_r = [90.0 - s.el_deg for s in track_states]

        # Center marker.
        center_az = ref_state.az_true_deg
        center_r = 90.0 - ref_state.el_deg

        # Window endpoints.
        s_start = sat_aim.state_at(win.t_start_utc)
        s_stop = sat_aim.state_at(win.t_stop_utc)

        fig_sky = go.Figure()

        # Track line.
        fig_sky.add_trace(
            go.Scatterpolar(
                r=track_r,
                theta=track_az,
                mode="lines",
                name="Track",
                line=dict(color="royalblue", width=2),
            )
        )

        # Center marker.
        fig_sky.add_trace(
            go.Scatterpolar(
                r=[center_r],
                theta=[center_az],
                mode="markers",
                name="Scene center",
                marker=dict(color="green", size=12, symbol="x"),
            )
        )

        # Window endpoints.
        fig_sky.add_trace(
            go.Scatterpolar(
                r=[90.0 - s_start.el_deg, 90.0 - s_stop.el_deg],
                theta=[s_start.az_true_deg, s_stop.az_true_deg],
                mode="markers",
                name="Window endpoints",
                marker=dict(color="orange", size=10, symbol="diamond"),
            )
        )

        fig_sky.update_layout(
            polar=dict(
                radialaxis=dict(range=[0, 90], title="Elevation offset (deg)"),
                angularaxis=dict(direction="clockwise", rotation=90),
            ),
            title="Sky Plot (North up, clockwise)",
            height=550,
        )

        st.plotly_chart(fig_sky, width="stretch")


# ===== Tab 4 -- Map ========================================================

with tab4:
    if not HAS_FOLIUM:
        st.warning("Install folium and streamlit-folium to see the map tab.")
    elif _results is None:
        st.info("Press **Compute** in the sidebar to generate results.")
    else:
        ref_state: LosState = _results["ref_state"]
        win: Window = _results["window"]

        m = folium.Map(location=[lat, lon], zoom_start=10)

        # Site marker.
        popup_html = (
            f"<b>Site</b><br>"
            f"Az: {ref_state.az_true_deg:.2f} deg<br>"
            f"El: {ref_state.el_deg:.2f} deg<br>"
            f"Slant: {ref_state.slant_km:.1f} km"
        )
        folium.Marker(
            [lat, lon],
            popup=folium.Popup(popup_html, max_width=250),
            tooltip="Observer site",
            icon=folium.Icon(color="blue", icon="crosshairs", prefix="fa"),
        ).add_to(m)

        # Line from site in the direction of azimuth.
        # Draw ~100 km line in the pointing direction.
        az_rad = math.radians(ref_state.az_true_deg)
        # Approximate: 1 deg lat ~ 111 km, 1 deg lon ~ 111*cos(lat) km.
        dist_deg = 1.0  # roughly 111 km
        end_lat = lat + dist_deg * math.cos(az_rad)
        end_lon = lon + dist_deg * math.sin(az_rad) / max(math.cos(math.radians(lat)), 0.01)
        folium.PolyLine(
            locations=[[lat, lon], [end_lat, end_lon]],
            color="red",
            weight=3,
            tooltip=f"Az {ref_state.az_true_deg:.1f} deg",
        ).add_to(m)

        st_folium(m, width=700, height=500)


# ===== Tab 5 -- Raw Table ==================================================

with tab5:
    if _results is None:
        st.info("Press **Compute** in the sidebar to generate results.")
    else:
        df_raw: pd.DataFrame = _results["df_raw"]

        st.subheader("Raw Data Table (1-second intervals)")
        st.dataframe(
            df_raw,
            width="stretch",
            height=600,
        )

        # CSV download.
        csv_raw_bytes = export_csv_raw(_results["raw_states"])
        st.download_button(
            "Download Raw CSV",
            data=csv_raw_bytes,
            file_name="raw_geometry.csv",
            mime="text/csv",
        )


# ===== Tab 6 -- Methodology ================================================


def _render_methodology(
    ref_state: LosState,
    sat_aim: SatAim,
    t_center: datetime,
    half_width: float,
    window: Window,
    mag_dec: float,
) -> None:
    """Render the methodology tab content with dynamic diagrams."""

    # --- Section 1: Scene Center Geometry ------------------------------------
    st.subheader("1. Scene Center Geometry")
    st.markdown(
        r"""
The satellite line-of-sight (LOS) vector is expressed in the local **ENU**
(East-North-Up) coordinate frame centred on the observer site:

$$
\hat{\mathbf{u}}_{\text{ENU}} =
\begin{pmatrix}
\sin A_z \cos E_l \\
\cos A_z \cos E_l \\
\sin E_l
\end{pmatrix}
$$

where $A_z$ is the true azimuth (clockwise from north) and $E_l$ is the
elevation angle above the local horizon.
"""
    )

    # --- Section 2: Off-boresight Angle -------------------------------------
    st.subheader("2. Off-boresight Angle")
    st.markdown(
        r"""
The off-boresight angle between two LOS unit vectors $\hat{\mathbf{u}}_{\text{ref}}$
(at scene center) and $\hat{\mathbf{u}}(t)$ (at any other time) is:

$$
\theta_{\text{OB}} = \arccos\!\left(
    \hat{\mathbf{u}}_{\text{ref}} \cdot \hat{\mathbf{u}}(t)
\right)
$$

This is the primary criterion used to define the pointing window.
"""
    )

    # --- Section 3: Solver Diagram ------------------------------------------
    st.subheader("3. Solver Diagram")

    # Build f(t) = off_boresight_deg over the search range.
    half_dur = max(window.duration_s / 2.0 + 5.0, 30.0)
    dt_offsets = np.arange(-half_dur, half_dur + 1.0, 1.0)
    f_values = []
    for dt_s in dt_offsets:
        t_i = t_center + timedelta(seconds=float(dt_s))
        st_i = sat_aim.state_at(t_i)
        val = sat_aim.off_boresight_deg(ref_state.los_enu_unit, st_i.los_enu_unit)
        f_values.append(val)

    fig_solver = go.Figure()

    # f(t) curve.
    fig_solver.add_trace(
        go.Scatter(
            x=dt_offsets,
            y=f_values,
            mode="lines",
            name="f(t) = off-boresight (deg)",
            line=dict(color="royalblue", width=2),
        )
    )

    # Coarse scan points (every 1 s) -- orange dots.
    fig_solver.add_trace(
        go.Scatter(
            x=dt_offsets[::1],
            y=f_values[::1],
            mode="markers",
            name="Coarse scan (1 s)",
            marker=dict(color="orange", size=4),
        )
    )

    # Threshold line.
    fig_solver.add_hline(
        y=half_width,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Threshold = {half_width} deg",
    )

    # Green shaded regions for brackets (where f(t) < half_width).
    in_bracket = [v < half_width for v in f_values]
    # Find contiguous bracket regions.
    bracket_starts: list[int] = []
    bracket_ends: list[int] = []
    for i, ib in enumerate(in_bracket):
        if ib and (i == 0 or not in_bracket[i - 1]):
            bracket_starts.append(i)
        if ib and (i == len(in_bracket) - 1 or not in_bracket[i + 1]):
            bracket_ends.append(i)
    for bs, be in zip(bracket_starts, bracket_ends):
        fig_solver.add_vrect(
            x0=dt_offsets[bs],
            x1=dt_offsets[be],
            fillcolor="LightGreen",
            opacity=0.3,
            line_width=0,
        )

    # Blue vertical line at scene center (dt=0).
    fig_solver.add_vline(x=0, line_dash="solid", line_color="blue", line_width=2)

    # Vertical lines at window boundaries.
    fig_solver.add_vline(
        x=-window.minus_s, line_dash="dot", line_color="purple",
        annotation_text="Window start",
    )
    fig_solver.add_vline(
        x=window.plus_s, line_dash="dot", line_color="purple",
        annotation_text="Window stop",
    )

    fig_solver.update_layout(
        xaxis_title="Time offset from scene center (s)",
        yaxis_title="Off-boresight angle (deg)",
        title="Solver: f(t) and bracket detection",
        height=450,
    )
    st.plotly_chart(fig_solver, width="stretch")

    # --- Section 4: Magnetic Declination ------------------------------------
    st.subheader("4. Magnetic Declination")
    lat_site = sat_aim.lat
    lon_site = sat_aim.lon
    st.markdown(
        f"""
The **World Magnetic Model (WMM 2025)** is used to convert true azimuth to
magnetic azimuth:

$$
A_{{z,\\text{{mag}}}} = (A_{{z,\\text{{true}}}} + \\delta_{{mag}}) \\mod 360
$$

At the observer location (lat={lat_site:.2f} / lon={lon_site:.2f}), the model gives:

| Parameter | Value |
|-----------|-------|
| Magnetic declination | **{mag_dec:.2f} deg** |
| True azimuth | {ref_state.az_true_deg:.2f} deg |
| Magnetic azimuth | {ref_state.az_mag_deg:.2f} deg |
"""
    )

    # --- Section 5: Assumptions & Limitations --------------------------------
    st.subheader("5. Assumptions & Limitations")
    assumptions = pd.DataFrame(
        {
            "Item": [
                "Propagation model",
                "Earth model",
                "Atmosphere",
                "Magnetic model",
                "Sunlit test",
                "Horizon",
                "TLE freshness",
            ],
            "Assumption / Limitation": [
                "SGP4 (sufficient for LEO pass prediction from TLEs).",
                "WGS-84 ellipsoid via Skyfield.",
                "Refraction not modelled; true geometric elevation used.",
                "WMM 2025; accuracy ~1 deg, degrades near magnetic poles.",
                "Geometric shadow (no penumbra).",
                "Local horizon at 0 deg elevation (no terrain masking).",
                "TLEs degrade over days/weeks; re-fetch for accuracy.",
            ],
        }
    )
    st.table(assumptions)

    # --- Section 6: References ----------------------------------------------
    st.subheader("6. References")
    st.markdown(
        """
- [CelesTrak](https://celestrak.org/) -- TLE data source.
- [Skyfield](https://rhodesmill.org/skyfield/) -- Python astronomy library for SGP4 and coordinate transforms.
- [World Magnetic Model (WMM)](https://www.ngdc.noaa.gov/geomag/WMM/) -- NOAA National Centers for Environmental Information.
- [SGP4/SDP4 Theory](https://celestrak.org/NORAD/documentation/spacetrk.pdf) -- Hoots & Roehrich, NORAD.
- [pygeomag](https://pypi.org/project/pygeomag/) -- Python WMM implementation.
"""
    )


with tab6:
    if _results is None:
        st.info("Press **Compute** in the sidebar to generate results.")
    else:
        _render_methodology(
            ref_state=_results["ref_state"],
            sat_aim=_results["sat_aim"],
            t_center=_results["t_center"],
            half_width=_results["half_width"],
            window=_results["window"],
            mag_dec=_results["mag_dec"],
        )
