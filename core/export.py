"""CSV and PDF export for satellite pointing cards and raw state tables."""

from __future__ import annotations

import csv
import io

from fpdf import FPDF

from core.propagator import LosState, Window


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

def export_csv_card(
    window: Window,
    ref_state: LosState,
    sat_name: str,
    site_lat: float,
    site_lon: float,
    site_h: float,
) -> str:
    """Return a CSV string with header and one data row for a pointing card.

    Parameters
    ----------
    window : Window
        Solved visibility window.
    ref_state : LosState
        Line-of-sight state at the window centre.
    sat_name : str
        Satellite name.
    site_lat : float
        Observer geodetic latitude in degrees.
    site_lon : float
        Observer geodetic longitude in degrees.
    site_h : float
        Observer height above WGS-84 ellipsoid in metres.

    Returns
    -------
    str
        CSV text (header + one row).
    """
    header = [
        "Satellite",
        "Site Lat",
        "Site Lon",
        "Site Height (m)",
        "Criterion",
        "Window Start UTC",
        "Window Stop UTC",
        "Window Duration (s)",
        "Minus Offset (s)",
        "Plus Offset (s)",
        "Half-Width (deg)",
        "Center Az True (deg)",
        "Center Az Mag (deg)",
        "Center El (deg)",
        "Center Slant (km)",
        "Center Sunlit",
        "Center Ascending",
        "Clamped by Horizon",
    ]

    row = [
        sat_name,
        site_lat,
        site_lon,
        site_h,
        window.criterion,
        window.t_start_utc.isoformat(),
        window.t_stop_utc.isoformat(),
        window.duration_s,
        window.minus_s,
        window.plus_s,
        window.half_width_deg,
        ref_state.az_true_deg,
        ref_state.az_mag_deg,
        ref_state.el_deg,
        ref_state.slant_km,
        ref_state.sunlit,
        ref_state.ascending,
        window.clamped_by_horizon,
    ]

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(header)
    writer.writerow(row)
    return buf.getvalue()


def export_csv_raw(states: list[LosState]) -> str:
    """Return a CSV string with header and one row per line-of-sight state.

    Parameters
    ----------
    states : list[LosState]
        Sequence of line-of-sight states.

    Returns
    -------
    str
        CSV text (header + one row per state).
    """
    header = [
        "Time UTC",
        "Az True (deg)",
        "Az Mag (deg)",
        "El (deg)",
        "Slant (km)",
        "Sunlit",
        "Ascending",
    ]

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(header)

    for s in states:
        writer.writerow([
            s.t_utc.isoformat(),
            s.az_true_deg,
            s.az_mag_deg,
            s.el_deg,
            s.slant_km,
            s.sunlit,
            s.ascending,
        ])

    return buf.getvalue()


# ---------------------------------------------------------------------------
# PDF export
# ---------------------------------------------------------------------------

def export_pdf_card(
    window: Window,
    ref_state: LosState,
    sat_name: str,
    site_lat: float,
    site_lon: float,
    site_h: float,
) -> bytes:
    """Return PDF bytes for a two-column pointing card.

    Parameters
    ----------
    window : Window
        Solved visibility window.
    ref_state : LosState
        Line-of-sight state at the window centre.
    sat_name : str
        Satellite name.
    site_lat : float
        Observer geodetic latitude in degrees.
    site_lon : float
        Observer geodetic longitude in degrees.
    site_h : float
        Observer height above WGS-84 ellipsoid in metres.

    Returns
    -------
    bytes
        PDF document.
    """
    label_width = 75
    value_width = 60
    row_height = 8

    fields = [
        ("Satellite", sat_name),
        ("Site Lat", site_lat),
        ("Site Lon", site_lon),
        ("Site Height (m)", site_h),
        ("Criterion", window.criterion),
        ("Window Start UTC", window.t_start_utc.isoformat()),
        ("Window Stop UTC", window.t_stop_utc.isoformat()),
        ("Window Duration (s)", window.duration_s),
        ("Minus Offset (s)", window.minus_s),
        ("Plus Offset (s)", window.plus_s),
        ("Half-Width (deg)", window.half_width_deg),
        ("Center Az True (deg)", ref_state.az_true_deg),
        ("Center Az Mag (deg)", ref_state.az_mag_deg),
        ("Center El (deg)", ref_state.el_deg),
        ("Center Slant (km)", ref_state.slant_km),
        ("Center Sunlit", ref_state.sunlit),
        ("Center Ascending", ref_state.ascending),
        ("Clamped by Horizon", window.clamped_by_horizon),
    ]

    pdf = FPDF(orientation="L", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=False)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 12, f"Pointing Card - {sat_name}", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(6)

    pdf.set_font("Helvetica", "", 10)

    # Two-column layout: left half and right half of the fields.
    mid = (len(fields) + 1) // 2

    x_left = 10
    x_right = x_left + label_width + value_width + 10

    for i in range(mid):
        y_row = pdf.get_y()

        # Left column
        label_l, value_l = fields[i]
        pdf.set_xy(x_left, y_row)
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(label_width, row_height, label_l, border=1)
        pdf.set_font("Helvetica", "", 10)
        pdf.cell(value_width, row_height, str(value_l), border=1)

        # Right column
        j = i + mid
        if j < len(fields):
            label_r, value_r = fields[j]
            pdf.set_xy(x_right, y_row)
            pdf.set_font("Helvetica", "B", 10)
            pdf.cell(label_width, row_height, label_r, border=1)
            pdf.set_font("Helvetica", "", 10)
            pdf.cell(value_width, row_height, str(value_r), border=1)

        pdf.set_y(y_row + row_height)

    return bytes(pdf.output())
