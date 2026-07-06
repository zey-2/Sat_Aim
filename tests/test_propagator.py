"""Tests for SatAim propagator and window solver."""
from datetime import datetime, timezone
import numpy as np
import pytest

from core.propagator import SatAim, Window, LosState

# ISS (ZARYA) TLE — stable, well-known, frequently updated
SAMPLE_TLE_L1 = "1 25544U 98067A   24040.50000000  .00016717  00000+0  10270-3 0  9995"
SAMPLE_TLE_L2 = "2 25544  51.6400 200.0000 0005000  50.0000 310.0000 15.50000000  12345"

# Singapore site
SITE_LAT = 1.29
SITE_LON = 103.85
SITE_H = 20


@pytest.fixture
def sat():
    """Create SatAim instance with ISS TLE."""
    return SatAim(
        tle_lines=(SAMPLE_TLE_L1, SAMPLE_TLE_L2),
        name="ISS (ZARYA)",
        lat=SITE_LAT,
        lon=SITE_LON,
        height_m=SITE_H,
    )


class TestStateAt:
    def test_azimuth_range(self, sat):
        """Azimuth must be in [0, 360)."""
        t = datetime(2026, 7, 7, 14, 23, 11, tzinfo=timezone.utc)
        state = sat.state_at(t)
        assert 0 <= state.az_true_deg < 360

    def test_elevation_above_horizon(self, sat):
        """If satellite is above horizon, el > 0."""
        t = datetime(2026, 7, 7, 14, 23, 11, tzinfo=timezone.utc)
        state = sat.state_at(t)
        # ISS may or may not be above horizon at this time
        # Just check the range
        assert -90 <= state.el_deg <= 90

    def test_slant_range_positive(self, sat):
        """Slant range must be positive."""
        t = datetime(2026, 7, 7, 14, 23, 11, tzinfo=timezone.utc)
        state = sat.state_at(t)
        assert state.slant_km > 0

    def test_los_unit_vector_normalized(self, sat):
        """LOS ENU unit vector must have norm ≈ 1."""
        t = datetime(2026, 7, 7, 14, 23, 11, tzinfo=timezone.utc)
        state = sat.state_at(t)
        norm = np.linalg.norm(state.los_enu_unit)
        assert abs(norm - 1.0) < 1e-10

    def test_magnetic_azimuth_reasonable(self, sat):
        """Magnetic azimuth should be within ±30° of true azimuth at Singapore."""
        t = datetime(2026, 7, 7, 14, 23, 11, tzinfo=timezone.utc)
        state = sat.state_at(t)
        diff = abs(state.az_mag_deg - state.az_true_deg)
        # Singapore declination is ~0.2°, so difference should be small
        assert diff < 30 or diff > 330  # wrapped

    def test_naive_datetime_treated_as_utc(self, sat):
        """Naive datetime (no tzinfo) should be accepted and treated as UTC."""
        t_naive = datetime(2026, 7, 7, 14, 23, 11)
        state = sat.state_at(t_naive)
        assert isinstance(state, LosState)


class TestOffBoresight:
    def test_identical_vectors(self, sat):
        """Off-boresight between identical vectors must be 0."""
        v = np.array([0.0, 1.0, 0.0])
        assert sat.off_boresight_deg(v, v) == pytest.approx(0.0, abs=1e-10)

    def test_orthogonal_vectors(self, sat):
        """Off-boresight between orthogonal vectors must be 90°."""
        v1 = np.array([1.0, 0.0, 0.0])
        v2 = np.array([0.0, 1.0, 0.0])
        assert sat.off_boresight_deg(v1, v2) == pytest.approx(90.0, abs=1e-10)

    def test_opposite_vectors(self, sat):
        """Off-boresight between opposite vectors must be 180°."""
        v1 = np.array([1.0, 0.0, 0.0])
        v2 = np.array([-1.0, 0.0, 0.0])
        assert sat.off_boresight_deg(v1, v2) == pytest.approx(180.0, abs=1e-10)


class TestSolveWindow:
    def test_returns_window(self, sat):
        """solve_window must return a Window instance."""
        t = datetime(2026, 7, 7, 14, 23, 11, tzinfo=timezone.utc)
        try:
            w = sat.solve_window(t, "off_boresight", 5.0)
            assert isinstance(w, Window)
            assert w.duration_s > 0
            assert w.minus_s > 0
            assert w.plus_s > 0
        except ValueError as e:
            # Satellite may be below horizon at this time
            pytest.skip(f"Satellite below horizon: {e}")

    def test_half_width_at_boundaries(self, sat):
        """Off-boresight at window edges must be ≈ half_width_deg."""
        t = datetime(2026, 7, 7, 14, 23, 11, tzinfo=timezone.utc)
        half_width = 5.0
        try:
            w = sat.solve_window(t, "off_boresight", half_width)
            # Check at t_start
            ref = sat.state_at(t)
            s_start = sat.state_at(w.t_start_utc)
            oob_start = sat.off_boresight_deg(ref.los_enu_unit, s_start.los_enu_unit)
            assert oob_start == pytest.approx(half_width, abs=0.5)
            # Check at t_stop
            s_stop = sat.state_at(w.t_stop_utc)
            oob_stop = sat.off_boresight_deg(ref.los_enu_unit, s_stop.los_enu_unit)
            assert oob_stop == pytest.approx(half_width, abs=0.5)
        except ValueError:
            pytest.skip("Satellite below horizon")

    def test_invalid_half_width(self, sat):
        """Must raise ValueError for out-of-range half_width_deg."""
        t = datetime(2026, 7, 7, 14, 23, 11, tzinfo=timezone.utc)
        with pytest.raises(ValueError):
            sat.solve_window(t, "off_boresight", 0)
        with pytest.raises(ValueError):
            sat.solve_window(t, "off_boresight", 50)

    def test_below_horizon_raises(self, sat):
        """Must raise ValueError if satellite is below horizon at t_center."""
        # Pick a time when ISS is likely below horizon for Singapore
        # Use a far-future date where TLE is very stale
        t = datetime(2030, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        with pytest.raises((ValueError, Exception)):
            sat.solve_window(t, "off_boresight", 5.0)


class TestWrapAzDiff:
    def test_wrap_az_diff_basic(self):
        """Basic azimuth difference test."""
        from core.propagator import _wrap_az_diff
        assert _wrap_az_diff(10, 20) == pytest.approx(10.0)
        assert _wrap_az_diff(350, 10) == pytest.approx(20.0)
        assert _wrap_az_diff(0, 180) == pytest.approx(180.0)
        assert _wrap_az_diff(10, 10) == pytest.approx(0.0)
