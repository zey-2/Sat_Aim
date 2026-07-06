"""Magnetic declination lookup using the World Magnetic Model (WMM).

Uses the pygeomag library which bundles WMM2025 coefficients. The WMM is
produced by NOAA/NGA and is accurate to roughly 1 degree of declination
for dates within the model's valid epoch (currently 2025-2030). Accuracy
degrades for locations near the magnetic poles and for dates outside the
valid epoch.
"""

from datetime import datetime

from pygeomag import GeoMag


def magnetic_declination(
    lat: float,
    lon: float,
    height_m: float = 0.0,
    year: int | None = None,
) -> float:
    """Return magnetic declination in degrees at a given location and time.

    Positive values indicate east declination (magnetic north is east of
    true north); negative values indicate west declination.

    Parameters
    ----------
    lat : float
        Geodetic latitude in degrees, range [-90, 90].
    lon : float
        Geodetic longitude in degrees, range [-180, 180].
    height_m : float, optional
        Height above the WGS-84 ellipsoid in metres. Defaults to 0.
    year : int or None, optional
        Decimal year (e.g. 2026 or 2026.5). If None, the current
        calendar year is used.

    Returns
    -------
    float
        Magnetic declination in degrees.

    Raises
    ------
    ValueError
        If lat or lon are outside their valid ranges.
    """
    if not -90.0 <= lat <= 90.0:
        raise ValueError(
            f"Latitude must be between -90 and 90 degrees, got {lat}"
        )
    if not -180.0 <= lon <= 180.0:
        raise ValueError(
            f"Longitude must be between -180 and 180 degrees, got {lon}"
        )

    if year is None:
        year = datetime.now().year

    geo = GeoMag()
    result = geo.calculate(glat=lat, glon=lon, alt=height_m / 1000.0, time=year)
    return result.d
