"""TLE (Two-Line Element) parsing, validation, and fetching utilities."""

from __future__ import annotations

import requests
import streamlit as st

CELESTRAK_URL = "https://celestrak.org/NORAD/elements/gp.php"
CELESTRAK_GROUP = "sar"

# Common SAR satellite name prefixes for filtering.
# Each entry is a case-insensitive substring matched against the TLE name.
SAR_SATELLITES: list[str] = [
    "Sentinel-1",
    "RADARSAT",
    "RCM",
    "Capella",
    "UMBRA",
    "ICEYE",
    "HawkEye",
    "Kondor",
    "NovaSAR",
    "NOVASAR",
    "SAOCOM",
    "COSMO-SkyMed",
    "TerraSAR",
    "TanDEM",
    "CSG",
    "PAZ",
    "ALOS",
    "GAOFEN",
    "RISAT",
    "SAR-LUPE",
]


def parse_tle(text: str) -> tuple[tuple[str, str], str | None]:
    """Parse a TLE block from raw text.

    Args:
        text: Raw text containing one TLE set (optionally with a name line).

    Returns:
        A tuple of ((line1, line2), name_or_none).

    Raises:
        ValueError: If the text does not contain valid TLE lines.
    """
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    if len(lines) < 2:
        raise ValueError(
            f"Expected at least 2 non-empty lines, got {len(lines)}"
        )

    line1: str | None = None
    line2: str | None = None
    name: str | None = None

    for line in lines:
        if line.startswith("1 ") and line1 is None:
            line1 = line
        elif line.startswith("2 ") and line2 is None:
            line2 = line

    if line1 is None or line2 is None:
        raise ValueError("Could not find lines starting with '1 ' and '2 '")

    # If there is a line before line1 that is not a TLE line, treat it as the name
    for line in lines:
        if line is line1:
            break
        if not line.startswith("1 ") and not line.startswith("2 "):
            name = line

    return (line1, line2), name


def parse_tle_file(path: str) -> list[tuple[tuple[str, str], str | None]]:
    """Parse a file containing one or more TLE sets.

    Supports files with or without a name line preceding each TLE pair.
    Groups of 3 lines are treated as (name, line1, line2); groups of 2 lines
    are treated as (line1, line2) without a name.

    Args:
        path: Path to the TLE file.

    Returns:
        A list of ((line1, line2), name_or_none) tuples.
    """
    with open(path, encoding="utf-8") as fh:
        raw_lines = [line.rstrip("\n\r") for line in fh]

    # Strip blank lines but keep ordering
    lines: list[str] = [line for line in raw_lines if line.strip()]

    results: list[tuple[tuple[str, str], str | None]] = []
    i = 0
    while i < len(lines):
        # Peek ahead to determine if current line is a name line
        if i + 1 < len(lines) and lines[i + 1].startswith("1 "):
            # Three-line group: name + line1 + line2
            if i + 2 >= len(lines):
                raise ValueError(
                    f"Incomplete TLE group starting at line {i + 1}"
                )
            name = lines[i]
            line1 = lines[i + 1]
            line2 = lines[i + 2]
            results.append(((line1, line2), name))
            i += 3
        elif lines[i].startswith("1 "):
            # Two-line group: line1 + line2 (no name)
            if i + 1 >= len(lines):
                raise ValueError(
                    f"Incomplete TLE group starting at line {i + 1}"
                )
            line1 = lines[i]
            line2 = lines[i + 1]
            results.append(((line1, line2), None))
            i += 2
        else:
            # Skip unrecognised lines
            i += 1

    return results


def validate_tle(line1: str, line2: str) -> bool:
    """Validate a TLE line pair.

    Checks:
    - Each line is exactly 69 characters long.
    - Line1 starts with '1 ' and line2 starts with '2 '.
    - NORAD catalogue numbers match between the two lines.
    - Inclination is in [0, 180] degrees.
    - RAAN is in [0, 360) degrees.
    - Eccentricity is in [0, 1) (stored without leading decimal point).

    Args:
        line1: TLE line 1.
        line2: TLE line 2.

    Returns:
        True if both lines are valid, False otherwise.
    """
    try:
        if len(line1) != 69 or len(line2) != 69:
            return False
        if not line1.startswith("1 ") or not line2.startswith("2 "):
            return False

        # NORAD catalogue number (columns 3-7)
        norad1 = line1[2:7].strip()
        norad2 = line2[2:7].strip()
        if norad1 != norad2:
            return False

        # Inclination (columns 9-16 of line 2)
        inclination = float(line2[8:16].strip())
        if not (0.0 <= inclination <= 180.0):
            return False

        # RAAN (columns 18-25 of line 2)
        raan = float(line2[17:25].strip())
        if not (0.0 <= raan < 360.0):
            return False

        # Eccentricity (columns 27-33 of line 2, stored without leading dot)
        ecc_str = line2[26:33].strip()
        eccentricity = float(f"0.{ecc_str}")
        if not (0.0 <= eccentricity < 1.0):
            return False

        return True
    except (ValueError, IndexError):
        return False


@st.cache_data(ttl=3600)
def fetch_tle_celestrak(
    satellite: str | None = None,
) -> list[tuple[tuple[str, str], str]]:
    """Fetch SAR satellite TLE data from CelesTrak.

    Fetches from the CelesTrak 'sar' group and optionally filters by
    satellite name substring.

    Args:
        satellite: Optional satellite name substring to filter results
            (case-insensitive). If None, returns all SAR satellites.

    Returns:
        A list of ((line1, line2), name) tuples. The name is always a
        non-empty string (satellites without a name line are excluded).

    Raises:
        ValueError: If the fetch returns zero matching TLEs.
        RuntimeError: If the HTTP request fails.
    """
    params: dict[str, str] = {"GROUP": CELESTRAK_GROUP, "FORMAT": "tle"}
    response = requests.get(CELESTRAK_URL, params=params, timeout=30)
    if not response.ok:
        raise RuntimeError(
            f"CelesTrak request failed with status {response.status_code}: "
            f"{response.text}"
        )

    parsed = parse_tle_file_text(response.text)

    # Ensure every entry has a name
    results: list[tuple[tuple[str, str], str]] = []
    for (line1, line2), name in parsed:
        if name is None:
            continue
        if satellite is not None:
            if satellite.lower() not in name.lower():
                continue
        results.append(((line1, line2), name))

    if not results:
        filter_msg = f" for satellite '{satellite}'" if satellite else ""
        raise ValueError(
            f"No TLEs found in CelesTrak '{CELESTRAK_GROUP}' group{filter_msg}"
        )

    return results


def parse_tle_file_text(text: str) -> list[tuple[tuple[str, str], str | None]]:
    """Parse TLE data from a raw multi-line string.

    This mirrors the logic of parse_tle_file but operates on an in-memory
    string rather than a file path.

    Args:
        text: Raw text containing one or more TLE sets.

    Returns:
        A list of ((line1, line2), name_or_none) tuples.
    """
    lines: list[str] = [line.rstrip() for line in text.splitlines() if line.strip()]

    results: list[tuple[tuple[str, str], str | None]] = []
    i = 0
    while i < len(lines):
        if i + 1 < len(lines) and lines[i + 1].startswith("1 "):
            if i + 2 >= len(lines):
                break
            name = lines[i]
            line1 = lines[i + 1]
            line2 = lines[i + 2]
            results.append(((line1, line2), name))
            i += 3
        elif lines[i].startswith("1 "):
            if i + 1 >= len(lines):
                break
            line1 = lines[i]
            line2 = lines[i + 1]
            results.append(((line1, line2), None))
            i += 2
        else:
            i += 1

    return results
