"""Shared Streamlit session-state keys and helpers."""

from __future__ import annotations

from collections.abc import MutableMapping
from datetime import datetime
from typing import Any, Final


SCENE_DATE_KEY: Final = "_scene_date"
SCENE_TIME_KEY: Final = "_scene_time"
PASS_SUGGESTION_KEY: Final = "_sat_aim_pass_suggestion"
RESULTS_KEY: Final = "_sat_aim_results"


def prefill_scene_center_from_peak(
    session_state: MutableMapping[str, Any],
    peak_utc: datetime,
) -> None:
    """Set scene-center widget state from a suggested pass peak."""
    session_state[SCENE_DATE_KEY] = peak_utc.date()
    session_state[SCENE_TIME_KEY] = peak_utc.time().replace(microsecond=0)
    session_state.pop(PASS_SUGGESTION_KEY, None)
    session_state.pop(RESULTS_KEY, None)
