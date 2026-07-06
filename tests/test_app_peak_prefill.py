"""Regression tests for the Streamlit peak-time prefill flow."""

import ast
import importlib
import importlib.util
from datetime import datetime, time, timezone
from pathlib import Path


APP_PATH = Path(__file__).parents[1] / "app.py"


def _node_text_contains(node: ast.AST, text: str) -> bool:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return text in node.value
    if isinstance(node, ast.JoinedStr):
        return any(_node_text_contains(value, text) for value in node.values)
    return False


def _is_peak_time_button(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "button"
        and bool(node.args)
        and _node_text_contains(node.args[0], "Use peak time")
    )


def _mentions_name(node: ast.AST, name: str) -> bool:
    return any(
        isinstance(child, ast.Name) and child.id == name for child in ast.walk(node)
    )


def test_peak_time_button_is_not_gated_by_compute_click():
    tree = ast.parse(APP_PATH.read_text())
    compute_blocks = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.If) and _mentions_name(node.test, "compute_btn")
    ]
    assert compute_blocks, "expected a compute button branch in app.py"

    for block in compute_blocks:
        button_inside_compute = any(
            _is_peak_time_button(child)
            for statement in block.body
            for child in ast.walk(statement)
        )
        assert not button_inside_compute


def test_peak_prefill_callback_updates_scene_inputs_and_clears_stale_state():
    spec = importlib.util.find_spec("core.ui_state")
    assert spec is not None, "expected Streamlit session-state helpers"

    ui_state = importlib.import_module("core.ui_state")
    peak_utc = datetime(2026, 7, 8, 12, 34, 56, 789, tzinfo=timezone.utc)
    session_state = {
        ui_state.PASS_SUGGESTION_KEY: object(),
        ui_state.RESULTS_KEY: object(),
    }

    ui_state.prefill_scene_center_from_peak(session_state, peak_utc)

    assert session_state[ui_state.SCENE_DATE_KEY] == peak_utc.date()
    assert session_state[ui_state.SCENE_TIME_KEY] == time(12, 34, 56)
    assert ui_state.PASS_SUGGESTION_KEY not in session_state
    assert ui_state.RESULTS_KEY not in session_state
