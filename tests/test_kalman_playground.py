"""Smoke-test the Panel Kalman playground entrypoints."""

from conftest import load_script_module

kalman_filter_playground = load_script_module("kalman_filter_playground.py")


def test_create_app_builds_panel_layout() -> None:
    """Verify the Kalman playground app can be constructed.

    Inputs:
        None.

    Returns:
        None. The test asserts the Panel application factory returns a layout object.
    """
    app = kalman_filter_playground.create_app()

    assert app is not None
    assert type(app).__name__ == "Row"
