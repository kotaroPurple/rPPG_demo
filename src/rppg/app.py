"""DearPyGUI entry point (placeholder).

Run with: `uv run python -m rppg.app`
"""

from __future__ import annotations


def main() -> None:
    """Launch a minimal DearPyGUI window as a sanity check."""
    # Import locally to avoid hard dependency at import time
    import dearpygui.dearpygui as dpg

    dpg.create_context()
    dpg.create_viewport(title="rPPG Demo", width=960, height=640)

    primary_tag = "primary_window"
    with dpg.window(tag=primary_tag, label="rPPG Demo", width=940, height=600):
        dpg.add_text("Hello from rPPG demo. UI coming soon.")

    dpg.setup_dearpygui()
    dpg.show_viewport()
    # Use explicit tag to set primary window; avoid using last_item()
    dpg.set_primary_window(primary_tag, True)
    dpg.start_dearpygui()
    dpg.destroy_context()


if __name__ == "__main__":  # pragma: no cover - manual entry
    main()
