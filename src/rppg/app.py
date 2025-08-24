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

    with dpg.window(label="rPPG Demo", width=940, height=600):
        dpg.add_text("Hello from rPPG demo. UI coming soon.")

    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.set_primary_window(dpg.last_item(), True)
    dpg.start_dearpygui()
    dpg.destroy_context()


if __name__ == "__main__":  # pragma: no cover - manual entry
    main()

