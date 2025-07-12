"""Main entry point for the playNano GUI application."""

import sys
from importlib.resources import files

from PySide6.QtWidgets import QApplication

from playNano.gui import styles
from playNano.gui.window import MainWindow


def gui_entry(
    afm_stack,
    output_dir=None,
    output_name="playNano_export",
    steps_with_kwargs=None,
    scale_bar_nm=100,
    zmin="auto",
    zmax="auto",
):
    """Launch the GUI with the provided AFM stack and parameters."""
    app = QApplication(sys.argv)

    qss_file = files(styles) / "dark_bluegreen.qss"
    if qss_file.is_file():
        app.setStyleSheet(qss_file.read_text(encoding="utf-8"))

    wnd = MainWindow(
        afm_stack=afm_stack,
        processing_steps=steps_with_kwargs,
        output_dir=output_dir,
        output_name=output_name,
        scale_bar_nm=scale_bar_nm,
        zmin=zmin,
        zmax=zmax,
    )
    wnd.show()
    sys.exit(app.exec())
