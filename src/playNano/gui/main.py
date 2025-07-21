"""Main entry point for the playNano GUI application."""

import sys
from importlib.resources import files

from PySide6.QtWidgets import QApplication

from playNano.gui import styles
from playNano.gui.window import MainWindow


def gui_entry(
    afm_stack,
    output_dir=None,
    output_name="",
    steps_with_kwargs=None,
    scale_bar_nm=100,
    zmin="auto",
    zmax="auto",
):
    """
    Launch the playNano GUI with a given AFM stack and visualization/export settings.

    Parameters
    ----------
    afm_stack : AFMImageStack
        The AFM image stack to display and process.
    output_dir : str or None, optional
        Directory in which to save exported files (default: None, which uses
        internal defaults).
    output_name : str, optional
        Base filename for any exports (default: "" - this triggers the default
        behaviour when utils.io_utils.sanitize_output_name is called).
    steps_with_kwargs : list of (str, dict) or None, optional
        A list of (filter_name, kwargs) tuples for the default processing pipeline.
        If None, the application's built-in defaults are used.
    scale_bar_nm : int, optional
        Physical length of the scale bar in nanometers (default: 100).
    zmin : float, "auto", or None, optional
        Minimum height value for the initial colormap range. If "auto", uses the
        1st percentile of the data; if None, uses the data minimum; if a float,
        uses that value (default: "auto").
    zmax : float, "auto", or None, optional
        Maximum height value for the initial colormap range. If "auto", uses the
        99th percentile of the data; if None, uses the data maximum; if a float,
        uses that value (default: "auto").

    Returns
    -------
    None

    Notes
    -----
    This function will start a Qt event loop and will not return to the caller.
    """
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
