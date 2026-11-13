"""Main entry point for the playNano GUI application."""

import logging
import sys
from importlib.resources import files

from PySide6.QtCore import QSize
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication

from playnano.gui import styles
from playnano.gui.window import MainWindow
from playnano.utils.colormaps import DEFAULT_CMAP

logger = logging.getLogger(__name__)

#: Pixel sizes of the bundled application icon PNGs. The 16 and 32 px icons are
#: simplified, pixel-aligned versions of the logo; larger sizes use the full mark.
ICON_SIZES: tuple[int, ...] = (16, 32, 48, 64, 128, 256, 512)

#: Windows taskbar identity for the GUI process (see ``_set_windows_app_id``).
WINDOWS_APP_ID = "playnano.gui"


def load_app_icon() -> QIcon:
    """
    Build the playNano application icon from the bundled PNG files.

    Each size is added separately so Qt picks the hand-tuned bitmap for small
    sizes (title bar, taskbar) instead of downscaling a large image.

    Returns
    -------
    QIcon
        The application icon. It is null (``QIcon.isNull()``) if no icon
        files could be found, in which case a warning is logged.
    """
    icon = QIcon()
    icons_dir = files("playnano.resources.icons")
    for size in ICON_SIZES:
        path = icons_dir.joinpath(f"playnano_icon_{size}.png")
        if path.is_file():
            icon.addFile(str(path), QSize(size, size))
    if icon.isNull():
        logger.warning("playNano icon files not found; using the default icon.")
    return icon


def _set_windows_app_id() -> None:
    """
    Give the GUI its own taskbar identity on Windows.

    Without this, Windows groups the window under the Python interpreter and
    shows Python's icon in the taskbar instead of the playNano icon. Does
    nothing on other platforms.
    """
    if sys.platform != "win32":
        return
    try:
        import ctypes

        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(WINDOWS_APP_ID)
    except (AttributeError, OSError) as exc:
        logger.debug("Could not set Windows AppUserModelID: %s", exc)


def gui_entry(
    afm_stack,
    output_dir=None,
    output_name="",
    steps_with_kwargs=None,
    scale_bar_nm=100,
    zmin="auto",
    zmax="auto",
    cmap=DEFAULT_CMAP,
    fps=5.0,
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
    cmap : str, optional
        Name of the colormap to use for visualisation on launch (default=DEFAULT_CMAP).
    fps : float, optional
        Initial frame rate for the GIF in frames per second (default=5.0).

    Returns
    -------
    None

    Notes
    -----
    This function will start a Qt event loop and will not return to the caller.
    """
    _set_windows_app_id()
    app = QApplication(sys.argv)
    app.setWindowIcon(load_app_icon())

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
        cmap=cmap,
        fps=fps,
    )
    wnd.show()
    sys.exit(app.exec())
