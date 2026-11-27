"""Start the playNano GUI (``playnano play``)."""

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
    PySide6.QtGui.QIcon
        The icon. It is empty (``QIcon.isNull()``) if no icon files were found,
        in which case a warning is logged.
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
    Open the playNano window for a stack and run it until it is closed.

    Called by ``playnano play`` (see
    :func:`playnano.cli.actions.play_pipeline_mode`). It creates the Qt
    application, applies the dark stylesheet and shows a
    :class:`~playnano.gui.window.MainWindow`.

    Parameters
    ----------
    afm_stack : AFMImageStack
        The stack to show.
    output_dir : str, optional
        Folder for exports. If None, ``./output``. Default is None.
    output_name : str, optional
        Start of the exported file names. If empty, the name of the file the
        stack was loaded from. Default is ``""``.
    steps_with_kwargs : list of tuple, optional
        Processing steps, as ``(name, parameters)``, run when F is pressed. If
        None or empty, F runs a built-in default pipeline. Default is None.
    scale_bar_nm : int, optional
        Scale bar length, in nm; 0 opens the window with the scale bar off.
        Default is 100.
    zmin, zmax : float or str, optional
        Initial colour scale limits: a number, ``"auto"`` (1st and 99th
        percentiles) or None (data minimum or maximum). Default is ``"auto"``.
    cmap : str, optional
        Colormap the window opens with. Names not in the window's list are
        replaced by ``"afm_brown"``, with a warning. Default is ``"afm_brown"``.
    fps : float, optional
        Initial playback rate, in frames per second; also used for animated
        exports. Default is 5.0.

    Raises
    ------
    SystemExit
        Always, when the window is closed, with Qt's exit status. The function
        never returns, so it cannot be used from a notebook.
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
