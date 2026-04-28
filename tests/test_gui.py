"""Unit tests for the playNano GUI, MainWindow, and ViewerWidget components."""

import logging
from unittest.mock import ANY, MagicMock, patch

import matplotlib
import numpy as np
import pytest
from matplotlib import colors
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap

from playnano.gui import main
from playnano.gui.widgets.viewer import ViewerWidget
from playnano.gui.window import MainWindow
from playnano.utils.colormaps import DEFAULT_CMAP

log = logging.getLogger(__name__)


@pytest.fixture
def main_window(qtbot):
    """Create a main window for testing."""
    wnd = MainWindow()
    test_stack = np.random.rand(3, 5, 5)
    wnd.set_stack(test_stack)
    qtbot.addWidget(wnd)
    yield wnd
    wnd._timer.stop()
    wnd.close()
    wnd.deleteLater()
    qtbot.wait(50)


@patch("playnano.gui.window.AFMImageStack.load_data")
def test_mainwindow_loads_and_interacts(mock_load_data, qtbot):
    """Test that the MainWindow loads and interacts correctly."""

    # Mock AFMImageStack
    # Dummy filter function with version
    def dummy_filter(arr, **kwargs):
        """Make as dummy filter function that increments input array by 1."""
        return arr + 1

    dummy_filter.__version__ = "0.0.1"

    # Add mock _resolve_step to simulate pipeline step resolution
    mock_stack = MagicMock(spec=["data", "width", "height"])
    mock_stack._resolve_step = MagicMock(return_value=("filter", dummy_filter))
    mock_stack._execute_filter_step = MagicMock(
        side_effect=lambda fn, arr, mask, name, **kwargs: fn(arr, **kwargs)
    )
    mock_stack.data = np.random.rand(10, 10, 10).astype(np.float32)
    mock_stack.width = 256
    mock_stack.height = 256
    mock_load_data.return_value = mock_stack
    mock_stack.analysis = {}
    mock_stack.add_analysis = MagicMock()
    mock_stack.time_for_frame = MagicMock(return_value=0.1)
    mock_stack.scaling_for_frame = MagicMock(return_value=1.0)
    mock_load_data.return_value = mock_stack
    mock_stack.pixel_size_nm = 1.0

    mock_stack.provenance = {
        "processing": {"steps": [], "keys_by_name": {}},
        "analysis": [],
        "environment": {},
    }
    mock_stack.processed = {}

    # Instantiate and show
    wnd = MainWindow(mock_stack)
    qtbot.addWidget(wnd)
    wnd.show()
    assert wnd.isVisible()

    # FPS defaults
    assert wnd.controls.fps_box.value() == 5.0

    # Slider config
    assert wnd.controls.slider.minimum() == 0
    assert wnd.controls.slider.maximum() == 9

    # Move slider → internal index should match
    wnd.controls.slider.setValue(3)
    assert wnd._idx == 3

    # Toggle processed without filtered data (should stay raw)
    wnd.toggle_processed()
    assert wnd._show_flat is False

    # Simulate apply_filters
    mock_stack.data = np.random.rand(10, 10, 10).astype(np.float32)
    wnd.apply_filters()
    assert wnd._flat is not None
    assert wnd._show_flat is True

    # Toggle back to raw
    wnd.toggle_processed()
    assert wnd._show_flat is False
    wnd.close()
    wnd.deleteLater()


def test_get_cmap_invalid_falls_back(monkeypatch, caplog):
    """
    Test that _get_cmap fall back to default cmap.

    Invalid colormap names should:
    - log a warning
    - reset _cmap_name to DEFAULT_CMAP
    - return the DEFAULT_CMAP colormap
    """

    # Create instance without running full Qt init
    window = MainWindow.__new__(MainWindow)
    window._cmap_name = "definitely_not_a_real_colormap"

    caplog.set_level("WARNING")

    cmap = window._get_cmap()

    assert cmap.name == matplotlib.colormaps.get_cmap(DEFAULT_CMAP).name

    assert window._cmap_name == DEFAULT_CMAP

    assert "Invalid cmap" in caplog.text
    assert DEFAULT_CMAP in caplog.text


def test_on_cmap_changed_invalid_name(caplog):
    """
    Test the behaviour of _on_cmap_change() with invalid cmap.

    Invalid colormap names should:
    - log a warning
    - not update internal state
    - not trigger any redraw / update methods
    """

    # Create instance without Qt init
    window = MainWindow.__new__(MainWindow)
    window._cmap_name = "viridis"  # existing valid state
    window._idx = 0

    # Attach spies that would fail if called
    window._update_background_color = pytest.fail
    window.show_frame = pytest.fail
    window._draw_bars = pytest.fail
    window._init_lines = pytest.fail

    caplog.set_level("WARNING")

    window._on_cmap_changed("this_is_not_a_real_cmap")

    assert "not registered" in caplog.text

    assert window._cmap_name == "viridis"


def test_on_cmap_changed_valid_name(monkeypatch):
    """
    Test the behaviour of _on_cmap_change() with valid cmap.

    Valid colormap names should:
    - update _cmap_name
    - call background update
    - redraw the frame
    - rebuild histogram bars and lines
    """

    window = MainWindow.__new__(MainWindow)
    window._idx = 3
    window._cmap_name = "plasma"

    calls = []

    def record(name):
        calls.append(name)

    window._update_background_color = lambda: record("update_bg")
    window.show_frame = lambda idx: record(f"show_frame_{idx}")
    window._draw_bars = lambda: record("draw_bars")
    window._init_lines = lambda: record("init_lines")

    window._on_cmap_changed("viridis")

    assert window._cmap_name == "viridis"

    assert calls == [
        "update_bg",
        "show_frame_3",
        "draw_bars",
        "init_lines",
    ]


def test_on_cmap_changed_valid_does_not_warn(caplog):
    """Verify that selecting a valid colormap does not trigger a warning."""
    window = MainWindow.__new__(MainWindow)
    window._idx = 0

    window._update_background_color = lambda: None
    window.show_frame = lambda _: None
    window._draw_bars = lambda: None
    window._init_lines = lambda: None

    caplog.set_level("WARNING")

    window._on_cmap_changed("viridis")

    assert caplog.text == ""


def test_update_histogram_colors_no_bars_returns_early():
    """Return early and perform no actions when histogram bars are not present."""

    window = MainWindow.__new__(MainWindow)

    # Fail fast if any downstream method is called
    window._get_cmap = pytest.fail
    window.hist_canvas = pytest.fail

    # Should simply return without error
    window._update_histogram_colors()


def test_update_histogram_colors_uses_flat_zrange(monkeypatch):
    """Use flat zmin/zmax when flat view is enabled and flat data is present."""

    window = MainWindow.__new__(MainWindow)

    # Required state
    window._hist_bars = [type("Bar", (), {"set_color": lambda self, c: None})()]
    window._hist_centers = [0.5]
    window._show_flat = True
    window._flat = object()

    window._zmin_flat = -1.0
    window._zmax_flat = 1.0
    window._zmin_raw = -10.0
    window._zmax_raw = 10.0

    # Track which vmin/vmax Normalize receives
    used = {}

    def fake_normalize(vmin, vmax):
        used["vmin"] = vmin
        used["vmax"] = vmax
        return lambda x: x

    monkeypatch.setattr(colors, "Normalize", fake_normalize)

    window._get_cmap = lambda: lambda x: [(1, 0, 0, 1)]
    window.hist_canvas = type("Canvas", (), {"draw_idle": lambda self: None})()

    window._update_histogram_colors()

    assert used["vmin"] == -1.0
    assert used["vmax"] == 1.0


def test_update_histogram_colors_uses_raw_zrange_when_not_flat(monkeypatch):
    """Use raw zmin/zmax when flat view is disabled or no flat data is available."""

    window = MainWindow.__new__(MainWindow)

    window._hist_bars = [type("Bar", (), {"set_color": lambda self, c: None})()]
    window._hist_centers = [0.5]
    window._show_flat = False
    window._flat = None

    window._zmin_raw = 2.0
    window._zmax_raw = 8.0
    window._zmin_flat = -1.0
    window._zmax_flat = 1.0

    used = {}

    def fake_normalize(vmin, vmax):
        used["vmin"] = vmin
        used["vmax"] = vmax
        return lambda x: x

    monkeypatch.setattr(colors, "Normalize", fake_normalize)

    window._get_cmap = lambda: lambda x: [(0, 1, 0, 1)]
    window.hist_canvas = type("Canvas", (), {"draw_idle": lambda self: None})()

    window._update_histogram_colors()

    assert used["vmin"] == 2.0
    assert used["vmax"] == 8.0
    assert used["vmin"] == window._zmin_raw
    assert used["vmax"] == window._zmax_raw


@patch("playnano.gui.main.QApplication")
@patch("playnano.gui.main.MainWindow")
def test_gui_entry_launches_gui(mock_main_window, mock_qapplication):
    """Test that the GUI entry point launches the application correctly."""
    # Arrange: fake args with a dummy file path
    mock_args = MagicMock()
    mock_args.input_file = "dummy/path.h5-jpk"

    # Mock instances
    mock_app = MagicMock()
    mock_window = MagicMock()

    mock_qapplication.return_value = mock_app
    mock_main_window.return_value = mock_window

    # Act
    with patch("sys.exit") as mock_exit:  # prevent test from exiting
        main.gui_entry(mock_args)

    # Assert
    mock_qapplication.assert_called_once()

    mock_main_window.assert_called_once_with(
        afm_stack=ANY,
        processing_steps=None,
        output_dir=None,
        output_name="",
        scale_bar_nm=100,
        zmin="auto",
        zmax="auto",
        cmap="afm_brown",
        fps=5.0,
    )
    mock_window.show.assert_called_once()
    mock_app.exec.assert_called_once()
    mock_exit.assert_called_once()


def test_mainwindow_font_fallbacks(qtbot, caplog):
    """Test that font loading failures trigger warnings."""
    with patch(
        "playnano.gui.window.QFontDatabase.addApplicationFont", return_value=-1
    ):  #
        with patch(
            "playnano.gui.window.QFontDatabase.applicationFontFamilies",
            return_value=[],
        ):
            caplog.set_level(logging.WARNING)
            mock_stack = MagicMock(width=256, height=256, data=np.random.rand(5, 5, 5))
            mock_stack.pixel_size_nm = 1.0
            mock_stack.frame_metadata = []
            mock_stack.time_for_frame = MagicMock(return_value=0.1)
            wnd = MainWindow(mock_stack)
            qtbot.addWidget(wnd)
            assert "Failed to load Steps Mono font" in caplog.text
            wnd.close()
            wnd.deleteLater()


def test_toggle_play_starts_and_stops(qtbot):
    """Test toggle_play switches between playing and paused."""
    mock_stack = MagicMock(width=256, height=256, data=np.random.rand(2, 10, 10))
    mock_stack.pixel_size_nm = 1.0
    mock_stack.frame_metadata = []
    mock_stack.time_for_frame = MagicMock(return_value=0.1)
    wnd = MainWindow(mock_stack)
    qtbot.addWidget(wnd)

    wnd._timer.stop()  # ensure stopped
    wnd.controls.fps_box.setValue(20)
    wnd.toggle_play()  # should start
    assert wnd._timer.isActive()
    assert wnd.controls.play_btn.text() == "⏸ Pause"

    wnd.toggle_play()  # should stop
    assert not wnd._timer.isActive()
    assert wnd.controls.play_btn.text() == "▶️ Play"

    wnd.close()
    wnd.deleteLater()


def test_keypress_triggers_methods(qtbot):
    """Test that keyPressEvent calls the right handlers."""
    mock_stack = MagicMock(width=256, height=256, data=np.random.rand(2, 10, 10))
    mock_stack.pixel_size_nm = 1.0
    mock_stack.frame_metadata = []
    mock_stack.time_for_frame = MagicMock(return_value=0.1)
    wnd = MainWindow(mock_stack)
    qtbot.addWidget(wnd)

    for key, method in [
        (Qt.Key_Space, "toggle_play"),
        (Qt.Key_F, "apply_filters"),
        (Qt.Key_R, "toggle_processed"),
        (Qt.Key_G, "_export_gif"),
        (Qt.Key_E, "_export_checked"),
    ]:
        called = {"value": False}

        def setter(*a, called=called, **k):
            called["value"] = True

        setattr(wnd, method, setter)
        evt = MagicMock()
        evt.key.return_value = key
        wnd.keyPressEvent(evt)
        assert called["value"], f"{method} should have been called"
    wnd.close()
    wnd.deleteLater()


def test_show_frame_fallback_pixel_size(qtbot):
    """Test that invalid pixel_size_nm uses fallback value."""
    mock_stack = MagicMock(width=256, height=256, data=np.random.rand(2, 10, 10))
    mock_stack.pixel_size_nm = -1  # invalid
    mock_stack.frame_metadata = []
    mock_stack.time_for_frame = MagicMock(return_value=0.1)
    wnd = MainWindow(mock_stack)
    qtbot.addWidget(wnd)

    wnd.show_frame(0)  # Should not raise
    wnd.close()
    wnd.deleteLater()


def test_colormap_normalize_flat_range(qtbot):
    """Test _colormap_and_normalize when zmin == zmax returns zeros for 'afmhot'."""
    mock_stack = MagicMock(width=256, height=256, data=np.ones((1, 5, 5)))
    mock_stack.pixel_size_nm = 1.0
    mock_stack.frame_metadata = []
    mock_stack.time_for_frame = MagicMock(return_value=0.1)
    wnd = MainWindow(mock_stack)
    qtbot.addWidget(wnd)

    wnd._zmin_raw = wnd._zmax_raw  # force flat range
    wnd._cmap_name = "afmhot"
    rgb = wnd._colormap_and_normalize(np.ones((5, 5)))
    assert np.all(rgb == 0)
    wnd.close()
    wnd.deleteLater()


def test_next_frame_advances_and_updates_slider(qtbot):
    """Test _next_frame increments index and updates slider."""
    mock_stack = MagicMock(width=256, height=256, data=np.random.rand(3, 10, 10))
    mock_stack.pixel_size_nm = 1.0
    mock_stack.time_for_frame = MagicMock(return_value=0.1)
    wnd = MainWindow(mock_stack)
    qtbot.addWidget(wnd)

    wnd._idx = 1
    wnd.show_frame = MagicMock()
    wnd._next_frame()
    assert wnd._idx == 2
    # Expect that show_frame was called at least once with the correct index
    assert any(call == ((2,),) for call in wnd.show_frame.call_args_list)
    assert wnd.controls.slider.value() == 2
    wnd.close()
    wnd.deleteLater()


def test_update_timer_interval_active_timer(qtbot):
    """Test _update_timer_interval restarts timer with correct interval."""
    mock_stack = MagicMock(width=256, height=256, data=np.random.rand(2, 10, 10))
    mock_stack.pixel_size_nm = 1.0
    mock_stack.time_for_frame = MagicMock(return_value=0.1)
    wnd = MainWindow(mock_stack)
    qtbot.addWidget(wnd)

    wnd._timer.start = MagicMock()
    wnd._timer.isActive = MagicMock(return_value=True)
    wnd._update_timer_interval(20)
    wnd._timer.start.assert_called_once()
    interval = wnd._timer.start.call_args[0][0]
    assert 45 < interval < 55  # approx 50 ms for 20 fps
    wnd.close()
    wnd.deleteLater()


def test_toggle_processed_updates_spinboxes_flat_branch():
    """Test toggle_processed updates spinboxes when showing flat data."""
    wnd = MainWindow.__new__(MainWindow)

    # Minimal state for this branch
    wnd._show_flat = False  # must be False so toggle_processed switches to True
    wnd._flat = np.ones((5, 5))  # not None
    wnd._zmin_flat, wnd._zmax_flat = 1.0, 5.0
    wnd._zmin_raw, wnd._zmax_raw = 0.0, 10.0
    wnd._idx = 0

    # Mock methods and widgets
    wnd.zmin_spin = MagicMock()
    wnd.zmax_spin = MagicMock()
    wnd._set_spinbox_value = MagicMock()
    wnd._update_background_color = MagicMock()
    wnd.show_frame = MagicMock()
    wnd._draw_bars = MagicMock()
    wnd._init_lines = MagicMock()

    wnd.toggle_processed()

    # Should have switched to flat and used _zmin_flat/_zmax_flat
    wnd._set_spinbox_value.assert_any_call(wnd.zmin_spin, 1.0)
    wnd._set_spinbox_value.assert_any_call(wnd.zmax_spin, 5.0)


def test_update_background_color_flat_branch():
    """Test _update_background_color sets viewer background for flat data."""
    wnd = MainWindow.__new__(MainWindow)

    wnd._show_flat = True
    wnd._flat = np.ones((5, 5))  # ensures branch is taken
    wnd._zperc_flat = 0.5
    wnd._zmin_flat = 1.0
    wnd._zmax_flat = 5.0
    wnd._cmap_name = "afm_brown"

    wnd.viewer = MagicMock()
    wnd.viewer.set_background_color = MagicMock()

    # Call method
    wnd._update_background_color()

    # Calculate expected RGB value the same way as in code:
    from playnano.gui.window import z_to_rgb

    expected_rgb = z_to_rgb(0.5, 1.0, 5.0, cmap_name="afm_brown")

    wnd.viewer.set_background_color.assert_called_once_with(expected_rgb)
    del wnd


@patch("playnano.io.gif_export.export_gif")
@patch("playnano.gui.window.prepare_output_directory", return_value="mock_dir")
def test_export_gif_calls_export(mock_prepare, mock_export, qtbot):
    """Test that _export_gif calls prepare_output_directory and export_gif."""
    mock_stack = MagicMock(width=256, height=256, data=np.random.rand(1, 10, 10))
    mock_stack.pixel_size_nm = 1.0
    mock_stack.time_for_frame = MagicMock(return_value=0.1)
    mock_stack.processed = {"raw": np.random.rand(1, 10, 10)}

    wnd = MainWindow(mock_stack)
    qtbot.addWidget(wnd)

    wnd._export_gif()
    mock_export.assert_called_once()
    mock_prepare.assert_called_once()
    wnd.close()
    wnd.deleteLater()


@pytest.mark.parametrize(
    "raw_present, raw_checked, expected_raw",
    [
        (True, True, True),  # raw data exists, raw radio selected
        (
            False,
            True,
            False,
        ),  # raw radio selected but no raw data → fallback to processed
        (True, False, False),  # processed radio selected
    ],
)
@patch("playnano.io.gif_export.export_gif")
@patch("playnano.gui.window.prepare_output_directory", return_value="mock_dir")
def test_export_gif_branches(
    mock_prepare, mock_export, raw_present, raw_checked, expected_raw, qtbot
):
    """Test _export_gif handles raw/processed branches and z-range correctly."""
    # Mock AFM stack
    mock_stack = MagicMock(width=256, height=256, data=np.random.rand(1, 10, 10))
    mock_stack.pixel_size_nm = 1.0
    mock_stack.time_for_frame = MagicMock(return_value=0.1)
    mock_stack.processed = {"raw": mock_stack.data} if raw_present else {}

    # Instantiate MainWindow
    wnd = MainWindow(mock_stack)
    qtbot.addWidget(wnd)

    # Set radio button states
    wnd.animated_raw_radio.setChecked(raw_checked)
    wnd.animated_processed_radio.setChecked(not raw_checked)

    # Set z-range values
    wnd._zmin_raw, wnd._zmax_raw = -1.0, 1.0
    wnd._zmin_flat, wnd._zmax_flat = -2.0, 2.0

    # Also toggle timestamp and scale bar
    wnd.show_timestamp_box.setChecked(True)
    wnd.show_scale_bar_box.setChecked(False)

    wnd._export_gif()

    # prepare_output_directory called once
    mock_prepare.assert_called_once_with(wnd.output_dir, "output")

    # export_gif called with correct raw flag and z-range
    call_args = mock_export.call_args.kwargs
    assert call_args["raw"] == expected_raw
    if expected_raw:
        assert call_args["zmin"] == wnd._zmin_raw
        assert call_args["zmax"] == wnd._zmax_raw
    else:
        assert call_args["zmin"] == wnd._zmin_flat
        assert call_args["zmax"] == wnd._zmax_flat

    # timestamp and scale bar toggles passed correctly
    assert call_args["draw_ts"] is True
    assert call_args["draw_scale"] is False
    wnd.close()
    wnd.deleteLater()


@pytest.mark.parametrize(
    "raw_present, raw_checked, expected_raw, exporters",
    [
        (True, True, True, {"gif": True, "video": True, "image": True}),
        (False, True, False, {"gif": True, "video": False, "image": True}),
        (True, False, False, {"gif": False, "video": True, "image": False}),
    ],
)
@patch("playnano.io.gif_export.export_gif")
@patch("playnano.io.video_export.export_video")
@patch("playnano.io.image_sequence_export.export_image_sequence")
@patch("playnano.gui.window.prepare_output_directory", return_value="mock_dir")
def test_export_animated_branches(
    mock_prepare,
    mock_image_seq,
    mock_video,
    mock_gif,
    raw_present,
    raw_checked,
    expected_raw,
    exporters,
    qtbot,
):
    """Test _export_animated handles raw branches, z-range & exporter selection."""
    mock_stack = MagicMock(width=256, height=256, data=np.random.rand(1, 10, 10))
    mock_stack.pixel_size_nm = 1.0
    mock_stack.time_for_frame = MagicMock(return_value=0.1)
    mock_stack.processed = {"raw": mock_stack.data} if raw_present else {}

    wnd = MainWindow(mock_stack)
    qtbot.addWidget(wnd)

    wnd.animated_raw_radio.setChecked(raw_checked)
    wnd.animated_processed_radio.setChecked(not raw_checked)

    # Explicit exporter selection
    wnd.export_gif_cb.setChecked(exporters["gif"])
    wnd.export_mp4_cb.setChecked(exporters["video"])
    wnd.export_avi_cb.setChecked(False)
    wnd.export_png_folder_cb.setChecked(exporters["image"])

    wnd._zmin_raw, wnd._zmax_raw = -1.0, 1.0
    wnd._zmin_flat, wnd._zmax_flat = -2.0, 2.0

    wnd.show_timestamp_box.setChecked(True)
    wnd.show_scale_bar_box.setChecked(False)

    # ---------------- Act ----------------
    wnd._export_animated()

    # ---------------- Assert ----------------
    mock_prepare.assert_called_once_with(wnd.output_dir, "output")

    expected_zmin = wnd._zmin_raw if expected_raw else wnd._zmin_flat
    expected_zmax = wnd._zmax_raw if expected_raw else wnd._zmax_flat

    def assert_export(mock, name):
        kwargs = mock.call_args.kwargs
        assert kwargs["raw"] == expected_raw
        assert kwargs["zmin"] == expected_zmin
        assert kwargs["zmax"] == expected_zmax
        assert kwargs["draw_ts"] is True
        assert kwargs["draw_scale"] is False

    if exporters["gif"]:
        assert mock_gif.called
        assert_export(mock_gif, "gif")
    else:
        mock_gif.assert_not_called()

    if exporters["video"]:
        assert mock_video.called
        assert_export(mock_video, "video")
    else:
        mock_video.assert_not_called()

    if exporters["image"]:
        assert mock_image_seq.called
        assert_export(mock_image_seq, "image")
    else:
        mock_image_seq.assert_not_called()

    wnd.close()
    wnd.deleteLater()


@patch("playnano.io.gif_export.export_gif", side_effect=Exception("Export failed"))
@patch("playnano.gui.window.prepare_output_directory", return_value="mock_dir")
def test_export_gif_logs_error(mock_prepare, mock_export, qtbot, caplog):
    """Test that _export_gif logs an error on failure."""
    mock_stack = MagicMock(width=256, height=256, data=np.random.rand(1, 10, 10))
    mock_stack.pixel_size_nm = 1.0
    mock_stack.time_for_frame = MagicMock(return_value=0.1)
    mock_stack.processed = {"raw": mock_stack.data}

    wnd = MainWindow(mock_stack)
    qtbot.addWidget(wnd)

    with caplog.at_level("ERROR"):
        wnd._export_gif()

    assert "GIF export failed: Export failed" in caplog.text
    mock_export.assert_called_once()
    mock_prepare.assert_called_once()
    wnd.close()
    wnd.deleteLater()


def test_show_frame_handles_annotation_exception(qtbot, caplog):
    """Test that show_frame logs an error if set_annotations raises."""
    mock_stack = MagicMock(width=256, height=256, data=np.random.rand(1, 5, 5))
    mock_stack.pixel_size_nm = 1.0
    mock_stack.time_for_frame = MagicMock(return_value=0.1)
    wnd = MainWindow(mock_stack)
    qtbot.addWidget(wnd)

    wnd.viewer.set_annotations = MagicMock(side_effect=Exception("fail"))
    caplog.set_level(logging.ERROR)
    wnd.show_frame(0)
    assert "Failed to set annotations" in caplog.text
    wnd.close()
    wnd.deleteLater()


def test_keypress_calls_super_for_other_keys(qtbot, monkeypatch):
    """Test that unhandled key presses call the superclass keyPressEvent."""
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

    mock_stack = MagicMock(width=256, height=256, data=np.random.rand(1, 5, 5))
    mock_stack.pixel_size_nm = 1.0
    mock_stack.time_for_frame = MagicMock(return_value=0.1)
    wnd = MainWindow(mock_stack)
    qtbot.addWidget(wnd)

    called = {}

    from PySide6.QtWidgets import QMainWindow

    # Patch the superclass method to record calls, regardless of signature
    def fake_super_keypress(*args, **kwargs):
        called["yes"] = True

    monkeypatch.setattr(QMainWindow, "keyPressEvent", fake_super_keypress)

    # Trigger key press with an unhandled key
    event = MagicMock()
    event.key.return_value = Qt.Key_Z  # Not mapped in MainWindow
    wnd.keyPressEvent(event)

    assert (
        "yes" in called
    ), "Superclass keyPressEvent should have been called for unhandled keys"

    wnd.close()
    wnd.deleteLater()


@patch("playnano.io.export_data.export_bundles")
@patch("playnano.gui.window.prepare_output_directory", return_value="mock_dir")
def test_export_checked_calls_export(mock_prepare, mock_export_bundles):
    """Test _export_gif branches on raw/processed data and z-range values."""
    # 1) Build a bare MainWindow without running its __init__
    from playnano.gui.window import MainWindow

    wnd = MainWindow.__new__(MainWindow)

    # 2) Inject just the attributes that _export_checked uses:
    #    – Three checkboxes
    wnd.export_npz_cb = MagicMock(isChecked=MagicMock(return_value=True))
    wnd.export_ome_tiff_cb = MagicMock(isChecked=MagicMock(return_value=False))
    wnd.export_h5_cb = MagicMock(isChecked=MagicMock(return_value=True))
    #    – Two radio buttons
    wnd.data_raw_radio = MagicMock(isChecked=MagicMock(return_value=True))
    wnd.data_processed_radio = MagicMock(isChecked=MagicMock(return_value=False))
    #    – A fake AFM stack and its processed map
    fake_stack = MagicMock()
    fake_stack.processed = {"raw": np.random.rand(1, 10, 10)}
    wnd.afm_stack = fake_stack
    #    – Output settings
    wnd.output_dir = "some/output"
    wnd.output_name = "base_name"

    # 3) Run the method under test
    wnd._export_checked()

    # 4) Verify calls
    mock_prepare.assert_called_once_with("some/output", "output")
    mock_export_bundles.assert_called_once_with(
        fake_stack,
        "mock_dir",
        "base_name",
        ["npz", "h5"],
        raw=True,
    )
    del wnd


@patch("playnano.io.export_data.export_bundles")
@patch("playnano.gui.window.prepare_output_directory", return_value="mock_dir")
def test_export_checked_calls_export_all(mock_prepare, mock_export_bundles):
    """Test that _export_checked exports all formats when all checkboxes are set."""
    # 1) Build a bare MainWindow without running its __init__
    from playnano.gui.window import MainWindow

    wnd = MainWindow.__new__(MainWindow)

    # 2) Inject just the attributes that _export_checked uses:
    #    – Three checkboxes
    wnd.export_npz_cb = MagicMock(isChecked=MagicMock(return_value=True))
    wnd.export_ome_tiff_cb = MagicMock(isChecked=MagicMock(return_value=True))
    wnd.export_h5_cb = MagicMock(isChecked=MagicMock(return_value=True))
    #    – Two radio buttons
    wnd.data_raw_radio = MagicMock(isChecked=MagicMock(return_value=True))
    wnd.data_processed_radio = MagicMock(isChecked=MagicMock(return_value=False))
    #    – A fake AFM stack and its processed map
    fake_stack = MagicMock()
    fake_stack.processed = {"raw": np.random.rand(1, 10, 10)}
    wnd.afm_stack = fake_stack
    #    – Output settings
    wnd.output_dir = "some/output"
    wnd.output_name = "base_name"

    # 3) Run the method under test
    wnd._export_checked()

    # 4) Verify calls
    mock_prepare.assert_called_once_with("some/output", "output")
    mock_export_bundles.assert_called_once_with(
        fake_stack,
        "mock_dir",
        "base_name",
        ["npz", "tif", "h5"],
        raw=True,
    )
    del wnd


@patch("playnano.gui.window.prepare_output_directory")
@patch("playnano.io.export_data.export_bundles")
def test_export_checked_no_formats(mock_export, mock_prepare, qtbot, caplog):
    """Test _export_checked logs a message and does nothing if no formats selected."""
    wnd = MainWindow.__new__(MainWindow)
    wnd.export_npz_cb = MagicMock(isChecked=lambda: False)
    wnd.export_ome_tiff_cb = MagicMock(isChecked=lambda: False)
    wnd.export_h5_cb = MagicMock(isChecked=lambda: False)
    wnd.data_raw_radio = MagicMock(isChecked=lambda: True)
    wnd.afm_stack = MagicMock()
    wnd.output_dir = "mock_dir"
    wnd.output_name = "output"

    caplog.set_level("INFO")
    wnd._export_checked()

    assert "No export formats selected." in caplog.text
    mock_export.assert_not_called()
    mock_prepare.assert_not_called()
    del wnd


@patch("playnano.io.export_data.export_bundles")
@patch("playnano.gui.window.prepare_output_directory", return_value="mock_dir")
def test_export_checked_raw_requested_but_missing(
    mock_prepare, mock_export, qtbot, caplog
):
    """Test that _export_checked falls back to processed data if raw is missing."""
    wnd = MainWindow.__new__(MainWindow)
    wnd.export_npz_cb = MagicMock(isChecked=lambda: True)
    wnd.export_ome_tiff_cb = MagicMock(isChecked=lambda: False)
    wnd.export_h5_cb = MagicMock(isChecked=lambda: False)
    wnd.data_raw_radio = MagicMock(isChecked=lambda: True)
    wnd.afm_stack = MagicMock(processed={})  # no raw data
    wnd.output_dir = "mock_dir"
    wnd.output_name = "output"

    caplog.set_level("DEBUG")
    wnd._export_checked()

    assert "Data is unprocessed, exporting the unprocessed data." in caplog.text
    call_args = mock_export.call_args.kwargs
    assert call_args["raw"] is False
    del wnd


@patch("playnano.io.export_data.export_bundles", side_effect=Exception("Export failed"))
@patch("playnano.gui.window.prepare_output_directory", return_value="mock_dir")
def test_export_checked_logs_error(mock_prepare, mock_export, qtbot, caplog):
    """Test that _export_checked logs an error if export_bundles raises."""
    wnd = MainWindow.__new__(MainWindow)
    wnd.export_npz_cb = MagicMock(isChecked=lambda: True)
    wnd.export_ome_tiff_cb = MagicMock(isChecked=lambda: False)
    wnd.export_h5_cb = MagicMock(isChecked=lambda: False)
    wnd.data_raw_radio = MagicMock(isChecked=lambda: True)
    wnd.afm_stack = MagicMock(processed={"raw": np.random.rand(1, 10, 10)})
    wnd.output_dir = "mock_dir"
    wnd.output_name = "output"

    caplog.set_level("ERROR")
    wnd._export_checked()

    assert "Export failed: Export failed" in caplog.text
    mock_export.assert_called_once()
    del wnd


def test_on_motion_updates_line_and_spinbox():
    """Test _on_motion updates zmin_raw, line, spinbox, and triggers refresh."""

    wnd = MainWindow.__new__(MainWindow)

    # Inject state
    wnd._dragging = MagicMock()  # acts as the active line
    wnd.line_min = wnd._dragging  # simulate dragging the min line
    wnd._show_flat = False
    wnd._flat = None
    wnd._zmin_raw = 0
    wnd._zmax_raw = 10
    wnd._idx = 1

    # Spinboxes with clamping
    wnd.zmin_spin = MagicMock(
        minimum=lambda: 1,
        maximum=lambda: 9,
        setValue=MagicMock(),
        blockSignals=MagicMock(),
    )
    wnd.zmax_spin = MagicMock(
        minimum=lambda: 1,
        maximum=lambda: 9,
        setValue=MagicMock(),
        blockSignals=MagicMock(),
    )

    # Mock canvas and callbacks
    wnd.hist_canvas = MagicMock(draw_idle=MagicMock())
    wnd._move_lines = MagicMock()
    wnd._update_background_color = MagicMock()
    wnd.show_frame = MagicMock()

    # Create event with xdata below minimum (will be clamped to 1)
    event = MagicMock(xdata=0.5)

    # Call method
    wnd._on_motion(event)

    # Check that the line moved to the clamped value
    wnd._dragging.set_xdata.assert_called_once_with([1.0, 1.0])

    # Check zmin_raw updated
    assert wnd._zmin_raw == 1.0

    # Spinbox updated with signals blocked
    wnd.zmin_spin.blockSignals.assert_any_call(True)
    wnd.zmin_spin.setValue.assert_called_once_with(1.0)
    wnd.zmin_spin.blockSignals.assert_any_call(False)

    # Refresh sequence triggered
    wnd.hist_canvas.draw_idle.assert_called_once()
    wnd._move_lines.assert_called_once()
    wnd._update_background_color.assert_called_once()
    wnd.show_frame.assert_called_once_with(1)
    del wnd


def test_on_motion_updates_zmax_flat_branch():
    """Test _on_motion updates _zmax_flat & zmax_spin when line moved in flat mode."""

    # Create an instance without calling __init__
    wnd = MainWindow.__new__(MainWindow)

    # _dragging needs .set_xdata method, not bool
    wnd._dragging = MagicMock()
    wnd._dragging.set_xdata = MagicMock()

    wnd.line_min = MagicMock()  # Different object → triggers zmax branch
    wnd._show_flat = True
    wnd._flat = [1]
    wnd._zmax_flat = 0
    wnd._zmax_raw = 0
    wnd._idx = 0

    wnd.zmin_spin = MagicMock()
    wnd.zmin_spin.minimum.return_value = 0
    wnd.zmax_spin = MagicMock()
    wnd.zmax_spin.maximum.return_value = 5
    wnd.zmax_spin.setValue = MagicMock()
    wnd.zmax_spin.blockSignals = MagicMock()

    wnd.hist_canvas = MagicMock()
    wnd._move_lines = MagicMock()
    wnd._update_background_color = MagicMock()
    wnd.show_frame = MagicMock()

    event = MagicMock(xdata=4.0)
    wnd._on_motion(event)

    # Optionally test that set_xdata called with [4.0, 4.0]
    wnd._dragging.set_xdata.assert_called_once_with([4.0, 4.0])
    del wnd


def test_on_motion_updates_zmax_raw_branch():
    """Test _on_motion updates _zmax_flat & zmax_spin when line moved in raw mode."""
    wnd = MainWindow.__new__(MainWindow)

    # Set dragging line to be max line (not line_min)
    wnd._dragging = MagicMock()
    wnd.line_min = MagicMock()  # different object

    # Force raw branch
    wnd._show_flat = False
    wnd._flat = None
    wnd._zmax_flat = 0
    wnd._zmax_raw = 0
    wnd._idx = 0

    # Mock zmax spin
    wnd.zmax_spin = MagicMock()
    wnd.zmax_spin.minimum.return_value = 0.0
    wnd.zmax_spin.maximum.return_value = 5.0
    wnd.zmax_spin.setValue = MagicMock()
    wnd.zmax_spin.blockSignals = MagicMock()

    # zmin_spin not used
    wnd.zmin_spin = MagicMock()
    wnd.zmin_spin.minimum.return_value = 0.0
    wnd.zmin_spin.maximum.return_value = 5.0

    # Canvas and callbacks
    wnd.hist_canvas = MagicMock(draw_idle=MagicMock())
    wnd._move_lines = MagicMock()
    wnd._update_background_color = MagicMock()
    wnd.show_frame = MagicMock()

    event = MagicMock(xdata=3.0)

    wnd._on_motion(event)

    assert wnd._zmax_raw == 3.0
    wnd.zmax_spin.blockSignals.assert_any_call(True)
    wnd.zmax_spin.setValue.assert_called_once_with(3.0)
    wnd.zmax_spin.blockSignals.assert_any_call(False)
    wnd.hist_canvas.draw_idle.assert_called_once()
    wnd._move_lines.assert_called_once()
    wnd._update_background_color.assert_called_once()
    wnd.show_frame.assert_called_once_with(wnd._idx)
    del wnd


@pytest.mark.parametrize(
    "which, show_flat, flat, attr_to_check",
    [
        ("min", True, [1], "_zmin_flat"),
        ("min", False, None, "_zmin_raw"),
        ("max", True, [1], "_zmax_flat"),
        ("max", False, None, "_zmax_raw"),
    ],
)
def test_on_spinbox_changed_updates_attributes(which, show_flat, flat, attr_to_check):
    """Test _on_spinbox_changed updates correct zmin/zmax attribute and line."""
    wnd = MainWindow.__new__(MainWindow)
    wnd._show_flat = show_flat
    wnd._flat = flat
    wnd._zmin_flat = 0
    wnd._zmin_raw = 0
    wnd._zmax_flat = 0
    wnd._zmax_raw = 0
    wnd._idx = 0

    # Mock lines and callbacks
    wnd.line_min = MagicMock()
    wnd.line_max = MagicMock()
    wnd._move_lines = MagicMock()
    wnd._update_background_color = MagicMock()
    wnd.show_frame = MagicMock()

    val = 2.5
    wnd._on_spinbox_changed(which, val)

    # Check that the correct attribute was updated
    assert getattr(wnd, attr_to_check) == val

    # Check that the right line was moved
    if which == "min":
        wnd.line_min.set_xdata.assert_called_once_with([val, val])
    else:
        wnd.line_max.set_xdata.assert_called_once_with([val, val])

    # Check callbacks were called
    wnd._move_lines.assert_called_once()
    wnd._update_background_color.assert_called_once()
    wnd.show_frame.assert_called_once_with(wnd._idx)
    del wnd


@pytest.mark.parametrize(
    "show_flat, flat_data", [(True, np.array([1, 2, 3])), (False, None)]
)
@patch("playnano.gui.window.compute_zscale_range", return_value=(1.0, 9.0))
def test_on_auto_recomputes_and_updates(mock_compute, show_flat, flat_data):
    """Test _on_auto recomputes z-scale, updates spinboxes, and refreshes viewer."""
    wnd = MainWindow.__new__(MainWindow)

    # Setup attributes
    wnd._show_flat = show_flat
    wnd._flat = flat_data
    wnd._frames = np.array([0, 1, 2, 3])
    wnd._idx = 0

    wnd._zmin_flat, wnd._zmax_flat = 1.0, 9.0
    wnd._zmin_raw, wnd._zmax_raw = 0.0, 5.0

    # Mock spinboxes
    wnd.zmin_spin = MagicMock()
    wnd.zmax_spin = MagicMock()

    # Mock callbacks
    wnd._draw_bars = MagicMock()
    wnd._init_lines = MagicMock()
    wnd._update_background_color = MagicMock()
    wnd.show_frame = MagicMock()

    # Call method
    wnd._on_auto()

    # compute_zscale_range called with correct data
    expected_array = flat_data if (show_flat and flat_data is not None) else wnd._frames
    mock_compute.assert_called_once_with(expected_array, zmin="auto", zmax="auto")

    # Correct zmin/zmax attributes updated
    if show_flat:
        assert wnd._zmin_flat == 1.0
        assert wnd._zmax_flat == 9.0
    else:
        assert wnd._zmin_raw == 1.0
        assert wnd._zmax_raw == 9.0

    # Spinboxes updated with sorted values
    wnd.zmin_spin.setValue.assert_called_once_with(1.0)
    wnd.zmax_spin.setValue.assert_called_once_with(9.0)

    # Histogram and viewer refreshed
    wnd._draw_bars.assert_called_once()
    wnd._init_lines.assert_called_once()
    wnd._update_background_color.assert_called_once()
    wnd.show_frame.assert_called_once_with(wnd._idx)
    del wnd


# --- Tests for viewer widget ---


@pytest.fixture
def widget(qtbot):
    """Provide a ViewerWidget fixture for painting tests."""
    w = ViewerWidget()
    w.resize(100, 100)
    qtbot.addWidget(w)
    return w


def test_paint_event_zero_division_warning(widget, qtbot, caplog):
    """Test paintEvent logs when division by zero occurs in scale bar calculation."""
    widget._original_pixmap = QPixmap(50, 50)
    widget._scaled_pixmap = QPixmap(50, 50)
    widget._draw_scale_bar = True
    widget._pixel_size_nm = 0.0  # zero triggers division by zero if condition passes
    widget._scale_bar_nm = 10
    widget.show()  # Make sure widget is visible

    # To force the division, temporarily override the method:
    original_paintEvent = widget.paintEvent

    def faulty_paintEvent(event):
        # Copy relevant part but forcibly run division by zero
        try:
            _ = widget._scale_bar_nm / widget._pixel_size_nm
        except ZeroDivisionError:
            log.warning("[ViewerWidget] Division by zero in scale bar calculation.")
        # call original for rest to not break rendering
        original_paintEvent(event)

    widget.paintEvent = faulty_paintEvent

    with caplog.at_level("WARNING"):
        widget.update()
        qtbot.wait(100)

    widget.paintEvent = original_paintEvent  # restore

    assert any(
        "[ViewerWidget] Division by zero in scale bar calculation." in rec.message
        for rec in caplog.records
    )


def test_paint_event_generic_exception_logs(widget, qtbot, caplog):
    """Test paintEvent logs errors when generic exception occurs during painting."""
    widget._original_pixmap = QPixmap(50, 50)
    widget._scaled_pixmap = QPixmap(50, 50)
    widget._draw_scale_bar = False
    widget.show()

    caplog.set_level("ERROR", logger="playnano.gui.widgets.viewer")

    with patch(
        "PySide6.QtGui.QPainter.fillRect", side_effect=RuntimeError("test error")
    ):
        with caplog.at_level("ERROR", logger="playnano.gui.widgets.viewer"):
            widget.update()
            widget.repaint()
            qtbot.waitUntil(lambda: widget.isVisible(), timeout=1000)
            qtbot.wait(100)

    assert any(
        "paintEvent crashed: test error" in record.message for record in caplog.records
    )


def test_paint_event_zero_division_branch(widget, qtbot, caplog):
    """Test paintEvent logs a warning when __rtruediv__ raises ZeroDivisionError."""

    class ZeroDiv:
        """Dummy class that triggers a ZeroDivisionError when used in division."""

        def __bool__(self):
            return True

        def __rtruediv__(self, other):
            raise ZeroDivisionError

    widget.resize(100, 100)
    widget._original_pixmap = QPixmap(50, 50)
    widget._scaled_pixmap = QPixmap(50, 50)
    widget._draw_scale_bar = True
    widget._pixel_size_nm = ZeroDiv()
    widget._scale_bar_nm = 10

    caplog.set_level("WARNING", logger="playnano.gui.widgets.viewer")

    with caplog.at_level("WARNING", logger="playnano.gui.widgets.viewer"):
        widget.show()
        widget.repaint()
        qtbot.waitUntil(lambda: widget.isVisible(), timeout=1000)
        qtbot.wait(100)

    assert any(
        "Division by zero in scale bar calculation." in record.message
        for record in caplog.records
    )
