"""Test for playNano minimal UI."""

from unittest.mock import MagicMock, patch

import numpy as np

from playNano.gui import main
from playNano.gui.window import MainWindow


@patch("playNano.gui.window.AFMImageStack.load_data")
def test_mainwindow_loads_and_interacts(mock_load_data, qtbot):
    """Test that the MainWindow loads and interacts correctly."""

    # Mock AFMImageStack
    # Dummy filter function with version
    def dummy_filter(arr, **kwargs):
        """Make a dummy a filter function."""
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
    mock_load_data.return_value = mock_stack
    mock_stack.pixel_size_nm = 1.0

    mock_stack.provenance = {
        "processing": {"steps": [], "keys_by_name": {}},
        "analysis": [],
        "environment": {},
    }
    mock_stack.processed = {}

    # Instantiate and show
    wnd = MainWindow("dummy")
    qtbot.addWidget(wnd)
    wnd.show()
    assert wnd.isVisible()

    # FPS defaults
    assert wnd.controls.fps_box.value() == 10

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


@patch("playNano.gui.main.QApplication")
@patch("playNano.gui.main.MainWindow")
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
    mock_main_window.assert_called_once_with("dummy/path.h5-jpk")
    mock_window.show.assert_called_once()
    mock_app.exec.assert_called_once()
    mock_exit.assert_called_once()
