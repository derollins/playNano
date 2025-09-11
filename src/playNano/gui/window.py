"""
Main GUI window for browsing and exporting HS-AFM image stacks.

Provides:
- Frame playback controls (play/pause, slider, FPS).
- Raw vs. processed toggling and filter application.
- Z-scale histogram with draggable range lines.
- Export tabs for GIF, NPZ, OME-TIFF, HDF5.
"""

import logging
from importlib.resources import files
from typing import Optional

import matplotlib
import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QFont, QFontDatabase
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QRadioButton,
    QSizePolicy,
    QSlider,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from playNano.afm_stack import AFMImageStack
from playNano.gui.widgets.controls import PlaybackControls
from playNano.gui.widgets.viewer import ViewerWidget
from playNano.processing.pipeline import ProcessingPipeline
from playNano.utils.constants import default_steps_with_kwargs
from playNano.utils.io_utils import compute_zscale_range, prepare_output_directory

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MainWindow(QMainWindow):
    """Main window for the playNano GUI application."""

    def __init__(
        self,
        afm_stack: AFMImageStack,
        processing_steps: Optional[list[tuple[str, dict]]] = None,
        output_dir: Optional[str] = None,
        output_name: str = "",
        scale_bar_nm: int = 100,
        zmin: str = "auto",
        zmax: str = "auto",
    ):
        """
        Initialize the main application window.

        Parameters
        ----------
        afm_stack : AFMImageStack
            The loaded AFM image stack to display and process.
        processing_steps : list of (str, dict), optional
            A list of (filter_name, kwargs) tuples defining the default
            processing pipeline steps.  If None, defaults will be used.
        output_dir : str, optional
            Base directory in which to write exported files.
        output_name : str, default="playNano_export"
            Base filename to use for exports.
        scale_bar_nm : int, default=100
            Physical length (nm) of the scale bar to draw on images.
        zmin, zmax : 'auto' or float, default="auto"
            Display range endpoints; if "auto", they will be computed
            from the data.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If zmin/zmax cannot be parsed (not 'auto' or float).

        """
        if not QMainWindow.__init__.__call__:
            raise RuntimeError("Base QMainWindow not properly initialized")
        super().__init__()
        self.setWindowTitle("playNano Player")

        steps_path = files("playNano.fonts").joinpath("Steps-Mono/Steps-Mono.otf")
        steps_id = QFontDatabase.addApplicationFont(str(steps_path))

        basic_path = files("playNano.fonts").joinpath("basic/basic_regular.ttf")
        basic_id = QFontDatabase.addApplicationFont(str(basic_path))

        steps_family = (
            QFontDatabase.applicationFontFamilies(steps_id)[0]
            if steps_id != -1
            else None
        )
        basic_family = (
            QFontDatabase.applicationFontFamilies(basic_id)[0]
            if basic_id != -1
            else None
        )

        if not steps_family:
            logger.warning(
                "Failed to load Steps Mono font. Falling back to Arial for annotations."
            )
            steps_family = "Arial"

        if not basic_family:
            logger.warning("Failed to load basic font. GUI stylesheet will fallback.")

        self.annotation_font = QFont(steps_family, 18)

        self.afm_stack: AFMImageStack = afm_stack

        # if this stack was loaded from one of our bundles, it will have
        # a 'raw' snapshot in processed.  Treat that as the "true" raw
        # and the stack.data as the "flat" (processed) frames.
        if "raw" in self.afm_stack.processed:
            # raw frames (unfiltered)
            self._frames = self.afm_stack.processed["raw"]
            # processed / “flat” frames
            self._flat = self.afm_stack.data.copy()
            # start in processed view by default
            self._show_flat = True
        else:
            # standard case: no prior bundle, stack.data is raw
            self._frames = self.afm_stack.data
            self._flat = None
            self._show_flat = False

        self.resize(
            int(self.afm_stack.width * 1.5),
            self.afm_stack.height + 200,
        )
        self.processing_steps: list[tuple[str, dict]] = processing_steps or []
        self.output_dir = output_dir
        self.output_name = output_name
        self.scale_bar_nm = scale_bar_nm
        self.zmin = zmin
        self.zmax = zmax
        self._idx = 0
        self._percentile_P = 25

        # Raw view z-scale
        self._zmin_raw, self._zmax_raw = compute_zscale_range(
            self._frames, zmin=zmin, zmax=zmax
        )
        self._zperc_raw = float(np.percentile(self._frames, self._percentile_P))

        # Processed (flat) view z-scale
        if self._flat is not None:
            self._zmin_flat, self._zmax_flat = compute_zscale_range(
                self._flat, zmin=zmin, zmax=zmax
            )
            self._zperc_flat = float(np.percentile(self._flat, self._percentile_P))
        else:
            # No processed stack yet: initialize to raw values
            self._zmin_flat, self._zmax_flat = self._zmin_raw, self._zmax_raw
            self._zperc_flat = self._zperc_raw

        # Prepare Matplotlib Figure and Canvas
        self.hist_fig = Figure(figsize=(4, 2))
        self.hist_fig.patch.set_facecolor("#252525")
        self.hist_canvas = FigureCanvas(self.hist_fig)
        self.hist_ax = self.hist_fig.add_subplot(111)

        # Create spin boxes for numeric zmin/zmax control
        RANGE_MIN = -1e5
        RANGE_MAX = +1e5

        self.zmin_spin = QDoubleSpinBox()
        self.zmin_spin.setRange(RANGE_MIN, RANGE_MAX)
        self.zmin_spin.setSingleStep(0.1)
        self.zmin_spin.setValue(self._zmin_raw)
        self.zmin_spin.setDecimals(1)

        self.zmax_spin = QDoubleSpinBox()
        self.zmax_spin.setRange(RANGE_MIN, RANGE_MAX)
        self.zmax_spin.setSingleStep(0.1)
        self.zmax_spin.setValue(self._zmax_raw)
        self.zmax_spin.setDecimals(1)

        self._init_ui()

    def _init_ui(self):
        """
        Construct and lay out all GUI widgets.

        This method builds the left-hand viewer panel, playback controls,
        annotation toggles, and right-hand export tabs.

        Returns
        -------
        None
        """

        # ─── Top-level container ─────────────────────────────────
        container = QWidget()
        main_layout = QHBoxLayout(container)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # ─── Left Panel ─────────────────────────────────────
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)

        # Set zero margins and spacing so viewer is flush
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(0)

        # Viewer widget stays at the top, no padding
        self.viewer = ViewerWidget()
        self.viewer.setObjectName("viewer")
        self.viewer.setMinimumSize(min(self.afm_stack.width, 256), 256)
        self.viewer.set_annotation_font(self.annotation_font)
        self.viewer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.viewer.set_background_color(
            z_to_rgb(self._zperc_raw, self._zmin_raw, self._zmax_raw)
        )
        left_layout.addWidget(self.viewer)

        # Wrap the controls in a container widget with padding
        controls_container = QWidget()
        controls_layout = QVBoxLayout(controls_container)
        controls_layout.setContentsMargins(10, 10, 10, 10)  # <-- padding here
        controls_layout.setSpacing(8)  # spacing between controls

        # Annotation controls
        self.show_timestamp_box = QCheckBox("Show Timestamp")
        self.show_timestamp_box.setChecked(True)
        self.show_timestamp_box.toggled.connect(lambda: self.show_frame(self._idx))
        self.show_scale_bar_box = QCheckBox("Show Scale Bar")
        self.show_scale_bar_box.setChecked(True)
        self.show_scale_bar_box.toggled.connect(lambda: self.show_frame(self._idx))

        annotation_hbox = QHBoxLayout()
        annotation_hbox.addWidget(self.show_timestamp_box)
        annotation_hbox.addWidget(self.show_scale_bar_box)
        controls_layout.addLayout(annotation_hbox)

        self.controls = PlaybackControls()
        play_btn = self.controls.play_btn
        fps_label = QLabel("FPS:")
        fps_label.setAlignment(Qt.AlignVCenter | Qt.AlignRight)

        n_frames = self._frames.shape[0]
        slider = self.controls.slider
        slider.setRange(0, n_frames - 1)
        slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        slider.setTickInterval(max(1, n_frames // 10))
        slider.valueChanged.connect(self.show_frame)
        slider.setValue(0)

        play_btn.setFixedSize(78, 30)
        play_btn.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        slider.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        slider.setMinimumWidth(200)

        fps_container = QWidget()
        fps_layout = QHBoxLayout(fps_container)
        fps_layout.setContentsMargins(0, 0, 0, 0)
        fps_layout.setSpacing(5)
        fps_layout.addWidget(fps_label)
        fps_layout.addWidget(self.controls.fps_box)
        fps_layout.addStretch(1)
        self.controls.fps_box.setFixedWidth(80)

        playback_hbox = QHBoxLayout()
        playback_hbox.addWidget(play_btn)
        playback_hbox.addSpacing(10)
        playback_hbox.addWidget(slider, 1)
        playback_hbox.addWidget(fps_container)

        controls_layout.addLayout(playback_hbox)

        self.controls.play_btn.clicked.connect(self.toggle_play)
        self.controls.fps_box.valueChanged.connect(self._update_timer_interval)

        self.apply_btn = QPushButton("Apply Filters (F)")
        self.toggle_proc_btn = QPushButton("Toggle Raw/Processed (R)")

        filter_hbox = QHBoxLayout()
        filter_hbox.addWidget(self.apply_btn)
        filter_hbox.addWidget(self.toggle_proc_btn)
        controls_layout.addLayout(filter_hbox)

        self.apply_btn.clicked.connect(self.apply_filters)
        self.toggle_proc_btn.clicked.connect(self.toggle_processed)

        # Add the controls container (with padding) below the viewer
        left_layout.addWidget(controls_container)

        main_layout.addWidget(left_panel, 2)

        # ─── Right Panel ─────────────────────────────────────────────────────
        right_tabs = QTabWidget()
        right_tabs.setMinimumWidth(250)
        right_tabs.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)

        # ── Export Tab ───────────────────────────────────────────────────────
        export_tab = QWidget()
        export_layout = QVBoxLayout(export_tab)

        # ── Group: GIF Export ────────────────────────────────────────────────
        gif_group = QGroupBox("Save Animated GIF")
        gif_layout = QVBoxLayout()
        gif_layout.setContentsMargins(10, 25, 10, 10)

        self.gif_raw_radio = QRadioButton("Save Raw")
        self.gif_processed_radio = QRadioButton("Save Processed")
        self.gif_processed_radio.setChecked(True)

        gif_radio_group = QButtonGroup(self)
        gif_radio_group.addButton(self.gif_raw_radio)
        gif_radio_group.addButton(self.gif_processed_radio)

        self.save_gif_btn = QPushButton("Save GIF")

        radio_row = QHBoxLayout()
        radio_row.addWidget(self.gif_raw_radio)
        radio_row.addWidget(self.gif_processed_radio)

        gif_layout.addLayout(radio_row)
        gif_layout.addWidget(self.save_gif_btn)
        gif_group.setLayout(gif_layout)

        # ── Group: Z-Scale Histogram ─────────────────────────────
        hist_group = QGroupBox("Z-Scale Histogram")
        hist_layout = QVBoxLayout(hist_group)
        hist_layout.setContentsMargins(10, 25, 10, 10)
        hist_group.setFixedHeight(150)

        # Matplotlib canvas
        hist_layout.addWidget(self.hist_canvas)

        # Spinboxes beneath the canvas
        spin_layout = QHBoxLayout()
        spin_layout.addWidget(QLabel("zmin:"))
        spin_layout.addWidget(self.zmin_spin)
        spin_layout.addSpacing(10)
        spin_layout.addWidget(QLabel("zmax:"))
        spin_layout.addWidget(self.zmax_spin)
        hist_layout.addLayout(spin_layout)

        # Auto button on its own line, right-aligned
        self.auto_btn = QPushButton("Auto")
        self.auto_btn.setToolTip("Reset to automatic (1st/99th percentile) z-range")
        hist_layout.addWidget(self.auto_btn, alignment=Qt.AlignRight)

        hist_group.setLayout(hist_layout)
        export_layout.addWidget(hist_group)

        self.zmin_spin.setFixedWidth(80)
        self.zmax_spin.setFixedWidth(80)
        self.auto_btn.setFixedHeight(30)
        self.auto_btn.setFixedWidth(60)

        # ── Group: Data Export ───────────────────────────────────────────────
        data_group = QGroupBox("Data Export")
        data_layout = QVBoxLayout()
        data_layout.setContentsMargins(10, 25, 10, 10)

        # Add radio buttons for processed/raw selection
        self.data_raw_radio = QRadioButton("Export Raw")
        self.data_processed_radio = QRadioButton("Export Processed")
        self.data_processed_radio.setChecked(True)

        data_radio_group = QButtonGroup(self)
        data_radio_group.addButton(self.data_raw_radio)
        data_radio_group.addButton(self.data_processed_radio)

        radio_row = QHBoxLayout()
        radio_row.addWidget(self.data_raw_radio)
        radio_row.addWidget(self.data_processed_radio)
        data_layout.addLayout(radio_row)

        # Format checkboxes in a horizontal layout
        format_hbox = QHBoxLayout()
        self.export_npz_cb = QCheckBox("NPZ")
        self.export_ome_tiff_cb = QCheckBox("OME-TIFF")
        self.export_h5_cb = QCheckBox("HDF5")

        for cb in [self.export_npz_cb, self.export_ome_tiff_cb, self.export_h5_cb]:
            cb.setChecked(True)
            format_hbox.addWidget(cb)

        data_layout.addLayout(format_hbox)

        # Export button
        self.export_btn = QPushButton("Export Selected")
        data_layout.addWidget(self.export_btn)
        data_group.setLayout(data_layout)

        # ── Add to right tab layout ──────────────────────────────────────────
        export_layout.addWidget(gif_group)
        export_layout.addSpacing(10)
        export_layout.addWidget(data_group)
        export_layout.addStretch(1)

        right_tabs.addTab(export_tab, "Export")
        main_layout.addWidget(right_tabs, 1)

        self.setCentralWidget(container)

        self.save_gif_btn.clicked.connect(self._export_gif)
        self.export_btn.clicked.connect(self._export_checked)

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._next_frame)
        self._update_export_options()
        self.show_frame(0)

        # Draw histogram & connect interactions
        QTimer.singleShot(0, self._draw_bars)
        QTimer.singleShot(0, self._init_lines)
        QTimer.singleShot(0, self._connect_hist_events)

        # Connect spin boxes to slot
        self.zmin_spin.valueChanged.connect(
            lambda v: self._on_spinbox_changed("min", v)
        )
        self.zmax_spin.valueChanged.connect(
            lambda v: self._on_spinbox_changed("max", v)
        )
        self.auto_btn.clicked.connect(self._on_auto)

    def apply_filters(self):
        """
        Apply the selected processing pipeline to the AFM stack.

        Builds a ProcessingPipeline from `self.processing_steps` (or defaults),
        runs it, updates the flattened data (`self._flat`), recomputes display
        ranges, and refreshes the viewer.

        Returns
        -------
        None

        Raises
        ------
        RuntimeError
            If the processing pipeline fails.

        """
        # choose which steps
        steps = self.processing_steps or default_steps_with_kwargs

        # execute pipeline
        pipeline = ProcessingPipeline(self.afm_stack)
        for name, params in steps:
            pipeline.add_filter(name, **params)
        pipeline.run()

        # stash filtered frames
        self._flat = self.afm_stack.data

        # recompute display range for the new (filtered) data
        self._zmin_flat, self._zmax_flat = compute_zscale_range(
            self._flat,
            zmin=self.zmin,
            zmax=self.zmax,
        )
        self._zperc_flat = float(np.percentile(self._flat, self._percentile_P))
        # switch to showing flattened
        self._show_flat = True  # ← this ensures flattened view is active
        # Sync spin-boxes to new flat range
        self.zmin_spin.blockSignals(True)
        self.zmin_spin.blockSignals(False)
        self.zmax_spin.blockSignals(True)
        self.zmax_spin.blockSignals(False)
        # and for z scale
        lo, hi = sorted((self._zmin_flat, self._zmax_flat))
        self.zmin_spin.blockSignals(True)
        self.zmax_spin.blockSignals(True)

        self.zmin_spin.setValue(lo)
        self.zmax_spin.setValue(hi)

        self.zmin_spin.blockSignals(False)
        self.zmax_spin.blockSignals(False)

        # then refresh histogram & viewer
        self._draw_bars()
        self._init_lines()
        self._move_lines()
        self.show_frame(self._idx)

        self._update_export_options()
        # Select processed for both GIF and data after filtering
        self.gif_processed_radio.setChecked(True)
        self.data_processed_radio.setChecked(True)

    def toggle_play(self):
        """
        Start or stop automatic frame advancement.

        If playing, stops the QTimer; if paused, starts it at the current FPS.

        Returns
        -------
        None

        """
        if self._timer.isActive():
            self._timer.stop()
            self.controls.play_btn.setText("▶️ Play")
        else:
            fps = self.controls.fps_box.value()
            interval_ms = int(1000 / fps) if fps > 0 else 50
            self._timer.start(interval_ms)
            self.controls.play_btn.setText("⏸ Pause")

    def _next_frame(self):
        """
        Advance to the next frame and update the display.

        Called on each QTimer timeout when playing.

        Returns
        -------
        None
        """
        self._idx = (self._idx + 1) % len(self._frames)
        self.show_frame(self._idx)
        self.controls.slider.setValue(self._idx)

    def show_frame(self, idx: int):
        """
        Render a specific frame in the viewer.

        Parameters
        ----------
        idx : int
            Index of the frame to display (raw or processed, depending on state).

        Returns
        -------
        None

        Raises
        ------
        IndexError
            If `idx` is out of bounds.
        """
        logger.debug(f"[show_frame] Showing index {idx}")
        self._idx = idx
        arr = (
            self._flat if (self._show_flat and self._flat is not None) else self._frames
        )[idx]
        rgb = self._colormap_and_normalize(arr)

        # Read timestamp
        timestamp = self.afm_stack.time_for_frame(idx)

        pixel_size_nm = self.afm_stack.pixel_size_nm
        if not isinstance(pixel_size_nm, (float, int)) or pixel_size_nm <= 0:
            pixel_size_nm = 1.0  # fallback or disable scale bar

        # Draw with annotations
        try:
            self.viewer.set_annotations(
                timestamp=timestamp,
                draw_ts=self.show_timestamp_box.isChecked(),
                draw_scale=self.show_scale_bar_box.isChecked(),
                draw_raw_label=not self._show_flat,
                pixel_size_nm=self.afm_stack.pixel_size_nm,
                scale_bar_nm=self.scale_bar_nm,
            )
        except Exception as e:
            logger.error(f"[MainWindow] Failed to set annotations: {e}")

        self.viewer.display_frame(rgb)

    def _colormap_and_normalize(self, arr):
        """
        Map a 2D array to an RGB uint8 image via z-scaling and colormap.

        Parameters
        ----------
        arr : (H, W) array
            Height map to render.

        Returns
        -------
        rgb : (H, W, 3) uint8 array
            Rendered RGB image.
        """
        if self._show_flat and self._flat is not None:
            zmin, zmax = self._zmin_flat, self._zmax_flat
        else:
            zmin, zmax = self._zmin_raw, self._zmax_raw

        if zmin == zmax:
            norm8 = np.zeros_like(arr, dtype=np.uint8)
        else:
            clipped = np.clip(arr, zmin, zmax)
            norm8 = ((clipped - zmin) / (zmax - zmin) * 255).astype(np.uint8)

        cmap = matplotlib.colormaps.get_cmap("afmhot")
        rgba = cmap(norm8 / 255.0)
        return (rgba[..., :3] * 255).astype(np.uint8)

    def keyPressEvent(self, ev):
        """
        Handle key press events for shortcuts.

        Parameters
        ----------
        ev : QKeyEvent
            The key event to handle.

        Returns
        -------
        None

        Raises
        ------
        None

        Notes
        -----
        Key mapping:
        Space → toggle play/pause
        F → apply filters
        R → toggle raw/processed view
        G → export GIF
        E → export data in checked formats
        """
        k = ev.key()
        if k == Qt.Key_Space:
            self.toggle_play()
        elif k == Qt.Key_F:
            self.apply_filters()
        elif k == Qt.Key_R:
            self.toggle_processed()
        elif k == Qt.Key_G:
            self._export_gif()
        elif k == Qt.Key_E:
            self._export_checked()
        else:
            super().keyPressEvent(ev)

    def toggle_processed(self):
        """
        Flip between raw and processed (flattened) data views.

        If no processed data exists yet, does nothing.

        Returns
        -------
        None
        """
        # If we've never applied filters, nothing to toggle
        if self._flat is None:
            return

        # flip a flag
        self._show_flat = not getattr(self, "_show_flat", False)
        # choose which z-range to use
        zmin, zmax = (
            (self._zmin_flat, self._zmax_flat)
            if self._show_flat
            else (self._zmin_raw, self._zmax_raw)
        )
        # update spinboxes (block signals to avoid recursive updates)
        if self._show_flat:
            lo, hi = self._zmin_flat, self._zmax_flat
        else:
            lo, hi = self._zmin_raw, self._zmax_raw

        lo, hi = sorted((zmin, zmax))
        self.zmin_spin.blockSignals(True)
        self.zmax_spin.blockSignals(True)

        self._set_spinbox_value(self.zmin_spin, lo)
        self._set_spinbox_value(self.zmax_spin, hi)

        self.zmin_spin.blockSignals(False)
        self.zmax_spin.blockSignals(False)
        # rebuild histogram for the newly-selected data
        self._draw_bars()
        self._init_lines()
        # then update viewer
        self._update_background_color()
        self.show_frame(self._idx)

    def _set_spinbox_value(self, spinbox, value):
        """
        Safely set a QDoubleSpinBox's value without emitting signals.

        Parameters
        ----------
        spinbox : QDoubleSpinBox
            The widget to update.
        value : float
            New value to set.

        Returns
        -------
        None
        """

        spinbox.blockSignals(True)
        spinbox.setValue(value)
        spinbox.clearFocus()  # force repaint
        spinbox.repaint()  # in case it didn't
        spinbox.blockSignals(False)

    def _update_background_color(self):
        """
        Update the viewer background color based on current z-percentile.

        Chooses raw vs. flat background depending on `self._show_flat`.
        """
        if self._show_flat and self._flat is not None:
            z_bg = self._zperc_flat
            zmin, zmax = self._zmin_flat, self._zmax_flat
        else:
            z_bg = self._zperc_raw
            zmin, zmax = self._zmin_raw, self._zmax_raw

        rgb = z_to_rgb(z_bg, zmin, zmax, cmap_name="afmhot")
        self.viewer.set_background_color(rgb)

    def _update_timer_interval(self, fps: int):
        """
        Change the playback timer interval when FPS is adjusted.

        Parameters
        ----------
        fps : int
            New frames-per-second rate.
        """
        if self._timer.isActive():
            interval_ms = int(1000 / fps) if fps > 0 else 50
            self._timer.start(interval_ms)

    def _export_gif(self):
        """
        Export the current stack view as an animated GIF.

        Honors the “Raw vs Processed” radio button and writes to
        `self.output_dir`/“output” with filename “gui_export.gif”.

        Returns
        -------
        None

        Raises
        ------
        Exception
            If GIF export fails.
        """
        from playNano.io.gif_export import export_gif

        raw = self.gif_raw_radio.isChecked()
        save_dir = prepare_output_directory(self.output_dir, "output")

        # Check for presence of raw data if user requests it
        if raw and "raw" not in self.afm_stack.processed:
            logger.debug("Data is unprocessed, exporting the unprocessed data.")
            raw = False

        # Grab exactly what the viewer is using right now:
        if raw:
            zmin, zmax = self._zmin_raw, self._zmax_raw
        else:
            zmin, zmax = self._zmin_flat, self._zmax_flat

        try:
            export_gif(
                self.afm_stack,
                True,
                save_dir,
                output_name=self.output_name,
                scale_bar_nm=self.scale_bar_nm,
                raw=raw,
                zmin=zmin,
                zmax=zmax,
                draw_ts=self.show_timestamp_box.isChecked(),
                draw_scale=self.show_scale_bar_box.isChecked(),
            )
            logger.info("Exported GIF.")
        except Exception as e:
            logger.error(f"GIF export failed: {e}")

    def _export_checked(self):
        """
        Export the AFM stack in the formats selected (NPZ, OME-TIFF, HDF5).

        Uses the “Export Raw” / “Export Processed” radio buttons to decide
        which data to write.
        """
        from playNano.io.export_data import export_bundles

        formats = []
        if self.export_npz_cb.isChecked():
            formats.append("npz")
        if self.export_ome_tiff_cb.isChecked():
            formats.append("tif")
        if self.export_h5_cb.isChecked():
            formats.append("h5")

        if not formats:
            logger.info("No export formats selected.")
            return

        raw = self.data_raw_radio.isChecked()

        # Check for presence of raw data if user requests it
        if raw and "raw" not in self.afm_stack.processed:
            logger.debug("Data is unprocessed, exporting the unprocessed data.")
            raw = False

        save_dir = prepare_output_directory(self.output_dir, "output")
        try:
            export_bundles(
                self.afm_stack,
                save_dir,
                self.output_name,
                formats,
                raw=raw,
            )
            logger.info(f"Exported: {', '.join(formats)}")
        except Exception as e:
            logger.error(f"Export failed: {e}")

    def _update_export_options(self):
        """
        Enable/disable export-format controls based on processing state.

        Disables processed-data options if no filters have been applied yet.
        """
        has_filtered = "raw" in self.afm_stack.processed and any(
            key != "raw" for key in self.afm_stack.processed
        )

        # For GIF export
        self.gif_processed_radio.setEnabled(has_filtered)
        if not has_filtered:
            self.gif_raw_radio.setChecked(True)

        # For data export
        self.data_processed_radio.setEnabled(has_filtered)
        if not has_filtered:
            self.data_raw_radio.setChecked(True)

    def _draw_bars(self):
        """Draw only the main histogram bars (with outlier clipping)."""
        ax = self.hist_ax
        ax.clear()
        ax.set_facecolor("#252525")

        # pick raw vs. flat
        data = (
            self._flat if (self._show_flat and self._flat is not None) else self._frames
        )
        vals = np.nan_to_num(data).ravel()

        # clip to 1–99th percentiles (+ small buffer)
        p_low, p_high = np.percentile(vals, [1, 99])
        hist_min = p_low - abs(p_low) / 3
        hist_max = p_high + abs(p_high) / 3

        counts, edges = np.histogram(vals, bins=250, range=(hist_min, hist_max))
        centers = (edges[:-1] + edges[1:]) / 2

        # draw bars & style
        ax.bar(
            centers,
            counts,
            width=edges[1] - edges[0],
            color="lightgray",
            edgecolor="none",
        )
        ax.set_xlim(hist_min, hist_max)
        ax.set_ylim(0, counts.max() * 1.1)
        ax.set_facecolor("#252525")
        ax.grid(True, color="#444", linestyle="--", linewidth=0.5)
        ax.yaxis.set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="x", colors="#888", labelsize=8)
        ax.spines["bottom"].set_color("#555")
        self.hist_fig.subplots_adjust(left=0.05, right=0.98, top=0.95, bottom=0.18)
        self.hist_canvas.draw_idle()

    def _init_lines(self):
        """Add two vertical red lines & shaded spans at the current zmin/zmax."""
        ax = self.hist_ax
        xmin, xmax = ax.get_xlim()

        zmin = self._zmin_flat if self._show_flat else self._zmin_raw
        zmax = self._zmax_flat if self._show_flat else self._zmax_raw

        # lines
        self.line_min = ax.axvline(zmin, color="#ff0000", linewidth=2, picker=5)
        self.line_max = ax.axvline(zmax, color="#3700ff", linewidth=2, picker=5)

        # spans
        self.span_left = ax.axvspan(
            xmin, zmin, facecolor="#209ba5", alpha=0.2, linewidth=0
        )
        self.span_right = ax.axvspan(
            zmax, xmax, facecolor="#209ba5", alpha=0.2, linewidth=0
        )

        self.hist_canvas.draw_idle()

    def _move_lines(self):
        """Reposition the red lines & update their shaded spans—no clearing axes."""
        ax = self.hist_ax
        xmin, xmax = ax.get_xlim()
        zmin = self._zmin_flat if self._show_flat else self._zmin_raw
        zmax = self._zmax_flat if self._show_flat else self._zmax_raw

        # move lines
        self.line_min.set_xdata([zmin, zmin])
        self.line_max.set_xdata([zmax, zmax])

        # refresh spans
        self.span_left.remove()
        self.span_right.remove()
        self.span_left = ax.axvspan(
            xmin, zmin, facecolor="#209ba5", alpha=0.2, linewidth=0
        )
        self.span_right = ax.axvspan(
            zmax, xmax, facecolor="#209ba5", alpha=0.2, linewidth=0
        )

        self.hist_canvas.draw_idle()

    def _connect_hist_events(self):
        """Wire up picking and dragging of the histogram lines."""
        self._dragging = None
        self.hist_canvas.mpl_connect("pick_event", self._on_pick)
        self.hist_canvas.mpl_connect("motion_notify_event", self._on_motion)
        self.hist_canvas.mpl_connect("button_release_event", self._on_release)

    def _on_pick(self, event):
        """
        Start dragging if a line is picked.

        Parameters
        ----------
        event : MatplotlibEvent
            The event object.

        Returns
        -------
        None
        """
        if event.artist in (self.line_min, self.line_max):
            self._dragging = event.artist

    def _on_motion(self, event):
        """
        While dragging, move line, update spinbox, and refresh viewer.

        Parameters
        ----------
        event : MatplotlibEvent
            The event object.

        Returns
        -------
        None
        """
        if self._dragging and event.xdata is not None:
            x = float(event.xdata)
            # Clamp within data range
            x = max(self.zmin_spin.minimum(), min(self.zmax_spin.maximum(), x))

            # Update the line
            self._dragging.set_xdata([x, x])
            self.hist_canvas.draw_idle()

            # Update the corresponding zmin/zmax and spinbox
            if self._dragging is self.line_min:
                if self._show_flat and self._flat is not None:
                    self._zmin_flat = x
                else:
                    self._zmin_raw = x
                self.zmin_spin.blockSignals(True)
                self.zmin_spin.setValue(x)
                self.zmin_spin.blockSignals(False)
            else:
                if self._show_flat and self._flat is not None:
                    self._zmax_flat = x
                else:
                    self._zmax_raw = x
                self.zmax_spin.blockSignals(True)
                self.zmax_spin.setValue(x)
                self.zmax_spin.blockSignals(False)

            # rebuild histogram (so shading and lines update)
            self._move_lines()
            self._update_background_color()
            self.show_frame(self._idx)

    def _on_release(self, event):
        """Stop dragging when mouse release."""
        self._dragging = None

    def _on_spinbox_changed(self, which: str, val: float) -> None:
        """
        Handle changes from the zmin/zmax spinboxes: update histogram lines and viewer.

        Parameters
        ----------
        which : str
            Either 'min' or 'max', indicating which spinbox changed.
        val : float
            The new value from the spinbox.

        Returns
        -------
        None
        """
        val = float(val)
        if which == "min":
            if self._show_flat and self._flat is not None:
                self._zmin_flat = val
            else:
                self._zmin_raw = val
            self.line_min.set_xdata([val, val])
        else:
            if self._show_flat and self._flat is not None:
                self._zmax_flat = val
            else:
                self._zmax_raw = val
            self.line_max.set_xdata([val, val])

        # rebuild histogram (so shading and lines update)
        self._move_lines()
        self._update_background_color()
        self.show_frame(self._idx)

    def _on_auto(self) -> None:
        """
        Reset the z-scale to automatic (1st/99th percentile) and refresh the display.

        This method:
        - Determines whether to use raw data or processed (“flat”) data.
        - Recomputes the zmin and zmax using :func:`compute_zscale_range` with "auto".
        - Updates the zmin/zmax spinboxes without emitting value-changed signals.
        - Redraws the histogram bars and vertical lines.
        - Updates the viewer background color.
        - Redisplays the current frame.

        Returns
        -------
        None
        """
        # recompute auto-ranges for raw or flat
        arr = (
            self._flat if (self._show_flat and self._flat is not None) else self._frames
        )
        # use compute_zscale_range utility
        zmin, zmax = compute_zscale_range(arr, zmin="auto", zmax="auto")
        # store
        if self._show_flat:
            self._zmin_flat, self._zmax_flat = zmin, zmax
        else:
            self._zmin_raw, self._zmax_raw = zmin, zmax

        # Update widgets
        # Sync spin-boxes
        lo, hi = sorted((self._zmin_flat, self._zmax_flat))
        self.zmin_spin.blockSignals(True)
        self.zmax_spin.blockSignals(True)

        self.zmin_spin.setValue(lo)
        self.zmax_spin.setValue(hi)

        self.zmin_spin.blockSignals(False)
        self.zmax_spin.blockSignals(False)

        # Rebuild histogram for the newly-selected data
        self._draw_bars()
        self._init_lines()
        # then update viewer
        self._update_background_color()
        self.show_frame(self._idx)


def z_to_rgb(z_value, zmin, zmax, cmap_name="afmhot"):
    """
    Map a single height value to an RGB triple via a matplotlib colormap.

    Parameters
    ----------
    z_value : float
        Height value to map.
    zmin, zmax : float
        Data range for normalization.  If zmax == zmin, returns black.
    cmap_name : str, default="afmhot"
        Name of the matplotlib colormap to use.

    Returns
    -------
    rgb : tuple of int
        (R, G, B) values in [0, 255].
    """
    span = zmax - zmin
    if span <= 0:
        return (0, 0, 0)
    normed = np.clip((z_value - zmin) / span, 0, 1)
    cmap = matplotlib.colormaps.get_cmap(cmap_name)
    rgba = cmap(normed)
    return tuple(int(255 * c) for c in rgba[:3])
