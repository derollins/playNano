"""Main window for the playNano GUI application."""

import logging
import sys
from importlib.resources import files
from typing import Optional

import matplotlib
import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QFont, QFontDatabase
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
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
from playNano.utils.io_utils import compute_zscale_range

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MainWindow(QMainWindow):
    """Main window for the playNano GUI application."""

    def __init__(self, afm_path: str):
        """Initialize the main window with the given AFM path."""
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

        self.afm_stack: AFMImageStack = AFMImageStack.load_data(afm_path)

        self.resize(
            int(self.afm_stack.width * 1.5),
            self.afm_stack.height + 200,
        )

        self.processing_steps: list[tuple[str, dict]] = []
        self._idx = 0
        self._frames = self.afm_stack.data
        self._vmin_raw, self._vmax_raw = compute_zscale_range(
            self._frames, "auto", "auto"
        )
        self._percentile_P = 25
        self._zperc_raw = float(np.percentile(self._frames, self._percentile_P))
        self._zperc_flat = None
        self._vmin_flat, self._vmax_flat = None, None
        self._flat: Optional[np.ndarray] = None
        self._show_flat = False

        self._init_ui()

    def _init_ui(self):
        """Set up the main window UI."""

        # ─── Top‐level container ─────────────────────────────────
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
            z_to_rgb(self._zperc_raw, self._vmin_raw, self._vmax_raw)
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

    def apply_filters(self):
        """
        Build and run a ProcessingPipeline over the AFM stack.

        Uses self.processing_steps (or defaults),
        then refresh the viewer.
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
        self._vmin_flat, self._vmax_flat = compute_zscale_range(
            self._flat, "auto", "auto"
        )
        self._zperc_flat = float(np.percentile(self._flat, self._percentile_P))
        # switch to showing flattened
        self._show_flat = True  # ← this ensures flattened view is active
        self._update_background_color()
        # redraw current frame
        self.show_frame(self._idx)

        self._update_export_options()
        # Select processed for both GIF and data after filtering
        self.gif_processed_radio.setChecked(True)
        self.data_processed_radio.setChecked(True)

    def toggle_play(self):
        """Start or stop the automatic frame-advancing timer."""
        if self._timer.isActive():
            self._timer.stop()
            self.controls.play_btn.setText("▶️ Play")
        else:
            fps = self.controls.fps_box.value()
            interval_ms = int(1000 / fps) if fps > 0 else 50
            self._timer.start(interval_ms)
            self.controls.play_btn.setText("⏸ Pause")

    def _next_frame(self):
        """Advance to the next frame in the stack."""
        self._idx = (self._idx + 1) % len(self._frames)
        self.show_frame(self._idx)
        self.controls.slider.setValue(self._idx)

    def show_frame(self, idx: int):
        """Render frame #idx (filtered if available, else raw) in viewer widget."""
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
                pixel_size_nm=self.afm_stack.pixel_size_nm,
                scale_bar_nm=100,
            )
        except Exception as e:
            logger.error(f"[MainWindow] Failed to set annotations: {e}")

        self.viewer.display_frame(rgb)

    def _colormap_and_normalize(self, arr):
        """
        Convert a 2D array to RGB uint8 using a colormap.

        Apply z-scaling, normalize to 0-255, apply a matplotlib
        colormap, and return a HxWx3 uint8.
        """
        if self._show_flat and self._flat is not None:
            zmin, zmax = self._vmin_flat, self._vmax_flat
        else:
            zmin, zmax = self._vmin_raw, self._vmax_raw

        if zmin == zmax:
            norm8 = np.zeros_like(arr, dtype=np.uint8)
        else:
            clipped = np.clip(arr, zmin, zmax)
            norm8 = ((clipped - zmin) / (zmax - zmin) * 255).astype(np.uint8)

        cmap = matplotlib.colormaps.get_cmap("afmhot")
        rgba = cmap(norm8 / 255.0)
        return (rgba[..., :3] * 255).astype(np.uint8)

    def keyPressEvent(self, ev):
        """Mirror key presses to the same methods as our buttons."""
        k = ev.key()
        if k == Qt.Key_Space:
            self.toggle_play()
        elif k == Qt.Key_F:
            self.apply_filters()
        elif k == Qt.Key_R:
            self.toggle_processed()
        elif k == Qt.Key_G:
            self._export_gif()
        # add more keys here (e.g. 'T' → export TIFF, etc.)
        else:
            super().keyPressEvent(ev)

    def toggle_processed(self):
        """Flip between processed frames (self._flat) and raw frames (self._frames)."""
        # If we've never applied filters, nothing to toggle
        if self._flat is None:
            return

        # flip a flag
        self._show_flat = not getattr(self, "_show_flat", False)
        # re‐draw the current frame and bg
        self._update_background_color()
        self.show_frame(self._idx)

    def _update_background_color(self):
        """Update viewer background based on current view (raw or filtered)."""
        if self._show_flat and self._flat is not None:
            z_bg = self._zperc_flat
            vmin, vmax = self._vmin_flat, self._vmax_flat
        else:
            z_bg = self._zperc_raw
            vmin, vmax = self._vmin_raw, self._vmax_raw

        rgb = z_to_rgb(z_bg, vmin, vmax, cmap_name="afmhot")
        self.viewer.set_background_color(rgb)

    def _update_timer_interval(self, fps: int):
        """Update playback timer interval if playing."""
        if self._timer.isActive():
            interval_ms = int(1000 / fps) if fps > 0 else 50
            self._timer.start(interval_ms)

    def _export_gif(self):
        """Export current view as an animated GIF."""
        from playNano.io.gif_export import export_gif
        from playNano.utils.io_utils import prepare_output_directory

        raw = self.gif_raw_radio.isChecked()
        save_dir = prepare_output_directory(".", "output")

        export_gif(
            self.afm_stack,
            True,
            save_dir,
            "gui_export",
            scale_bar_nm=100,
            raw=raw,
            zmin=self._vmin_flat if not raw else self._vmin_raw,
            zmax=self._vmax_flat if not raw else self._vmax_raw,
        )
        logger.info("Exported GIF.")

    def _export_checked(self):
        """Export selected formats (NPZ, OME-TIFF, HDF5)."""
        from playNano.io.export import export_bundles
        from playNano.utils.io_utils import prepare_output_directory

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

        raw = not self._show_flat

        # Check for presence of raw data if user requests it
        if raw and "raw" not in self.afm_stack.processed:
            logger.debug("Data is unprocessed, exporting the unprocessed data.")
            raw = False

        save_dir = prepare_output_directory(".", "output")

        try:
            export_bundles(
                self.afm_stack,
                save_dir,
                "gui_export",
                formats,
                raw=raw,
            )
            logger.info(f"Exported: {', '.join(formats)}")
        except Exception as e:
            logger.error(f"Export failed: {e}")

    def _update_export_options(self):
        """Enable or disable processed export options based on processing state."""
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


def z_to_rgb(z_value, vmin, vmax, cmap_name="afmhot"):
    """Map a data value z_value → RGB via the colormap."""
    span = vmax - vmin
    if span <= 0:
        return (0, 0, 0)
    normed = np.clip((z_value - vmin) / span, 0, 1)
    cmap = matplotlib.colormaps.get_cmap(cmap_name)
    rgba = cmap(normed)
    return tuple(int(255 * c) for c in rgba[:3])


# If you want to launch this window standalone:
if __name__ == "__main__":
    from pathlib import Path

    from PySide6.QtWidgets import QApplication

    app = QApplication(sys.argv)

    qss_path = Path("src/playNano/gui/styles/dark_bluegreen.qss").resolve()
    if qss_path.exists():
        with open(qss_path) as f:
            app.setStyleSheet(f.read())
    win = MainWindow(
        afm_path=r"C:\\Users\\ggjh246\\OneDrive - University of Leeds\\Code\\playNano_testdata\\save-2025.05.20-12.57.06.187.h5-jpk"  # noqa
    )
    win.show()
    sys.exit(app.exec())
