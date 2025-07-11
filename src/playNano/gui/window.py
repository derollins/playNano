"""Main window for the playNano GUI application."""

import logging
import sys
from typing import Optional
from importlib.resources import files

import matplotlib
import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSizePolicy,
    QSlider,
    QVBoxLayout,
    QWidget,
)
from PySide6.QtGui import QFontDatabase, QFont

from playNano.afm_stack import AFMImageStack
from playNano.gui.widgets.controls import PlaybackControls
from playNano.gui.widgets.viewer import ViewerWidget
from playNano.processing.pipeline import ProcessingPipeline
from playNano.utils.constants import (  # define or import your defaults
    default_steps_with_kwargs,
)
from playNano.utils.io_utils import compute_zscale_range

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MainWindow(QMainWindow):
    """Main window for the playNano GUI application."""

    def __init__(self, afm_path: str):
        """Initialize the main window with the given AFM path."""
        super().__init__()
        self.setWindowTitle("playNano Player")

        font_path = files("playNano.fonts").joinpath("Steps-Mono.otf")
        font_id = QFontDatabase.addApplicationFont(str(font_path))

        # Load your AFM stack here (replace with however you load)
        self.afm_stack: AFMImageStack = AFMImageStack.load_data(afm_path)

        margin_w = 150  # for UI controls horizontally
        margin_h = 200  # for UI controls vertically, sliders etc.

        initial_width = self.afm_stack.width + margin_w
        initial_height = self.afm_stack.height + margin_h

        self.resize(initial_width, initial_height)

        # keep track of which filters the user has configured
        # e.g. list of tuples: [("gaussian_filter", {"sigma":2.0}), ...]
        self.processing_steps: list[tuple[str, dict]] = []

        # viewer state
        self._idx = 0
        self._frames = self.afm_stack.data  # loaded frames (N, H, W)
        self._vmin_raw, self._vmax_raw = compute_zscale_range(
            self._frames, "auto", "auto"
        )
        # percentile (0–100) used for background color
        self._percentile_P = 25
        self._zperc_raw = float(np.percentile(self._frames, self._percentile_P))
        self._zperc_flat = None
        self._vmin_flat, self._vmax_flat = None, None  # not available yet
        self._flat: Optional[np.ndarray] = None  # will hold filtered stack
        self._show_flat = False  # Start in raw view mode
        if font_id == -1:
            logger.warning("Failed to load Steps Mono! Falling back to Arial.")
            self.custom_font = QFont("Arial", 12)
        else:
            font_family = QFontDatabase.applicationFontFamilies(font_id)[0]
            logger.debug(f"Loaded font: {font_family}")
            self.custom_font = QFont(font_family, 12)

        # set up UI
        self._init_ui()

    def _init_ui(self):
        """Set up the main window UI."""

        # ─── Top‐level container ─────────────────────────────────
        container = QWidget()
        main_layout = QVBoxLayout(container)

        # ─── Left panel: viewer + playback controls below ───────────
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)

        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(4)  # optional, make it neat

        # 1) The image viewer (start at 100% image size)
        self.viewer = ViewerWidget()
        image_w, image_h = self.afm_stack.width, self.afm_stack.height

        # Set minimum size (what you had before, e.g., 256)
        min_size = min(image_w, 256)
        self.viewer.setMinimumSize(min_size, min_size)

        # Set initial size to 100% image size
        self.viewer.resize(image_w, image_h)

        self.viewer.set_annotation_font(self.custom_font)

        self.viewer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        bg_rgb = z_to_rgb(self._zperc_raw, self._vmin_raw, self._vmax_raw)
        self.viewer.set_background_color(bg_rgb)

        left_layout.addWidget(self.viewer, 1)

        # 2) Tickboxes for annotations
        # ── Add timestamp + scale bar toggles in a horizontal layout ──
        annotation_hbox = QHBoxLayout()
        annotation_hbox.setSpacing(5)  # Optional: space between checkboxes

        self.show_timestamp_box = QCheckBox("Show Timestamp")
        self.show_timestamp_box.setChecked(True)

        self.show_scale_bar_box = QCheckBox("Show Scale Bar")
        self.show_scale_bar_box.setChecked(True)

        annotation_hbox.addWidget(self.show_timestamp_box)
        annotation_hbox.addWidget(self.show_scale_bar_box)

        left_layout.addLayout(annotation_hbox)

        # 3) Your playback controls (fixed height)
        # ── Playback controls layout: [Play] [FPS:] [SpinBox] ──
        playback_hbox = QHBoxLayout()
        playback_hbox_left = QHBoxLayout()
        playback_hbox_right = QHBoxLayout()
        self.controls = PlaybackControls()

        # Access subwidgets (assumes you expose them in PlaybackControls)
        play_btn = self.controls.play_btn
        fps_box = self.controls.fps_box

        fps_label = QLabel("FPS:")
        fps_label.setAlignment(Qt.AlignVCenter | Qt.AlignRight)

        # Add widgets horizontally
        playback_hbox_left.addWidget(play_btn, 1)
        playback_hbox_right.addWidget(fps_label, 1)
        playback_hbox_right.addWidget(fps_box, 1)

        playback_hbox.addLayout(playback_hbox_left)
        playback_hbox.addLayout(playback_hbox_right)

        left_layout.addLayout(playback_hbox)

        self.controls.play_btn.clicked.connect(self.toggle_play)
        self.controls.fps_box.valueChanged.connect(self._update_timer_interval)

        # 4) Slider with 0—N ticks below and numeric end‐labels
        n_frames = self._frames.shape[0]

        # Refresh the current frame when tickbox checked or unchecked
        self.show_timestamp_box.toggled.connect(lambda: self.show_frame(self._idx))
        self.show_scale_bar_box.toggled.connect(lambda: self.show_frame(self._idx))

        # configure the slider itself
        slider = self.controls.slider
        slider.setRange(0, n_frames - 1)
        slider.setTickPosition(QSlider.TickPosition.TicksBelow)  # draw tick marks
        slider.setTickInterval(max(1, n_frames // 10))  # Tick every 10 frames
        slider.valueChanged.connect(self.show_frame)
        slider.setValue(0)

        # wrap it with numeric labels
        slider_hbox = QHBoxLayout()
        slider_hbox.addWidget(QLabel("0"))
        slider_hbox.addWidget(slider, 1)  # stretch in middle
        slider_hbox.addWidget(QLabel(str(n_frames - 1)))

        left_layout.addLayout(slider_hbox)

        # 5) Button row: Apply Filters + Toggle Raw/Processed ──
        button_hbox = QHBoxLayout()
        self.apply_btn = QPushButton("Apply Filters (F)")
        self.toggle_proc_btn = QPushButton("Toggle Raw/Processed (R)")

        button_hbox.addWidget(self.apply_btn)
        button_hbox.addWidget(self.toggle_proc_btn)
        left_layout.addLayout(button_hbox)

        # Add the left panel to the main layout
        main_layout.addWidget(left_panel, 1)

        container.setLayout(main_layout)

        # Set the central widget so it appears!
        self.setCentralWidget(container)

        # Connect buttons
        self.apply_btn.clicked.connect(self.apply_filters)
        self.toggle_proc_btn.clicked.connect(self.toggle_processed)

        # ── Timer for playback ───────────────────────────────────────────────────

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._next_frame)

        # show the very first frame
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
    from PySide6.QtWidgets import QApplication

    app = QApplication(sys.argv)
    win = MainWindow(
        afm_path=r"C:\Users\ggjh246\OneDrive - University of Leeds\Code\playNano_testdata\save-2025.05.20-12.57.06.187.h5-jpk"  # noqa
    )
    win.show()
    sys.exit(app.exec())
