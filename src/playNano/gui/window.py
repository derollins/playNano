"""Main window for the playNano GUI application."""

import sys
from typing import Optional

import matplotlib
import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSizePolicy,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from playNano.afm_stack import AFMImageStack
from playNano.gui.widgets.controls import PlaybackControls
from playNano.gui.widgets.viewer import ViewerWidget
from playNano.processing.pipeline import ProcessingPipeline
from playNano.utils.constants import (  # define or import your defaults
    default_steps_with_kwargs,
)
from playNano.utils.io_utils import compute_zscale_range


class MainWindow(QMainWindow):
    """Main window for the playNano GUI application."""

    def __init__(self, afm_path: str):
        """Initialize the main window with the given AFM path."""
        super().__init__()
        self.setWindowTitle("playNano Player")

        # Load your AFM stack here (replace with however you load)
        self.afm_stack: AFMImageStack = AFMImageStack.load_data(afm_path)

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

        # set up UI
        self._init_ui()

        self.toggle_proc_btn.clicked.connect(self.toggle_processed)

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

        # 1) The image viewer (expands)
        self.viewer = ViewerWidget()
        image_width_px = self.afm_stack.width
        # Set minimum size to the image size in pixels or 512, whichever is smaller.
        max_min_image_size = min(image_width_px, 512)
        self.viewer.setMinimumSize(max_min_image_size, max_min_image_size)
        self.viewer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        bg_rgb = z_to_rgb(self._zperc_raw, self._vmin_raw, self._vmax_raw)
        self.viewer.set_background_color(bg_rgb)

        left_layout.addWidget(self.viewer, 1)

        # 2) Your playback controls (fixed height)
        self.controls = PlaybackControls()
        # insert an “FPS:” label before the fps spin‑box
        fps_label = QLabel("FPS:")
        # assume PlaybackControls uses a QHBoxLayout as its top‐level layout:
        self.controls.layout().insertWidget(
            2,  # right after the play button
            fps_label,  # the new label
        )
        left_layout.addWidget(self.controls, 0)
        self.controls.play_btn.clicked.connect(self.toggle_play)
        # 3) Slider with 0—N ticks below and numeric end‐labels
        n_frames = self._frames.shape[0]

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

        # 3) Add toggle to swap between raw and filtered
        self.toggle_proc_btn = QPushButton("Toggle Raw/Processed (R)")
        left_layout.addWidget(self.toggle_proc_btn, 0)

        main_layout.addWidget(left_panel)
        left_panel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.setCentralWidget(container)

        self._update_background_color()

        # “Apply Filters” button:
        self.apply_btn = QPushButton("Apply Filters (F)")
        left_layout.addWidget(self.apply_btn)
        self.apply_btn.clicked.connect(self.apply_filters)

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
        self._idx = idx
        self._update_background_color()  # ensure background tracks toggle state
        arr = (
            self._flat if (self._show_flat and self._flat is not None) else self._frames
        )[idx]
        # convert arr → RGB uint8 with your colormap & normalization
        rgb = self._colormap_and_normalize(arr)

        # hand it off to the viewer canvas
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
        afm_path=r"C:\Users\ggjh246\OneDrive - University of Leeds\Code\playNano_testdata\save-2025.06.06-17.47.19.349.h5-jpk"  # noqa
    )
    win.show()
    sys.exit(app.exec())
