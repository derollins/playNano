"""
Defines ViewerWidget, a Qt widget for viewing AFM height map stacks.

The data is viewed as a resizable image with optional timestamp,
scale-bar and “RAW” label overlays. Playback controls allow video
playback of a image stack.

Classes
-------
ViewerWidget
    QWidget subclass that renders one frame of AFM data and draws annotations.
"""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDoubleSpinBox, QPushButton, QSlider, QVBoxLayout, QWidget


class PlaybackControls(QWidget):
    """Widget containing playback controls; play button, slider, and FPS control."""

    def __init__(self):
        """
        Initialize the ViewerWidget.

        - Sets up placeholder pixmaps.
        - Initializes annotation toggles (timestamp, scale bar, RAW label).
        - Chooses a default fallback font.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        super().__init__()
        layout = QVBoxLayout(self)
        self.play_btn = QPushButton("▶️ Play")
        self.slider = QSlider(Qt.Horizontal)
        self.fps_box = QDoubleSpinBox()
        self.fps_box.setRange(0.1, 60)
        self.fps_box.setValue(10)
        layout.addWidget(self.play_btn)
        layout.addWidget(self.slider)
        layout.addWidget(self.fps_box)
