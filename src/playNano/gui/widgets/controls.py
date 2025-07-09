"""Widget for playback controls in the playNano GUI."""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDoubleSpinBox, QPushButton, QSlider, QVBoxLayout, QWidget


class PlaybackControls(QWidget):
    """Widget containing playback controls; play button, slider, and FPS control."""

    def __init__(self):
        """Initialize the playback controls widget."""
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
