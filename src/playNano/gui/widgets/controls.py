"""Widget for playback controls in the playNano GUI."""

from PySide6.QtWidgets import QPushButton, QSlider, QDoubleSpinBox, QVBoxLayout, QWidget
from PySide6.QtCore import Qt

class PlaybackControls(QWidget):
    def __init__(self):
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
