"""The playback controls of the playNano window."""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDoubleSpinBox, QPushButton, QSlider, QVBoxLayout, QWidget


class PlaybackControls(QWidget):
    """
    Play button, frame slider and playback rate box.

    Parameters
    ----------
    initial_fps : float, optional
        Starting playback rate, in frames per second (0.1 to 60). Default is
        5.0.

    Attributes
    ----------
    play_btn : PySide6.QtWidgets.QPushButton
        Play/pause button.
    slider : PySide6.QtWidgets.QSlider
        Frame slider.
    fps_box : PySide6.QtWidgets.QDoubleSpinBox
        Playback rate.
    """

    def __init__(self, initial_fps: float = 5.0):
        """Build the controls. The parameters are described in the class docstring."""
        super().__init__()
        layout = QVBoxLayout(self)
        self.play_btn = QPushButton("▶️ Play")
        self.slider = QSlider(Qt.Horizontal)
        self.fps_box = QDoubleSpinBox()
        self.fps_box.setRange(0.1, 60)
        self.fps_box.setValue(initial_fps)
        layout.addWidget(self.play_btn)
        layout.addWidget(self.slider)
        layout.addWidget(self.fps_box)
