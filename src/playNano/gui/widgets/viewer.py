"""GUI widget for view AFM video data"""

from PySide6.QtWidgets import QLabel
from PySide6.QtGui import QPixmap, QImage, QResizeEvent, QPainter, QColor
from PySide6.QtCore import Qt
import numpy as np


class ViewerWidget(QLabel):
    """Displays a single frame as a resizable QPixmap, with background color."""

    def __init__(self):
        super().__init__()
        self.setScaledContents(False)
        self.setAlignment(Qt.AlignCenter)
        self._original_pixmap: QPixmap | None = None
        self._bg_rgb = (0, 0, 0)  # Default background color

    def display_frame(self, arr: np.ndarray):
        """Display a frame with arr: HxWx3 uint8 RGB."""
        h, w, _ = arr.shape
        img = QImage(arr.data, w, h, 3*w, QImage.Format_RGB888)
        self._original_pixmap = QPixmap.fromImage(img)
        self._rescale()


    def _rescale(self):
        if self._original_pixmap:
            scaled = self._original_pixmap.scaled(
                self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            self.setPixmap(scaled)

    def set_background_color(self, rgb: tuple[int, int, int]):
        """Set the background color using an RGB tuple."""
        self._bg_rgb = rgb
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        color = QColor(*self._bg_rgb)
        painter.fillRect(self.rect(), color)

        if self.pixmap():
            super().paintEvent(event)

    def resizeEvent(self, event: QResizeEvent):
        super().resizeEvent(event)
        self._rescale()