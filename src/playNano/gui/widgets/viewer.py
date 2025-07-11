"""GUI widget for viewing AFM video data."""

import logging
from typing import Optional

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QImage, QPainter, QPixmap, QResizeEvent, QFont
from PySide6.QtWidgets import QWidget

log = logging.getLogger(__name__)


class ViewerWidget(QWidget):
    """Displays a single frame as a resizable QPixmap, with background color."""

    def __init__(self):
        """Initialize the view widget."""
        super().__init__()
        self._original_pixmap: Optional[QPixmap] = None
        self._scaled_pixmap: Optional[QPixmap] = None
        self._bg_rgb = (0, 0, 0)
        self._timestamp: Optional[float] = None
        self._pixel_size_nm: Optional[float] = None
        self._scale_bar_nm: Optional[int] = None
        self._draw_timestamp = False
        self._draw_scale_bar = False
        self.custom_font = QFont("Arial", 14)  # fallback font

    def display_frame(self, arr: np.ndarray):
        """Display a frame with arr: HxWx3 uint8 RGB."""
        log.debug("[ViewerWidget] display_frame called.")
        h, w, _ = arr.shape
        img = QImage(arr.data, w, h, 3 * w, QImage.Format_RGB888)
        self._original_pixmap = QPixmap.fromImage(img)
        log.debug(
            f"[ViewerWidget] QPixmap created: {self._original_pixmap is not None}"
        )
        self._rescale()

    def _rescale(self):
        """Create a scaled pixmap matching current widget size."""
        if self._original_pixmap:
            self._scaled_pixmap = self._original_pixmap.scaled(
                self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            self.update()
        else:
            self._scaled_pixmap = None

    def set_background_color(self, rgb: tuple[int, int, int]):
        """Set the background color using an RGB tuple."""
        self._bg_rgb = rgb
        self.update()

    def paintEvent(self, event):
        """Set custom paint event: background, image, and overlays."""
        try:
            log.debug("[ViewerWidget] paintEvent triggered.")
            painter = QPainter(self)
            painter.fillRect(self.rect(), QColor(*self._bg_rgb))

            if self._scaled_pixmap:
                x = (self.width() - self._scaled_pixmap.width()) // 2
                y = (self.height() - self._scaled_pixmap.height()) // 2
                painter.drawPixmap(x, y, self._scaled_pixmap)

            painter.setRenderHint(QPainter.Antialiasing)
            painter.setPen(Qt.white)

            # Set font with desired size and family
            font = QFont(self.custom_font)
            font.setPointSize(18)
            painter.setFont(font)

            # Timestamp
            if self._draw_timestamp and self._timestamp is not None:
                painter.drawText(10, 30, f"{self._timestamp:.2f} s")

            # Scale bar
            if self._original_pixmap and self._draw_scale_bar:
                pix_width = self._original_pixmap.width()
                log.debug(
                    f"[ViewerWidget] Drawing scale bar: pix_width={pix_width}, widget_width={self.width()}"  # noqa: E501
                )
                if self._pixel_size_nm and self._scale_bar_nm:
                    try:
                        bar_px = self._scale_bar_nm / self._pixel_size_nm
                        if self._scaled_pixmap:
                            scaled_width = self._scaled_pixmap.width()
                            scale = scaled_width / self._original_pixmap.width()
                            bar_width = int(bar_px * scale)
                        bar_height = 5
                        x = 10
                        y = self.height() - 20
                        painter.fillRect(x, y, bar_width, bar_height, Qt.white)
                        painter.drawText(x, y - 5, f"{self._scale_bar_nm} nm")
                    except ZeroDivisionError:
                        log.warning(
                            "[ViewerWidget] Division by zero in scale bar calculation."
                        )
        except Exception as e:
            log.exception(f"[ViewerWidget] paintEvent crashed: {e}")

    def set_annotations(
        self,
        timestamp: Optional[float],
        draw_ts: bool,
        draw_scale: bool,
        pixel_size_nm: Optional[float],
        scale_bar_nm: Optional[int],
    ):
        """Set the annotation settings (timestamp + scale bar)."""
        log.debug(
            f"[ViewerWidget] set_annotations: ts={timestamp}, scale={scale_bar_nm}, px={pixel_size_nm}, "  # noqa: E501
            f"draw_ts={draw_ts}, draw_scale={draw_scale}"
        )
        self._timestamp = timestamp
        self._draw_timestamp = draw_ts
        self._draw_scale_bar = draw_scale
        self._pixel_size_nm = pixel_size_nm
        self._scale_bar_nm = scale_bar_nm
        self.update()

    def resizeEvent(self, event: QResizeEvent):
        """Handle resize events to rescale the pixmap."""
        super().resizeEvent(event)
        self._rescale()

    def set_annotation_font(self, font: QFont):
        """Set the font used for annotations like timestamp and scale bar."""
        self.custom_font = font
