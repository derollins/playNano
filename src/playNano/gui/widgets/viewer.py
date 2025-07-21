"""GUI widget for viewing AFM video data."""

import logging
from typing import Optional

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPixmap, QResizeEvent
from PySide6.QtWidgets import QWidget

log = logging.getLogger(__name__)


class ViewerWidget(QWidget):
    """Displays AFM height maps as a resizable image with optional overlays."""

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
        self._draw_raw_label = False
        self.custom_font = QFont("Arial", 14)  # fallback font

    def display_frame(self, arr: np.ndarray):
        """
        Load and show a new RGB frame.

        Parameters
        ----------
        arr : np.ndarray
            3D uint8 array of shape (H, W, 3) containing the RGB image data.

        Returns
        -------
        None
        """
        log.debug("[ViewerWidget] display_frame called.")
        h, w, _ = arr.shape
        img = QImage(arr.data, w, h, 3 * w, QImage.Format_RGB888)
        self._original_pixmap = QPixmap.fromImage(img)
        log.debug(
            f"[ViewerWidget] QPixmap created: {self._original_pixmap is not None}"
        )
        self._rescale()

    def _rescale(self):
        """
        Rescale the current pixmap to fit the widget's size.

        This updates `self._scaled_pixmap` and triggers a repaint.

        Returns
        -------
        None
        """
        if self._original_pixmap:
            self._scaled_pixmap = self._original_pixmap.scaled(
                self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            self.update()
        else:
            self._scaled_pixmap = None

    def set_background_color(self, rgb: tuple[int, int, int]):
        """
        Change the widget’s background fill color.

        Parameters
        ----------
        rgb : tuple of int
            (R, G, B) values in the range [0…255].

        Returns
        -------
        None
        """
        self._bg_rgb = rgb
        self.update()

    def paintEvent(self, event):
        """
        Create custom paint handler: draws background, frame, and overlays.

        Parameters
        ----------
        event : QPaintEvent
            The Qt paint event.

        Notes
        -----
        - Fills the widget rectangle with the background color.
        - Centers and draws the scaled pixmap, if any.
        - Optionally overlays timestamp, “RAW” label, and scale bar.

        Returns
        -------
        None
        """
        painter = QPainter(self)
        try:
            log.debug("[ViewerWidget] paintEvent triggered.")
            painter.fillRect(self.rect(), QColor(*self._bg_rgb))

            if self._scaled_pixmap:
                x = (self.width() - self._scaled_pixmap.width()) // 2
                y = (self.height() - self._scaled_pixmap.height()) // 2
                painter.drawPixmap(x, y, self._scaled_pixmap)

            painter.setRenderHint(QPainter.Antialiasing)
            painter.setPen(Qt.white)

            font = QFont(self.custom_font)
            font.setPointSize(18)
            painter.setFont(font)

            if self._draw_timestamp and self._timestamp is not None:
                painter.drawText(10, 30, f"{self._timestamp:.2f} s")

            if self._draw_raw_label:
                text = "RAW"
                font_metrics = painter.fontMetrics()
                text_width = font_metrics.horizontalAdvance(text)
                x = self.width() - text_width - 10
                y = 30
                painter.drawText(x, y, text)

            if self._original_pixmap and self._draw_scale_bar:
                try:
                    bar_px = (
                        self._scale_bar_nm / self._pixel_size_nm
                    )  # may raise ZeroDivisionError
                except ZeroDivisionError:
                    log.warning(
                        "[ViewerWidget] Division by zero in scale bar calculation."
                    )
                else:
                    if self._scaled_pixmap:
                        scaled_width = self._scaled_pixmap.width()
                        scale = scaled_width / self._original_pixmap.width()
                        bar_width = int(bar_px * scale)
                    else:
                        bar_width = int(bar_px)
                    bar_height = 5
                    x = 10
                    y = self.height() - 20
                    painter.fillRect(x, y, bar_width, bar_height, Qt.white)
                    painter.drawText(x, y - 5, f"{self._scale_bar_nm} nm")
            else:
                log.warning(
                    "[ViewerWidget] Skipped scale bar: pixel_size_nm is zero or None"
                )

        except Exception as e:
            log.exception(f"[ViewerWidget] paintEvent crashed: {e}")
        finally:
            painter.end()

    def set_annotations(
        self,
        timestamp: Optional[float],
        draw_ts: bool,
        draw_scale: bool,
        draw_raw_label: bool,
        pixel_size_nm: Optional[float],
        scale_bar_nm: Optional[int],
    ):
        """
        Update which overlays to draw on each frame.

        Parameters
        ----------
        timestamp : float or None
            The time (in seconds) to display, or None.
        draw_ts : bool
            Whether to show the timestamp.
        draw_scale : bool
            Whether to show a scale bar.
        draw_raw_label : bool
            Whether to draw a “RAW” label.
        pixel_size_nm : float or None
            Nanometers per pixel (needed to size the scale bar).
        scale_bar_nm : int or None
            Physical length (nm) of the scale bar to draw.

        Returns
        -------
        None
        """
        log.debug(
            f"[ViewerWidget] set_annotations: ts={timestamp}, scale={scale_bar_nm}, px={pixel_size_nm}, "  # noqa: E501
            f"draw_ts={draw_ts}, draw_scale={draw_scale}"
        )
        self._timestamp = timestamp
        self._draw_timestamp = draw_ts
        self._draw_scale_bar = draw_scale
        self._pixel_size_nm = pixel_size_nm
        self._scale_bar_nm = scale_bar_nm
        self._draw_raw_label = draw_raw_label
        self.update()

    def resizeEvent(self, event: QResizeEvent):
        """
        Handle widget resize: recompute scaled pixmap size.

        Parameters
        ----------
        event : QResizeEvent
            The Qt resize event.

        Returns
        -------
        None
        """
        super().resizeEvent(event)
        self._rescale()

    def set_annotation_font(self, font: QFont):
        """
        Override the font used for drawing text overlays.

        Parameters
        ----------
        font : QFont
            A Qt font object.

        Returns
        -------
        None
        """
        self.custom_font = font
