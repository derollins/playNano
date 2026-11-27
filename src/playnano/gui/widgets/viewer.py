"""The frame view of the playNano window."""

import logging
from typing import Optional

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPixmap, QResizeEvent
from PySide6.QtWidgets import QWidget

log = logging.getLogger(__name__)


class ViewerWidget(QWidget):
    """
    Shows one RGB frame, scaled to fit, with optional time, scale bar and "RAW" label.

    The frame keeps its aspect ratio and is enlarged or shrunk with Qt's smooth
    (bilinear) scaling, for display only. Space around it is filled with a
    background colour.
    """

    def __init__(self):
        """Create an empty view."""
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
        Show a new frame.

        Parameters
        ----------
        arr : numpy.ndarray
            RGB image, shape (height, width, 3), dtype uint8.
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
        Set the colour that fills the space around the frame.

        Parameters
        ----------
        rgb : tuple of int
            ``(R, G, B)``, each 0-255.
        """
        self._bg_rgb = rgb
        self.update()

    def paintEvent(self, event):
        """
        Draw the background, the frame and the overlays (Qt calls this).

        The time is drawn at the top left, "RAW" at the top right and the scale
        bar, sized from the pixel size and the display scale, at the bottom left.

        Parameters
        ----------
        event : PySide6.QtGui.QPaintEvent
            The paint event.
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
                px = self._pixel_size_nm
                # Checked before dividing, so a missing, zero, negative or NaN
                # pixel size skips the bar ("not px > 0" is True for NaN).
                if px is None or not px > 0 or not self._scale_bar_nm:
                    # Called on every repaint, so log at debug level only.
                    log.debug(
                        "[ViewerWidget] No scale bar: pixel size or bar length "
                        "is missing, zero or not a positive number."
                    )
                else:
                    bar_px = self._scale_bar_nm / self._pixel_size_nm
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
        Set the overlays drawn on the frame.

        Parameters
        ----------
        timestamp : float or None
            Time to show, in seconds; None shows none.
        draw_ts : bool
            Show the time.
        draw_scale : bool
            Show the scale bar.
        draw_raw_label : bool
            Show "RAW" at the top right.
        pixel_size_nm : float or None
            Pixel size of the frame, in nm, used to size the scale bar.
        scale_bar_nm : int or None
            Scale bar length, in nm.
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
        Scale the frame to the new size (Qt calls this).

        Parameters
        ----------
        event : PySide6.QtGui.QResizeEvent
            The resize event.
        """
        super().resizeEvent(event)
        self._rescale()

    def set_annotation_font(self, font: QFont):
        """
        Set the font of the overlay text.

        The font's family is used at 18 pt.

        Parameters
        ----------
        font : PySide6.QtGui.QFont
            The font.
        """
        self.custom_font = font
