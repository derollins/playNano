"""Initializes the AFMImageStack class."""

import logging
from typing import Any, Dict, List

import numpy as np

from playNano.processing.image_processing import flatten_stack

logger = logging.getLogger(__name__)


class AFMImageStack:
    """Class for managing AFM image stacks."""

    def __init__(
        self,
        image_stack: np.ndarray,
        pixel_size_nm: float,
        img_shape: tuple[int, int],
        line_rate: float,
        channel: str,
        file_path: str,
        frame_metadata: List[Dict[str, Any]] = None,
    ):
        """
        Represent a stack of AFM images and associated metadata.

        Parameters
        ----------
        image_stack : np.ndarray
            NumPy array of shape (num_frames, height, width) with image data.
        pixel_size_nm : float
            Physical pixel size in nanometers.
        img_shape : tuple[int, int]
            Image shape as (height, width).
        line_rate : float
            Scan line rate in lines per second.
        channel : str
            Channel name.
        file_path : str
            Path to source file.
        frame_metadata : list of dict, optional
            List of metadata dicts, one per frame, indexed in sync with image_stack.
        """
        self.image_stack = image_stack
        self.pixel_size_nm = pixel_size_nm
        self.img_shape = img_shape
        self.line_rate = line_rate
        self.channel = channel
        self.file_path = file_path
        self.frame_metadata = frame_metadata if frame_metadata is not None else []
        self.raw_image_stack = None

    def get_frame(self, index: int) -> np.ndarray:
        """Return the image data for frame at given index."""
        return self.image_stack[index]

    def get_frame_metadata(self, index: int) -> Dict[str, Any]:
        """Return the metadata dictionary for frame at given index."""
        if 0 <= index < len(self.frame_metadata):
            return self.frame_metadata[index]
        else:
            raise IndexError(f"Frame metadata index {index} out of range")

    def frames_with_metadata(self):
        """Yield (frame_index, image_frame, metadata_dict) tuples."""
        for idx, (image, meta) in enumerate(
            zip(self.image_stack, self.frame_metadata, strict=False)
        ):
            if image is not None:
                yield idx, image, meta
            else:
                print(f"Warning: Frame {idx} is None and skipped")

    def flatten_images(self, keep_raw=True):
        """
        Flatten the AFM image stack using TopoStats flattening filters.

        Parameters:
        - keep_raw (bool): If True, keep a copy of the original
        raw images in self.raw_image_stack
        """
        if keep_raw and self.raw_image_stack is None:
            self.raw_image_stack = self.image_stack.copy()

        logger.info("Flattening image stack...")
        self.image_stack = flatten_stack(
            self.image_stack, pixel_to_nm=self.pixel_size_nm
        )
        logger.info("Flattening complete.")

    @property
    def num_frames(self) -> int:
        """Return number of frames in the stack."""
        return self.image_stack.shape[0]
