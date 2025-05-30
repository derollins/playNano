"""Defines AFMImageStack for managing AFM time-series data and processing steps."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from playNano.processing.image_processing import flatten_stack

logger = logging.getLogger(__name__)


class AFMImageStack:
    """Class for managing AFM image stacks and applying processing steps."""

    def __init__(
        self,
        data: np.ndarray,
        pixel_size_nm: float,
        channel: str,
        file_path: Path,
        frame_metadata: list[dict[str, Any]] = None,
    ) -> None:
        """
        Represent a stack of AFM images and associated metadata.

        Parameters
        ----------
        data : np.ndarray
            NumPy array of shape (n_frames, height, width) with image data.
        pixel_size_nm : float
            Physical pixel size in nanometers.
        channel : str
            Channel name.
        file_path : str | Path
            Path to source file or folder.
        frame_metadata : list of dict, optional
            Per-frame metadata; will be padded or trimmed to length n_frames
        """
        self.data = data
        self.pixel_size_nm = pixel_size_nm
        self.channel = channel
        self.file_path = file_path

        # Validate and pad/trim metadata to match number of frames
        n = self.data.shape[0]
        if frame_metadata is None:
            self.frame_metadata = [{} for _ in range(n)]
        else:
            if len(frame_metadata) < n:
                self.frame_metadata = frame_metadata + [{}] * (n - len(frame_metadata))
            elif len(frame_metadata) > n:
                raise ValueError(
                    f"Metadata length ({len(frame_metadata)}) does not match number of frames ({n})."  # noqa
                )
            else:
                self.frame_metadata = frame_metadata

        # Store processed results; 'raw' is populated on first processing
        self.processed: dict[str, np.ndarray] = {}

    @classmethod
    def from_file(
        cls, path: str | Path, channel: str = "height_trace"
    ) -> AFMImageStack:
        """Load AFM data (file or folder) into an AFMImageStack."""
        from playNano.io.loader import load_afm_stack

        afm = load_afm_stack(path, channel)
        return cls(
            data=afm.data,
            pixel_size_nm=afm.pixel_size_nm,
            channel=channel,
            file_path=path,
            frame_metadata=afm.frame_metadata,
        )

    @property
    def n_frames(self) -> int:
        """Number of frames in the stack."""
        return self.data.shape[0]

    @property
    def height(self) -> int:
        """Return frame height."""
        return self.data.shape[1]

    @property
    def width(self) -> int:
        """Return frame width."""
        return self.data.shape[2]

    @property
    def image_shape(self) -> tuple[int, int]:
        """Return (height, width)."""
        return self.data.shape[1:]

    def get_frame(self, index: int) -> np.ndarray:
        """Return the image data for frame at given index."""
        return self.data[index]

    def get_frame_metadata(self, index: int) -> dict[str, Any]:
        """Return the metadata dictionary for frame at given index."""
        if 0 <= index < len(self.frame_metadata):
            return self.frame_metadata[index]
        else:
            raise IndexError(f"Frame metadata index {index} out of range")

    def frames_with_metadata(self):
        """Yield (frame_index, image_frame, metadata_dict) tuples."""
        for idx, (image, meta) in enumerate(
            zip(self.data, self.frame_metadata, strict=False)
        ):
            if image is not None:
                yield idx, image, meta
            else:
                print(f"Warning: Frame {idx} is None and skipped")

    def _snapshot_raw(self):
        """
        Store the very first raw data under 'raw' in processed dict.

        Subsequent calls do nothing.
        """
        if "raw" not in self.processed:
            # copy the array so later modifications to self.data don’t touch 'raw'
            self.processed["raw"] = self.data.copy()

    def apply(self, name: str, func: callable[..., np.ndarray], **kwargs) -> np.ndarray:
        """
        Apply a processing function to the current data.

        Parameters
        ----------
        name : str
            Key under which to store the result in self.processed.
        func : callable
            Function taking (array, **kwargs) and returning an array.

        Returns
        -------
        np.ndarray
            The processed data (also stored and set as current data).
        """
        # snapshot raw on the first processing call
        self._snapshot_raw()

        # run the filter on the *current* data
        result = func(self.data, **kwargs)

        # store under its name
        self.processed[name] = result

        # update current data
        self.data = result

        logger.info(f"{name} processes completed.")
        return self.data

    def flatten_images(self):
        """Flatten the AFM image stack using TopoStats flattening filters."""
        logger.info("Flattening image stack...")
        return self.apply(
            name="flattened",
            func=flatten_stack,
            pixel_to_nm=self.pixel_size_nm,
        )

    def time_for_frame(self, idx: int) -> Any:
        """Return timestamp for frame idx (may be None)."""
        return self.frame_metadata[idx].get("timestamp")

    def channel_for_frame(self, idx: int) -> str:
        """Return channel name for frame idx."""
        return self.frame_metadata[idx].get("channel", self.channel)
