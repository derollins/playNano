from topostats.filters import Filters
import numpy as np

import logging

logger = logging.getLogger(__name__)


def flatten_afm_frame(
    frame: np.ndarray,
    filename: str = "frame",
    pixel_to_nm: float = 1.0,
    filter_config: dict = None,
) -> np.ndarray:
    if filter_config is None:
        filter_config = {
            "row_alignment_quantile": 0.05,
            "threshold_method": "std_dev",
            "otsu_threshold_multiplier": 1.0,
            "threshold_std_dev": {"above": 0.5, "below": 10},
            "threshold_absolute": None,
            "gaussian_size": 2,
            "gaussian_mode": "nearest",
            "remove_scars": {"run": False},
        }

    try:
        logger.info(
            f"Flattening frame '{filename}': shape={frame.shape}, dtype={frame.dtype}"
        )
        logger.info(
            f"Frame stats before flattening: min={np.min(frame)}, max={np.max(frame)}, mean={np.mean(frame)}"
        )

        if frame.size == 0 or np.isnan(frame).any():
            logger.warning(
                f"Input frame invalid for {filename}: empty or contains NaNs"
            )
            return None

        filters = Filters(
            image=frame,
            filename=filename,
            pixel_to_nm_scaling=pixel_to_nm,
            **filter_config,
        )
        filters.filter_image()  # run filtering pipeline

        gaussian_filtered = filters.images.get("gaussian_filtered", None)
        if gaussian_filtered is None:
            logger.warning(
                f"Warning: 'gaussian_filtered' image not found in filters.images for {filename}"
            )
            return None

        return gaussian_filtered

    except Exception as e:
        logger.warning(f"Exception flattening frame {filename}: {e}")
        return None


def flatten_stack(image_stack: np.ndarray, pixel_to_nm: float = 1.0) -> np.ndarray:
    return np.stack(
        [
            flatten_afm_frame(frame, filename=f"frame_{i}", pixel_to_nm=pixel_to_nm)
            for i, frame in enumerate(image_stack)
        ]
    )
