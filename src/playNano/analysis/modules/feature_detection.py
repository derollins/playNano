from typing import Any, Optional

import numpy as np
from scipy.ndimage import binary_fill_holes
from skimage.measure import label, regionprops
from skimage.morphology import remove_small_holes

from playNano.analysis.base import AnalysisModule


class FeatureDetectionModule(AnalysisModule):
    @property
    def name(self) -> str:
        return "feature_detection"

    def run(
        self,
        stack,
        previous_results: Optional[dict[str, Any]] = None,
        *,
        # Mask input: either supply a mask function or refer to
        # existing mask in previous_results
        mask_fn: Optional[callable] = None,
        mask_key: Optional[str] = None,
        # Filtering criteria:
        min_size: int = 10,
        remove_edge: bool = True,
        # Hole-filling options:
        fill_holes: bool = False,
        hole_area: Optional[int] = None,
        # kwargs for mask_fn(frame, **mask_kwargs)
        **mask_kwargs,
    ) -> dict[str, Any]:
        """
        Detect contiguous features on each frame of stack.data.

        Parameters
        ----------
        stack : AFMImageStack
            The AFM stack whose `.data` (3D array) and `.time_for_frame()` are used.
        previous_results : dict[str, Any], optional
            Mapping of earlier analysis outputs. If `mask_key` is given,
            must contain a boolean mask array under that key.
        mask_fn : callable, optional
            Function frame->bool array for masking.
            Required if `mask_key` is None.
        mask_key : str, optional
            Key in `previous_results` whose value is a boolean
            mask array of same shape as `stack.data`.
        min_size : int, default 10
            Minimum area (in pixels) to keep a region.
        remove_edge : bool, default True
            If True, discard regions touching any image boundary.
        fill_holes : bool, default False
            Whether to fill holes before labeling.
        hole_area : int or None, default None
            If set, only fill holes smaller than this area.
        **mask_kwargs
            Passed to `mask_fn(frame, **mask_kwargs)`.

        Returns
        -------
        dict[str, Any]
            {
                "features_per_frame": List[List[dict[str, Any]]],
                "labeled_masks": List[np.ndarray],  # labeled mask per frame
                "summary": {
                    "total_frames": int,
                    "total_features": int,
                    "avg_features_per_frame": float,
                }
            }

        Raises
        ------
        ValueError
            If `stack.data` is None or not 3D, or mask array invalid,
            or neither `mask_fn` nor `mask_key` provided.
        KeyError
            If `mask_key` not found in `previous_results`.
        """
        data = stack.data
        if data is None:
            raise ValueError("AFMImageStack has no data")
        if not isinstance(data, np.ndarray) or data.ndim != 3:
            raise ValueError("stack.data must be a 3D numpy array (n_frames, H, W)")
        n_frames, H, W = data.shape

        # 1. Obtain or validate mask array
        if mask_key is not None:
            if not previous_results or mask_key not in previous_results:
                raise KeyError(f"mask_key '{mask_key}' not found in previous_results")
            mask_arr = previous_results[mask_key]
            if not (
                isinstance(mask_arr, np.ndarray)
                and mask_arr.dtype == bool
                and mask_arr.shape == data.shape
            ):
                raise ValueError(
                    f"previous_results[{mask_key}] must be a boolean ndarray of shape {data.shape}"  # noqa
                )
        else:
            if mask_fn is None:
                raise ValueError("Either mask_fn or mask_key must be provided")
            # compute mask frame-by-frame
            mask_arr = np.zeros_like(data, dtype=bool)
            for i in range(n_frames):
                frame = data[i]
                try:
                    mf = mask_fn(frame, **mask_kwargs)
                except TypeError:
                    mf = mask_fn(frame)
                # validate
                if not (
                    isinstance(mf, np.ndarray)
                    and mf.dtype == bool
                    and mf.shape == (H, W)
                ):
                    raise ValueError(f"mask_fn returned invalid mask for frame {i}")
                mask_arr[i] = mf

        features_per_frame: list[list[dict[str, Any]]] = []
        total_features = 0
        labeled_masks = []

        for i in range(n_frames):
            frame = data[i]
            ts = stack.time_for_frame(i)
            try:
                frame_ts = float(ts)
            except Exception:
                frame_ts = float(i)
            mask_frame = mask_arr[i].copy()

            # 2. Optionally fill holes
            if fill_holes:
                # mask_frame is boolean
                if hole_area is not None:
                    # fill holes smaller than hole_area
                    # remove_small_holes fills holes with area < area_threshold
                    # note: remove_small_holes expects a 2D array
                    mask_frame = remove_small_holes(
                        mask_frame, area_threshold=hole_area
                    )
                else:
                    # fill all holes regardless of size
                    mask_frame = binary_fill_holes(mask_frame)
                # ensure boolean dtype
                mask_frame = mask_frame.astype(bool)

            # 3. Label connected regions
            initial_labeled = label(mask_frame)
            filtered_mask = np.zeros_like(mask_frame, dtype=bool)

            for prop in regionprops(initial_labeled):
                area = prop.area
                minr, minc, maxr, maxc = prop.bbox
                if area < min_size:
                    continue
                if remove_edge and (minr == 0 or minc == 0 or maxr == H or maxc == W):
                    continue
                filtered_mask[initial_labeled == prop.label] = True

            # Now label the filtered mask
            labeled = label(filtered_mask)
            labeled_masks.append(labeled)
            props = regionprops(labeled, intensity_image=frame)

            kept_stats: list[dict[str, Any]] = []
            for prop in props:
                area = prop.area
                minr, minc, maxr, maxc = prop.bbox  # maxr/maxc are exclusive
                # Compute stats on underlying image
                mask_pixels = labeled == prop.label
                vals = frame[mask_pixels]
                if vals.size == 0:
                    continue
                stats = {
                    "frame_timestamp": frame_ts,
                    "label": int(prop.label),
                    "area": int(area),
                    "min": float(vals.min()),
                    "max": float(vals.max()),
                    "mean": float(vals.mean()),
                    "bbox": (int(minr), int(minc), int(maxr), int(maxc)),
                    "centroid": tuple(map(float, prop.centroid)),
                }
                kept_stats.append(stats)

            features_per_frame.append(kept_stats)
            total_features += len(kept_stats)

        summary = {
            "total_frames": n_frames,
            "total_features": total_features,
            "avg_features_per_frame": total_features / n_frames if n_frames > 0 else 0,
        }

        return {
            "features_per_frame": features_per_frame,
            "labeled_masks": labeled_masks,
            "summary": summary,
        }
