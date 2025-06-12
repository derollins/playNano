"""Module containing the ProcessingPipeline class for AFMImageStack processing."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from playNano.afm_stack import AFMImageStack

logger = logging.getLogger(__name__)


class ProcessingPipeline:
    """
    A simple orchestrator for running a sequence of masks/filters on an AFMImageStack.

    Steps are (step_name, kwargs) pairs.
    Available step types: 'clear', mask, filter, plugin, method.
    """

    def __init__(self, stack: AFMImageStack) -> None:
        """
        Initialize the processing pipeline with an AFMImageStack instance.

        Parameters
        ----------
        stack : AFMImageStack
            The AFMImageStack instance to process.
        """
        self.stack = stack
        self.steps: list[tuple[str, dict[str, Any]]] = []

    def add_mask(self, mask_name: str, **kwargs) -> ProcessingPipeline:
        """
        Add a mask step by name. Mask will be recomputed on current data.

        If a mask was added previously and not cleared this mask is added to
        the origonal.
        """
        self.steps.append((mask_name, kwargs))
        return self

    def add_filter(self, filter_name: str, **kwargs) -> ProcessingPipeline:
        """
        Add a filter step by name.

        If a mask is currently active (not cleared),
        will look for a masked version in MASK_FILTERS_MAP;
        otherwise, applies unmasked filter.
        """
        self.steps.append((filter_name, kwargs))
        return self

    def clear_mask(self) -> ProcessingPipeline:
        """Add a 'clear' step, which resets any active mask (unmasked mode)."""
        self.steps.append(("clear", {}))
        return self

    def run(self) -> np.ndarray:
        """
        Execute all configured steps on the underlying AFMImageStack.

        Returns the final processed array. Snapshots (including 'raw')
        are stored in stack.processed.
        """
        # 1) Snapshot raw data if not yet done
        if "raw" not in self.stack.processed:
            self.stack.processed["raw"] = self.stack.data.copy()

        arr = self.stack.data
        mask = None  # current boolean mask (3D array) or None
        step_idx = 0
        for step_name, kwargs in self.steps:
            step_idx += 1
            logger.info(
                f"[processing] Applying step {step_idx}: '{step_name}' with args {kwargs}"  # noqa
            )
            try:
                step_type, fn = self.stack._resolve_step(step_name)
            except Exception as e:
                logger.error(f"Failed to resolve step {step_idx}: {step_name}': {e}")
                raise

            if step_type == "clear":
                # Reset mask
                mask = None
                continue

            if step_type == "mask":
                if mask is None:
                    # Compute a mask over all frames
                    new_mask = self.stack._execute_mask_step(fn, arr, **kwargs)
                    # Save mask in oject and update
                    self.stack.masks[f"step_{step_idx}_{step_name}"] = new_mask.copy()
                    mask = new_mask
                    continue
                elif isinstance(mask, np.ndarray):
                    # Compute a new mask over all frames
                    new_mask = self.stack._execute_mask_step(fn, arr, **kwargs)
                    new_mask = np.logical_or(mask, new_mask)
                    try:
                        last_mask = list(self.stack.masks)[-1]
                        last_mask = "_".join(last_mask.split("_")[2:])
                    except IndexError:
                        logger.warning(
                            "No previous mask found when overlaying. Using fallback name 'overlay'."  # noqa
                        )
                        last_mask = "overlay"
                        raise ValueError("Previous mask not accessible.") from None
                    self.stack.masks[f"step_{step_idx}_{last_mask}_{step_name}"] = (
                        new_mask.copy()
                    )
                    mask = new_mask
                    continue
                continue

            # Otherwise, step is a filter/method/plugin
            try:
                new_arr = self.stack._execute_filter_step(
                    fn, arr, mask, step_name, **kwargs
                )
            except Exception as e:
                logger.error(f"Failed to apply filter/plugin/method '{step_name}': {e}")    # noqa
                raise
            # Snapshot
            self.stack.processed[f"step_{step_idx}_{step_name}"] = new_arr.copy()
            arr = new_arr

        # Overwrite stack.data
        self.stack.data = arr
        logger.info("Processing pipeline completed successfully.")
        return arr
