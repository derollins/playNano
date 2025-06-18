"""Module containing the ProcessingPipeline class for AFMImageStack processing."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
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
        Add a masking step to the pipeline.

        Parameters
        ----------
        mask_name : str
            The name of the registered mask function to apply.
        **kwargs
            Additional parameters passed to the mask function.

        Returns
        -------
        ProcessingPipeline
            The pipeline instance (for method chaining).

        Notes
        -----
        If a mask is currently active (i.e. not cleared), this new mask will be
        logically combined (ORed) with the existing one.
        """
        self.steps.append((mask_name, kwargs))
        return self

    def add_filter(self, filter_name: str, **kwargs) -> ProcessingPipeline:
        """
        Add a filter step to the pipeline.

        Parameters
        ----------
        filter_name : str
            The name of the registered filter function to apply.
        **kwargs
            Additional keyword arguments for the filter function.

        Returns
        -------
        ProcessingPipeline
            The pipeline instance (for method chaining).

        Notes
        -----
        If a mask is currently active, the pipeline will attempt to use a
        masked version of the filter (from `MASK_FILTERS_MAP`) if available.
        Otherwise, the unmasked filter is applied to the whole dataset.
        """
        self.steps.append((filter_name, kwargs))
        return self

    def clear_mask(self) -> ProcessingPipeline:
        """
        Add a step to clear the current mask.

        Returns
        -------
        ProcessingPipeline
            The pipeline instance (for method chaining).

        Notes
        -----
        Calling this resets the masking state, so subsequent filters will be
        applied to the entire dataset unless a new mask is added.
        """
        self.steps.append(("clear", {}))
        return self

    def run(self) -> np.ndarray:
        """
        Execute all configured steps on the underlying AFMImageStack.

        This method runs the processing pipeline step-by-step, applying filters,
        masks, or other operations in order. During execution:

        - A snapshot of the original data is saved under the key "raw"
        in `stack.processed` (if not already present).
        - Each processing step stores its output in `stack.processed` (for filters) or
        `stack.masks` (for masks), using a unique key with the step index.
        - A detailed, ordered history of all executed steps is recorded in
        `stack.processing_history`, where each entry contains metadata about the step
        (e.g., name, parameters, timestamp, output reference key).

        Returns
        -------
        np.ndarray
            The final processed array (last output of the pipeline). Also sets
            `stack.data = result`.
            Raises
        ------
        RuntimeError
            If a step cannot be resolved or executed due to misconfiguration.
        ValueError
            If a mask overlay is attempted without a valid previous mask key.
        Exception
            Any other exceptions raised by individual processing steps will be
            propagated with full tracebacks.

        Examples
        --------
        >>> stack = AFMImageStack(data)  # your AFM 3D array
        >>> pipeline = ProcessingPipeline(stack)
        >>> pipeline.add_filter("gaussian_filter", sigma=1.0)
        >>> pipeline.add_mask("mask_threshold", threshold=0.5)
        >>> result = pipeline.run()
        >>> result.shape
        (n_frames, height, width)

        >>> stack.processing_history[0]["name"]
        'gaussian_filter'
        >>> stack.processed[stack.processing_history[0]["processed_key"]].shape
        (n_frames, height, width)

        Notes
        -----
        - All snapshots are stored with unique keys (e.g., 'step_1_gaussian_filter').
        - This function also attaches a dictionary `stack.processing_keys_by_name`
        mapping step names to lists of snapshot keys for easy access.
        """
        # Snapshot raw once
        if "raw" not in self.stack.processed:
            self.stack.processed["raw"] = self.stack.data.copy()

        arr = self.stack.data
        mask = None  # current mask or None
        processing_history: list[dict[str, Any]] = []
        step_idx = 0

        for step_name, kwargs in self.steps:
            step_idx += 1
            logger.info(
                f"[processing] Applying step {step_idx}: '{step_name}' with args {kwargs}"  # noqa
            )
            # Prepare record for this step
            timestamp = datetime.now(UTC).isoformat().replace("+00:00", "Z")
            step_record: dict[str, Any] = {
                "index": step_idx,
                "name": step_name,
                "params": kwargs,
                "timestamp": timestamp,
                # we'll fill 'step_type' and reference keys and any summary
            }
            # Resolve step type
            try:
                step_type, fn = self.stack._resolve_step(step_name)
            except Exception as e:
                logger.error(f"Failed to resolve step {step_idx}: {step_name}: {e}")
                raise
            step_record["step_type"] = step_type

            if step_type == "clear":
                mask = None
                # No snapshot stored in stack.processed/masks
                # Record that mask was cleared
                step_record["mask_cleared"] = True
                processing_history.append(step_record)
                continue

            if step_type == "mask":
                # Compute new mask
                new_mask = self.stack._execute_mask_step(fn, arr, **kwargs)
                if mask is None:
                    # First mask: store under a unique key as before
                    key = f"step_{step_idx}_{step_name}"
                    self.stack.masks[key] = new_mask.copy()
                else:
                    # Overlay: combine and store under derived key
                    combined = np.logical_or(mask, new_mask)
                    try:
                        last_mask_key = list(self.stack.masks)[-1]
                        last_mask_part = "_".join(last_mask_key.split("_")[2:])
                    except IndexError:
                        last_mask_part = "overlay"
                        logger.warning(
                            "No previous mask found when overlaying; using 'overlay'"
                        )
                    key = f"step_{step_idx}_{last_mask_part}_{step_name}"
                    self.stack.masks[key] = combined.copy()
                    new_mask = combined
                mask = new_mask
                # Record the mask snapshot key in history (no need to duplicate array)
                step_record["mask_key"] = key
                # Optionally record mask shape/dtype summary
                step_record["mask_summary"] = {
                    "shape": new_mask.shape,
                    "dtype": str(new_mask.dtype),
                }
                processing_history.append(step_record)
                continue

            # Else: filter/method/plugin
            try:
                new_arr = self.stack._execute_filter_step(
                    fn, arr, mask, step_name, **kwargs
                )
            except Exception as e:
                logger.error(f"Failed to apply filter '{step_name}': {e}")
                raise
            # Store snapshot under unique key
            proc_key = f"step_{step_idx}_{step_name}"
            self.stack.processed[proc_key] = new_arr.copy()
            # Update arr for next steps
            arr = new_arr
            # Record processed_key in history
            step_record["processed_key"] = proc_key
            step_record["output_summary"] = {
                "shape": new_arr.shape,
                "dtype": str(new_arr.dtype),
            }
            processing_history.append(step_record)

        # After all steps, overwrite stack.data
        self.stack.data = arr
        logger.info("Processing pipeline completed successfully.")
        # Attach the ordered history list to stack for later inspection
        self.stack.processing_history = processing_history
        # Optionally also attach a name→list-of-keys view
        from collections import defaultdict

        keys_by_name: dict[str, list[str]] = defaultdict(list)
        for rec in processing_history:
            nm = rec["name"]
            if "processed_key" in rec:
                keys_by_name[nm].append(rec["processed_key"])
            elif "mask_key" in rec:
                keys_by_name[nm].append(rec["mask_key"])
        self.stack.processing_keys_by_name = dict(keys_by_name)
        return arr
