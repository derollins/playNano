"""Module containing the ProcessingPipeline class for AFMImageStack processing.

This module provides ProcessingPipeline, which runs a sequence of
mask/filter/method/plugin steps on an AFMImageStack. Each step's output is stored
in `stack.processed` (for filters) or `stack.masks` (for masks), and detailed
provenance (timestamps, parameters, step type, version info, keys) is recorded in
`stack.provenance["processing"]`. Environment metadata at pipeline start is recorded in
`stack.provenance["environment"]`.
"""

from __future__ import annotations

import importlib.metadata
import inspect
import logging
from typing import Any

import numpy as np

from playNano.afm_stack import AFMImageStack
from playNano.utils.system_info import gather_environment_info
from playNano.utils.time_utils import utc_now_iso

logger = logging.getLogger(__name__)


def _get_plugin_version(fn) -> str | None:
    """
    Attempt to determine the package version for a plugin function.

    This inspects the module in which `fn` is defined, extracts the top-level
    package name, and returns its version via importlib.metadata. If any step
    fails (e.g., module not found, package not installed), returns None.

    Parameters
    ----------
    fn : callable
        The function object for which to infer the package version. Typically
        a plugin filter or similar user-provided function.

    Returns
    -------
    str or None
        The version string of the package containing `fn`, or None if it cannot
        be determined.
    """
    try:
        module = inspect.getmodule(fn)
        if module and hasattr(module, "__name__"):
            pkg_name = module.__name__.split(".")[0]
            return importlib.metadata.version(pkg_name)
    except Exception:
        return None


class ProcessingPipeline:
    """
    Orchestrates a sequence of masking and filtering steps on an AFMImageStack.

    This pipeline records outputs and detailed provenance for each step. Each step is
    specified by a name and keyword arguments:

    - ``"clear"``: resets any active mask.
    - Mask steps: compute boolean masks stored in ``stack.masks[...]``.
    - Filter/method/plugin steps: apply to the current data (and mask if present),
      storing results in ``stack.processed[...]``.

    Provenance for each step, including index, name, parameters, timestamp, step type,
    version, keys, and summaries, is appended to
    ``stack.provenance["processing"]["steps"]``. Additionally, a mapping from step
    name to a list of snapshot keys is stored in
    ``stack.provenance["processing"]["keys_by_name"]``. The final processed array
    overwrites ``stack.data``, and environment metadata is captured once in
    ``stack.provenance["environment"]``.
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
        Execute configured steps on the AFMImageStack, storing outputs and provenance.

        The pipeline iterates through all added masks, filters, and plugins in order,
        applying each to the current data. Masks are combined if multiple are applied
        before a filter. Each step's output is stored in `stack.processed` (filters) or
        `stack.masks` (masks), and a detailed provenance record is saved in
        `stack.provenance["processing"]`.

        Behavior
        --------

        1. Record or update environment metadata via ``gather_environment_info()`` into
        ``stack.provenance["environment"]``.

        2. Reset previous processing provenance under
        ``stack.provenance["processing"]``, ensuring that keys ``"steps"`` (a list)
        and ``"keys_by_name"`` (a dictionary) exist and are cleared.

        3. If not already present, snapshot the original data as ``"raw"`` in
        ``stack.processed``.

        4. Iterate over ``self.steps`` in order (1-based index):

        - Resolve the step type via ``stack._resolve_step(step_name)``, which returns
          a tuple of the form (``step_type``, ``fn``).
        - Record a timestamp (from ``utc_now_iso()``), index, name, parameters,
          step type, function version (from ``fn.__version__`` or plugin lookup), and
          module name.

        - If ``step_type`` is ``"clear"``:
            - Reset the current mask to ``None``.
            - Record ``"mask_cleared": True`` in the provenance entry.

        - If ``step_type`` is ``"mask"``:
            - Call ``stack._execute_mask_step(fn, arr, **kwargs)`` to compute a boolean
              mask array.
            - If there is no existing mask, store it under a new key
              ``step_<idx>_<mask_name>`` in ``stack.masks``.
            - Otherwise, overlay it with the previous mask (logical OR) under a derived
              key.
            - Update the current mask and record ``"mask_key"`` and ``"mask_summary"``
              in provenance.

        - Else (filter/method/plugin):
            - Call ``stack._execute_filter_step(fn, arr, mask, step_name, **kwargs)``
              to obtain the new array.
            - Store the result under
              ``stack.processed["step_<idx>_<safe_name>"]`` and update ``arr``.
            - Record ``"processed_key"`` and ``"output_summary"`` in provenance.

        5. After all steps, overwrite ``stack.data`` with ``arr``.

        6. Build ``stack.provenance["processing"]["keys_by_name"]``, mapping each step
        name to the list of stored keys (``processed_key`` or ``mask_key``) in order.

        7. Return the final processed array.

        Returns
        -------
        np.ndarray
            The final processed data array (shape (n_frames, height, width)), now
            stored as `stack.data`.

        Raises
        ------
        RuntimeError
            If a step cannot be resolved or executed due to misconfiguration.

        ValueError
            If overlaying a mask fails due to missing previous mask key (propagated).

        Exception
            Any exception from individual steps is logged then re-raised.

        Examples
        --------
        >>> stack = AFMImageStack(data, pixel_size_nm=1.0, channel="h", file_path=".")
        >>> pipeline = ProcessingPipeline(stack)
        >>> pipeline.add_filter("gaussian_filter", sigma=1.0)
        >>> pipeline.add_mask("threshold_mask", threshold=0.5)
        >>> result = pipeline.run()
        >>> # Final array in stack.data:
        >>> result.shape
        (n_frames, height, width)
        >>> # Provenance entries:
        >>> for rec in stack.provenance["processing"]["steps"]:
        ...     print(
        ...         rec["index"],
        ...         rec["name"],
        ...         rec.get("processed_key") or rec.get("mask_key")
        ...     )
        >>> # Keys by name:
        >>> print(stack.provenance["processing"]["keys_by_name"])
        >>> # Environment info:
        >>> print(stack.provenance["environment"])
        """
        # Record environment info (overwrite or only if empty)
        env = gather_environment_info()
        self.stack.provenance["environment"] = env

        # Reset previous processing history
        proc_prov = self.stack.provenance["processing"]
        proc_prov["steps"].clear()
        proc_prov["keys_by_name"].clear()

        # Snapshot raw once
        if "raw" not in self.stack.processed:
            self.stack.processed["raw"] = self.stack.data.copy()

        arr = self.stack.data
        mask = None  # current mask or None
        step_idx = 0

        for step_name, kwargs in self.steps:
            step_idx += 1
            logger.info(
                f"[processing] Applying step {step_idx}: '{step_name}' with args {kwargs}"  # noqa
            )
            # Prepare record for this step
            timestamp = utc_now_iso()
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

            func_version = getattr(fn, "__version__", None)
            if func_version is None and step_type == "plugin":
                func_version = _get_plugin_version(fn)
            step_record["version"] = func_version
            step_record["function_module"] = getattr(fn, "__module__", None)

            if step_type == "clear":
                mask = None
                # No snapshot stored in stack.processed/masks
                # Record that mask was cleared
                step_record["mask_cleared"] = True
                self.stack.provenance["processing"]["steps"].append(step_record)
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
                self.stack.provenance["processing"]["steps"].append(step_record)
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
            safe_name = step_name.replace(" ", "_")
            proc_key = f"step_{step_idx}_{safe_name}"
            self.stack.processed[proc_key] = new_arr.copy()
            # Update arr for next steps
            arr = new_arr
            # Record processed_key in history
            step_record["processed_key"] = proc_key
            step_record["output_summary"] = {
                "shape": new_arr.shape,
                "dtype": str(new_arr.dtype),
            }
            self.stack.provenance["processing"]["steps"].append(step_record)

        # After all steps, overwrite stack.data
        self.stack.data = arr
        logger.info("Processing pipeline completed successfully.")
        # Optionally also attach a name→list-of-keys view
        from collections import defaultdict

        keys_by_name: dict[str, list[str]] = defaultdict(list)
        for rec in self.stack.provenance["processing"]["steps"]:
            name = rec["name"]
            if "processed_key" in rec:
                keys_by_name[name].append(rec["processed_key"])
            elif "mask_key" in rec:
                keys_by_name[name].append(rec["mask_key"])
        self.stack.provenance["processing"]["keys_by_name"] = dict(keys_by_name)
        return arr
