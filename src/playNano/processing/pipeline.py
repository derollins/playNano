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
from collections import defaultdict
from typing import Any, Optional, Tuple

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
            The final processed data array, now also stored in `stack.data`.

        Raises
        ------
        RuntimeError
            If a step cannot be resolved or executed due to misconfiguration.

        ValueError
            If overlaying a mask fails due to missing previous mask key (propagated).

        Exception
            Any exception raised by a step function is logged and re-raised.

        Notes
        -----
        - The method ensures a raw copy of the original stack exists under
          `stack.processed["raw"]`.
        - Mask steps may be overlaid with previous masks using logical OR.
        - Non-drop_frames stack_edit steps automatically delegate to drop_frames
          to maintain provenance consistency.
        """
        self._prepare_environment_and_provenance()
        arr = self._snapshot_raw_data()
        mask = None

        for step_idx, (step_name, kwargs) in enumerate(self.steps, start=1):
            arr, mask = self._run_single_step(step_idx, step_name, kwargs, arr, mask)

        self.stack.data = arr
        self._finalize_provenance()
        logger.info("Processing pipeline completed successfully.")
        return arr

    # -------------------------------------------------------------------------
    # Core setup and teardown helpers
    # -------------------------------------------------------------------------

    def _prepare_environment_and_provenance(self) -> None:
        """
        Record environment metadata and reset processing provenance.

        This method overwrites the previous environment information and clears
        all processing history under `stack.provenance["processing"]`.

        Side Effects
        ------------
        - Modifies `stack.provenance["environment"]`.
        - Clears `stack.provenance["processing"]["steps"]`.
        - Clears `stack.provenance["processing"]["keys_by_name"]`.
        """
        env = gather_environment_info()
        self.stack.provenance["environment"] = env
        proc_prov = self.stack.provenance["processing"]
        proc_prov["steps"].clear()
        proc_prov["keys_by_name"].clear()

    def _snapshot_raw_data(self) -> np.ndarray:
        """
        Ensure a copy of the original stack data exists.

        If 'raw' is not present in `stack.processed`, this method stores a
        copy of the current stack data for provenance purposes.

        Returns
        -------
        np.ndarray
            A reference to the current stack data array for processing.

        Side Effects
        ------------
        - Modifies `stack.processed["raw"]` if it does not already exist.
        """
        if "raw" not in self.stack.processed:
            self.stack.processed["raw"] = self.stack.data.copy()
        return self.stack.data

    def _finalize_provenance(self) -> None:
        """
        Populate `keys_by_name` mapping for processed and mask keys.

        This method iterates over all recorded steps in
        `stack.provenance["processing"]["steps"]` and builds a dictionary
        mapping each step name to a list of processed or mask keys in order
        of execution.

        Side Effects
        ------------
        - Modifies `stack.provenance["processing"]["keys_by_name"]`.
        """
        keys_by_name: dict[str, list[str]] = defaultdict(list)
        for rec in self.stack.provenance["processing"]["steps"]:
            if key := rec.get("processed_key") or rec.get("mask_key"):
                keys_by_name[rec["name"]].append(key)
        self.stack.provenance["processing"]["keys_by_name"] = dict(keys_by_name)

    # -------------------------------------------------------------------------
    # Step dispatch and resolution
    # -------------------------------------------------------------------------

    def _run_single_step(
        self,
        step_idx: int,
        step_name: str,
        kwargs: dict[str, Any],
        arr: np.ndarray,
        mask: Optional[np.ndarray],
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Execute a single pipeline step based on its resolved type.

        Parameters
        ----------
        step_idx : int
            1-based index of the current step.
        step_name : str
            Name of the step to execute.
        kwargs : dict
            Keyword arguments to pass to the step function.
        arr : np.ndarray
            Current working array.
        mask : np.ndarray or None
            Current mask array or None.

        Returns
        -------
        tuple
            Tuple of (updated array, updated mask). Mask is None for non-mask
            steps.

        Notes
        -----
        - Resolves the step type using `stack._resolve_step`.
        - Logs execution start and provenance information.
        - Delegates execution to specialized `_handle_*` functions based on type.
        """
        logger.info(f"[processing] Step {step_idx}: '{step_name}' with args {kwargs}")
        step_record = self._init_step_record(step_idx, step_name, kwargs)

        step_type, fn = self._resolve_step_with_logging(step_idx, step_name)
        step_record["step_type"] = step_type
        step_record["version"] = self._get_step_version(fn, step_type)
        step_record["function_module"] = getattr(fn, "__module__", None)

        if step_type == "clear":
            return self._handle_clear_step(step_record)
        elif step_type == "mask":
            return self._handle_mask_step(
                step_idx, step_name, fn, arr, mask, step_record
            )
        elif step_type in {"filter", "method", "plugin"}:
            return self._handle_filter_step(
                step_idx, step_name, fn, arr, mask, step_record, kwargs
            )
        elif step_type == "video_filter":
            return self._handle_video_filter_step(
                step_idx, step_name, fn, arr, step_record, kwargs
            )
        elif step_type == "stack_edit":
            return self._handle_stack_edit_step(
                step_idx, step_name, fn, arr, step_record, kwargs
            )
        else:
            logger.warning(f"Unrecognized step_type '{step_type}' for {step_name}")
            return arr, mask

    def _init_step_record(
        self, step_idx: int, step_name: str, kwargs: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Initialize a provenance dictionary for a processing step.

        Parameters
        ----------
        step_idx : int
            The 1-based step index.
        step_name : str
            Name of the step.
        kwargs : dict
            Parameters passed to the step.

        Returns
        -------
        dict
            A dictionary with fields 'index', 'name', 'params', and 'timestamp'.
        """
        return {
            "index": step_idx,
            "name": step_name,
            "params": kwargs,
            "timestamp": utc_now_iso(),
        }

    def _resolve_step_with_logging(self, step_idx: int, step_name: str):
        """
        Resolve a processing step to its type and callable, with error logging.

        This method attempts to determine what kind of processing step is
        requested (mask, filter, method, plugin, video filter, stack edit),
        and returns the step type along with a callable implementing it.

        If the step cannot be resolved, the original exception is logged and
        re-raised, preserving its type.

        Parameters
        ----------
        step_idx : int
            The 1-based index of the step within the pipeline.
        step_name : str
            Name of the step to resolve.

        Returns
        -------
        tuple[str, callable | None]
            - step type: one of "clear", "mask", "method", "plugin", "filter",
            "video_filter", "stack_edit"
            - callable implementing the step, or None if step_type == "clear"

        Raises
        ------
        ValueError
            If the step name is not recognized among masks, filters, methods,
            plugins, video filters, or stack edits.
        """
        try:
            return self.stack._resolve_step(step_name)
        except Exception as e:
            # Log the failure for debugging
            logger.error(f"Failed to resolve step {step_idx}: {step_name}: {e}")
            # Re-raise the original exception so that its type is preserved
            raise

    def _get_step_version(self, fn, step_type: str) -> Optional[str]:
        """
        Get the version of a step function or plugin.

        Parameters
        ----------
        fn : callable
            Step function.
        step_type : str
            Step type ('plugin' triggers special lookup).

        Returns
        -------
        str or None
            Version string if available.
        """
        version = getattr(fn, "__version__", None)
        if version is None and step_type == "plugin":
            version = _get_plugin_version(fn)
        return version

    # -------------------------------------------------------------------------
    # Step handlers (all with detailed docstrings)
    # -------------------------------------------------------------------------

    def _handle_clear_step(
        self, step_record: dict[str, Any]
    ) -> Tuple[np.ndarray, None]:
        """
        Handle a 'clear' step by resetting the current mask.

        Parameters
        ----------
        step_record : dict
            Provenance dictionary for this step.

        Returns
        -------
        tuple
            (current array, None) since mask is cleared.

        Side Effects
        ------------
        - Appends the step record to `stack.provenance["processing"]["steps"]`.
        """
        step_record["mask_cleared"] = True
        self._record_step(step_record)
        return self.stack.data, None

    def _handle_mask_step(
        self,
        step_idx: int,
        step_name: str,
        fn,
        arr: np.ndarray,
        mask: Optional[np.ndarray],
        step_record: dict[str, Any],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute a new mask or overlay with previous masks.

        Parameters
        ----------
        step_idx : int
            Index of the step.
        step_name : str
            Step name.
        fn : callable
            Mask-generating function.
        arr : np.ndarray
            Current array.
        mask : np.ndarray or None
            Current mask.
        step_record : dict
            Step provenance dictionary.

        Returns
        -------
        tuple
            (arr, new_mask)

        Side Effects
        ------------
        - Updates `stack.masks` with new mask.
        - Updates step record with 'mask_key' and 'mask_summary'.
        """
        new_mask = self.stack._execute_mask_step(fn, arr, **step_record["params"])
        if mask is None:
            key = f"step_{step_idx}_{step_name}"
            self.stack.masks[key] = new_mask.copy()
        else:
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

        step_record["mask_key"] = key
        step_record["mask_summary"] = {
            "shape": new_mask.shape,
            "dtype": str(new_mask.dtype),
        }
        self._record_step(step_record)
        return arr, new_mask

    def _handle_filter_step(
        self,
        step_idx: int,
        step_name: str,
        fn,
        arr: np.ndarray,
        mask: Optional[np.ndarray],
        step_record: dict[str, Any],
        kwargs: dict[str, Any],
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Execute a 'filter', 'method', or 'plugin' step.

        Parameters
        ----------
        step_idx : int
        step_name : str
        fn : callable
        arr : np.ndarray
        mask : np.ndarray or None
        step_record : dict
        kwargs : dict

        Returns
        -------
        tuple
            (updated array, mask unchanged)
        """
        try:
            new_arr = self.stack._execute_filter_step(
                fn, arr, mask, step_name, **kwargs
            )
        except Exception as e:
            logger.error(f"Failed to apply filter '{step_name}': {e}")
            raise
        safe_name = step_name.replace(" ", "_")
        proc_key = f"step_{step_idx}_{safe_name}"
        self.stack.processed[proc_key] = new_arr.copy()
        step_record["processed_key"] = proc_key
        step_record["output_summary"] = {
            "shape": new_arr.shape,
            "dtype": str(new_arr.dtype),
        }
        self._record_step(step_record)
        return new_arr, mask

    def _handle_video_filter_step(
        self,
        step_idx: int,
        step_name: str,
        fn,
        arr: np.ndarray,
        step_record: dict[str, Any],
        kwargs: dict[str, Any],
    ) -> Tuple[np.ndarray, Optional[dict]]:
        """
        Execute a 'video_filter' step, producing a new array for all frames.

        Parameters
        ----------
        step_idx : int
            Index of the step within the pipeline.
        step_name : str
            Name of the filter function being executed.
        fn : callable
            The filter function to apply.
        arr : np.ndarray
            Input 3D image stack or 2D frame array.
        step_record : dict
            Provenance record to update.
        kwargs : dict
            Additional keyword arguments for the filter function.

        Returns
        -------
        new_arr : np.ndarray
            The filtered array.
        metadata : dict or None
            Metadata returned by the filter function, if available.
        """
        try:
            result = self.stack._execute_video_processing_step(fn, arr, **kwargs)

            if isinstance(result, tuple) and len(result) == 2:
                new_arr, metadata = result
            else:
                new_arr, metadata = result, {}

        except Exception as e:
            logger.error(f"Video filter '{step_name}' failed: {e}")
            raise

        proc_key = f"step_{step_idx}_{step_name}"
        self.stack.processed[proc_key] = new_arr.copy()

        step_record["processed_key"] = proc_key
        step_record["metadata"] = metadata
        step_record["output_summary"] = {
            "shape": new_arr.shape,
            "dtype": str(new_arr.dtype),
        }

        self._record_step(step_record)
        return new_arr, metadata

    def _handle_stack_edit_step(
        self,
        step_idx: int,
        step_name: str,
        fn,
        arr: np.ndarray,
        step_record: dict[str, Any],
        kwargs: dict[str, Any],
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Execute a 'stack_edit' step.

        If the step is not 'drop_frames', delegates to 'drop_frames' to maintain
        consistent processing provenance.

        Parameters
        ----------
        step_idx : int
        step_name : str
        fn : callable
        arr : np.ndarray
        step_record : dict
        kwargs : dict

        Returns
        -------
        tuple
            (arr after edit, None)

        Raises
        ------
        TypeError
            If a non-drop_frames stack_edit does not return a list or array of indices.
        """
        try:
            if step_name == "drop_frames":
                new_arr = self.stack._execute_stack_edit_step(fn, arr, **kwargs)
                delegated_to = None
            else:
                indices_to_drop = fn(self.stack.data, **kwargs)
                if not isinstance(indices_to_drop, (list, np.ndarray)):
                    raise TypeError(
                        f"Stack edit '{step_name}' must return list or array of indices, "  # noqa
                        f"got {type(indices_to_drop).__name__}"
                    )
                delegated_to = "drop_frames"
                drop_fn = self.stack._resolve_step(delegated_to)[1]
                new_arr = self.stack._execute_stack_edit_step(
                    drop_fn, arr, indices_to_drop=indices_to_drop
                )

        except Exception as e:
            logger.error(f"Stack edit '{step_name}' failed: {e}")
            raise

        proc_key = f"step_{step_idx}_drop_frames"
        self.stack.processed[proc_key] = new_arr.copy()
        step_record["processed_key"] = proc_key
        step_record["output_summary"] = {
            "stack_edit_function_used": step_name,
            "delegated_to": delegated_to,
            "shape": new_arr.shape,
            "dtype": str(new_arr.dtype),
        }
        self._record_step(step_record)
        return new_arr, None

    # -------------------------------------------------------------------------
    # Provenance recording
    # -------------------------------------------------------------------------

    def _record_step(self, step_record: dict[str, Any]) -> None:
        """
        Append a step record to the processing provenance.

        Parameters
        ----------
        step_record : dict
            Metadata dictionary describing the executed step.

        Side Effects
        ------------
        - Updates `stack.provenance["processing"]["steps"]`.
        """
        self.stack.provenance["processing"]["steps"].append(step_record)
