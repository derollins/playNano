"""
The ProcessingPipeline class, which runs processing steps on an AFMImageStack.

A pipeline runs masks, 2D filters, video processing and stack edits in order.
It stores each step's output in ``stack.processed`` (or ``stack.masks`` for
masks) and records what was done in ``stack.provenance["processing"]``, and the
software environment in ``stack.provenance["environment"]``.
"""

from __future__ import annotations

import importlib.metadata
import inspect
import logging
from collections import defaultdict
from typing import Any, Optional, Tuple

import numpy as np

from playnano.afm_stack import AFMImageStack
from playnano.utils.system_info import gather_environment_info
from playnano.utils.time_utils import utc_now_iso

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
    Run a sequence of processing steps on an AFMImageStack and record provenance.

    Steps are added with :meth:`add_filter`, :meth:`add_mask` and
    :meth:`clear_mask`, and run in order by :meth:`run`. Each step is a name and
    keyword arguments; the name decides what kind of step it is:

    - ``"clear"``: removes the current mask.
    - A mask (see :mod:`~playnano.processing.mask_generators`): makes a boolean
      mask, stored in ``stack.masks``. Masks made one after another are combined
      with logical OR until the next ``clear``.
    - A 2D filter (see :mod:`~playnano.processing.filters`): applied to every
      frame. While a mask is active, the masked version from
      :mod:`~playnano.processing.masked_filters` is used if there is one.
    - A video processing step (see
      :mod:`~playnano.processing.video_processing`): applied to the whole stack.
    - A stack edit (see :mod:`~playnano.processing.stack_edit`): removes frames
      and the matching frame metadata. It also clears the current mask.
    - A method of :class:`~playnano.afm_stack.AFMImageStack`, or a filter or
      video processing plugin registered under the ``playnano.filters`` or
      ``playnano.video_processing`` entry point group.

    Parameters
    ----------
    stack : AFMImageStack
        The stack to process. It is changed in place by :meth:`run`.

    See Also
    --------
    playnano.afm_stack.AFMImageStack.apply : Run a list of steps in one call.

    Examples
    --------
    >>> from playnano.afm_stack import AFMImageStack
    >>> from playnano.processing.pipeline import ProcessingPipeline
    >>> stack = AFMImageStack.load_data("tests/resources/sample_0.h5-jpk")
    >>> pipeline = ProcessingPipeline(stack)
    >>> (pipeline.add_filter("remove_plane")
    ...          .add_mask("mask_threshold", threshold=1.0)
    ...          .add_filter("polynomial_flatten", order=2)
    ...          .clear_mask())  # doctest: +ELLIPSIS
    <playnano.processing.pipeline.ProcessingPipeline object at ...>
    >>> result = pipeline.run()
    >>> sorted(stack.processed)
    ['raw', 'step_1_remove_plane', 'step_3_polynomial_flatten']
    >>> sorted(stack.masks)
    ['step_2_mask_threshold']
    """

    def __init__(self, stack: AFMImageStack) -> None:
        """Store the stack to process; see the class description."""
        self.stack = stack
        self.steps: list[tuple[str, dict[str, Any]]] = []

    def add_mask(self, mask_name: str, **kwargs) -> ProcessingPipeline:
        """
        Add a mask step.

        Parameters
        ----------
        mask_name : str
            Name of the mask, for example ``"mask_threshold"``.
        **kwargs
            Arguments for the mask function, for example ``threshold=1.0``.

        Returns
        -------
        ProcessingPipeline
            This pipeline, so calls can be chained.

        Notes
        -----
        If a mask is already active, the new mask is combined with it by logical
        OR. The step is identified by its name when the pipeline runs, so this does
        the same as :meth:`add_filter`; the two names only make pipelines easier to
        read.
        """
        self.steps.append((mask_name, kwargs))
        return self

    def add_filter(self, filter_name: str, **kwargs) -> ProcessingPipeline:
        """
        Add a processing step: a 2D filter, video processing step, stack edit or plugin.

        Parameters
        ----------
        filter_name : str
            Name of the step, for example ``"remove_plane"`` or ``"align_frames"``.
        **kwargs
            Arguments for the step, for example ``order=2``.

        Returns
        -------
        ProcessingPipeline
            This pipeline, so calls can be chained.

        Notes
        -----
        If a mask is active when a 2D filter runs, the masked version of the filter
        is used if there is one; otherwise the filter is applied to the whole frame.
        The step is identified by its name when the pipeline runs, so this does the
        same as :meth:`add_mask`.
        """
        self.steps.append((filter_name, kwargs))
        return self

    def clear_mask(self) -> ProcessingPipeline:
        """
        Add a step that removes the current mask.

        Filters after this step are applied to whole frames until a new mask is
        added.

        Returns
        -------
        ProcessingPipeline
            This pipeline, so calls can be chained.
        """
        self.steps.append(("clear", {}))
        return self

    def run(self) -> np.ndarray:
        """
        Run the steps in order on the stack.

        Before the first step, the current data is saved as
        ``stack.processed["raw"]`` if there is no such entry yet. After the last step
        the result replaces ``stack.data``.

        Each step's output is stored under a key ``"step_<n>_<name>"``, where ``n``
        counts from 1:

        - 2D filters and video processing steps: in ``stack.processed``.
        - Stack edits: in ``stack.processed``, always named ``"step_<n>_drop_frames"``.
        - Masks: in ``stack.masks``. A mask combined with an earlier one is named
          after both, for example ``"step_3_mask_threshold_mask_below_threshold"``.

        The provenance of each run replaces that of any earlier run:

        - ``stack.provenance["processing"]["steps"]``: one record per step, with its
          index, name, parameters, timestamp, step type, version, function module,
          the key of its output (``processed_key`` or ``mask_key``) and a summary of
          the output's shape and dtype. Video processing steps also store their
          metadata.
        - ``stack.provenance["processing"]["keys_by_name"]``: step name to the keys
          of its outputs, in order.
        - ``stack.provenance["environment"]``: Python, platform and package
          versions.

        Returns
        -------
        numpy.ndarray
            The processed stack, also stored in ``stack.data``.

        Raises
        ------
        ValueError
            If a step name is not recognised.
        Exception
            Any error raised by a step is logged and raised again; steps already
            run keep their outputs.
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
            return self._handle_clear_step(step_record, arr)
        elif step_type == "mask":
            return self._handle_mask_step(
                step_idx, step_name, fn, arr, mask, step_record
            )
        elif step_type in {"filter", "method", "plugin"}:
            return self._handle_filter_step(
                step_idx, step_name, fn, arr, mask, step_record, kwargs
            )
        elif step_type in {"video_filter", "video_plugin"}:
            return self._handle_video_filter_step(
                step_idx, step_name, fn, arr, mask, step_record, kwargs
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
        requested (mask, filter, method, plugin, video filter, video plugin,
        stack edit), and returns the step type along with a callable implementing it.

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
            "video_plugin", "video_filter", "stack_edit"
            - callable implementing the step, or None if step_type == "clear"

        Raises
        ------
        ValueError
            If the step name is not recognized among masks, filters, methods,
            plugins, video filters, video plugins, or stack edits.
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
        if version is None and step_type in {"plugin", "video_plugin"}:
            version = _get_plugin_version(fn)
        return version

    # -------------------------------------------------------------------------
    # Step handlers (all with detailed docstrings)
    # -------------------------------------------------------------------------

    def _handle_clear_step(
        self, step_record: dict[str, Any], arr: np.ndarray
    ) -> Tuple[np.ndarray, None]:
        """
        Handle a 'clear' step in the processing pipeline safely.

        This step clears the current mask but does not modify the working array.
        Returns the array currently being processed (arr), not stack.data.

        Parameters
        ----------
        step_record : dict
            Provenance dictionary for this step.
        arr : np.ndarray
            Current working array in the pipeline.

        Returns
        -------
        tuple
            (arr, None)
            - arr : np.ndarray
                The array currently being processed (unchanged).
            - None :
                Indicates that the mask has been cleared.

        Side Effects
        ------------
        - Appends the step record to `stack.provenance["processing"]["steps"]`.
        """
        step_record["mask_cleared"] = True
        self._record_step(step_record)
        return arr, None

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
            Index of the step within the pipeline.
        step_name : str
            Name of the filter function being executed.
        fn : callable
            The filter function to apply.
        arr : np.ndarray
            Input 3D image stack or 2D frame array.
        mask : np.ndarray or None
            Any masks present in the pipeline, not used but passed.
        step_record : dict
            Provenance record to update.
        kwargs : dict
            Additional keyword arguments for the filter function.

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
        mask: Optional[np.ndarray],
        step_record: dict[str, Any],
        kwargs: dict[str, Any],
    ) -> Tuple[np.ndarray, Optional[dict]]:
        """
        Execute 'video_filter' or 'video_plugin' steps.

        This function runs filters that apply to and return a 3D numpy array and
        creates a new array for all frames.

        Parameters
        ----------
        step_idx : int
            Index of the step within the pipeline.
        step_name : str
            Name of the filter function being executed.
        fn : callable
            The video filter function (built in or plugin) to apply.
        arr : np.ndarray
            Input 3D image stack or 2D frame array.
        mask : np.ndarray or None
            Any masks present in the pipeline, not used but passed.
        step_record : dict
            Provenance record to update.
        kwargs : dict
            Additional keyword arguments for the filter function.

        Returns
        -------
        new_arr : np.ndarray
            The filtered array.
        mask : np.ndarray or None
            Mask if present

        Notes
        -----
        Although masks are not currently supported for video filters any masks present
        are passed to align with the other handling functions.
        Video filter output metadata is not returned but is saved in provenance as
        a side effect.
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
        return new_arr, mask

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
