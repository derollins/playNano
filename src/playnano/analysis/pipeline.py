"""
Run a sequence of analysis modules on an AFM image stack.

:class:`AnalysisPipeline` runs analysis modules (subclasses of
:class:`~playnano.analysis.base.AnalysisModule`) in order and records their
outputs and provenance. It is what ``playnano analyze`` uses.
"""

import importlib.metadata as metadata
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

from playnano.afm_stack import AFMImageStack
from playnano.analysis import BUILTIN_ANALYSIS_MODULES
from playnano.analysis.base import AnalysisModule
from playnano.analysis.utils.common import sanitize_analysis_for_logging
from playnano.processing.mask_generators import register_masking
from playnano.utils.system_info import gather_environment_info
from playnano.utils.time_utils import utc_now_iso

MASKING_FUNCS = register_masking()

logger = logging.getLogger(__name__)

AnalysisRecord = dict[str, Any]
"""
Type of the record :meth:`AnalysisPipeline.run` returns: a dict with the keys
``"environment"``, ``"analysis"`` and ``"provenance"``, described there.
"""


class AnalysisPipeline:
    """
    Run a list of analysis modules on a stack, in order.

    Add steps with :meth:`add`, then call :meth:`run`. Each step names an
    analysis module, either a built-in one (see
    :data:`playnano.analysis.BUILTIN_ANALYSIS_MODULES`) or a plugin registered
    under the ``playnano.analysis`` entry point group. Each step's outputs are
    stored in ``stack.analysis["step_<n>_<module_name>"]``, and its parameters,
    time and module version in ``stack.provenance["analysis"]``.

    Attributes
    ----------
    steps : list of tuple
        The steps added, as ``(module_name, params)`` pairs.

    Examples
    --------
    Detect a particle that moves across three frames, then link it into a
    track:

    >>> import numpy as np
    >>> from playnano.afm_stack import AFMImageStack
    >>> from playnano.analysis.pipeline import AnalysisPipeline
    >>> data = np.zeros((3, 64, 64))
    >>> for t in range(3):
    ...     data[t, 10 + t : 16 + t, 10:16] = 3.0
    >>> stack = AFMImageStack(
    ...     data,
    ...     pixel_size_nm=2.0,
    ...     channel="height_trace",
    ...     file_path="demo.h5-jpk",
    ...     frame_metadata=[{"timestamp": 0.5 * t} for t in range(3)],
    ... )
    >>> pipeline = AnalysisPipeline()
    >>> pipeline.add("feature_detection", mask_fn="mask_threshold", threshold=1.0)
    >>> pipeline.add("particle_tracking", max_distance=5)
    >>> record = pipeline.run(stack)
    >>> list(record["analysis"])
    ['step_1_feature_detection', 'step_2_particle_tracking']
    >>> record["analysis"]["step_1_feature_detection"]["summary"]
    {'total_frames': 3, 'total_features': 3, 'avg_features_per_frame': 1.0}
    >>> record["analysis"]["step_2_particle_tracking"]["n_tracks"]
    1
    """

    def __init__(self):
        """Create an empty pipeline."""
        # Each entry: (module_name: str, params:   dict)
        self.steps: list[tuple[str, dict[str, Any]]] = []
        # Cache instantiated modules: name -> instance
        self._module_cache: dict[str, AnalysisModule] = {}

    def add(self, module_name: str, **params) -> None:
        """
        Add a step that runs an analysis module.

        Parameters
        ----------
        module_name : str
            Name of the analysis module, for example ``"feature_detection"``. It is
            not checked until :meth:`run`.
        **params
            Keyword arguments passed to the module's ``run`` method. If
            ``mask_fn`` is a str, it is replaced by the masking function of that
            name from :mod:`playnano.processing.mask_generators` when the step runs.

        Examples
        --------
        >>> pipeline = AnalysisPipeline()
        >>> pipeline.add("feature_detection", mask_fn="mask_threshold", threshold=1.0)
        >>> pipeline.add("particle_tracking", max_distance=5)
        >>> pipeline.steps[1]
        ('particle_tracking', {'max_distance': 5})
        """
        self.steps.append((module_name, params))

    def clear(self) -> None:
        """Remove all steps, so the pipeline can be set up again."""
        self.steps.clear()
        self._module_cache.clear()

    def _load_module(self, module_name: str) -> AnalysisModule:
        """
        Load and instantiate an analysis module given its name.

        Modules are first looked up in a built-in registry, then via entry points
        registered under the group 'playnano.analysis'. Loaded modules are cached
        to avoid re-instantiation on repeated `run()` calls.

        Parameters
        ----------
        module_name : str
            The name of the analysis module to load.

        Returns
        -------
        AnalysisModule
            The loaded and initialized module instance.

        Raises
        ------
        ValueError
            If the module name cannot be resolved from the registry or entry points.
        TypeError
            If the loaded module is not an instance of `AnalysisModule`.
        """
        # Fast path: cached
        if module_name in self._module_cache:
            return self._module_cache[module_name]

        # 1) Internal registry
        cls = None
        try:
            cls = BUILTIN_ANALYSIS_MODULES[module_name]
        except Exception:
            cls = None

        # 2) Entry points (plugin mechanism)
        if cls is None:
            # Modern API available on Python 3.10–3.13
            eps = metadata.entry_points().select(
                group="playnano.analysis", name=module_name
            )
            ep = next(iter(eps), None)
            if ep is None:
                raise ValueError(
                    f"Analysis module '{module_name}' not found in registry or entry points"  # noqa
                )

            try:
                cls = ep.load()
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load entry point for analysis module '{module_name}': {e}"  # noqa
                ) from e

        # 3) Instantiate and type-check

        # First: check class/type without instantiating
        if not callable(cls):
            raise TypeError(
                f"Loaded analysis module '{module_name}' is not instantiable "
                f"(got {cls!r})"
            )

        # Create the instance
        try:
            instance = cls()
        except Exception as e:
            raise RuntimeError(
                f"Failed to instantiate analysis module '{module_name}' ({cls}): {e}"
            ) from e

        # Now validate that instance is correct type
        if not isinstance(instance, AnalysisModule):
            raise TypeError(
                f"Loaded module, '{module_name}', is not an AnalysisModule subclass;"
                f"got {type(instance)!r}"
            )

        # Cache and return
        self._module_cache[module_name] = instance
        return instance

    def _resolve_mask_fn(self, params: dict[str, Any]) -> dict[str, Any]:
        """
        Resolve a string-based 'mask_fn' to its registered callable, if applicable.

        If `params["mask_fn"]` is a string matching a registered masking function,
        replaces it with the corresponding callable. Returns a shallow copy of `params`
        with the resolved function.

        Parameters
        ----------
        params : dict of str to Any
            Dictionary of parameters passed to an analysis module.
            Must include 'mask_fn'.

        Returns
        -------
        dict
            A shallow copy of `params` with 'mask_fn' resolved to a callable.

        Raises
        ------
        ValueError
            If the provided string does not correspond to a registered masking function.
        """
        mask_key = params["mask_fn"]
        if mask_key in MASKING_FUNCS:
            params = params.copy()
            params["mask_fn"] = MASKING_FUNCS[mask_key]
            return params
        else:
            raise ValueError(
                f"mask_fn '{mask_key}' not found in registered masking functions"
            )

    def run(self, stack: AFMImageStack, log_to: Optional[str] = None) -> AnalysisRecord:
        """
        Run every step on a stack and return the results.

        Each step's module receives the stack, the step's parameters and
        ``previous_results``: a dict mapping the name of each module run so far to
        its latest outputs. Results from an earlier call of ``run`` are discarded:
        ``stack.analysis`` and ``stack.provenance["analysis"]`` are replaced.

        Parameters
        ----------
        stack : AFMImageStack
            The stack to analyse.
        log_to : str, optional
            If given, also write the record to this JSON file, creating missing
            folders. Arrays are summarised in the file (shape, dtype, minimum,
            maximum and mean), not written in full; see
            :func:`~playnano.analysis.utils.common.sanitize_analysis_for_logging`.
            Default is None.

        Returns
        -------
        dict
            The analysis record, with these keys:

            ``"environment"``
                Python, platform and package versions; the same dict as
                ``stack.provenance["environment"]``, which is only set if empty.
            ``"analysis"``
                The outputs of each step, keyed ``"step_<n>_<module_name>"``, with
                ``n`` counting from 1 and spaces in the name replaced by ``_``. The
                same dict as ``stack.analysis``.
            ``"provenance"``
                The same dict as ``stack.provenance["analysis"]``, with:

                - ``"steps"``: one dict per step, with ``index``, ``name``,
                  ``params`` (as passed to the module, so a ``mask_fn`` name appears
                  as the function), ``timestamp`` (ISO 8601, UTC), ``version`` (the
                  module's ``version``, or None) and ``analysis_key``.
                - ``"results_by_name"``: each module name mapped to a list with one
                  ``{"analysis_key": ..., "outputs": ...}`` dict per time it ran.
                - ``"frame_times"``: the stack's frame times, in seconds.

        Raises
        ------
        ValueError
            If a module name is not found, or ``mask_fn`` names no masking
            function.
        RuntimeError
            If a module declares ``requires`` and none of those modules ran
            before it, or a plugin module fails to load.
        TypeError
            If a plugin is not an :class:`~playnano.analysis.base.AnalysisModule`.
        Exception
            Any error raised by a module is logged and raised again.
        """
        # Ensure stack.provenance is a dict
        if not hasattr(stack, "provenance") or not isinstance(stack.provenance, dict):
            stack.provenance = {}
        # Ensure 'analysis' sub-dict exists
        if "analysis" not in stack.provenance or not isinstance(
            stack.provenance["analysis"], dict
        ):
            stack.provenance["analysis"] = {
                "steps": [],
                "results_by_name": {},
                "frame_times": None,
            }
        # Start each run with empty results, so steps from an earlier, longer run
        # do not linger. A new dict leaves records returned earlier untouched.
        stack.analysis = {}

        env = gather_environment_info()
        # If provenance already has environment from processing, you may choose to
        # merge or keep first;
        # For simplicity, you can overwrite or check if empty.
        if not stack.provenance.get("environment"):
            stack.provenance["environment"] = env

        # Clear analysis provenance record
        stack.provenance["analysis"]["steps"].clear()
        stack.provenance["analysis"]["results_by_name"].clear()
        stack.provenance["analysis"]["frame_times"] = None

        step_results: list[dict[str, Any]] = []
        results_by_name: defaultdict[str, list] = defaultdict(list)
        previous_latest: dict[str, dict[str, Any]] = {}
        # module cache unchanged
        for idx, (module_name, params) in enumerate(self.steps, start=1):
            logger.info(
                f"Running analysis step {idx}: {module_name} with params {params!r}"
            )
            module = self._load_module(module_name)
            # check any declared requirements
            reqs = getattr(module, "requires", ())
            if reqs:
                if not any(r in previous_latest for r in reqs):
                    raise RuntimeError(
                        f"Analysis step '{module_name}' requires one of {reqs!r}; "
                        f"make sure to add at least one before '{module_name}'."
                    )
            # timestamp
            timestamp = utc_now_iso()

            if "mask_fn" in params and isinstance(params["mask_fn"], str):
                params = self._resolve_mask_fn(params)

            # run; pass in previous_latest so module can read latest outputs by name
            try:
                outputs = module.run(stack, previous_results=previous_latest, **params)
            except Exception as e:
                logger.error(
                    f"Module '{module_name}' failed at step {idx}: {e}", exc_info=True
                )
                raise
            # Store snapshot under unique key
            safe_name = module_name.replace(" ", "_")
            analysis_key = f"step_{idx}_{safe_name}"
            stack.analysis[analysis_key] = outputs
            # record this step
            step_record: dict[str, Any] = {
                "index": idx,
                "name": module_name,
                "params": params,
                "timestamp": timestamp,
                "version": getattr(module, "version", None),
                "analysis_key": analysis_key,
            }
            step_results.append(step_record)
            # update previous_results structures
            results_by_name[module_name].append(
                {"analysis_key": analysis_key, "outputs": outputs}
            )
            # allow downstream modules to use latest result
            previous_latest[module_name] = outputs

        # Build the updated provenance record
        stack.provenance["analysis"]["steps"] = step_results
        stack.provenance["analysis"]["results_by_name"] = dict(results_by_name)

        frame_times = (
            stack.get_frame_times() if hasattr(stack, "get_frame_times") else None
        )
        stack.provenance["analysis"]["frame_times"] = frame_times

        record = {
            "environment": stack.provenance["environment"],
            "analysis": stack.analysis,
            "provenance": stack.provenance["analysis"],
        }
        # write to file if requested
        # Swapped from using NumpyEncoder because it could not handle the size
        # of the full analysis record.
        if log_to:
            import json

            safe_record = {
                "environment": sanitize_analysis_for_logging(record["environment"]),
                "analysis": sanitize_analysis_for_logging(record["analysis"]),
                "provenance": sanitize_analysis_for_logging(record["provenance"]),
            }
            Path(log_to).parent.mkdir(parents=True, exist_ok=True)
            with open(log_to, "w") as file:
                json.dump(safe_record, file, indent=2)
        # Add record
        return record
