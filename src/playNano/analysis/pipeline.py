"""Module for the AnalysisPipeline class for orchastration of analysis workflows."""

import importlib.metadata
import logging
from collections import defaultdict
from typing import Any, Optional

from playNano.afm_stack import AFMImageStack
from playNano.analysis import BUILTIN_ANALYSIS_MODULES
from playNano.analysis.base import AnalysisModule
from playNano.analysis.utils.common import sanitize_analysis_for_logging
from playNano.processing.mask_generators import register_masking
from playNano.utils.system_info import gather_environment_info
from playNano.utils.time_utils import utc_now_iso

MASKING_FUNCS = register_masking()

logger = logging.getLogger(__name__)

AnalysisRecord = dict[str, Any]
"""
Structured output of an AnalysisPipeline run.

This record contains:
- environment : dict
    Metadata about the runtime environment (e.g. Python version, library versions).
- analysis : dict
    Results of each analysis module run, with keys 'step_<n>_<module_name>'.
- provenance : dict
    Metadata about the provenance of the analysis steps, with keys:
      - steps : list of dict
          Ordered list of executed analysis steps. Each entry contains:
            - index : int
                1-based index of the step in the pipeline.
            - name : str
                The name of the analysis module used.
            - params : dict
                Parameters passed to the module.
            - timestamp : str
                ISO 8601 UTC timestamp when the step was executed.
            - version : str or None
                Optional version string provided by the module instance.
            - analysis_key : str
                Key under which this step's outputs are stored in the `analysis` dict.
      - results_by_name : dict[str, list]
          Maps module names to lists of outputs from each occurrence.
      - frame_times : list[float] or None
          Timestamps for each frame in the stack, from `stack.get_frame_times()`,
          or None if unavailable.

Examples
--------
>>> pipeline = AnalysisPipeline()
>>> pipeline.add("feature_detection", threshold=5)
>>> record = pipeline.run(stack, log_to="out.json")
>>> # Access outputs:
>>> record["analysis"]["step_1_feature_detection"]["summary"]
{'total_features': 23, 'avg_per_frame': 3.8}
>>> # Inspect provenance:
>>> record["provenance"]["results_by_name"]["feature_detection"][0]["summary"]
{'total_features': 23, 'avg_per_frame': 3.8}
"""


class AnalysisPipeline:
    """
    Orchestrates a sequence of analysis steps on an AFMImageStack.

    Each step corresponds to an AnalysisModule (built-in or entry-point), invoked
    in order with the given parameters. Outputs of each step are stored in
    `stack.analysis` under keys 'step_<n>_<module_name>'. Detailed provenance
    (timestamps, parameters, version, linking keys) is recorded in
    `stack.provenance["analysis"]`. The run() method returns a dict containing
    environment info, the `analysis` dict, and its `provenance`.
    """

    def __init__(self):
        """
        Initialize an empty analysis pipeline.

        Steps are stored as a list of (module_name, params) tuples.
        Modules are loaded on demand using an internal registry or entry points.
        """
        # Each entry: (module_name: str, params:   dict)
        self.steps: list[tuple[str, dict[str, Any]]] = []
        # Cache instantiated modules: name -> instance
        self._module_cache: dict[str, AnalysisModule] = {}

    def add(self, module_name: str, **params) -> None:
        """
        Add an analysis module to the pipeline.

        Parameters
        ----------
        module_name : str
            The name of the analysis module to add (must be registered).
        **params
            Keyword arguments passed to the module's `run()` method.

        Returns
        -------
        None

        Examples
        --------
        >>> pipeline.add("particle_detect", threshold=5, min_size=10)
        >>> pipeline.add("track_particles", max_jump=3)
        """
        self.steps.append((module_name, params))

    def clear(self) -> None:
        """
        Remove all scheduled analysis steps and clear module cache.

        This allows reconfiguration of the pipeline without creating a new instance.
        """
        self.steps.clear()
        self._module_cache.clear()

    def _load_module(self, module_name: str) -> AnalysisModule:
        """
        Load and instantiate an analysis module given its name.

        Modules are first looked up in a built-in registry, then via entry points
        registered under the group 'playNano.analysis'. Loaded modules are cached
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
        if module_name in self._module_cache:
            return self._module_cache[module_name]

        # 1) Internal registry
        cls = None
        try:
            cls = BUILTIN_ANALYSIS_MODULES[module_name]
        except Exception:
            cls = None

        if cls is None:
            # 2) Try entry points
            eps = importlib.metadata.entry_points().select(
                group="playNano.analysis", name=module_name
            )
            # In older importlib.metadata: entry_points().get('playNano.analysis', [])
            if not eps:
                raise ValueError(
                    f"Analysis module '{module_name}' not found in registry or entry points"  # noqa
                )
            # If multiple, pick first
            ep = eps[0]
            cls = ep.load()
        # Instantiate
        instance = cls()
        # Optionally check it's subclass of AnalysisModule
        if not isinstance(instance, AnalysisModule):
            raise TypeError(
                f"Loaded module for '{module_name}' is not an AnalysisModule subclass"
            )
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
        Execute all added analysis steps on the given AFMImageStack.

        Each step:

        - is resolved to an AnalysisModule instance
        - invoked with `(stack, previous_results=..., **params)`
        - its outputs are stored under `stack.analysis["step_<n>_<module_name>"]`
        - provenance is recorded in `stack.provenance["analysis"]["steps"]`

        The overall provenance sub-dict also collects:

        - results_by_name: mapping module name to list of outputs
        - frame_times: result of `stack.get_frame_times()`, or None

        The environment info (via gather_environment_info) is stored at
        `stack.provenance["environment"]` (if not already set).

        Parameters
        ----------
        stack : AFMImageStack
            The AFMImageStack to analyze.
        log_to : str, optional
            Path to a JSON file where the combined record will be saved.

        Returns
        -------
        AnalysisRecord :     dict

        {
            "environment": <dict of environment metadata>,
            "analysis": <dict of outputs per step>,
            "provenance": <dict with keys "steps", "results_by_name", "frame_times">
        }

        Notes
        -----
        - Raw outputs: accessible via `stack.analysis["step_<n>_<module_name>"]`.
        - Provenance: in `stack.provenance["analysis"]`, with a list of step records.
        - If stack.provenance or stack.analysis is absent, they are created.
        - If log_to is provided, the same record dict is JSON-dumped using NumpyEncoder.

        Raises
        ------
        Exception
            Propagates any exception from module.run(...), after logging.

        Examples
        --------
        >>> pipeline = AnalysisPipeline()
        >>> pipeline.add("count_nonzero")
        >>> pipeline.add("feature_detection", mask_fn="threshold_mask", min_size=5)
        >>> record = pipeline.run(stack, log_to="out.json")
        >>> # Access the outputs:
        >>> record["analysis"]["step_1_count_nonzero"]
        {'counts': [...], ...}
        >>> # Inspect provenance:
        >>> for step_info in record["provenance"]["steps"]:
        ...     print(step_info["name"], step_info["analysis_key"])
        count_nonzero step_1_count_nonzero
        feature_detection step_2_feature_detection
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
        if not hasattr(stack, "analysis") or not isinstance(stack.analysis, dict):
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

        # Build the updated provedence record
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
        # Swapped from using NumpyEncoder becuase it could not handle the size
        # of the full analysis record.
        if log_to:
            import json

            safe_record = {
                "environment": sanitize_analysis_for_logging(record["environment"]),
                "analysis": sanitize_analysis_for_logging(record["analysis"]),
                "provenance": sanitize_analysis_for_logging(record["provenance"]),
            }
            with open(log_to, "w") as file:
                json.dump(safe_record, file, indent=2)
        # Add record
        return record
