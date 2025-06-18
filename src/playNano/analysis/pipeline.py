"""Module for the AnalysisPipeline class for orchastration of analysis workflows."""

import importlib.metadata
import logging
from collections import defaultdict
from datetime import UTC, datetime
from typing import Any, Optional

from playNano.afm_stack import AFMImageStack
from playNano.analysis import BUILTIN_ANALYSIS_MODULES
from playNano.analysis.base import AnalysisModule
from playNano.analysis.metadata import gather_environment_info
from playNano.analysis.utils import NumpyEncoder

logger = logging.getLogger(__name__)

AnalysisRecord = dict[str, Any]  # could refine with a dataclass later


class AnalysisPipeline:
    """
    Orchestrates a sequence of analysis steps on an AFMImageStack.

    This class manages the configuration and execution of modular, reusable
    analysis routines. Each step is defined by a registered analysis module
    (subclass of `AnalysisModule`) and its parameters.

    Analysis results are collected in order and optionally logged to disk.
    Outputs from previous steps can be accessed by subsequent modules.
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

    def run(self, stack: AFMImageStack, log_to: Optional[str] = None) -> AnalysisRecord:
        """
        Execute all added analysis steps on the given AFMImageStack.

        Each module is called with the current `stack` and the results of
        previous steps (accessible by name). Outputs are stored in a list
        of ordered step records and grouped by module name.

        Parameters
        ----------
        stack : AFMImageStack
            The AFM data object to analyze.
        log_to : str, optional
            Path to a JSON file where the analysis record will be saved.

        Returns
        -------
        AnalysisRecord : dict
            A dictionary containing:
            - "environment": snapshot of software/library versions
            - "steps": list of step records with outputs and metadata
            - "results_by_name": mapping from module names to output lists

        Notes
        -----
        Step outputs are stored inline in the "steps" list, and grouped in
        "results_by_name" for easy access.

        Raises
        ------
        Exception
            Any exception raised by individual analysis modules will be propagated.

        Examples
        --------
        >>> stack = AFMImageStack(data)
        >>> pipeline = AnalysisPipeline()
        >>> pipeline.add("count_nonzero")
        >>> pipeline.add("particle_detect", threshold=4)
        >>> results = pipeline.run(stack)
        >>> results["results_by_name"]["particle_detect"][0]["coords"].shape
        (n_detections, 3)
        """
        env_info = gather_environment_info()
        step_results: list[dict[str, Any]] = []
        results_by_name: defaultdict[str, list] = defaultdict(list)
        previous_latest: dict[str, dict[str, Any]] = {}
        # module cache unchanged
        for idx, (module_name, params) in enumerate(self.steps, start=1):
            logger.info(
                f"Running analysis step {idx}: {module_name} with params {params!r}"
            )
            module = self._load_module(module_name)
            # timestamp
            timestamp = datetime.now(UTC).isoformat() + "Z"
            # run; pass in previous_latest so module can read latest outputs by name
            try:
                outputs = module.run(stack, previous_results=previous_latest, **params)
            except Exception as e:
                logger.error(
                    f"Module '{module_name}' failed at step {idx}: {e}", exc_info=True
                )
                raise

            # record this step
            step_record: dict[str, Any] = {
                "index": idx,
                "name": module_name,
                "params": params,
                "timestamp": timestamp,
                "module_version": getattr(module, "version", None),
                "outputs": outputs,
            }
            step_results.append(step_record)

            # update previous_results structures
            results_by_name[module_name].append(outputs)
            # allow downstream modules to use latest result
            previous_latest[module_name] = outputs

        # build final record
        analysis_record: AnalysisRecord = {
            "environment": env_info,
            "steps": step_results,
            "results_by_name": results_by_name,
        }
        # attach to stack
        stack.analysis_results = analysis_record
        # write to file if requested
        if log_to:
            import json
            import os
            os.makedirs(os.path.dirname(log_to), exist_ok=True)
            with open(log_to, "w") as file:
                json.dump(analysis_record, file, indent=2, cls=NumpyEncoder)
        return analysis_record
