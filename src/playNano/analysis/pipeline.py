"""Module for the AnalysisPipeline class for orchastration of analysis workflows."""

import importlib.metadata
import logging
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
    """Orchestrates a sequence of analysis steps on an AFMImageStack."""

    def __init__(self):
        """Initiallize the AnalysisPipeline class"""
        # Each entry: (module_name: str, params:   dict)
        self.steps: list[tuple[str, dict[str, Any]]] = []
        # Cache instantiated modules: name -> instance
        self._module_cache: dict[str, AnalysisModule] = {}

    def add(self, module_name: str, **params) -> None:
        """
        Enqueue a module by name with given parameters.

        Example: pipeline.add("particle_detect", threshold=5, min_size=10)
        """
        self.steps.append((module_name, params))

    def clear(self) -> None:
        """Remove all scheduled steps."""
        self.steps.clear()
        self._module_cache.clear()

    def _load_module(self, module_name: str) -> AnalysisModule:
        """
        Load and instantiate a module given its name.

        First check an internal registry, then entry points.
        Caches instances to avoid reloading if run() called multiple times.
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
        Execute all added steps on the AFMImageStack.

        Returns a  dict with keys:
        - "environment": info  dict
        - "steps": ordered list of step records
        - optionally "results_by_name":  dict mapping name->list of outputs
        Optionally writes JSON log to log_to path.
        """
        env_info = gather_environment_info()
        step_results: list[dict[str, Any]] = []
        # previous_results_all: name -> list of outputs  dicts
        previous_results_all: dict[str, list[dict[str, Any]]] = {}
        # previous_latest: name -> latest outputs  dict
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
            rec: dict[str, Any] = {
                "index": idx,
                "name": module_name,
                "params": params,
                "module_version": getattr(module, "version", None),
                "timestamp": timestamp,
                "outputs": outputs,
            }
            step_results.append(rec)

            # update previous_results structures
            previous_results_all.setdefault(module_name, []).append(outputs)
            previous_latest[module_name] = outputs

        # optionally build results_by_name
        results_by_name: dict[str, list[dict[str, Any]]] = dict(previous_results_all)

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
            with open(log_to, "w") as f:
                json.dump(analysis_record, f, indent=2, cls=NumpyEncoder)
        return analysis_record
