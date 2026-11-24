"""The base class that every analysis module subclasses."""

from abc import ABC, abstractmethod
from typing import Any, Optional

from playnano.afm_stack import AFMImageStack

AnalysisOutputs = dict[str, Any]


class AnalysisModule(ABC):
    """
    Base class for analysis modules.

    An analysis module measures something in a stack (for example, detecting
    features or tracking particles) and returns its results as a dict.
    :class:`~playnano.analysis.pipeline.AnalysisPipeline` runs modules in order.
    Built-in modules are in :mod:`playnano.analysis.modules`; plugins are
    registered under the ``playnano.analysis`` entry point group.

    A subclass must define:

    - :attr:`name`: the name used to add the module to a pipeline.
    - :meth:`run`: the analysis.

    and can set these class attributes:

    - ``version`` (str): recorded in the provenance of each step.
    - ``requires`` (list of str): names of modules, at least one of which must
      run earlier in the pipeline; the pipeline raises ``RuntimeError``
      otherwise.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Name used to add the module to a pipeline, for example ``"feature_detection"``.

        Must be unique among the installed modules.
        """
        raise NotImplementedError("Subclasses must implement 'name' property")

    @abstractmethod
    def run(
        self,
        stack: AFMImageStack,
        previous_results: Optional[dict[str, Any]] = None,
        **params,
    ) -> AnalysisOutputs:
        """
        Run the analysis on a stack.

        Parameters
        ----------
        stack : AFMImageStack
            The stack to analyse.
        previous_results : dict, optional
            Outputs of the modules run earlier in the pipeline: each module name
            mapped to the dict its latest run returned. Default is None.
        **params
            Parameters of this module, as given to
            :meth:`~playnano.analysis.pipeline.AnalysisPipeline.add`.

        Returns
        -------
        dict
            Outputs, keyed by str. The values can be anything, for example::

                {"features_per_frame": [...], "summary": {"total_features": 23}}

            Later modules receive this dict through ``previous_results``.
        """
        raise NotImplementedError("Subclasses must implement 'run' method")
