"""Module holding the AnalysisModule base class."""

from abc import ABC, abstractmethod
from typing import Any, Optional

from playNano.afm_stack import AFMImageStack

AnalysisOutputs = dict[str, Any]


class AnalysisModule(ABC):
    """
    Abstract base class for analysis steps.
    Subclasses must implement:
      - a `name` property returning a unique string identifier
      - a `run(stack, previous_results=None, **params) -> dict` method
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Unique name for this analysis module, e.g. "particle_detect".
        Used by pipeline to identify and refer to the module.
        """
        ...

    @abstractmethod
    def run(
        self,
        stack: AFMImageStack,
        previous_results: Optional[dict[str, Any]] = None,
        **params,
    ) -> AnalysisOutputs:
        """
        Perform the analysis on the given AFMImageStack.

        Parameters
        ----------
        stack : AFMImageStack
            The AFMImageStack instance, containing `.data` and metadata.
        previous_results : dict or None
            Outputs from earlier modules in the pipeline, if any.
        **params : module-specific parameters
            E.g. threshold, min_size, etc.

        Returns
        -------
        dict
            A dictionary mapping output names (strings) to results, e.g.:
            {
                "coords": numpy array of shape (N,3),
                "masks": numpy array of shape (n_frames, H, W),
                ...
            }
        """
        ...
