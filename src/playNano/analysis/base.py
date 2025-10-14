"""Module holding the AnalysisModule base class."""

from abc import ABC, abstractmethod
from typing import Any, Optional

from playNano.afm_stack import AFMImageStack

AnalysisOutputs = dict[str, Any]


class AnalysisModule(ABC):
    """
    Abstract base class for analysis steps.

    Subclasses must implement:

    - a ``name`` property returning a unique string identifier
    - a ``run(stack, previous_results=None, **params) -> dict`` method

    Inherits from :class:`abc.ABC`.

    """

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Unique name for this analysis module, e.g. "particle_detect".

        Used by pipeline to identify and refer to the module.
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
        Perform the analysis on the given AFMImageStack.

        Parameters
        ----------
        stack : AFMImageStack
            The AFMImageStack instance, containing `.data` and metadata.

        previous_results : dict[str, Any] or None, optional
            Outputs from earlier modules in the pipeline, if any.

        **params : dict
            Module-specific parameters, e.g., threshold, min_size, etc.

        Returns
        -------
        AnalysisOutputs
            Dictionary mapping output names (strings) to results. Example::

                {
                    "coords": numpy.ndarray of shape (N, 3),
                    "masks": numpy.ndarray of shape (n_frames, H, W)
                }

        Notes
        -----
        Subclasses must implement this method. The returned dictionary can
        contain any data the analysis module produces, but must be keyed by
        unique output names.
        """
        raise NotImplementedError("Subclasses must implement 'run' method")
