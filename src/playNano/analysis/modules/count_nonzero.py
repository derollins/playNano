"""Analysis module for counting non-zero data points in an array."""

from typing import Any

from playNano.analysis.base import AnalysisModule


class CountNonzeroModule(AnalysisModule):
    """
    Count non-zero pixels in each frame of an AFMImageStack.

    This simple analysis module computes the number of non-zero pixels
    per frame and returns the result as a 1D array.

    Version
    -------
    0.1.0

    Examples
    --------
    >>> module = CountNonzeroModule()
    >>> result = module.run(stack)
    >>> result['counts'].shape
    (n_frames,)
    """

    version = "0.1.0"

    @property
    def name(self) -> str:
        """
        Name of the analysis module.

        Returns
        -------
        str
            The string identifier for this module.
        """
        return "count_nonzero"

    def run(self, stack, previous_results=None, **params) -> dict[str, Any]:
        """
        Count non-zero pixels per frame in the AFMImageStack.

        Parameters
        ----------
        stack : AFMImageStack
            Stack of AFM frames with `.data` of shape (n_frames, H, W).

        previous_results : dict[str, Any], optional
            Ignored by this module. Included for API compatibility.

        **params : dict
            Additional parameters (unused).

        Returns
        -------
        dict
            Dictionary with key:

                - "counts": np.ndarray of shape (n_frames,), number of non-zero
                  pixels per frame.
        """
        data = stack.data  # shape (n_frames, H, W)
        # Compute counts
        import numpy as np

        counts = np.count_nonzero(data, axis=(1, 2))  # array of length n_frames
        return {"counts": counts}
