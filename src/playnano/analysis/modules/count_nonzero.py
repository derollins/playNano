"""Analysis module that counts the non-zero pixels in each frame."""

from typing import Any

from playnano.analysis.base import AnalysisModule


class CountNonzeroModule(AnalysisModule):
    """
    Count the non-zero pixels in each frame.

    Pipeline name: ``"count_nonzero"``. It has no parameters. NaN pixels count
    as non-zero.

    Examples
    --------
    >>> import numpy as np
    >>> from playnano.afm_stack import AFMImageStack
    >>> data = np.zeros((3, 64, 64))
    >>> for t in range(3):
    ...     data[t, 10 + t : 16 + t, 10:16] = 3.0  # a 6 x 6 pixel particle
    >>> stack = AFMImageStack(
    ...     data,
    ...     pixel_size_nm=2.0,
    ...     channel="height_trace",
    ...     file_path="demo.h5-jpk",
    ...     frame_metadata=[{"timestamp": 0.5 * t} for t in range(3)],
    ... )
    >>> from playnano.analysis.modules.count_nonzero import CountNonzeroModule
    >>> CountNonzeroModule().run(stack)
    {'counts': array([36, 36, 36])}
    """

    version = "0.1.0"

    @property
    def name(self) -> str:
        """Name of this module in a pipeline, ``"count_nonzero"``."""
        return "count_nonzero"

    def run(self, stack, previous_results=None, **params) -> dict[str, Any]:
        """
        Count the non-zero pixels in each frame of ``stack.data``.

        Parameters
        ----------
        stack : AFMImageStack
            The stack to analyse.
        previous_results : dict, optional
            Not used. Default is None.
        **params
            Not used; ignored.

        Returns
        -------
        dict
            ``{"counts": numpy.ndarray}``, the number of non-zero pixels in each
            frame, shape (n_frames,).
        """
        data = stack.data  # shape (n_frames, H, W)
        # Compute counts
        import numpy as np

        counts = np.count_nonzero(data, axis=(1, 2))  # array of length n_frames
        return {"counts": counts}
