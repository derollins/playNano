"""Analysis module for counting non-zero data points in an array."""

from typing import Any

from playNano.analysis.base import AnalysisModule


class CountNonzeroModule(AnalysisModule):
    @property
    def name(self) -> str:
        return "count_nonzero"

    def run(self, stack, previous_results=None, **params) -> dict[str, Any]:
        """
        Count non-zero pixels per frame in the AFMImageStack.
        Returns:
          - "counts": numpy array of shape (n_frames,)
        """
        data = stack.data  # shape (n_frames, H, W)
        # Compute counts
        import numpy as np

        counts = np.count_nonzero(data, axis=(1, 2))  # array of length n_frames
        return {"counts": counts}
