"""Functions for exporting ananlysis results."""

import json
import os
from typing import Any

from playNano.analysis.utils.common import NumpyEncoder


def export_analysis_to_json(out_path: str, analysis_record: dict[str, Any]) -> None:
    """
    Write the analysis_record (returned by AnalysisPipeline.run) to JSON.

    Parameters
    ----------
    out_path : str
        Output file path.

    analysis_record : dict
        Analysis record to serialize.

    Returns
    -------
    None

    Raises
    ------
    OSError
        If the file cannot be written.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(analysis_record, f, indent=2, cls=NumpyEncoder)


# Later, extend to HDF5.
