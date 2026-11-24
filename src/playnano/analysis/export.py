"""Write an analysis record to a JSON file."""

import json
import os
from typing import Any

from playnano.analysis.utils.common import NumpyEncoder


def export_analysis_to_json(out_path: str, analysis_record: dict[str, Any]) -> None:
    """
    Write an analysis record to a JSON file, with arrays written in full.

    Unlike ``AnalysisPipeline.run(log_to=...)``, which summarises arrays, this
    writes every value: arrays become nested lists and NumPy numbers become
    Python numbers (see :class:`~playnano.analysis.utils.common.NumpyEncoder`).

    Parameters
    ----------
    out_path : str
        The JSON file to write. Missing folders are created.
    analysis_record : dict
        The record returned by
        :meth:`~playnano.analysis.pipeline.AnalysisPipeline.run`.

    Raises
    ------
    TypeError
        If the record holds a value that cannot be written as JSON.
    OSError
        If the file cannot be written.
    """
    parent = os.path.dirname(out_path)
    if parent:  # a bare filename writes to the current folder
        os.makedirs(parent, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(analysis_record, f, indent=2, cls=NumpyEncoder)


# Later, extend to HDF5.
