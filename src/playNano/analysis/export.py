"""Functions for exporting ananlysis results"""

import json
import os
from typing import Any, Dict

from playNano.analysis.utils import NumpyEncoder


def export_analysis_to_json(out_path: str, analysis_record: Dict[str, Any]) -> None:
    """
    Write the analysis_record (as returned by AnalysisPipeline.run) to a JSON file.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(analysis_record, f, indent=2, cls=NumpyEncoder)


# Later, extend to HDF5.
