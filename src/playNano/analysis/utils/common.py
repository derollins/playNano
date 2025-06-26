"""Common utility functions for analysis."""

import json
from pathlib import Path
from typing import Any, Mapping

import h5py
import numpy as np


class NumpyEncoder(json.JSONEncoder):
    """
    Custom JSON encoder for serializing NumPy ndarray objects.

    This encoder converts NumPy arrays to native Python lists so they can be
    serialized by the standard `json` module. It can be used with `json.dump`
    or `json.dumps` by passing it as the `cls` argument.

    Example:
        json.dump(data, file, cls=NumpyEncoder)
    """

    def default(self, obj):
        """
        Override the default method to convert NumPy arrays to lists.

        Parameters:
            obj (Any): The object to be serialized.

        Returns:
            A JSON-serializable version of the object. If the object is a NumPy
            ndarray, it is converted to a list. Otherwise, the superclass's
            default method is used.

        Raises:
            TypeError: If the object cannot be serialized by the superclass.
        """
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def export_to_hdf5(
    record: Mapping[str, Any], out_path: Path, dataset_name: str = "analysis_record"
) -> None:
    """
    Save a nested dict of lists/arrays to HDF5.

    Parameters
    ----------
    record : dict
        E.g. the AnalysisRecord dict returned by `AnalysisPipeline.run()`.
    out_path : Path
    dataset_name : str
        Root group name.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_path, "w") as f:

        def recurse(group, obj):
            if isinstance(obj, Mapping):
                for k, v in obj.items():
                    recurse(group.create_group(k), v)
            elif isinstance(obj, (list, tuple, np.ndarray)):
                arr = np.array(obj, dtype=object)
                # store variable‐length array of JSON‐encoded strings
                dt = h5py.string_dtype(encoding="utf-8")
                group.create_dataset(
                    "values",
                    data=[json.dumps(i, cls=NumpyEncoder) for i in arr],
                    dtype=dt,
                )
            else:
                group.attrs["value"] = obj

        recurse(f.create_group(dataset_name), record)
