"""Common utility functions for analysis."""

import json
import logging
from pathlib import Path
from typing import Any, Mapping

import h5py
import numpy as np

logger = logging.getLogger(__name__)


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
        if callable(obj):
            return f"<function {obj.__name__}>"
        return super().default(obj)


def safe_json_dumps(obj):
    try:
        return json.dumps(obj, cls=NumpyEncoder)
    except TypeError:
        # fallback for anything not serializable, like functions
        return str(obj)


def export_to_hdf5(
    record: Mapping[str, Any], out_path: Path, dataset_name: str = "analysis_record"
) -> None:
    """
    Save a nested dict/list/array structure to HDF5 with full breakdown.

    - Dicts become groups.
    - Lists of primitives become datasets.
    - Lists of complex objects become subgroups item_0, item_1, etc.
    - NumPy arrays become datasets.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_path, "w") as h5file:

        def recurse(group, obj):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    recurse(group.create_group(str(key)), value)

            elif isinstance(obj, list):
                if len(obj) == 0:
                    group.create_dataset("empty", data=[])
                elif all(isinstance(item, (int, float, np.number)) for item in obj):
                    group.create_dataset("values", data=np.array(obj, dtype=float))
                elif all(isinstance(item, (str, bytes)) for item in obj):
                    dt = h5py.string_dtype(encoding="utf-8")
                    group.create_dataset("values", data=np.array(obj, dtype=dt))
                else:
                    for idx, item in enumerate(obj):
                        recurse(group.create_group(f"item_{idx}"), item)

            elif isinstance(obj, np.ndarray):
                try:
                    group.create_dataset("values", data=obj)
                except TypeError:
                    dt = h5py.string_dtype(encoding="utf-8")
                    group.create_dataset("values", data=obj.astype(str), dtype=dt)

            else:
                try:
                    group.attrs["value"] = obj
                except TypeError:
                    group.attrs["value"] = str(obj)

        recurse(h5file.create_group(dataset_name), record)


def sanitize_analysis_for_logging(obj, path="root", _depth=0, _max_depth=5):
    """Return a JSON-safe version of any object, printing any functions it finds."""
    import numpy as np

    if callable(obj):
        logger.debug(
            f"Removing function at {path}: {getattr(obj, '__name__', 'anonymous')}"
        )
        return f"<function {getattr(obj, '__name__', 'anonymous')}>"

    if _depth > _max_depth:
        return f"<... truncated depth {_depth} ...>"

    if isinstance(obj, dict):
        return {
            k: sanitize_analysis_for_logging(
                v, path=f"{path}.{k}", _depth=_depth + 1, _max_depth=_max_depth
            )
            for k, v in obj.items()
        }

    if isinstance(obj, (list, tuple, set)):
        return [
            sanitize_analysis_for_logging(
                v, path=f"{path}[{i}]", _depth=_depth + 1, _max_depth=_max_depth
            )
            for i, v in enumerate(obj)
        ]

    if isinstance(obj, np.ndarray):
        return {
            "_array_shape": obj.shape,
            "_array_dtype": str(obj.dtype),
            "_summary": {
                "min": float(np.min(obj)) if obj.size else None,
                "max": float(np.max(obj)) if obj.size else None,
                "mean": float(np.mean(obj)) if obj.size else None,
            },
        }

    if isinstance(obj, bytes):
        return obj.decode("utf-8", errors="replace")

    return obj


def make_json_safe(record: dict) -> dict:
    """
    Make a AnalysisRecord safe for export through JSON dumping.

    Prepare an AnalysisRecord for JSON dumping by stripping out or summarizing
    any non-JSON-serializable entries.

    Parameters
    ----------
    record : dict
        The full AnalysisRecord returned by AnalysisPipeline.run().

    Returns
    -------
    dict
        A new dict with the same top-level keys ("environment", "analysis",
        "provenance"), but with each value run through your sanitizers so
        that they contain only numbers, strings, lists, and dicts.
    """
    return {
        "environment": sanitize_analysis_for_logging(record["environment"]),
        "analysis": sanitize_analysis_for_logging(record["analysis"]),
        "provenance": sanitize_analysis_for_logging(record["provenance"]),
    }
