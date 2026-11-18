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
    Custom JSON encoder for serializing NumPy ndarray and scalar objects.

    This encoder converts NumPy arrays to native Python lists and NumPy scalar
    types (e.g., float32, int64) to their native Python equivalents so they can be
    serialized by the standard `json` module. It can be used with `json.dump`
    or `json.dumps` by passing it as the `cls` argument.

    Example:
        json.dump(data, file, cls=NumpyEncoder)
    """

    def default(self, obj):
        """
        Override default method to convert NumPy arrays and scalar types.

        Convert NumPy arrays ad scalar types to JSON-serializable forms.

        Parameters:
            obj (Any): The object to be serialized.

        Returns:
            A JSON-serializable version of the object. If the object is a NumPy
            ndarray, it is converted to a list. If the object is a NumPy scalar
            (e.g., np.float32, np.int64), it is converted to the equivalent Python
            scalar. Otherwise, the superclass's default method is used.

        Raises:
            TypeError: If the object cannot be serialized by the superclass.
        """
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        if callable(obj):
            return f"<function {obj.__name__}>"
        return super().default(obj)


def safe_json_dumps(obj):
    """Serialize an object to JSON safely, falling back to str() on failure."""
    try:
        return json.dumps(obj, cls=NumpyEncoder)
    except TypeError as e:
        # fallback for anything not serializable, like functions
        print(f"Fallback triggered for object: {obj!r} with error {e}")
        return str(obj)


#: HDF5 attribute recording what Python type each group was written from, so
#: :func:`load_analysis_from_hdf5` can rebuild it exactly.
TYPE_ATTR = "_playnano_type"


def export_to_hdf5(
    record: Mapping[str, Any], out_path: Path, dataset_name: str = "analysis_record"
) -> None:
    """
    Save a nested dict/list/array structure to HDF5 with full breakdown.

    - Dicts become groups, one subgroup per key. Keys are stored as strings.
    - Lists of numbers, or of strings, become a ``values`` dataset.
    - Other lists become subgroups ``item_0``, ``item_1``, etc.
    - NumPy arrays become a ``values`` dataset with their original dtype.
    - Scalars and strings are stored in a ``value`` attribute.

    Each group also records the type it came from (attribute ``_playnano_type``),
    so :func:`load_analysis_from_hdf5` returns the same structure: ``None``,
    empty dicts, float lists and keys named ``values`` all survive the round trip.
    Tuples and sets are stored as lists.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    str_dtype = h5py.string_dtype(encoding="utf-8")

    def is_number(item):
        return isinstance(item, (int, float, np.number)) and not isinstance(
            item, (bool, np.bool_)
        )

    with h5py.File(out_path, "w") as h5file:

        def recurse(group, obj):
            if isinstance(obj, dict):
                group.attrs[TYPE_ATTR] = "dict"
                for key, value in obj.items():
                    recurse(group.create_group(str(key)), value)

            elif isinstance(obj, (list, tuple, set)):
                obj = list(obj)
                group.attrs[TYPE_ATTR] = "list"
                if len(obj) == 0:
                    group.create_dataset("empty", data=[])
                elif all(is_number(item) for item in obj):
                    # np.asarray keeps ints as ints and floats as floats.
                    group.create_dataset("values", data=np.asarray(obj))
                elif all(isinstance(item, str) for item in obj):
                    group.create_dataset(
                        "values", data=np.array(obj, dtype=object), dtype=str_dtype
                    )
                else:
                    for idx, item in enumerate(obj):
                        recurse(group.create_group(f"item_{idx}"), item)

            elif isinstance(obj, np.ndarray):
                if obj.dtype.kind in ("U", "O"):
                    group.attrs[TYPE_ATTR] = "str_array"
                    group.create_dataset(
                        "values", data=obj.astype(str).astype(object), dtype=str_dtype
                    )
                else:
                    group.attrs[TYPE_ATTR] = "array"
                    group.create_dataset("values", data=obj)

            elif obj is None:
                group.attrs[TYPE_ATTR] = "none"

            else:
                try:
                    group.attrs["value"] = obj
                    group.attrs[TYPE_ATTR] = "scalar"
                except TypeError:
                    group.attrs["value"] = str(obj)
                    group.attrs[TYPE_ATTR] = "str"

        recurse(h5file.create_group(dataset_name), record)


def _sorted_items(group) -> list:
    """Return the ``item_N`` children of a group in index order."""
    keys = sorted(group.keys(), key=lambda x: int(x.split("_")[1]))
    return [_load_group(group[k]) for k in keys]


def _load_group(group) -> Any:
    """Rebuild the value stored in an HDF5 group by :func:`export_to_hdf5`."""
    kind = group.attrs.get(TYPE_ATTR)
    if kind is None:
        return _load_legacy_group(group)
    if isinstance(kind, bytes):
        kind = kind.decode("utf-8")
    return _load_typed_group(group, kind)


def _load_typed_group(group, kind: str) -> Any:
    """Rebuild a value from a group that records its type."""
    if kind == "dict":
        return {k: _load_group(group[k]) for k in group.keys()}
    if kind == "list":
        if "values" in group:
            ds = group["values"]
            if h5py.check_string_dtype(ds.dtype) is not None:
                return list(ds.asstr()[()])
            return ds[()].tolist()
        if "empty" in group:
            return []
        return _sorted_items(group)
    if kind == "array":
        return group["values"][()]
    if kind == "str_array":
        return np.array(group["values"].asstr()[()], dtype=str)
    if kind == "none":
        return None
    if kind == "str":
        return str(group.attrs["value"])
    # "scalar"
    val = group.attrs["value"]
    if isinstance(val, bytes):
        return val.decode("utf-8")
    return val.item() if isinstance(val, np.generic) else val


def _whole_float_to_int(val):
    """Return ``int(val)`` for integer-valued NumPy numbers, else ``val``."""
    if isinstance(val, (np.integer, np.floating)) and float(val).is_integer():
        return int(val)
    return val


def _load_legacy_group(group) -> Any:
    """
    Rebuild a value from a group written before type records were added.

    The type is inferred from the layout, and integer-valued floats are converted
    to ints, as earlier versions did.
    """
    if "values" in group:
        data = group["values"][()]
        if isinstance(data, np.ndarray) and data.dtype.kind == "S":
            return data.astype(str)
        if isinstance(data, np.ndarray) and data.shape == ():
            return _whole_float_to_int(data.item())
        if isinstance(data, np.ndarray) and np.issubdtype(data.dtype, np.floating):
            if np.all(np.mod(data, 1) == 0):
                data = data.astype(int)
        return data

    if "empty" in group:
        return []

    if "value" in group.attrs:
        val = group.attrs["value"]
        return _whole_float_to_int(val) if isinstance(val, np.floating) else val

    keys = list(group.keys())
    if all(k.startswith("item_") for k in keys):
        return _sorted_items(group)
    return {k: _load_group(group[k]) for k in keys}


def load_analysis_from_hdf5(
    file_path: str | Path, dataset_name: str = "analysis_record"
) -> dict:
    """
    Load a nested dict/list/NumPy array structure from an HDF5 file.

    This reverses `export_to_hdf5`. Files written by this version record the type
    of every value, so dicts, lists, arrays, ``None`` and scalars come back as they
    were saved. Files written by earlier versions have no type records; for those,
    integer-valued floats are converted to Python ints (for use as list indices),
    and ``None`` and empty dicts cannot be told apart from other values.

    Parameters
    ----------
    file_path : str or Path
        Path to the HDF5 file containing the saved analysis dictionary.
    dataset_name : str
        Name of the top-level group in the HDF5 file where the dictionary is stored.
        Default is 'analysis_record'.

    Returns
    -------
    record : dict
        Nested dictionary reconstructed from the HDF5 file. The structure preserves:
        - dicts as dicts
        - lists as lists
        - NumPy arrays as np.ndarray
        - strings as str
        - empty lists as []
        - primitive values stored in attributes
        - integer-valued floats converted to Python int recursively

    Raises
    ------
    KeyError
        If `dataset_name` is not found in the HDF5 file.
    """

    with h5py.File(file_path, "r") as h5file:
        if dataset_name not in h5file:
            raise KeyError(f"Dataset '{dataset_name}' not found in HDF5 file.")
        return _load_group(h5file[dataset_name])


def _json_safe_key(key):
    """Convert a dict key to a type JSON accepts (str, int, float, bool or None)."""
    if isinstance(key, np.generic):
        key = key.item()
    if key is None or isinstance(key, (str, int, float, bool)):
        return key
    return str(key)


#: Nesting depth at which :func:`sanitize_analysis_for_logging` stops. Analysis
#: results are far shallower than this; the limit only guards against runaway
#: recursion.
MAX_SANITIZE_DEPTH = 50


def sanitize_analysis_for_logging(
    obj, path="root", _depth=0, _max_depth=MAX_SANITIZE_DEPTH
):
    """
    Return a JSON-safe version of any object, for the analysis JSON output.

    Dicts, lists, tuples and sets are converted recursively. NumPy arrays are
    replaced by a summary (shape, dtype, min, max, mean), functions by a
    placeholder string, and NumPy scalars by Python scalars. Numbers and
    strings are always kept, however deeply nested.

    Parameters
    ----------
    obj : Any
        The object to convert.
    path : str, optional
        Location of ``obj`` in the record, used in log messages.
    _depth, _max_depth : int
        Internal recursion counters. A container nested deeper than
        ``_max_depth`` is replaced by a placeholder string and a warning is
        logged.

    Returns
    -------
    Any
        A structure of dicts, lists, numbers, strings, booleans and None.
    """
    import numpy as np

    if callable(obj):
        logger.debug(
            f"Removing function at {path}: {getattr(obj, '__name__', 'anonymous')}"
        )
        return f"<function {getattr(obj, '__name__', 'anonymous')}>"

    # Only containers count towards the depth limit; scalars are always kept.
    if isinstance(obj, (dict, list, tuple, set)) and _depth > _max_depth:
        logger.warning(
            "Analysis output at %s is nested more than %d levels deep; "
            "it is replaced by a placeholder in the JSON.",
            path,
            _max_depth,
        )
        return f"<... truncated depth {_depth} ...>"

    if isinstance(obj, dict):
        return {
            _json_safe_key(k): sanitize_analysis_for_logging(
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

    if isinstance(obj, (np.generic, np.bool_)):
        return obj.item()

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
