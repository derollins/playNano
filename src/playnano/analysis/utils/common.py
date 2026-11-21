"""
Convert analysis results for saving as JSON or HDF5.

- :class:`NumpyEncoder` and :func:`safe_json_dumps` write NumPy values as JSON.
- :func:`sanitize_analysis_for_logging` and :func:`make_json_safe` make a
  smaller, JSON-safe copy in which arrays are summarised.
- :func:`export_to_hdf5` and :func:`load_analysis_from_hdf5` save and reload a
  record in full.
"""

import json
import logging
from pathlib import Path
from typing import Any, Mapping

import h5py
import numpy as np

logger = logging.getLogger(__name__)


class NumpyEncoder(json.JSONEncoder):
    """
    JSON encoder that also writes NumPy arrays, NumPy numbers and functions.

    Arrays become (nested) lists, NumPy integers and floats become Python numbers,
    and functions become the text ``"<function name>"``. Pass it as ``cls`` to
    :func:`json.dump` or :func:`json.dumps`.

    Examples
    --------
    >>> import json
    >>> import numpy as np
    >>> json.dumps({"a": np.arange(3), "b": np.float32(0.5)}, cls=NumpyEncoder)
    '{"a": [0, 1, 2], "b": 0.5}'
    """

    def default(self, obj):
        """
        Return a JSON-serialisable form of ``obj``.

        Parameters
        ----------
        obj : Any
            A value the standard encoder cannot write.

        Returns
        -------
        list or int or float or str
            The converted value.

        Raises
        ------
        TypeError
            If ``obj`` is none of the types above (for example, a NumPy bool).
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
    """
    Return ``obj`` as JSON text, or as ``str(obj)`` if it cannot be written as JSON.

    Parameters
    ----------
    obj : Any
        The value to convert, written with :class:`NumpyEncoder`.

    Returns
    -------
    str
        The JSON text, or ``str(obj)``. In that case a message is printed.
    """
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
    Save nested dicts, lists and arrays, such as an analysis record, to HDF5.

    Nothing is summarised, so :func:`load_analysis_from_hdf5` returns the same
    structure. Each value becomes an HDF5 group:

    - dicts: one subgroup per key, with keys stored as str;
    - lists of numbers, or of str, and arrays: a ``values`` dataset;
    - other lists: subgroups ``item_0``, ``item_1`` and so on;
    - numbers, str and other values: a ``value`` attribute (values HDF5 cannot
      store are saved as ``str(value)``);
    - None: an empty group.

    Each group's ``_playnano_type`` attribute records the type it came from.
    Tuples and sets are saved as lists.

    Parameters
    ----------
    record : dict
        The structure to save, for example the record returned by
        :meth:`~playnano.analysis.pipeline.AnalysisPipeline.run`.
    out_path : str or pathlib.Path
        The file to write; overwritten if it exists. Missing folders are
        created.
    dataset_name : str, optional
        Name of the top-level group. Default is ``"analysis_record"``.
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
    Load a structure saved by :func:`export_to_hdf5`.

    Parameters
    ----------
    file_path : str or pathlib.Path
        The HDF5 file.
    dataset_name : str, optional
        Name of the top-level group. Default is ``"analysis_record"``.

    Returns
    -------
    dict
        The saved structure, with dicts, lists, arrays, numbers, str and None
        as they were saved (tuples and sets come back as lists).

        Files written by playNano before 0.5.2 do not record types. For these,
        the type is inferred from the layout: integer-valued floats become
        ints, and None and empty dicts cannot be recovered.

    Raises
    ------
    KeyError
        If the file has no group called ``dataset_name``.
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
    Return a JSON-safe copy of ``obj`` in which arrays are summarised.

    Dicts, lists, tuples and sets are converted recursively, with dict keys
    made JSON-safe. Arrays are replaced by a summary dict with the keys
    ``"_array_shape"``, ``"_array_dtype"`` and ``"_summary"`` (``min``, ``max``
    and ``mean``, or None for an empty array). Functions become the text
    ``"<function name>"``, NumPy numbers become Python numbers and bytes are
    decoded as UTF-8.

    Parameters
    ----------
    obj : Any
        The value to convert.
    path : str, optional
        Location of ``obj`` in the record, used in log messages. Default is
        ``"root"``.
    _depth, _max_depth : int, optional
        Used internally. A container nested more than ``_max_depth`` levels
        deep (default 50) is replaced by a placeholder text, and a warning is
        logged.

    Returns
    -------
    Any
        The converted value, made only of dicts, lists, str, numbers, bools and
        None. Other objects are returned unchanged.
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
    Return a JSON-safe copy of an analysis record, with arrays summarised.

    Runs :func:`sanitize_analysis_for_logging` on each part of the record.

    Parameters
    ----------
    record : dict
        The record returned by
        :meth:`~playnano.analysis.pipeline.AnalysisPipeline.run`.

    Returns
    -------
    dict
        A new dict with the keys ``"environment"``, ``"analysis"`` and
        ``"provenance"``.

    Raises
    ------
    KeyError
        If one of those keys is missing.

    See Also
    --------
    playnano.utils.io_utils.make_json_safe : A different function with the same
        name, used for stack provenance.
    """
    return {
        "environment": sanitize_analysis_for_logging(record["environment"]),
        "analysis": sanitize_analysis_for_logging(record["analysis"]),
        "provenance": sanitize_analysis_for_logging(record["provenance"]),
    }
