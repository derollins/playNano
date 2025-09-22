"""Utility functions for the playNano CLI."""

import inspect
import json
import logging
import numbers
from importlib import metadata
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml

from playNano.analysis import BUILTIN_ANALYSIS_MODULES
from playNano.processing.filters import register_filters
from playNano.processing.mask_generators import register_masking
from playNano.processing.masked_filters import register_mask_filters

# Built-in filters and mask dictionaries
FILTER_MAP = register_filters()
MASK_MAP = register_masking()
MASK_FILTERS_MAP = register_mask_filters()

# Names of all entry-point plugins (if any third-party filters are installed)
_PLUGIN_ENTRYPOINTS = {
    ep.name: ep for ep in metadata.entry_points(group="playNano.filters")
}

# Names of all entry-point plugins (if any third-party filters are installed)
_ANALYSIS_PLUGIN_ENTRYPOINTS = {
    ep.name: ep for ep in metadata.entry_points(group="playNano.analysis")
}

INVALID_CHARS = r'\/:*?"<>|'
INVALID_FOLDER_CHARS = r'*?"<>|'

logger = logging.getLogger(__name__)


def is_valid_step(name: str) -> bool:
    """Return True if `name` is a built-in filter, mask, plugin or the 'clear' step."""
    return (
        name == "clear"
        or name in FILTER_MAP
        or name in MASK_MAP
        or name in _PLUGIN_ENTRYPOINTS
    )


def is_valid_analysis_step(name: str) -> bool:
    """Return True if `name` is a built-in analysis, plugin or the 'clear' step."""
    return (
        name == "clear"
        or name in BUILTIN_ANALYSIS_MODULES
        or name in _ANALYSIS_PLUGIN_ENTRYPOINTS
    )


def parse_processing_string(processing_str: str) -> list[tuple[str, dict[str, object]]]:
    """
    Parse a semicolon-delimited string of processing steps into a structured list.

    Each step in the string can optionally include parameters, separated by commas.
    Parameters are specified as key=value pairs.

    Parameters
    ----------
    processing_str : str
        Semicolon-delimited string specifying processing steps.
        Each step may have optional parameters (seperated by commas) after a colon,
        e.g., "remove_plane; gaussian_filter:sigma=2.0; threshold_mask:threshold=2"

    Returns
    -------
    list of tuple
        List of tuples, each containing:
        - step_name (str): the name of the processing step
        - kwargs (dict of str → object): dictionary of parameters for the step

    Examples
    --------
    >>> parse_processing_string("remove_plane")
    [('remove_plane', {})]

    >>> parse_processing_string("gaussian_filter:sigma=2.0,truncate=4.0")
    [('gaussian_filter', {'sigma': 2.0, 'truncate': 4.0})]

    >>> parse_processing_string(
    ...     "remove_plane; gaussian_filter:sigma=2.0; threshold_mask:threshold=2"
    ... )
    [
        ('remove_plane', {}),
        ('gaussian_filter', {'sigma': 2.0}),
        ('threshold_mask', {'threshold': 2})
    ]
    """
    steps: list[tuple[str, dict[str, object]]] = []

    # Split the input string into individual steps using ';' as the delimiter
    for segment in processing_str.split(";"):
        segment = segment.strip()
        if not segment:
            continue  # Skip empty segments

        # Check if the step includes parameters (indicated by ':')
        if ":" in segment:
            step_name, params_part = segment.split(":", 1)
            step_name = step_name.strip()

            # Validate the step name
            if not is_valid_step(step_name):
                raise ValueError(f"Unknown processing step: '{step_name}'")

            kwargs: dict[str, object] = {}

            # Split parameters by ',' and parse each key=value pair
            for pair in params_part.split(","):
                pair = pair.strip()
                if not pair:
                    continue  # Skip empty parameter entries

                if "=" not in pair:
                    raise ValueError(
                        f"Invalid parameter expression '{pair}' in step '{step_name}'"
                    )

                key, val_str = pair.split("=", 1)
                key = key.strip()
                val_str = val_str.strip()

                # Attempt to convert the value to a boolean, int, or float
                if val_str.lower() in ("true", "false"):
                    val = val_str.lower() == "true"
                else:
                    try:
                        val = float(val_str) if "." in val_str else int(val_str)
                    except ValueError:
                        val = val_str  # Leave as string if not numeric

                kwargs[key] = val

            steps.append((step_name, kwargs))

        else:
            # Step without parameters
            step_name = segment
            if not is_valid_step(step_name):
                raise ValueError(f"Unknown processing step: '{step_name}'")
            steps.append((step_name, {}))

    return steps


def parse_processing_file(path: str) -> list[tuple[str, dict[str, object]]]:
    """
    Parse a YAML (or JSON) processing file into a list of (step_name, kwargs) tuples.

    Expected YAML schema:
      filters:
        - name: remove_plane
        - name: gaussian_filter
          sigma: 2.0
        - name: threshold_mask
          threshold: 2
        - name: polynomial_flatten
          order: 2

    Returns a list in the order listed under `filters`.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"processing file not found: {path}")
    text = p.read_text()

    # Attempt to parse YAML first
    try:
        data = yaml.safe_load(text)
    except Exception:
        # If YAML parse fails, try JSON
        import json

        try:
            data = json.loads(text)
        except Exception as e:
            raise ValueError(
                f"Unable to parse processing file as YAML or JSON: {e}"
            ) from e

    if not isinstance(data, dict) or "filters" not in data:
        raise ValueError("processing file must contain top-level key 'filters'")

    filters_list = data["filters"]
    if not isinstance(filters_list, list):
        raise ValueError("'filters' must be a list in the processing file")

    steps: list[tuple[str, dict[str, object]]] = []
    for entry in filters_list:
        if not isinstance(entry, dict) or "name" not in entry:
            raise ValueError(
                "Each entry under 'filters' must be a dict containing 'name'"
            )  # noqa
        step_name = entry["name"]
        if not is_valid_step(step_name):
            raise ValueError(
                f"Unknown processing step in processing file: '{step_name}'"
            )

        # Build kwargs from all other key/value pairs in the dict
        kwargs: dict[str, object] = {}
        for k, v in entry.items():
            if k == "name":
                continue
            kwargs[k] = v

        steps.append((step_name, kwargs))

    return steps


def parse_analysis_string(analysis_str: str) -> list[tuple[str, dict[str, object]]]:
    """
    Parse ; delimited analysis strings into a list (analysis_step_name, kwargs) tuples.

    Each segment in `analysis_str` is of the form:
        analysis_module_name
        analysis_module_name:param=value
        analysis_module_name:param1=value1,param2=value2

    Example:
      "log_blob_detection:min_sigma=1.0,max_sigma=5.0;x_means_clustering:time_weight=0.2"

    Returns a list in the order encountered, e.g.:
      [("log_blob_detection", {"min_sigma":1.0,"max_sigma":5.0}),
       ("x_means_clustering", {"time_weight": 0.2})]
    """
    steps: list[tuple[str, dict[str, object]]] = []
    # Split on ';' (also accept ',' as alternate, just in case)
    for segment in analysis_str.split(";"):
        segment = segment.strip()
        if not segment:
            continue

        # If the segment contains ':', separate name from params
        if ":" in segment:
            name_part, params_part = segment.split(":", 1)
            step_name = name_part.strip()
            if not is_valid_analysis_step(step_name):
                raise ValueError(f"Unknown analysis step: '{step_name}'")

            # Parse params: they can be separated by ',' or ';' (but usually commas)
            kwargs: dict[str, object] = {}
            for pair in params_part.replace(";", ",").split(","):
                pair = pair.strip()
                if not pair:
                    continue
                if "=" not in pair:
                    raise ValueError(
                        f"Invalid parameter expression '{pair}' in analysis step '{step_name}'"  # noqa
                    )  # noqa
                key, val_str = pair.split("=", 1)
                key = key.strip()
                val_str = val_str.strip()

                # Convert to float or int if possible
                if val_str.lower() in ("true", "false"):
                    # Allow boolean parameters if needed
                    val = val_str.lower() == "true"
                else:
                    try:
                        if "." in val_str:
                            val = float(val_str)
                        else:
                            val = int(val_str)
                    except ValueError:
                        val = val_str  # leave it as string if it’s not numeric

                kwargs[key] = val

            steps.append((step_name, kwargs))

        else:
            # No colon → just the filter name
            step_name = segment
            if not is_valid_analysis_step(step_name):
                raise ValueError(f"Unknown analysis step: '{step_name}'")

            steps.append((step_name, {}))

    return steps


def parse_analysis_file(path: str) -> list[tuple[str, dict[str, object]]]:
    """
    Parse a YAML or JSON analysis file into a list of (name, parameters) tuples.

    This reads a saved analysis pipeline definition, validates its structure,
    and normalizes any complex types (e.g., tuples) into YAML/JSON-safe forms.

    Parameters
    ----------
    path : str
        Path to the YAML or JSON file containing the analysis definition.

    Returns
    -------
    list of tuple
        A list where each element is ``(analysis_step_name, kwargs_dict)``.
        The ``kwargs_dict`` contains parameters for that analysis step.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the file cannot be parsed as YAML or JSON, or if the
        top-level key ``analysis`` is missing.

    Examples
    --------
    Example YAML format::

        analysis:
          - name: count_nonzero
          - name: feature_detection
            mask_fn: mask_threshold
            min_size: 10
            remove_edge: true
          - name: particle_tracking
            coord_columns: [centroid_x, centroid_y]
            coord_key: features_per_frame
            detection_module: feature_detection
            max_distance: 5.0

    This would be parsed as::

        [
            ("count_nonzero", {}),
            ("feature_detection", {
                "mask_fn": "mask_threshold",
                "min_size": 10,
                "remove_edge": True
            }),
            ("particle_tracking", {
                "coord_columns": ["centroid_x", "centroid_y"],
                "coord_key": "features_per_frame",
                "detection_module": "feature_detection",
                "max_distance": 5.0
            })
        ]
    """
    p = Path(path)
    text = p.read_text(encoding="utf8")
    # Try JSON first if file extension suggests, otherwise YAML
    try:
        if p.suffix.lower() in (".json",):
            raw = json.loads(text)
        else:
            raw = yaml.safe_load(text)
    except (json.JSONDecodeError, yaml.YAMLError) as e:
        raise ValueError("Unable to parse analysis file as YAML or JSON") from e

    # Support two styles: {"analysis": [...]} or bare list [...]
    steps_raw = None
    if isinstance(raw, dict) and "analysis" in raw:
        steps_raw = raw["analysis"]
    elif isinstance(raw, list):
        steps_raw = raw
    else:
        raise ValueError(
            "Invalid analysis file: expected top-level 'analysis' or list of steps"
        )

    out = []
    for i, step in enumerate(steps_raw):
        if not isinstance(step, dict) or "name" not in step:
            raise ValueError(
                f"Invalid step #{i}: each step must be a mapping with a 'name' key"
            )
        name = step["name"]

        if not is_valid_analysis_step(name):
            raise ValueError(f"Unknown analysis step: {name}")

        # Copy kwargs excluding the name to keep file dict intact
        kwargs = {k: v for k, v in step.items() if k != "name"}
        out.append((name, kwargs))
    return out


def _get_analysis_class(module_name: str):
    """
    Retrieve an analysis class by name from built-in modules or registered entry points.

    This function first checks the `BUILTIN_ANALYSIS_MODULES` dictionary. If the
    module is not found there, it attempts to load it from Python package entry points
    registered under the `playNano.analysis` group.

    Parameters
    ----------
    module_name : str
        The name of the analysis module to retrieve.

    Returns
    -------
    type
        The analysis class corresponding to the requested module.

    Raises
    ------
    ValueError
        If the module cannot be found in built-ins or entry points.
    Exception
        If any other error occurs during module loading, the exception is logged
        and re-raised.
    """
    # mirror pipeline._load_module logic or import from registry
    try:
        cls = BUILTIN_ANALYSIS_MODULES.get(module_name)
        if cls is None:
            # try to load entry point
            eps = metadata.entry_points().select(
                group="playNano.analysis", name=module_name
            )
            if not eps:
                raise ValueError(f"Analysis module '{module_name}' not found")
            cls = eps[0].load()
        return cls
    except Exception as e:
        logger.exception(f"Failed to load analysis module '{module_name}': {e}")
        raise


def _cast_input(s: str, expected_type: type, default: Any):
    """
    Convert a string input into a specified Python type, with fallback defaults.

    Performs a best-effort conversion based on the expected type. Supports basic
    types such as `str`, `bool`, `int`, `float`, `tuple`, and `list`. If the
    conversion fails or the string is empty, returns the provided default value.

    Parameters
    ----------
    s : str
        The string to convert.
    expected_type : type
        The Python type to convert the string into. If `None` or `inspect._empty`,
        the string is returned as-is.
    default : Any
        Value to return if the string is empty or conversion is not possible.

    Returns
    -------
    Any
        The converted value, or the default if conversion fails.

    Notes
    -----
    - Boolean conversion recognizes '1', 'true', 'yes', 'y', 't'
      (case-insensitive) as True.
    - Tuple and list types assume comma-separated values in the string.
    """
    # best-effort conversion
    if s == "":
        return default
    if expected_type in (str, None) or expected_type is inspect._empty:
        return s
    if expected_type is bool:
        s2 = s.lower()
        return s2 in ("1", "true", "yes", "y", "t")
    try:
        # common numeric types
        if expected_type is int:
            return int(s)
        if expected_type is float:
            return float(s)
        if expected_type is tuple:
            # comma separated
            return tuple(x.strip() for x in s.split(",") if x.strip())
        if expected_type is list:
            return [x.strip() for x in s.split(",") if x.strip()]
    except Exception:
        # fallback
        return s
    return s


def ask_for_analysis_params(module_name: str) -> dict[str, Any]:
    """
    Introspect module_name.run signature and interactively ask for param values.

    Returns kwargs dict.
    """
    cls = _get_analysis_class(module_name)
    # Prefer a module-provided spec if available
    if hasattr(cls, "parameters") and callable(cls.parameters):
        # parameters() -> list of (name, type, default, help) would be ideal
        spec = cls.parameters()
        kwargs = {}
        for entry in spec:
            name = entry["name"]
            typ = entry.get("type", str)
            default = entry.get("default", "")
            prompt = f"  Enter {name} (default={default}): "
            val_str = input(prompt).strip()
            kwargs[name] = _cast_input(val_str, typ, default)
        return kwargs

    # Fallback to signature of run()
    sig = inspect.signature(cls.run)
    kwargs = {}
    for pname, param in sig.parameters.items():
        # skip positional-only params: skip stack, previous_results
        if pname in ("self", "stack", "previous_results"):
            continue
        # skip **kwargs maybe, but still allow user to add via raw spec
        if param.kind == inspect.Parameter.VAR_KEYWORD:
            # can't introspect further — ask for none or allow raw later
            continue
        default = param.default if param.default is not inspect._empty else None
        ann = param.annotation if param.annotation is not inspect._empty else None
        # Show the parameter and default
        prompt_default = default if default is not None else ""
        prompt = f"  Enter {pname} (type={getattr(ann,'__name__',str(ann))}, default={prompt_default}): "  # noqa
        val_str = input(prompt).strip()
        try:
            val = _cast_input(val_str, ann, default)
        except Exception:
            val = val_str or default
        # Only add if user provided something or default exists
        if val is not None:
            kwargs[pname] = val
    return kwargs


def _get_processing_callable(step_name: str):
    """
    Return a processing callable filter, mask generator, or mask filter.

    These are from built-ins or plugins.
    """
    try:
        if step_name in FILTER_MAP:
            return FILTER_MAP[step_name]
        if step_name in MASK_MAP:
            return MASK_MAP[step_name]
        if step_name in MASK_FILTERS_MAP:
            return MASK_FILTERS_MAP[step_name]
        if step_name in _PLUGIN_ENTRYPOINTS:
            return _PLUGIN_ENTRYPOINTS[step_name].load()
        raise ValueError(f"Processing step '{step_name}' not found")
    except Exception as e:
        logger.exception(f"Failed to load processing step '{step_name}': {e}")
        raise


def get_processing_step_type(step_name: str) -> str:
    """Return the type of a processing step."""
    if step_name in FILTER_MAP:
        return "filter"
    if step_name in MASK_MAP:
        return "mask generator"
    if step_name in MASK_FILTERS_MAP:
        return "mask filter"
    if step_name in _PLUGIN_ENTRYPOINTS:
        return "plugin filter"
    return "unknown"


def ask_for_processing_params(step_name: str) -> dict[str, Any]:
    """
    Introspect a processing callable's parameters and ask interactively.

    Skips the first positional arguments (data, mask).
    """
    func = _get_processing_callable(step_name)
    sig = inspect.signature(func)
    kwargs = {}
    for pname, param in sig.parameters.items():
        # Skip data/mask args
        if pname in ("data", "image", "arr", "mask"):
            continue

        default = param.default if param.default is not inspect._empty else None
        ann = param.annotation if param.annotation is not inspect._empty else None
        prompt = f"  Enter {pname} (type={getattr(ann, '__name__', str(ann))}, default={default}): "  # noqa
        val_str = input(prompt).strip()
        kwargs[pname] = _cast_input(val_str, ann, default)
    return kwargs


def _sanitize_for_dump(obj: Any) -> Any:
    """
    Convert Python objects into JSON/YAML-safe types suitable for safe_dump/json.dump.

    - tuple -> list
    - numpy types -> native Python types (numbers, lists)
    - pathlib.Path -> str
    - recursively applied to lists/dicts
    """
    # Paths -> strings
    if isinstance(obj, Path):
        return str(obj)
    # numpy scalars -> python scalars
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    # numpy array -> nested lists
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    # numbers including Python ints/floats -> keep
    if (
        isinstance(obj, numbers.Number)
        or isinstance(obj, str)
        or isinstance(obj, bool)
        or obj is None
    ):
        return obj
    # tuple -> list (important to avoid !!python/tuple)
    if isinstance(obj, tuple):
        return [_sanitize_for_dump(x) for x in obj]
    # list -> map recursively
    if isinstance(obj, list):
        return [_sanitize_for_dump(x) for x in obj]
    # dict-like -> sanitize values
    if isinstance(obj, Mapping):
        return {k: _sanitize_for_dump(v) for k, v in obj.items()}
    # fallback to string representation
    return str(obj)


def _normalize_loaded(obj: Any) -> Any:
    """
    Normalize objects returned by yaml.safe_load / json.load.

    - Convert tuples to lists (some YAML loaders can still produce tuples)
    - Recurse into dicts/lists
    """
    if isinstance(obj, tuple):
        return [_normalize_loaded(x) for x in obj]
    if isinstance(obj, list):
        return [_normalize_loaded(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _normalize_loaded(v) for k, v in obj.items()}
    return obj
