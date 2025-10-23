"""Utility functions for the playNano CLI."""

import inspect
import json
import logging
import numbers
from importlib import metadata
from pathlib import Path
from typing import Any, Mapping, Union, get_args, get_origin

import numpy as np
import yaml

from playNano.analysis import BUILTIN_ANALYSIS_MODULES
from playNano.processing.filters import register_filters
from playNano.processing.mask_generators import register_masking
from playNano.processing.masked_filters import register_mask_filters
from playNano.processing.stack_edit import register_stack_edit_processing
from playNano.processing.video_processing import register_video_processing

# Built-in filters and mask dictionaries
FILTER_MAP = register_filters()
MASK_MAP = register_masking()
MASK_FILTERS_MAP = register_mask_filters()
VIDEO_FILTER_MAP = register_video_processing()
STACK_EDIT_MAP = register_stack_edit_processing()

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
SKIP_PARAM_NAMES = {"data", "image", "arr", "mask", "stack", "debug"}

logger = logging.getLogger(__name__)


def is_valid_step(name: str) -> bool:
    """Return True if `name` is a built-in filter, mask, plugin or the 'clear' step."""
    return (
        name == "clear"
        or name in FILTER_MAP
        or name in MASK_MAP
        or name in _PLUGIN_ENTRYPOINTS
        or name in VIDEO_FILTER_MAP
        or name in STACK_EDIT_MAP
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


def _cast_input(s: str, expected_type: Any, default: Any):
    """
    Convert a string input into a specified Python type, with fallback defaults.

    Handles `Optional` and `Union` annotations, and performs best-effort conversion
    for standard Python types (str, bool, int, float, tuple, list). If the
    conversion fails or the string is empty, returns the provided default value.

    Parameters
    ----------
    s : str
        The string to convert.
    expected_type : type | Any
        The Python type or type annotation to convert the string into.
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
    - `Optional[T]` and `Union[T, NoneType]` are treated as `T`.
    - For generic types like `list[int]`, the element type hint is ignored,
      but the container conversion still applies.
    """
    if s == "":
        return default

    # handle missing annotations
    if expected_type is None or expected_type is inspect._empty:
        return s

    origin = get_origin(expected_type)
    args = get_args(expected_type)

    # Union / Optional
    if origin is None and isinstance(expected_type, type):
        base = expected_type
    else:
        base = None

    if origin is Union:
        # try each non-None option in order
        non_none = [a for a in args if not isinstance(a, type(None))]
        for opt in non_none:
            try:
                return _cast_input(s, opt, default)
            except Exception:
                continue
        # fallback
        return default if type(None) in args else s

    # Handle plain tuple/list types
    if origin is None:
        if expected_type is tuple:
            items = [item.strip() for item in s.split(",") if item.strip()]
            return tuple(items)
        if expected_type is list:
            items = [item.strip() for item in s.split(",") if item.strip()]
            return items

    # simple types
    try:
        if base is bool:
            s2 = s.lower()
            return s2 in ("1", "true", "yes", "y", "t")
        if base is int:
            return int(s)
        if base is float:
            return float(s)
        if base is str:
            return s
    except Exception:
        # fall through to return default/string
        return s

    # fallback
    return s


def ask_for_analysis_params(module_name: str) -> dict[str, Any]:
    """Introspect a module's `run()` or parameter spec and ask for values."""
    cls = _get_analysis_class(module_name)

    if hasattr(cls, "parameters") and callable(cls.parameters):
        spec = cls.parameters()
        return _ask_with_spec(spec)
    else:
        return _ask_with_signature(cls)


def _ask_with_spec(spec: list[dict[str, Any]]) -> dict[str, Any]:
    """Prompt user for parameter values based on a module-provided spec."""
    kwargs = {}
    pending = list(spec)
    while pending:
        progressed, to_retry = _process_pending_entries(pending, kwargs)
        if not progressed:
            _prompt_remaining(to_retry, kwargs)
            break
        pending = to_retry
    return kwargs


def _ask_with_signature(cls) -> dict[str, Any]:
    """Prompt user for parameter values based on `run()` signature."""
    sig = inspect.signature(cls.run)
    kwargs = {}
    conds = getattr(cls.run, "_param_conditions", {})

    pending = [
        (name, param)
        for name, param in sig.parameters.items()
        if name not in ("self", "stack", "previous_results")
        and param.kind != inspect.Parameter.VAR_KEYWORD
    ]
    while pending:
        progressed, to_retry = _process_signature_pending(pending, kwargs, conds)
        if not progressed:
            _prompt_signature_remaining(to_retry, kwargs, conds)
            break
        pending = to_retry
    return kwargs


# === Internal shared helpers ===


def _process_pending_entries(pending, kwargs):
    """Process pending spec entries with conditions."""
    progressed = False
    to_retry = []
    for entry in pending:
        name, typ, default, cond = (
            entry["name"],
            entry.get("type", str),
            entry.get("default", ""),
            entry.get("condition"),
        )
        should_ask = _resolve_condition(cond, kwargs)
        if should_ask is None:
            to_retry.append(entry)
            continue
        if not should_ask:
            progressed = True
            continue
        val = _prompt_and_cast(name, typ, default)
        if val is not None:
            kwargs[name] = val
        progressed = True
    return progressed, to_retry


def _process_signature_pending(pending, kwargs, conds):
    """Process pending signature parameters with conditions."""
    progressed = False
    to_retry = []
    for pname, param in pending:
        cond = conds.get(pname, getattr(param, "condition", None))
        should_ask = _resolve_condition(cond, kwargs)
        if should_ask is None:
            to_retry.append((pname, param))
            continue
        if not should_ask:
            progressed = True
            continue
        default = param.default if param.default is not inspect._empty else None
        ann = param.annotation if param.annotation is not inspect._empty else None
        kwargs[pname] = _prompt_and_cast(pname, ann, default)
        progressed = True
    return progressed, to_retry


def _prompt_remaining(to_retry, kwargs):
    """Prompt user for remaining spec entries."""
    for entry in to_retry:
        name, typ, default, cond = (
            entry["name"],
            entry.get("type", str),
            entry.get("default", ""),
            entry.get("condition"),
        )
        if cond and not _resolve_condition(cond, kwargs):
            continue
        val = _prompt_and_cast(name, typ, default)
        if val is not None:
            kwargs[name] = val


def _prompt_signature_remaining(to_retry, kwargs, conds):
    """Prompt user for remaining signature parameters."""
    for pname, param in to_retry:
        cond = conds.get(pname, getattr(param, "condition", None))
        if cond and not _resolve_condition(cond, kwargs):
            continue
        default = param.default if param.default is not inspect._empty else None
        ann = param.annotation if param.annotation is not inspect._empty else None
        val = _prompt_and_cast(pname, ann, default)
        kwargs[pname] = val


def _resolve_condition(cond, kwargs):
    """Safely evaluate a conditional parameter dependency."""
    if cond is None:
        return True
    try:
        return bool(cond(kwargs))
    except KeyError:
        return None
    except Exception:
        return True


def _prompt_and_cast(name, typ, default):
    """Prompt user for a value and cast appropriately."""
    prompt = f"  Enter {name} (type={getattr(typ, '__name__', str(typ))}, default={default}): "  # noqa
    val_str = input(prompt).strip()
    return _cast_input(val_str, typ, default)


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
        if step_name in VIDEO_FILTER_MAP:
            return VIDEO_FILTER_MAP[step_name]
        if step_name in STACK_EDIT_MAP:
            return STACK_EDIT_MAP[step_name]
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
    if step_name in VIDEO_FILTER_MAP:
        return "video filter"
    if step_name in STACK_EDIT_MAP:
        return "stack edit"
    return "unknown"


def ask_for_processing_params(step_name: str) -> dict[str, Any]:
    """
    Introspect a processing callable's parameters and ask interactively.

    Skips the first positional arguments (data, mask).
    """
    func = _get_processing_callable(step_name)  # your existing resolver
    sig = inspect.signature(func)
    conditions = getattr(func, "_param_conditions", {})

    # parameters in signature order, excluding data-like args
    params = [(n, p) for n, p in sig.parameters.items() if n not in SKIP_PARAM_NAMES]

    kwargs: dict[str, Any] = {}
    pending = params[:]  # list of (name, param)
    # keep trying until done or no progress
    while pending:
        progressed = False
        to_retry = []
        for name, param in pending:
            cond = conditions.get(name)
            # If there is no condition -> we should ask it
            if cond is None:
                should_ask = True
            else:
                try:
                    should_ask = bool(cond(kwargs))
                except KeyError:
                    # condition depends on missing answers; postpone
                    should_ask = None
                except Exception:
                    # if condition raises, ask to be safe
                    should_ask = True

            if should_ask is None:
                to_retry.append((name, param))
                continue

            progressed = True
            # Remove from pending implicitly by not adding to to_retry

            # If should_ask is False, skip parameter (do not include)
            if not should_ask:
                continue

            default = param.default if param.default is not inspect._empty else None
            ann = param.annotation if param.annotation is not inspect._empty else None
            prompt = f"  Enter {name} (type={getattr(ann,'__name__', str(ann))}, default={default}): "  # noqa
            val_str = input(prompt).strip()
            kwargs[name] = _cast_input(val_str, ann, default)

        if not progressed:
            # nothing progressed – break and ask remaining params defensively
            # (prevents infinite loop if conditions depend on each other circularly)
            for name, param in to_retry:
                default = param.default if param.default is not inspect._empty else None
                ann = (
                    param.annotation if param.annotation is not inspect._empty else None
                )
                prompt = f"  Enter {name} (type={getattr(ann,'__name__', str(ann))}, default={default}): "  # noqa
                val_str = input(prompt).strip()
                kwargs[name] = _cast_input(val_str, ann, default)
            break
        pending = to_retry

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
