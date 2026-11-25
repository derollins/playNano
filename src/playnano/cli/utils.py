"""
Helpers for the command line.

Checking step names, reading processing and analysis steps from text or
files, and asking for parameters interactively.
"""

import ast
import inspect
import json
import logging
import numbers
import types
from collections.abc import Sequence
from importlib import metadata
from pathlib import Path
from typing import Any, Callable, Mapping, Union, get_args, get_origin

import numpy as np
import yaml

from playnano.analysis import BUILTIN_ANALYSIS_MODULES
from playnano.processing.filters import register_filters
from playnano.processing.mask_generators import register_masking
from playnano.processing.masked_filters import register_mask_filters
from playnano.processing.stack_edit import register_stack_edit_processing
from playnano.processing.video_processing import register_video_processing

# Built-in filters and mask dictionaries
FILTER_MAP = register_filters()
MASK_MAP = register_masking()
MASK_FILTERS_MAP = register_mask_filters()
VIDEO_FILTER_MAP = register_video_processing()
STACK_EDIT_MAP = register_stack_edit_processing()

# Names of all entry-point plugins (if any third-party filters are installed)
_PLUGIN_ENTRYPOINTS = {
    ep.name: ep for ep in metadata.entry_points(group="playnano.filters")
}
_VIDEO_PLUGIN_ENTRYPOINTS = {
    ep.name: ep for ep in metadata.entry_points(group="playnano.video_processing")
}

# Names of all entry-point plugins (if any third-party filters are installed)
_ANALYSIS_PLUGIN_ENTRYPOINTS = {
    ep.name: ep for ep in metadata.entry_points(group="playnano.analysis")
}

INVALID_CHARS = r'\/:*?"<>|'
INVALID_FOLDER_CHARS = r'*?"<>|'
SKIP_PARAM_NAMES = {
    "data",
    "image",
    "arr",
    "mask",
    "stack",
    "debug",
    "imarray",
    "array",
    "frame",
    "video",
}

logger = logging.getLogger(__name__)


def is_valid_step(name: str) -> bool:
    """
    Check whether a name is a processing step.

    Parameters
    ----------
    name : str
        Step name.

    Returns
    -------
    bool
        True for ``"clear"``, built-in filters, masks, video filters and stack
        edits, and plugins registered under ``playnano.filters`` or
        ``playnano.video_processing``. Masked filters are not steps of their
        own: they are used automatically when a mask is set.
    """
    return (
        name == "clear"
        or name in FILTER_MAP
        or name in MASK_MAP
        or name in _PLUGIN_ENTRYPOINTS
        or name in _VIDEO_PLUGIN_ENTRYPOINTS
        or name in VIDEO_FILTER_MAP
        or name in STACK_EDIT_MAP
    )


def is_valid_analysis_step(name: str) -> bool:
    """
    Check whether a name is an analysis module.

    Parameters
    ----------
    name : str
        Module name.

    Returns
    -------
    bool
        True for built-in modules and plugins registered under
        ``playnano.analysis``. Also True for ``"clear"``, although the analysis
        pipeline has no such step, so it fails when the analysis runs.
    """
    return (
        name == "clear"
        or name in BUILTIN_ANALYSIS_MODULES
        or name in _ANALYSIS_PLUGIN_ENTRYPOINTS
    )


_NONE_WORDS = ("none", "null")
_NONE_TYPE = type(None)


def parse_param_value(text: str) -> Any:
    """
    Convert one parameter value from the command line into a Python value.

    Used for ``--processing`` and ``--analysis-steps`` strings and for values
    typed into the wizard, so all three read values the same way.

    Parameters
    ----------
    text : str
        The value as typed, for example ``"2"``, ``"1e3"``, ``"[0, 1]"`` or
        ``"none"``.

    Returns
    -------
    Any
        ``True``/``False`` for ``true``/``false`` and ``None`` for
        ``none``/``null`` (any case); otherwise a Python literal (number, list,
        tuple, dict or quoted string) if the text is one, a float for ``nan``
        or ``inf``, and the text itself otherwise.

    Notes
    -----
    Python literal syntax is used rather than YAML because PyYAML follows
    YAML 1.1, which reads ``1e3`` as a string and ``on``/``off``/``yes``/``no``
    as booleans. To keep a value that looks like a number as text, quote it:
    ``name='1'``.

    Examples
    --------
    >>> parse_param_value("1e3")
    1000.0
    >>> parse_param_value("[0, 1]")
    [0, 1]
    >>> parse_param_value("None")
    >>> parse_param_value("otsu")
    'otsu'
    """
    value = text.strip()
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered in _NONE_WORDS:
        return None
    try:
        return ast.literal_eval(value)
    except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
        pass
    for convert in (int, float):
        try:
            return convert(value)
        except ValueError:
            continue
    return value


def _split_top_level(text: str, sep: str) -> list[str]:
    """
    Split ``text`` on ``sep``, ignoring separators inside brackets or quotes.

    ``"a=[0,1],b=2"`` split on ``","`` gives ``["a=[0,1]", "b=2"]``.
    """
    parts: list[str] = []
    current: list[str] = []
    depth = 0
    quote: str | None = None
    for ch in text:
        if quote:
            if ch == quote:
                quote = None
        elif ch in "'\"":
            quote = ch
        elif ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth = max(depth - 1, 0)
        elif ch == sep and depth == 0:
            parts.append("".join(current))
            current = []
            continue
        current.append(ch)
    parts.append("".join(current))
    return parts


def _parse_step_string(
    text: str,
    is_valid: Callable[[str], bool],
    unknown_message: str,
    step_label: str,
) -> list[tuple[str, dict[str, object]]]:
    """
    Parse ``"step:key=value,key=value; step2"`` into ``[(name, kwargs), ...]``.

    Shared by :func:`parse_processing_string` and
    :func:`parse_analysis_string`. Steps are separated by ``;`` and
    parameters by ``,``, except inside brackets or quotes, so list and tuple
    values such as ``indices_to_drop=[0,1]`` stay whole. Each value is
    converted with :func:`parse_param_value`.

    Parameters
    ----------
    text : str
        The step string.
    is_valid : callable
        Returns True if a step name is known.
    unknown_message : str
        Start of the error message for an unknown step name.
    step_label : str
        How a step is named in the error for a malformed parameter.

    Returns
    -------
    list of tuple
        ``(step_name, kwargs)`` for each step, in order.

    Raises
    ------
    ValueError
        If a step name is unknown or a parameter is not ``key=value``.
    """
    steps: list[tuple[str, dict[str, object]]] = []
    for segment in _split_top_level(text, ";"):
        segment = segment.strip()
        if not segment:
            continue
        step_name, _, params_part = segment.partition(":")
        step_name = step_name.strip()
        if not is_valid(step_name):
            raise ValueError(f"{unknown_message}: '{step_name}'")
        kwargs: dict[str, object] = {}
        for pair in _split_top_level(params_part, ","):
            pair = pair.strip()
            if not pair:
                continue
            if "=" not in pair:
                raise ValueError(
                    f"Invalid parameter expression '{pair}' in {step_label} "
                    f"'{step_name}'"
                )
            key, val_str = pair.split("=", 1)
            kwargs[key.strip()] = parse_param_value(val_str)
        steps.append((step_name, kwargs))
    return steps


def parse_processing_string(processing_str: str) -> list[tuple[str, dict[str, object]]]:
    """
    Read processing steps from text, as given to ``--processing``.

    Steps are separated by ``;``. A step can be followed by ``:`` and
    parameters as ``key=value``, separated by ``,``. Separators inside brackets
    or quotes do not count, so ``indices_to_drop=[0,1]`` stays whole. Values are
    read with :func:`parse_param_value`.

    Parameters
    ----------
    processing_str : str
        For example ``"remove_plane; gaussian_filter:sigma=2.0"``.

    Returns
    -------
    list of tuple
        ``(step_name, parameters)`` for each step, in order.

    Raises
    ------
    ValueError
        If a step name is not a processing step (see :func:`is_valid_step`), or
        a parameter is not ``key=value``.

    Examples
    --------
    >>> from playnano.cli.utils import parse_processing_string
    >>> parse_processing_string("remove_plane; gaussian_filter:sigma=2.0")
    [('remove_plane', {}), ('gaussian_filter', {'sigma': 2.0})]
    >>> parse_processing_string("drop_frames:indices_to_drop=[0,1]")
    [('drop_frames', {'indices_to_drop': [0, 1]})]
    """
    return _parse_step_string(
        processing_str, is_valid_step, "Unknown processing step", "step"
    )


def parse_processing_file(path: str) -> list[tuple[str, dict[str, object]]]:
    """
    Read processing steps from a YAML or JSON file, as given to ``--processing-file``.

    The file has a ``filters`` list; each entry has a ``name`` and its
    parameters::

        filters:
          - name: remove_plane
          - name: gaussian_filter
            sigma: 2.0
          - name: polynomial_flatten
            order: 2

    Values keep the types YAML gives them. YAML reads some values differently
    from ``--processing``: write ``1.0e3`` rather than ``1e3``, which YAML
    reads as text.

    Parameters
    ----------
    path : str or pathlib.Path
        The file. It is read as YAML, then as JSON if that fails.

    Returns
    -------
    list of tuple
        ``(step_name, parameters)`` for each step, in order.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the file cannot be read, has no ``filters`` list, or an entry has no
        ``name`` or names an unknown step.
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
    Read analysis steps from text, as given to ``--analysis-steps``.

    Written in the same way as for :func:`parse_processing_string`.

    Parameters
    ----------
    analysis_str : str
        For example
        ``"feature_detection:mask_fn=mask_threshold,threshold=1;particle_tracking"``.

    Returns
    -------
    list of tuple
        ``(module_name, parameters)`` for each step, in order.

    Raises
    ------
    ValueError
        If a name is not an analysis module (see
        :func:`is_valid_analysis_step`), or a parameter is not ``key=value``.

    Examples
    --------
    >>> from playnano.cli.utils import parse_analysis_string
    >>> parse_analysis_string("feature_detection:threshold=1;particle_tracking")
    [('feature_detection', {'threshold': 1}), ('particle_tracking', {})]
    """
    return _parse_step_string(
        analysis_str, is_valid_analysis_step, "Unknown analysis step", "analysis step"
    )


def parse_analysis_file(path: str) -> list[tuple[str, dict[str, object]]]:
    """
    Read analysis steps from a YAML or JSON file, as given to ``--analysis-file``.

    The file has an ``analysis`` list, or is just the list; each entry has a
    ``name`` and its parameters::

        analysis:
          - name: feature_detection
            mask_fn: mask_threshold
            threshold: 1.0
            min_size: 10
          - name: particle_tracking
            max_distance: 5.0

    Values keep the types YAML or JSON gives them (in YAML, write ``1.0e3``
    rather than ``1e3``, which is read as text).

    Parameters
    ----------
    path : str or pathlib.Path
        The file: read as JSON if it ends in ``.json``, otherwise as YAML.

    Returns
    -------
    list of tuple
        ``(module_name, parameters)`` for each step, in order.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the file cannot be read, has neither an ``analysis`` list nor a
        list of steps, or an entry has no ``name`` or names an unknown module.
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
    registered under the `playnano.analysis` group.

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
                group="playnano.analysis", name=module_name
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
    - Tuple and list types (also ``list[int]``, ``tuple[int, ...]``,
      ``Sequence[str]``) accept
      ``1,2`` or ``[1, 2]``; each item is converted with `parse_param_value`.
    - `Optional[T]`, `Union[T, None]` and ``T | None`` are treated as `T`, and
      ``none``/``null`` gives None.
    - Unions of several types, other annotations and missing annotations use
      `parse_param_value`; `str` keeps the text as typed.
    """
    if s == "":
        return default

    # no annotation: read the value as the command-line strings do
    if expected_type is None or expected_type is inspect._empty:
        return parse_param_value(s)

    origin = get_origin(expected_type)

    # Union / Optional, including the ``float | None`` form
    if origin in (Union, types.UnionType):
        args = get_args(expected_type)
        options = [a for a in args if a is not _NONE_TYPE]
        if len(options) < len(args) and s.strip().lower() in _NONE_WORDS:
            return None
        if len(options) == 1:
            return _cast_input(s, options[0], default)
        return parse_param_value(s)

    # sequences: "1,2" or "[1, 2]", with each item converted
    if expected_type in (list, tuple) or origin in (list, tuple, Sequence):
        container = tuple if tuple in (expected_type, origin) else list
        parsed = parse_param_value(s)
        if isinstance(parsed, (list, tuple)):
            return container(parsed)
        items = [item for item in _split_top_level(s, ",") if item.strip()]
        return container(parse_param_value(item) for item in items)

    if expected_type is str:
        return s
    if expected_type is bool:
        return s.strip().lower() in ("1", "true", "yes", "y", "t")
    if expected_type in (int, float):
        try:
            return expected_type(s)
        except ValueError:
            value = parse_param_value(s)
            if expected_type is float and isinstance(value, (int, float)):
                return float(value)
            if expected_type is int and isinstance(value, float) and value.is_integer():
                return int(value)
            return value

    return parse_param_value(s)


def ask_for_analysis_params(module_name: str) -> dict[str, Any]:
    """
    Ask in the terminal for each parameter of an analysis module.

    Uses the module's ``parameters()`` list if it has one, otherwise the
    arguments of its ``run`` method. Parameters whose condition (see
    :func:`playnano.utils.param_utils.param_conditions`) is not met are not
    asked. Pressing Enter uses the default; typed values are converted to the
    parameter's type.

    Parameters
    ----------
    module_name : str
        Built-in analysis module or ``playnano.analysis`` plugin.

    Returns
    -------
    dict
        Parameter names and values.

    Raises
    ------
    ValueError
        If no module has that name.
    """
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
        if step_name in _VIDEO_PLUGIN_ENTRYPOINTS:
            return _VIDEO_PLUGIN_ENTRYPOINTS[step_name].load()
        raise ValueError(f"Processing step '{step_name}' not found")
    except Exception as e:
        logger.exception(f"Failed to load processing step '{step_name}': {e}")
        raise


def get_processing_step_type(step_name: str) -> str:
    """
    Name the kind of a processing step, for messages.

    Parameters
    ----------
    step_name : str
        Step name.

    Returns
    -------
    str
        ``"filter"``, ``"mask generator"``, ``"mask filter"``,
        ``"plugin filter"``, ``"video plugin"``, ``"video filter"``,
        ``"stack edit"`` or ``"unknown"``.
    """
    if step_name in FILTER_MAP:
        return "filter"
    if step_name in MASK_MAP:
        return "mask generator"
    if step_name in MASK_FILTERS_MAP:
        return "mask filter"
    if step_name in _PLUGIN_ENTRYPOINTS:
        return "plugin filter"
    if step_name in _VIDEO_PLUGIN_ENTRYPOINTS:
        return "video plugin"
    if step_name in VIDEO_FILTER_MAP:
        return "video filter"
    if step_name in STACK_EDIT_MAP:
        return "stack edit"
    return "unknown"


def ask_for_processing_params(step_name: str) -> dict[str, Any]:
    """
    Ask in the terminal for each parameter of a processing step.

    The parameters are read from the step's function. Data arguments (such as
    ``data``, ``frame``, ``mask`` and ``stack``) are skipped, as are
    parameters whose condition is not met. Pressing Enter uses the default
    (None if there is none); typed values are converted to the parameter's
    type.

    Parameters
    ----------
    step_name : str
        Processing step name.

    Returns
    -------
    dict
        Parameter names and values.

    Raises
    ------
    ValueError
        If no processing step has that name.
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
