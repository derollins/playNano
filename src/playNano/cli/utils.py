"""Utility functions for the playNano CLI."""

import logging
from importlib import metadata
from pathlib import Path

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
    Parses a semicolon-delimited string of processing steps into a structured list of
    (step_name, parameters) tuples.

    Each step in the string can optionally include parameters, separated by commas.
    Parameters are specified as key=value pairs.

    Format examples:
        - "remove_plane"
        - "gaussian_filter:sigma=2.0"
        - "threshold_mask:threshold=2,mode=soft"

    Full example:
        "remove_plane; gaussian_filter:sigma=2.0; threshold_mask:threshold=2"

    Returns:
        A list of tuples, each containing:
            - step_name (str): the name of the processing step
            - kwargs (dict[str, object]): a dictionary of parameters for the step

    Example output:
        [
            ("remove_plane", {}),
            ("gaussian_filter", {"sigma": 2.0, "truncate": 4.0}),
            ("threshold_mask", {"threshold": 2})
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
    Parse YAML or JSON analysis files into a list (analysis_step_name, kwargs) tuples.

    Expected YAML schema:
      analysis:
        - name:
        - name:
          sigma: 2.0
        - name:
          threshold: 2
        - name:
          order: 2
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

    if not isinstance(data, dict) or "analysis" not in data:
        raise ValueError("analysis file must contain top-level key 'filters'")

    analysis_list = data["analysis"]
    if not isinstance(analysis_list, list):
        raise ValueError("'analysis' must be a list in the processing file")

    steps: list[tuple[str, dict[str, object]]] = []
    for entry in analysis_list:
        if not isinstance(entry, dict) or "name" not in entry:
            raise ValueError(
                "Each entry under 'analysis' must be a dict containing 'name'"
            )  # noqa
        step_name = entry["name"]
        if not is_valid_analysis_step(step_name):
            raise ValueError(f"Unknown analysis step in analysis file: '{step_name}'")

        # Build kwargs from all other key/value pairs in the dict
        kwargs: dict[str, object] = {}
        for k, v in entry.items():
            if k == "name":
                continue
            kwargs[k] = v

        steps.append((step_name, kwargs))

    return steps
