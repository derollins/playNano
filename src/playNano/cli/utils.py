"""Utility functions for the playNano CLI."""

import logging
from importlib import metadata
from pathlib import Path

import yaml

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


def parse_processing_string(processing_str: str) -> list[tuple[str, dict[str, object]]]:
    """
    Parse ; delimited processing strings into a list of (step_name, kwargs) tuples.

    Each segment in `processing_str` is of the form:
        filter_name
        filter_name:param=value
        filter_name:param1=value1,param2=value2

    Example:
      "remove_plane; gaussian_filter:sigma=2.0; threshold_mask:threshold=2"

    Returns a list in the order encountered, e.g.:
      [("remove_plane", {}),
       ("gaussian_filter", {"sigma": 2.0}),
       ("threshold_mask", {"threshold": 2})]
    """
    steps: list[tuple[str, dict[str, object]]] = []
    # Split on ';' (also accept ',' as alternate, just in case)
    for segment in processing_str.split(";"):
        segment = segment.strip()
        if not segment:
            continue

        # If the segment contains ':', separate name from params
        if ":" in segment:
            name_part, params_part = segment.split(":", 1)
            step_name = name_part.strip()
            if not is_valid_step(step_name):
                raise ValueError(f"Unknown processing step: '{step_name}'")

            # Parse params: they can be separated by ',' or ';' (but usually commas)
            kwargs: dict[str, object] = {}
            for pair in params_part.replace(";", ",").split(","):
                pair = pair.strip()
                if not pair:
                    continue
                if "=" not in pair:
                    raise ValueError(
                        f"Invalid parameter expression '{pair}' in step '{step_name}'"
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

                # Some filters use parameter names with underscores
                # e.g. 'pixel_to_nm' or 'threshold' or 'order' or 'sigma'
                # We just pass them through; the filter will validate internally
                kwargs[key] = val

            steps.append((step_name, kwargs))

        else:
            # No colon → just the filter name
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


def sanitize_output_name(name: str, default: str) -> str:
    """
    Sanitize output file names by removing extensions and stripping whitespace.

    Parameters
    ----------
    name : str
        The output file name provided by the user.
    default : str
        Default name to use if `name` is empty or None.

    Returns
    -------
    str
        Sanitized base file name without extension.
    """
    if not name:
        return default
    name = name.strip()
    # Remove extension if any
    try:
        name = Path(name).with_suffix("").name
    except ValueError:
        return default

    if any(c in name for c in INVALID_CHARS):
        raise ValueError(f"Invalid characters in output name: {INVALID_CHARS}")

    return name


def prepare_output_directory(folder: str | None, default: str = "output") -> Path:
    """
    Validate, resolve, and create the output directory if it doesn't exist.

    Parameters
    ----------
    folder : str or None
        User-provided output folder path. If None, use `default`.
    default : str, optional
        Default folder name to use if `folder` not specified.

    Returns
    -------
    Path
        A Path object pointing to the created output directory.

    Raises
    ------
    ValueError
        If any part of the folder path contains invalid characters.
    """
    folder_path = Path(folder.strip()) if folder else Path(default)
    for part in folder_path.parts:
        if any(c in part for c in INVALID_FOLDER_CHARS):
            raise ValueError(
                f"Invalid characters in output folder path: {INVALID_FOLDER_CHARS}"
            )
    folder_path.mkdir(parents=True, exist_ok=True)
    return folder_path
