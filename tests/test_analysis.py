"""Tests for the analsyis functions other than the pipeline."""

import json
import os
import re
import tempfile

import numpy as np
import pytest

from playNano.afm_stack import AFMImageStack
from playNano.analysis import export
from playNano.analysis.base import AnalysisModule
from playNano.utils import system_info


class DummyModule(AnalysisModule):
    """Create dummy class for testing."""

    @property
    def name(self):
        """Give name to dummy class."""
        return "dummy"

    def run(self, stack, previous_results=None, **params):
        """Run the dummy class."""
        return {"result": 123}


def test_cannot_instantiate_abstract_base_class():
    """Test that abstract base class can't be initiated without attributes."""
    with pytest.raises(TypeError):
        AnalysisModule()  # Cannot instantiate ABC


def test_dummy_module_name_and_run():
    """Test dummy modules has name and runs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        stack = AFMImageStack(
            data=np.zeros((1, 2, 2)),
            pixel_size_nm=1.0,
            channel="test_ch",
            file_path=tmpdir,
        )
        mod = DummyModule()
        assert mod.name == "dummy"
        results = mod.run(stack)
        assert isinstance(results, dict)
        assert results.get("result") == 123


def test_export_analysis_to_json_creates_file_and_dir():
    """Test that a folder anf file is created when export analysis to JSON is run."""
    data = {"array": np.array([[1, 2], [3, 4]]), "value": 42}

    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = os.path.join(tmpdir, "subdir", "analysis.json")
        export.export_analysis_to_json(out_path, data)

        assert os.path.isfile(out_path)

        with open(out_path) as f:
            loaded = json.load(f)

        assert loaded["value"] == 42
        # numpy array should be serialized as list of lists
        assert loaded["array"] == [[1, 2], [3, 4]]


def test_gather_environment_info_contains_expected_keys():
    """Test that enviroment info is gathered."""
    info = system_info.gather_environment_info()

    assert "timestamp" in info
    assert re.match(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}.*Z", info["timestamp"])

    assert "python_version" in info
    assert "platform" in info
    assert "playNano_version" in info

    # Versions for key packages may or may not exist
    for pkg in ("numpy_version", "h5py_version", "scipy_version"):
        assert pkg in info or True  # at least present or silently missing
