"""Tests for the analysis pipeline and moldule."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from playNano.afm_stack import AFMImageStack
from playNano.analysis.pipeline import AnalysisPipeline


@pytest.fixture
def dummy_stack():
    """Fixture for a dummy AFMImageStack."""
    data = np.ones((3, 4, 4), dtype=float)
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir)
        stack = AFMImageStack(
            data.copy(),
            pixel_size_nm=1.0,
            channel="h",
            file_path=path,
        )
        yield stack


@pytest.fixture
def dummy_module():
    """Fixture for a dummy AnalysisModule with a working run() method."""
    module = MagicMock()
    module.run.return_value = {"result": 42}
    module.version = "1.0"
    return module


def test_add_and_clear():
    """Test that steps can be added and cleared."""
    pipeline = AnalysisPipeline()
    pipeline.add("mock_module", x=1)
    assert pipeline.steps == [("mock_module", {"x": 1})]
    pipeline.clear()
    assert pipeline.steps == []


def test_run_executes_steps(dummy_stack, dummy_module):
    """Test that run executes steps and records outputs."""
    pipeline = AnalysisPipeline()
    pipeline._module_cache["dummy"] = dummy_module
    pipeline.add("dummy", a=5)

    result = pipeline.run(dummy_stack)

    assert "environment" in result
    assert "steps" in result
    assert result["steps"][0]["name"] == "dummy"
    assert result["steps"][0]["outputs"] == {"result": 42}
    assert dummy_stack.analysis_results == result
    assert "dummy" in result["results_by_name"]
    assert result["results_by_name"]["dummy"][0] == {"result": 42}


def test_module_is_cached(dummy_stack, dummy_module):
    """Test that modules are cached after loading."""
    pipeline = AnalysisPipeline()
    pipeline._module_cache["dummy"] = dummy_module
    pipeline.add("dummy")

    pipeline.run(dummy_stack)
    assert pipeline._module_cache["dummy"] is dummy_module


def test_run_propagates_exception(dummy_stack):
    """Test that exceptions from modules are raised and logged."""
    broken_module = MagicMock()
    broken_module.run.side_effect = RuntimeError("intentional fail")
    pipeline = AnalysisPipeline()
    pipeline._module_cache["fail"] = broken_module
    pipeline.add("fail")

    with pytest.raises(RuntimeError, match="intentional fail"):
        pipeline.run(dummy_stack)


def test_load_module_from_cache(dummy_module):
    """Test that _load_module returns cached module if already loaded."""
    pipeline = AnalysisPipeline()
    pipeline._module_cache["cached"] = dummy_module
    result = pipeline._load_module("cached")
    assert result is dummy_module


def test_load_module_invalid_type():
    """Test that a loaded object not subclassing AnalysisModule raises TypeError."""
    pipeline = AnalysisPipeline()
    bad_instance = object()  # not an AnalysisModule

    mock_entry_point = MagicMock()
    mock_entry_point.load.return_value = bad_instance

    mock_eps = MagicMock()
    mock_eps.select.return_value = [mock_entry_point]

    with patch("importlib.metadata.entry_points", return_value=mock_eps):
        with pytest.raises(TypeError):
            pipeline._load_module("bad_type")


def test_run_saves_to_log_file(tmp_path, dummy_stack, dummy_module):
    """Test that run() writes output JSON if log_to is provided."""
    pipeline = AnalysisPipeline()
    pipeline._module_cache["dummy"] = dummy_module
    pipeline.add("dummy")

    log_path = tmp_path / "analysis_record.json"
    result = pipeline.run(dummy_stack, log_to=str(log_path))    # noqa

    assert log_path.exists()
    content = log_path.read_text()
    assert "dummy" in content
    assert "result" in content
