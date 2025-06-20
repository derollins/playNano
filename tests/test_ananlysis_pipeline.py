"""Tests for the AnalysisPipeline and related functionality."""

import importlib.metadata
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock

import numpy as np
import pytest

import playNano.analysis.pipeline as apipeline_mod
from playNano.afm_stack import AFMImageStack
from playNano.analysis.base import AnalysisModule
from playNano.analysis.pipeline import AnalysisPipeline


@pytest.fixture
def dummy_stack():
    """Fixture for a dummy AFMImageStack."""
    data = np.ones((3, 4, 4), dtype=float)
    with TemporaryDirectory() as tmpdir:
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

    # Define a simple AnalysisModule subclass
    class DummyMod(AnalysisModule):
        version = "1.0"

        @property
        def name(self) -> str:
            return "dummy"

        def run(self, stack, previous_results=None, **params):
            return {"result": 42, "params": params.copy()}

    return DummyMod()


def test_add_and_clear():
    """Test that steps can be added and cleared."""
    pipeline = AnalysisPipeline()
    pipeline.add("mock_module", x=1)
    assert pipeline.steps == [("mock_module", {"x": 1})]
    pipeline.clear()
    assert pipeline.steps == []


def test_run_executes_steps(dummy_stack, dummy_module):
    """Test that run executes steps and records outputs when module is in cache."""
    pipeline = AnalysisPipeline()
    # Put dummy_module instance in cache under name "dummy"
    pipeline._module_cache["dummy"] = dummy_module
    pipeline.add("dummy", a=5)

    result = pipeline.run(dummy_stack)

    # Top-level keys
    assert "environment" in result
    assert "analysis_steps" in result
    assert "results_by_name" in result

    # Check recorded step
    step0 = result["analysis_steps"][0]
    assert step0["name"] == "dummy"
    assert step0["outputs"]["result"] == 42
    assert step0["outputs"]["params"] == {"a": 5}

    # Check stack.analysis_results attached
    assert dummy_stack.analysis_results is result

    # Check results_by_name
    assert "dummy" in result["results_by_name"]
    assert result["results_by_name"]["dummy"][0]["result"] == 42


def test_module_is_cached(dummy_stack, dummy_module):
    """Test that modules are cached after loading or if pre-cached."""
    pipeline = AnalysisPipeline()
    pipeline._module_cache["dummy"] = dummy_module
    pipeline.add("dummy")
    # Running should use the same cached instance
    pipeline.run(dummy_stack)
    assert pipeline._module_cache["dummy"] is dummy_module


def test_run_propagates_exception(dummy_stack):
    """Test that exceptions from modules are raised and logged."""

    class BrokenMod(AnalysisModule):
        version = "1.0"

        @property
        def name(self) -> str:
            return "fail"

        def run(self, stack, previous_results=None, **params):
            raise RuntimeError("intentional fail")

    broken_instance = BrokenMod()
    pipeline = AnalysisPipeline()
    pipeline._module_cache["fail"] = broken_instance
    pipeline.add("fail")

    with pytest.raises(RuntimeError, match="intentional fail"):
        pipeline.run(dummy_stack)


def test_load_module_from_cache(dummy_module):
    """Test that _load_module returns cached module if already loaded."""
    pipeline = AnalysisPipeline()
    pipeline._module_cache["cached"] = dummy_module
    result = pipeline._load_module("cached")
    assert result is dummy_module


def test_load_module_invalid_type(monkeypatch):
    """Test that a loaded object not subclassing AnalysisModule raises TypeError."""
    pipeline = AnalysisPipeline()
    # Ensure BUILTIN_ANALYSIS_MODULES does not contain 'bad_type'
    monkeypatch.setitem(apipeline_mod.BUILTIN_ANALYSIS_MODULES, "bad_type", None)

    # Mock entry_points to return an entry point whose load() returns a plain object
    mock_entry_point = MagicMock()
    mock_entry_point.load.return_value = object()  # not AnalysisModule subclass

    mock_eps = MagicMock()
    mock_eps.select.return_value = [mock_entry_point]
    # Patch importlib.metadata.entry_points to return our mock_eps
    monkeypatch.setattr(importlib.metadata, "entry_points", lambda: mock_eps)

    with pytest.raises(TypeError):
        pipeline._load_module("bad_type")


def test_run_saves_to_log_file(tmp_path, dummy_stack, dummy_module):
    """Test that run() writes output JSON if log_to is provided."""
    pipeline = AnalysisPipeline()
    pipeline._module_cache["dummy"] = dummy_module
    pipeline.add("dummy")

    log_path = tmp_path / "analysis_record.json"
    result = pipeline.run(                          # noqa: F841
        dummy_stack, log_to=str(log_path)
    )
    assert log_path.exists()
    content = log_path.read_text()
    # Should contain module name and result key
    assert "dummy" in content
    assert "result" in content


@pytest.fixture
def stack_with_times():
    """AFMImageStack with explicit and implicit timestamps."""
    data = np.zeros((4, 2, 2), dtype=float)
    meta = [{"timestamp": 0.0}, {}, {"timestamp": 2.5}, {}]
    with TemporaryDirectory() as td:
        stack = AFMImageStack(
            data.copy(),
            pixel_size_nm=1.0,
            channel="h",
            file_path=Path(td),
            frame_metadata=meta,
        )
        yield stack


# --- Tests for _resolve_mask_fn ---


@pytest.fixture(autouse=True)
def ensure_masking_funcs(monkeypatch):
    """
    Ensure MASKING_FUNCS is a dict with known entries for testing.
    Here we monkeypatch the module-level MASKING_FUNCS.
    """
    # Example: register_masking() returns {"dummy_mask": callable}
    dummy = lambda arr, **kw: np.zeros_like(arr, dtype=bool)    # noqa
    monkeypatch.setattr(apipeline_mod, "MASKING_FUNCS", {"dummy_mask": dummy})
    yield


def test_resolve_mask_fn_success():
    """_resolve_mask_fn should replace string key with callable."""
    pipeline = AnalysisPipeline()
    params = {"mask_fn": "dummy_mask", "foo": 1}
    new = pipeline._resolve_mask_fn(params)
    assert new is not params  # must be a copy
    assert callable(new["mask_fn"])
    assert new["foo"] == 1


def test_resolve_mask_fn_invalid():
    """_resolve_mask_fn should raise if key not in MASKING_FUNCS."""
    pipeline = AnalysisPipeline()
    params = {"mask_fn": "not_exist"}
    with pytest.raises(ValueError):
        pipeline._resolve_mask_fn(params)


# --- Tests for AnalysisPipeline.run with registry entry ---


@pytest.fixture
def dummy_registry(monkeypatch):
    """Monkeypatch BUILTIN_ANALYSIS_MODULES to include 'dummy' -> DummyModule."""

    # Define DummyModule subclass
    class DummyModule(AnalysisModule):
        version = "0.1"

        @property
        def name(self) -> str:
            return "dummy"

        def run(self, stack, previous_results=None, **params):
            return {"marker": True, "params": params.copy()}

    monkeypatch.setitem(apipeline_mod.BUILTIN_ANALYSIS_MODULES, "dummy", DummyModule)
    yield


def test_run_empty_pipeline(stack_with_times):
    """Running with no steps should return environment and empty lists."""
    pipeline = AnalysisPipeline()
    record = pipeline.run(stack_with_times)
    assert "environment" in record
    assert record["analysis_steps"] == []
    assert record["results_by_name"] == {}
    # frame_times should be present
    assert "frame_times" in record and isinstance(record["frame_times"], list)
    assert record["frame_times"] == stack_with_times.get_frame_times()


def test_run_single_step(dummy_registry, stack_with_times, tmp_path):
    """Running one dummy module records output and frame_times, writes log if requested.""" # noqa
    pipeline = AnalysisPipeline()
    pipeline.add("dummy", alpha=5)
    record = pipeline.run(stack_with_times)
    # check structure
    assert "environment" in record
    assert record["analysis_steps"][0]["name"] == "dummy"
    assert record["analysis_steps"][0]["outputs"]["marker"] is True
    assert record["analysis_steps"][0]["outputs"]["params"] == {"alpha": 5}
    assert "dummy" in record["results_by_name"]
    # frame_times included
    assert record["frame_times"] == stack_with_times.get_frame_times()

    # test log_to writing
    log_file = tmp_path / "out.json"
    _ = pipeline.run(stack_with_times, log_to=str(log_file))
    assert log_file.exists()
    content = json.loads(log_file.read_text())
    assert content["analysis_steps"][0]["name"] == "dummy"
    log_file.unlink()


def test_run_multiple_steps(monkeypatch, stack_with_times):
    """Outputs of earlier step should appear in previous_results for next."""

    # Define two AnalysisModule subclasses
    class FirstMod(AnalysisModule):
        version = "1"

        @property
        def name(self) -> str:
            return "firstmod"

        def run(self, stack, previous_results=None, **params):
            return {"first": 1}

    class SecondMod(AnalysisModule):
        version = "1"

        @property
        def name(self) -> str:
            return "secondmod"

        def run(self, stack, previous_results=None, **params):
            # previous_results should contain key "firstmod"
            assert "firstmod" in previous_results
            return {"second": 2}

    # register in BUILTIN_ANALYSIS_MODULES
    monkeypatch.setitem(apipeline_mod.BUILTIN_ANALYSIS_MODULES, "firstmod", FirstMod)
    monkeypatch.setitem(apipeline_mod.BUILTIN_ANALYSIS_MODULES, "secondmod", SecondMod)

    pipeline = AnalysisPipeline()
    pipeline.add("firstmod")
    pipeline.add("secondmod")
    rec = pipeline.run(stack_with_times)
    assert "firstmod" in rec["results_by_name"]
    assert "secondmod" in rec["results_by_name"]


def test_run_missing_module(monkeypatch, stack_with_times):
    """If module not in BUILTIN_ANALYSIS_MODULES or entry points, ValueError."""
    pipeline = AnalysisPipeline()
    pipeline.add("no_such_mod")
    # monkeypatch entry_points to empty
    mock_eps = MagicMock()
    mock_eps.select.return_value = []
    monkeypatch.setattr(importlib.metadata, "entry_points", lambda: mock_eps)
    with pytest.raises(ValueError):
        pipeline.run(stack_with_times)


def test_run_module_wrong_type(monkeypatch, stack_with_times):
    """If entry point loads a non-AnalysisModule, TypeError raised."""
    pipeline = AnalysisPipeline()
    pipeline._module_cache.clear()
    # Ensure BUILTIN_ANALYSIS_MODULES does not have 'badmod'
    monkeypatch.setitem(apipeline_mod.BUILTIN_ANALYSIS_MODULES, "badmod", None)
    # Mock entry_points.select
    fake_ep = MagicMock()
    fake_ep.load.return_value = object()  # not AnalysisModule subclass
    mock_eps = MagicMock()
    mock_eps.select.return_value = [fake_ep]
    monkeypatch.setattr(importlib.metadata, "entry_points", lambda: mock_eps)
    pipeline.add("badmod")
    with pytest.raises(TypeError):
        pipeline.run(stack_with_times)
