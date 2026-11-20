"""
Tests for analyze output names, failed saves and CLI exit codes.

These run real analyses on ``tests/resources/sample_0.h5-jpk``, so they live
outside ``test_cli.py``, whose autouse fixtures replace ``load_data`` and
``Path.exists`` for every test in that module.
"""

import logging
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import playnano.cli.actions as actions
from playnano.cli.actions import Wizard, analyze_pipeline_mode
from playnano.cli.handlers import handle_analyze

SAMPLE_H5JPK = Path(__file__).parent / "resources" / "sample_0.h5-jpk"


@pytest.mark.parametrize(
    ("output_name", "expected_stem"),
    [("../escaped.name", "escaped.name"), ("run.json", "run"), ("run.h5", "run")],
)
def test_analyze_output_name_stays_in_output_folder(
    tmp_path, output_name, expected_stem
):
    """A folder part is dropped and a known extension removed, as for process."""
    out = tmp_path / "out"
    analyze_pipeline_mode(
        str(SAMPLE_H5JPK), "height_trace", "count_nonzero", None, str(out), output_name
    )
    assert sorted(p.name for p in out.iterdir()) == [
        f"{expected_stem}.h5",
        f"{expected_stem}.json",
    ]
    assert [p.name for p in tmp_path.iterdir()] == ["out"]


def test_wizard_analysis_export_sanitises_output_name(tmp_path):
    """The wizard's analysis export uses the same output-name rules."""
    written = []
    record = {"environment": {}, "analysis": {}, "provenance": {}}
    wizard_like = SimpleNamespace(
        output_folder=str(tmp_path),
        output_name="../run.json",
        input_file="in.jpk",
        _build_pipeline=lambda: SimpleNamespace(run=lambda stack, log_to=None: record),
        _atomic_write_json=lambda path, data: written.append(Path(path)),
        _atomic_write_hdf5=lambda path, record: written.append(Path(path)),
    )
    Wizard._run_analysis_and_export(wizard_like, stack=MagicMock())

    assert [p.name for p in written] == ["run.json", "run.h5"]
    assert all(p.parent.resolve() == tmp_path.resolve() for p in written)


@pytest.mark.parametrize("failing", ["json", "h5"])
def test_analyze_writes_what_it_can_then_raises(tmp_path, monkeypatch, failing):
    """If one output fails, the other is still written and an error is raised."""
    if failing == "h5":
        monkeypatch.setattr(
            actions, "export_to_hdf5", MagicMock(side_effect=OSError("disk full"))
        )
    else:
        monkeypatch.setattr(
            actions, "make_json_safe", MagicMock(side_effect=TypeError("bad value"))
        )
    out = tmp_path / "out"

    with pytest.raises(RuntimeError, match="could not be written") as exc:
        analyze_pipeline_mode(
            str(SAMPLE_H5JPK), "height_trace", "count_nonzero", None, str(out), "run"
        )

    other = "json" if failing == "h5" else "h5"
    assert [p.name for p in out.iterdir()] == [f"run.{other}"]
    assert f"run.{failing}" in str(exc.value)


def test_handle_analyze_exits_1_when_a_file_is_not_written(
    tmp_path, monkeypatch, caplog
):
    """A failed HDF5 save makes the analyze command exit with code 1, not 0."""
    monkeypatch.setattr(
        actions, "export_to_hdf5", MagicMock(side_effect=OSError("disk full"))
    )
    args = Namespace(
        input_file=str(SAMPLE_H5JPK),
        channel="height_trace",
        analysis_steps="count_nonzero",
        analysis_file=None,
        output_folder=str(tmp_path),
        output_name="run",
    )
    with pytest.raises(SystemExit) as exc:
        handle_analyze(args)

    assert exc.value.code == 1
    assert "could not be written" in caplog.text


@pytest.mark.parametrize(
    ("handler_name", "target"),
    [
        ("handle_play", "play_pipeline_mode"),
        ("handle_process", "process_pipeline_mode"),
        ("handle_wizard", "Wizard"),
        ("handle_analyze", "analyze_pipeline_mode"),
    ],
)
@pytest.mark.parametrize(
    ("level", "has_traceback"), [(logging.INFO, False), (logging.DEBUG, True)]
)
def test_handler_errors_include_traceback_only_at_debug(
    monkeypatch, caplog, handler_name, target, level, has_traceback
):
    """Every subcommand exits with code 1 and shows a traceback at DEBUG level."""
    from playnano.cli import handlers

    monkeypatch.setattr(handlers, target, MagicMock(side_effect=ValueError("bad")))
    caplog.set_level(level, logger="playnano.cli.handlers")
    args = Namespace(
        input_file="in.h5",
        channel="height_trace",
        processing=None,
        processing_file=None,
        analysis_steps="count_nonzero",
        analysis_file=None,
        export=None,
        make_gif=False,
        make_video=None,
        make_sequence=False,
        output_folder=None,
        output_name=None,
        scale_bar_nm=None,
        zmin="auto",
        zmax="auto",
        draw_ts=False,
        cmap="afm_brown",
        fps=None,
    )

    with pytest.raises(SystemExit) as exc:
        getattr(handlers, handler_name)(args)

    assert exc.value.code == 1
    records = [r for r in caplog.records if r.getMessage() == "bad"]
    assert len(records) == 1
    assert bool(records[0].exc_info) is has_traceback
