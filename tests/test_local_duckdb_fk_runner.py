from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def runner():
    spec=spec_from_file_location("local_duckdb_fk_runner",Path(__file__).resolve().parents[1]/"scripts"/"run_local_duckdb_fk.py")
    module=module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_shared_start_labels_become_fk_end_labels_without_changing_sessions(runner):
    raw=np.asarray(["2026-10-23T09:30","2026-10-23T11:29","2026-10-23T13:00","2026-10-23T14:59"],dtype="datetime64[ns]")
    expected=np.asarray(["2026-10-23T09:31","2026-10-23T11:30","2026-10-23T13:01","2026-10-23T15:00"],dtype="datetime64[ns]")
    np.testing.assert_array_equal(runner.shift_start_to_end(raw),expected)


def test_end_labelled_or_out_of_session_rows_are_not_silently_shifted(runner):
    for value in ["2026-10-23T11:30","2026-10-23T12:00","2026-10-23T15:00"]:
        with pytest.raises(ValueError,match="start-label convention"):
            runner.shift_start_to_end(np.asarray([value],dtype="datetime64[ns]"))


def test_auction_and_after_hours_are_excluded_from_continuous_panel(runner):
    values=np.asarray(["2026-10-23T09:29","2026-10-23T09:30","2026-10-23T11:29","2026-10-23T11:30","2026-10-23T13:00","2026-10-23T14:59","2026-10-23T15:00","2026-10-23T15:30"],dtype="datetime64[ns]")
    np.testing.assert_array_equal(runner.continuous_minute_mask(values),[False,True,True,False,True,True,False,False])
