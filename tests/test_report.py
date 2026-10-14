import numpy as np
import pandas as pd
import pytest

from motor_thermal import metrics as M
from motor_thermal.report import md_table, render


def test_render_replaces_and_rejects_unknown_placeholders():
    assert render("a {{x}} b {{ y }}", {"x": 1, "y": "z"}) == "a 1 b z"
    with pytest.raises(KeyError, match="missing_key"):
        render("{{missing_key}}", {})


def test_md_table_formats_floats_and_header():
    out = md_table(pd.DataFrame({"a": [1.23456], "b": ["x"]}), "{:.2f}")
    assert out.splitlines()[0] == "| a | b |"
    assert "1.23" in out.splitlines()[2]


def test_error_by_bins_is_ordered_by_bin_edge():
    y = np.array([10.0, 50.0, 70.0, 110.0, 130.0])
    t = M.error_by_bins(y, y, y + 1, [0, 40, 60, 100, 200], "rng")
    assert list(t["n"]) == [1, 1, 1, 2]
    assert t["rng"].tolist()[0].startswith("(-0.001, 40")
    assert t["rng"].tolist()[-1].startswith("(100")
