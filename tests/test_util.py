# # Distribution Statement A. Approved for public release. Distribution unlimited.
# # 
# # Author:
# # Naval Research Laboratory, Marine Meteorology Division
# # 
# # This program is free software:
# # you can redistribute it and/or modify it under the terms
# # of the NRLMMD License included with this program.
# # If you did not receive the license, see
# # https://github.com/U-S-NRL-Marine-Meteorology-Division/
# # for more information.
# # 
# # This program is distributed WITHOUT ANY WARRANTY;
# # without even the implied warranty of MERCHANTABILITY
# # or FITNESS FOR A PARTICULAR PURPOSE.
# # See the included license for more details.

import datetime
import os

import numpy as np
import pytest
import xnrl
import xnrl.constant as C


TEST_SELS = {
    "value": {
        "navgem": {"lat": [0, 1], "lon": [272, 180]},
        "coamps": {
            "lat": [33.29664612, 35.35521317],
            "lon": [278.54467773, 332.38165283],
        },
        "neptune": {"lat": [35.26438968, 0], "lon": [225, 41.1148538]},
    },
    "index": {
        "navgem": {"lat": ["0i", "1"], "lon": ["-1i", 2]},
        "coamps": {"lat": ["3i", "4i"], "lon": ["-1i", "-2i"]},
        "neptune": {"lat": ["-4i", "-5i"], "lon": ["-8i", "-11i"]},
    },
}


def _assert_within_time_limit(s, minutes=1):
    e = datetime.datetime.utcnow()
    assert e - s < datetime.timedelta(minutes=minutes)


@pytest.mark.parametrize("by", ["value", "index"])
@pytest.mark.parametrize("model", ["navgem", "coamps", "neptune"])
@pytest.mark.parametrize("coords", [("lat", "lon"), "lat", "lon"])
@pytest.mark.parametrize("method", ["exact", "nearest", "linear"])
@pytest.mark.parametrize(
    "type_", ["single", "single list", "multiple list", "slice", "slice none", "cross"]
)
def test_sel(by, model, coords, method, type_):
    open_kwds = {}
    if "lat" in coords:
        open_kwds["lats"] = np.array(TEST_SELS[by][model]["lat"])
    if "lon" in coords:
        open_kwds["lons"] = np.array(TEST_SELS[by][model]["lon"])

    if method == "nearest":
        if by == "index":
            return
        for coord in open_kwds:
            open_kwds[coord] = open_kwds[coord].astype(float) + np.random.rand(1)
    for coord in open_kwds:
        if "single" in type_:
            open_kwds[coord] = open_kwds[coord][0]
        elif "slice" in type_:
            open_kwds[coord] = slice(*open_kwds[coord])
            if "none" in type_:
                open_kwds[coord] = slice(open_kwds[coord].start, None)
        elif "cross" in type_:
            if coords != ("lat", "lon"):
                return
            open_kwds[coord] = [list(open_kwds[coord])]

    if model == "neptune":
        open_kwds["file_type"] = "hdf5"
        open_kwds["label"] = "restart"
        open_kwds["fields"] = "rho"

    s = datetime.datetime.utcnow()
    try:
        ds = xnrl.tutorial.open_dataset(model, **open_kwds)
    except:
        raise ValueError(open_kwds)
    _assert_within_time_limit(s)

    for coord in ["lat", "lon"]:
        if f"{coord}s" in open_kwds:
            sel = open_kwds[f"{coord}s"]
            values = ds[coord].values.ravel()
            if by == "value":
                if coord == "lon" and sel is not None:
                    lons_range = xnrl.util._detect_lons_range(ds)
                    sel = xnrl.util._shift_lons(sel, lons_range)

                if "single" in type_:
                    assert np.isclose(sel, values, atol=1).any()
                elif "multiple" in type_:
                    has_0 = np.isclose(sel[0], values, atol=1).any()
                    has_1 = np.isclose(sel[1], values, atol=1).any()
                    assert has_0 and has_1
                elif "slice" in type_:
                    sel_min = sel.start
                    sel_max = sel.stop

                    if sel_min is None:
                        sel_min = ds[coord].values.min()
                    if sel_max is None:
                        sel_max = ds[coord].values.max()

                    if sel_min > sel_max:
                        sel_min, sel_max = sel_max, sel_min
                    threshold = 25 if model == "coamps" else 1
                    has_0 = values.min() >= sel_min - threshold
                    has_1 = values.max() <= sel_max + threshold
                    assert has_0 and has_1

    if "slice" not in type_ and coord == ("lat", "lon"):
        try:
            assert "idx" in ds.dims
        except:
            raise ValueError(open_kwds)
