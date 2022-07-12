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

import os
import subprocess

import xarray as xr
import xnrl

XDUMP_DF_COLS = [
    "lev_type",
    "grid_dim",
    "field",
    "exp",
    "mbr",
    "ini",
    "tau",
    "lev",
    "model",
    "units",
    "long_name",
]

XDUMP_DS_COORDS = ["lat", "lon", "exp", "mbr", "ini", "tau", "lev", "time", "crs"]


def test_xdump_df():
    paths = xnrl.tutorial._get_paths("navgem", "flatfile", "basic")
    p = subprocess.Popen(["xdump", *paths], stdout=subprocess.PIPE)
    out, err = p.communicate()
    act = out.decode()
    for col in XDUMP_DF_COLS:
        assert col in act


def test_xdump_ds():
    paths = xnrl.tutorial._get_paths("navgem", "flatfile", "basic")
    p = subprocess.Popen(["xdump", "-om", "False", *paths], stdout=subprocess.PIPE)
    out, err = p.communicate()
    act = out.decode()
    for col in XDUMP_DS_COORDS:
        assert col in act


def test_xplot():
    paths = xnrl.tutorial._get_paths("navgem", "flatfile", "basic")[:1]
    p = subprocess.Popen(["xplot", "-od", "/tmp", *paths], stdout=subprocess.PIPE)
    out, err = p.communicate()
    assert os.path.exists("/tmp/airtmp_lonXlat.png")


def test_xoutf():
    path = xnrl.tutorial._get_paths("neptune", "hdf5", "basic")[0]
    p = subprocess.Popen(["xoutf", "-od", "/tmp", path], stdout=subprocess.PIPE)
    out, err = p.communicate()
    act = xnrl.open_dataset("/tmp/ci_xoutf_E0004P4L029_201210241800_PT000000H00M.hdf5")
    exp = xnrl.open_dataset(path)
    assert act.equals(exp)
    for ds1, ds2 in zip(act["ds"], exp["ds"]):
        xr.testing.assert_equal(ds1, ds2.drop_vars("terrain"))
