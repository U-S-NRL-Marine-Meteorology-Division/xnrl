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
import h5py
import numpy as np
import pandas as pd
import pygrib
import pytest
import xarray as xr
import xnrl
import xnrl.constant as C

EXP_COLS = set(
    ["lev_type", "grid_dim", "field", "exp", "mbr", "ini", "tau", "lev", "ds", "model"]
)
LABELS = "model, file_type, label"


def _get_paths(model, file_type, label):
    try:
        paths = xnrl.tutorial._get_paths(model, file_type, label)
    except FileNotFoundError:
        pytest.skip(f"{model} {file_type} {label} skipped")
    return paths


def _only_meta_test(only_meta, exp_cols, model, file_type, df):
    exp_cols = set(exp_cols)
    exp_cols.remove("ds")

    if only_meta == "list":
        exp_cols |= set(["num_xs", "num_ys", "directory", "file", "path", "model"])

    try:
        assert len(exp_cols - set(df.columns)) == 0
    except AssertionError:
        assert exp_cols == set(df.columns)


def _baseline_test(paths, model, file_type, label, **kwds):
    df = xnrl.open_dataset(paths, model=model, wrap_df=True, log_lev="debug", **kwds)
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0

    only_meta = kwds.get("only_meta")
    chunks = kwds.get("chunks")
    temporal_dim = kwds.get("temporal_dim")
    if temporal_dim == "time":
        exp_cols = ["time" if col == "tau" else col for col in EXP_COLS]
    else:
        exp_cols = EXP_COLS.copy()

    if only_meta in [True, "list"] and not chunks:
        _only_meta_test(only_meta, exp_cols, model, file_type, df)
        return
    elif only_meta in [True, "list"] and chunks or label != "basic":
        pytest.skip("")
    else:
        assert set(exp_cols) == set(df.columns)

    for ds in df["ds"]:
        assert isinstance(ds, xr.Dataset)
        assert "lev_type" in ds.attrs
        assert "grid_dim" in ds.attrs
        assert "model" in ds.attrs
        if file_type != C.HDF5:
            for coord in ds.coords:
                if coord == "crs":
                    continue
                key = ds.attrs["lev_type"] if coord == "lev" else coord
                if key == C.UNTITLED:
                    continue
                assert ds[coord].attrs == C.COORD_ATTR[key]
    return df


@pytest.mark.parametrize("model", [C.NEPTUNE, C.NAVGEM])
@pytest.mark.parametrize(
    "label", ["basic", "reanalysis", "restart", "native", "tgz", "xy"]
)
@pytest.mark.parametrize("chunks", [True, False])
@pytest.mark.parametrize("only_meta", [True, "list"])
@pytest.mark.parametrize("temporal_dim", ["tau", "time"])
def test_open_dataset_hdf5(model, label, chunks, only_meta, temporal_dim):
    file_type = C.HDF5
    paths = _get_paths(model, file_type, label)
    model_meta = C.MODEL_META[model][file_type]
    kwds = dict(chunks=chunks, only_meta=only_meta, temporal_dim=temporal_dim)
    if model == C.NEPTUNE:
        kwds["lons_range"] = 180
    df = _baseline_test(paths, model, file_type, label, **kwds)

    if only_meta:
        return

    for path in paths:
        with h5py.File(path, "r") as hf:
            for group in hf:
                if group not in model_meta["meta_lev_types"]:
                    try:
                        hds = hf[group]
                        ds = df.loc[df["lev_type"] == group, "ds"].iloc[0]
                    except IndexError:
                        continue
                    for var in hds:
                        if var in ["latitude", "longitude"]:
                            continue
                        ex = hds[var][:]
                        if group != "native" and model == C.NEPTUNE:
                            ex = np.rollaxis(ex, -1)  # make lev index 0
                        da = ds[var]
                        if "lev" in da.dims:
                            da = da.dropna("lev")
                        ac = da.values.squeeze()
                        if group == "pressure" and var != "terrain":
                            ac = ac[::-1]
                        if np.all(ex == -9999.0):
                            print(f"Skipping {var}")
                            continue
                        assert np.allclose(ex.ravel(), ac.ravel())


@pytest.mark.parametrize("label", ["basic"])
@pytest.mark.parametrize("chunks", [True, False])
@pytest.mark.parametrize("only_meta", [None, True, "list"])
@pytest.mark.parametrize("temporal_dim", ["tau", "time"])
def test_open_dataset_coamps_hdf5(label, chunks, only_meta, temporal_dim):
    model = C.COAMPS
    file_type = C.HDF5
    paths = _get_paths(model, file_type, label)
    kwds = dict(
        error_out=False, chunks=chunks, only_meta=only_meta, temporal_dim=temporal_dim
    )
    for path in paths:
        df = _baseline_test(path, model, file_type, label, **kwds)

    if only_meta:
        return

    meta_df = xnrl.open_dataset(paths, model="COAMPS", only_meta="list")
    for _, row in meta_df.iterrows():
        lev = row["lev"]
        path = row["path"]
        group = row["group"]
        field = row["field"]
        lev_type = row["lev_type"]
        grid_dim = row["grid_dim"]

        with h5py.File(path, "r") as hf:
            ex = hf[group][:]

        try:
            ds = df.loc[
                (df["lev_type"] == lev_type) & (df["grid_dim"] == grid_dim), "ds"
            ].iloc[0][field]
        except IndexError:
            continue

        if "lev" in ds.dims and "sig" not in lev_type:
            ds = ds.sel(**{"lev": lev})
        elif "sig" in lev_type:
            ds = ds.dropna("lev").sortby("lev", ascending=False)

        ac = ds.values.squeeze()
        assert np.allclose(ex.ravel(), ac.ravel())


@pytest.mark.parametrize("model", [C.NAVGEM, C.COAMPS, C.GFS])
@pytest.mark.parametrize(
    "label", ["basic", "sigma", "sigma_no_datahd", "coords", "no_latlons"]
)
@pytest.mark.parametrize("chunks", [True, False])
@pytest.mark.parametrize("only_meta", [None, True, "list"])
@pytest.mark.parametrize("temporal_dim", ["tau", "time"])
def test_open_dataset_flatfile(model, label, chunks, only_meta, temporal_dim):
    file_type = C.FLATFILE
    paths = _get_paths(model, file_type, label)
    kwds = dict(chunks=chunks, only_meta=only_meta, temporal_dim=temporal_dim)
    df = _baseline_test(paths, model, file_type, label, **kwds)

    if only_meta:
        return

    ds = df["ds"][0]
    meta_df = xnrl.open_dataset(paths, only_meta="list")
    for i, row in meta_df.iterrows():
        cols = ["ini", "tau", "lev"]
        if "sig_w" in ds.coords or "lev" not in ds.dims:
            cols.remove("lev")
        if temporal_dim == "time":
            cols = ["time" if col == "tau" else col for col in cols]
        da = ds.sel(row[cols])[row["field"]].squeeze()
        offset = 8 if model == C.GFS else 0
        ex = np.fromfile(row["path"], offset=offset, dtype=">f")
        ac = da.values.ravel()
        assert np.allclose(ex, ac)


@pytest.mark.parametrize("model", [C.NAVGEM, C.COAMPS, C.GENERIC])
@pytest.mark.parametrize("label", ["basic"])
@pytest.mark.parametrize("chunks", [True, False])
@pytest.mark.parametrize("only_meta", [None, True, "list"])
@pytest.mark.parametrize("temporal_dim", ["tau", "time"])
def test_open_dataset_grib(model, label, chunks, only_meta, temporal_dim):
    file_type = C.GRIB
    paths = _get_paths(model, file_type, label)
    kwds = dict(chunks=chunks, only_meta=only_meta, temporal_dim=temporal_dim)
    for path in paths:
        df = _baseline_test(paths, model, file_type, label, **kwds)

    if only_meta:
        return

    ds = df["ds"][0]
    meta_df = xnrl.open_dataset(paths, only_meta="list")
    for i, row in meta_df.iterrows():
        cols = ["ini", "tau", "lev"]
        if temporal_dim == "time":
            cols = ["time" if col == "tau" else col for col in cols]
        da = ds.sel(row[cols])[row["field"]].squeeze()
        with pygrib.open(row["path"]) as grbs:
            ex = grbs[1].values.ravel()
        ac = da.values.ravel()
        assert np.allclose(ex, ac)


@pytest.mark.parametrize("model", [C.GENERIC])
@pytest.mark.parametrize("label", ["basic"])
@pytest.mark.parametrize("chunks", [True, False])
@pytest.mark.parametrize("only_meta", [None, True, "list"])
@pytest.mark.parametrize("temporal_dim", ["tau", "time"])
def test_open_dataset_netcdf(model, label, chunks, only_meta, temporal_dim):
    file_type = C.NETCDF
    paths = _get_paths(model, file_type, label)
    kwds = dict(chunks=chunks, only_meta=only_meta, temporal_dim=temporal_dim)
    for path in paths:
        df = _baseline_test(paths, model, file_type, label, **kwds)
        ac = df["ds"][0]
        with xr.open_dataset(path, decode_cf=False).sortby("lat") as ex:
            for var in ex.data_vars:
                assert np.allclose(ex[var].values, ac[var].squeeze().values)


@pytest.mark.parametrize("model", [C.NAVGEM])
@pytest.mark.parametrize("label", ["basic"])
@pytest.mark.parametrize("chunks", [True, False])
@pytest.mark.parametrize("only_meta", [None, True, "list"])
@pytest.mark.parametrize("temporal_dim", ["tau", "time"])
def test_open_dataset_grads(model, label, chunks, only_meta, temporal_dim):
    file_type = C.GRADS
    paths = _get_paths(model, file_type, label)
    kwds = dict(chunks=chunks, only_meta=only_meta, temporal_dim=temporal_dim)
    df = _baseline_test(paths, model, file_type, label, **kwds)

    if only_meta:
        return

    ds_pre = df["ds"][0]
    ds_sfc = df["ds"][1]
    meta_df = xnrl.open_dataset(paths, only_meta="list")
    for i, row in meta_df.iterrows():
        ds = ds_sfc if row["lev"] == 0 else ds_pre
        cols = ["ini", "tau", "lev"]
        if row["lev"] == 0:
            cols = cols[:-1]
        if temporal_dim == "time":
            cols = ["time" if col == "tau" else col for col in cols]
        da = ds.sel(row[cols])[row["field"]].squeeze()
        ex = np.fromfile(
            row["path"], dtype=">f", count=np.prod(da.shape), offset=row["offset"]
        )
        ac = da.values.ravel()
        assert np.allclose(ex, ac)


@pytest.mark.parametrize("wrap_df", [True, None, False])
def test_open_dataset_merge_lev_types(wrap_df):
    df_ds = xnrl.tutorial.open_dataset(
        "NEPTUNE",
        wrap_df=wrap_df,
        lev_types=["model_level", "height_agl"],
        merge_lev_types=True,
    )
    lev_type_label = "height_agl+model_level"
    if not wrap_df:
        assert isinstance(df_ds, xr.Dataset)
        assert df_ds.attrs["lev_type"] == lev_type_label
    else:
        assert isinstance(df_ds, pd.DataFrame)
        assert (df_ds["lev_type"] == lev_type_label).all()


def test_datahd_path():
    datahd_path = os.path.join(
        C.TEST_DIR,
        "coamps",
        "flatfile",
        "sigma",
        "datahd_sfc_000000_000000_1a2000x0001_2014071500_00000000_infofld",
    )
    ds = xnrl.tutorial.open_dataset(
        model="COAMPS", label="datahd_path", datahd_path=datahd_path
    )
    assert "sig_m" in ds.dims


def test_datahd_path_dir():
    datahd_dir = os.path.join(C.TEST_DIR, "coamps", "flatfile", "sigma")
    ds = xnrl.tutorial.open_dataset(
        model="COAMPS", label="datahd_path", datahd_path=datahd_dir
    )
    assert "sig_m" in ds.dims


def test_datahd_path_glob():
    datahd_path = os.path.join(C.TEST_DIR, "coamps", "flatfile", "sigma", "datahd*")
    ds = xnrl.tutorial.open_dataset(
        model="COAMPS", label="datahd_path", datahd_path=datahd_path
    )
    assert "sig_m" in ds.dims
