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

from collections.abc import Iterable

import numpy as np
import pandas as pd
import xarray as xr
import xnrl.constant as C
from xnrl import packages, util


def _prod_y_x(sel_y, sel_x, force=False):
    try:
        if len(sel_y) != len(sel_x) or force:
            sel_y, sel_x = np.array(np.meshgrid(sel_y, sel_x)).reshape(2, -1)
    except TypeError:
        pass
    return sel_y, sel_x


def transform_points(crs, projection, input_x, input_y):
    xyz = crs.transform_points(projection, input_x, input_y)
    out_x = np.median(xyz[..., 0], axis=0)
    out_y = np.median(xyz[..., 1], axis=1)
    return out_x, out_y


def isclose(da, value):
    return xr.DataArray(np.isclose(da, value), dims=da.dims)


def _interp_to_regular_grid_metpy(da, var_y, var_x, regrid_kwds):
    regrid_kwds.pop("degrees", None)
    method = regrid_kwds.pop("method", "nearest")
    if "interp_type" not in regrid_kwds:
        regrid_kwds["interp_type"] = method

    interpolate = packages._import_interpolate()
    da_tmp, dims, coords = util._interp_prep(da, var_y, var_x)

    hres = regrid_kwds.get("hres", 1)
    if var_y == "y" and hres < 1000:
        factor = 10000
        hres *= factor
        regrid_kwds["hres"] = hres
        C.LOG.warning(
            f"Found an abnormally low horizontal regrid "
            f"resolution for {var_y} and {var_x}! "
            f"Increasing by {factor}; now hres={hres}"
        )

    data_list = []
    for tmp in da_tmp["tmp_dim"]:
        da_sub = da_tmp.sel(**{"tmp_dim": tmp})
        x = da_sub[var_x].values.copy()
        y = da_sub[var_y].values.copy()
        z = da_sub.values.copy()
        gx, gy, gz = interpolate.interpolate_to_grid(x, y, z, **regrid_kwds)
        data_list.append(gz)

    coords = {var_y: gy[:, 0], var_x: gx[0, :], **coords}
    da_interp = util._interp_rebuild(da, var_y, var_x, data_list, dims, coords)
    return da_interp


def _interp_to_regular_grid_xesmf(ds, regrid_kwds):
    if "hres" in regrid_kwds:
        regrid_kwds["degrees"] = regrid_kwds.pop("hres")
    if "interp_type" in regrid_kwds:
        interp_type = regrid_kwds.pop("interp_type")
        if interp_type not in ["nearest_s2d", "nearest_d2s"]:
            C.LOG.warning(f"xesmf only supports nearest_s2d!")
            regrid_kwds["method"] = "nearest_s2d"
        else:
            regrid_kwds["method"] = interp_type
    ds_interp = util.regrid(ds, **regrid_kwds)
    return ds_interp


def _interp_to_sel_grid(da, var_y, var_x, sel_y, sel_x, interp_kwds, to_irregular):
    interpolate = packages._import_interpolate()
    da_tmp, dims, coords = util._interp_prep(da, var_y, var_x)

    grid_ys = np.array(da[var_y].values.flat)
    grid_xs = np.array(da[var_x].values.flat)
    grid_coords = np.stack([grid_ys, grid_xs]).T

    num_ys = len(sel_y)
    num_xs = len(sel_x)

    force = False if to_irregular else True
    prod_y, prod_x = _prod_y_x(sel_y, sel_x, force=force)
    sel_coords = np.stack([prod_y, prod_x]).T

    method = interp_kwds.pop("method", None)
    if "interp_type" not in interp_kwds:
        interp_kwds["interp_type"] = method

    data_list = []
    for tmp in da_tmp["tmp_dim"]:
        da_sub = da_tmp.sel(**{"tmp_dim": tmp})
        grid_values = np.array(da_sub.values.flat)
        sel_values = interpolate.interpolate_to_points(
            grid_coords, grid_values, sel_coords, **interp_kwds
        )
        if not to_irregular:
            sel_values = sel_values.reshape(num_xs, num_ys).T
        data_list.append(sel_values)

    da_interp = util._interp_rebuild(
        da, var_y, var_x, data_list, dims, coords, to_irregular=to_irregular
    )

    if to_irregular:
        da_interp["idx"] = np.arange(len(da_interp["idx"]))
        da_interp[var_y] = "idx", prod_y
        da_interp[var_x] = "idx", prod_x
    else:
        da_interp[var_y] = sel_y
        da_interp[var_x] = sel_x
    return da_interp


def _interp_y_x(ds, var_y, var_x, sel_y, sel_x, interp_kwds):
    to_irregular = True
    if isinstance(sel_y, slice):
        sel_y = _slice_to_range(sel_y)
        to_irregular = False
    if isinstance(sel_x, slice):
        sel_x = _slice_to_range(sel_x)
        to_irregular = False

    for dim, sel in {var_y: sel_y, var_x: sel_x}.items():
        util._is_within(ds, dim, sel, "min")
        util._is_within(ds, dim, sel, "max")

    ds, base_coords, var_coords = util._reset_var_coords(ds, var_y, var_x)
    ds = ds.map(
        _interp_to_sel_grid,
        args=(var_y, var_x, sel_y, sel_x, interp_kwds, to_irregular),
    )
    ds = util._set_var_coords(ds, base_coords, var_coords)
    return ds


def _reorder_start_stop(sel):
    try:
        both_negative = sel.start < 0 and sel.stop < 0
        both_positive = sel.start >= 0 and sel.stop >= 0
        if (both_negative or both_positive) and sel.start > sel.stop:
            # make sure elements are in ascending order
            sel = slice(sel.stop, sel.start, sel.step)
    except (TypeError, ValueError):
        pass
    return sel


def _slice_sel(ds, sel, coord):  # lat/lon slice selection
    min_val = float(ds[coord].min())
    max_val = float(ds[coord].max())
    sel = _reorder_start_stop(sel)

    if sel.start == 0 and sel.stop == 0 and coord == var_x:
        sel = slice(min_val, max_val)

    sel_list = [
        sel.start if sel.start is not None else min_val,
        sel.stop if sel.stop is not None else max_val,
        sel.step,
    ]

    if np.isclose(sel_list[0], sel_list[1]):
        ds = ds.where(isclose(ds[coord], sel_list[0]), drop=True)
    else:
        ds = ds.where(
            (ds[coord] >= sel_list[0]) & (ds[coord] <= sel_list[1]), drop=True
        )
    return ds


def _slice_to_range(sel):
    return np.arange(sel.start, sel.stop, sel.step)


def _slice_to_point(ds, var_y, var_x, sel, coord):
    return np.unique(_slice_sel(ds, sel, coord)[coord])


def _slice_y_x(ds, var_y, var_x, sel_y, sel_x):
    for dim, sel in {var_y: sel_y, var_x: sel_x}.items():
        util._is_within(ds, dim, sel.start, "min")
        util._is_within(ds, dim, sel.stop, "max")

    if sel_y.step is not None or sel_x.step is not None:
        sel_y = _slice_to_range(sel_y)
        sel_x = _slice_to_range(sel_x)
        ds = _point_y_x(ds, var_y, var_x, sel_y, sel_x)
    else:
        ds = _slice_sel(ds, sel_y, var_y)
        ds = _slice_sel(ds, sel_x, var_x)

    return ds


def _haversine_np(lat1, lon1, lat2, lon2):
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return c


def _point_y_x_scipy(ds, var_y, var_x, sel_df, grid_df):
    scipy = packages._import_scipy()

    grid_coords = np.stack([grid_df[var_y], grid_df[var_x]]).T
    sel_coords = np.stack([sel_df[var_y], sel_df[var_x]]).T

    kdtree = scipy.spatial.cKDTree(grid_coords, leafsize=15)
    dist = 2 if var_y == "lat" else 2000
    _, grid_indices = kdtree.query(sel_coords, k=1, distance_upper_bound=dist)
    subset_indices = np.where(grid_indices < np.max(grid_coords.shape))
    grid_indices = grid_indices[subset_indices]
    if len(grid_indices.ravel()) == 0:
        dist_str = "{dist} degrees" if var_y == "lat" else dist
        raise ValueError(f"Found 0 points within a {dist_str} radius using kdtree!")

    query_df = pd.DataFrame(grid_coords[grid_indices], columns=[var_y, var_x])
    return query_df


def _point_y_x_numpy(ds, var_y, var_x, sel_df, grid_df):
    # subset so there's not a million elements to compare
    tol_y = np.diff(grid_df[var_y]).max() + 0.25
    tol_x = np.diff(grid_df[var_x]).max() + 0.25

    grid_df = grid_df.loc[
        (
            (grid_df[var_y] >= sel_df[var_y].min() - tol_y)
            & (grid_df[var_y] <= sel_df[var_y].max() + tol_y)
        )
        & (
            (grid_df[var_x] >= sel_df[var_x].min() - tol_x)
            & (grid_df[var_x] <= sel_df[var_x].max() + tol_x)
        )
    ]

    # calculate distance and find shortest distance
    sel_df.columns = "sel_" + sel_df.columns
    sel_df = sel_df.assign(key=1).merge(grid_df.assign(key=1)).drop("key", axis=1)
    sel_df["dist"] = _haversine_np(*[sel_df[col] for col in sel_df.columns])
    sel_df = sel_df.sort_values("dist").drop_duplicates(
        [f"sel_{var_y}", f"sel_{var_x}"]
    )
    return sel_df


def _point_finalize(ds, var_y, var_x, query_df, to_irregular=False):
    ds = ds.where(
        (ds[var_y].isin(query_df[var_y].unique()))
        & (ds[var_x].isin(query_df[var_x].unique())),
        drop=True,
    )

    if to_irregular:
        if "idx" not in ds.dims:
            ds = ds.stack({"idx": ["x", "y"]}).dropna("idx")

    return ds


def _point_y_x(ds, var_y, var_x, sel_y, sel_x, to_irregular=False):
    for dim, sel in {var_y: sel_y, var_x: sel_x}.items():
        util._is_within(ds, dim, sel, "min")
        util._is_within(ds, dim, sel, "max")

    sel_y, sel_x = _prod_y_x(sel_y, sel_x)
    sel_df = pd.DataFrame({var_y: sel_y, var_x: sel_x}).drop_duplicates()

    grid_df = pd.DataFrame(
        {var_y: ds[var_y].values.ravel(), var_x: ds[var_x].values.ravel()}
    )

    try:
        query_df = _point_y_x_scipy(ds, var_y, var_x, sel_df, grid_df)
    except (ImportError, ValueError) as e:
        C.LOG.warning(e)
        query_df = _point_y_x_numpy(ds, var_y, var_x, sel_df, grid_df)

    return _point_finalize(ds, var_y, var_x, query_df, to_irregular=to_irregular)


def _index_to_value(ds, sel, coord, indices, indices_2=None):
    if not indices and not indices_2:
        return sel

    is_xarray = isinstance(sel, xr.DataArray)
    if is_xarray:
        sel = sel.values.tolist()

    if isinstance(indices, slice):
        start = indices.start if indices.start is not None else 0
        stop = indices.stop if indices.start is not None else -1
        indices = slice(start, stop, indices.step)

    if isinstance(indices_2, slice):
        start_2 = indices_2.start if indices_2.start is not None else 0
        stop_2 = indices_2.stop if indices_2.stop is not None else -1
        indices_2 = slice(start_2, stop_2, indices_2.step)

    if isinstance(indices, slice) or isinstance(indices_2, slice):
        if indices_2 and not indices:
            indices = slice(0, -1)

        if indices.start is not None:
            try:
                start = ds[coord].values[indices.start]
            except IndexError:
                start = ds[coord].values[0]
        else:
            start = ds[coord].values[0]

        if len(np.atleast_1d(start)) > 1:
            if indices_2:
                try:
                    start = start[indices_2.start]
                except IndexError:
                    start = ds[coord].values[0]
            else:
                start = start[0]

        if indices.stop is not None:
            try:
                stop = ds[coord].values[indices.stop]
            except IndexError:
                stop = ds[coord].values[-1]
        else:
            stop = ds[coord].values[-1]

        if len(np.atleast_1d(stop)) > 1:
            if indices_2:
                try:
                    stop = stop[indices_2.stop]
                except IndexError:
                    stop = ds[coord].values[-1]
            else:
                stop = stop[-1]

        step = indices.step
        sel = slice(start, stop, step)
        sel = _reorder_start_stop(sel)

    elif isinstance(indices, list):
        try:
            to_cross = True if isinstance(indices[0], list) else False
        except IndexError:
            to_cross = False

        if indices is not None and indices_2 is not None and ds[coord].ndim > 1:
            if indices == []:
                indices = [-1]
            if indices_2 == []:
                indices_2 = [-1]
            sel = np.unique(ds[coord].values[indices, indices_2])
        elif indices is not None and indices_2 is None:
            try:
                if isinstance(indices[0], list):
                    indices = indices[0]
                sel = ds[coord].values[indices]
            except IndexError as e:
                C.LOG.warning(f"Unable to select {sel} due to {e}!")
                sel = ds[coord].values[[-1]]

        if to_cross:
            sel = [sel.tolist()]

    if is_xarray:
        sel = xr.DataArray(sel, dims="idx")
    return sel


def _preprocess_sel(ds, coord, sel, nulls, is_irregular):
    ds[coord] = ds[coord].load()
    if isinstance(sel, np.ndarray):
        sel = sel.tolist()
    elif isinstance(sel, (xr.DataArray, range)):
        sel = list(sel)

    if isinstance(sel, list):
        sel_flat = np.array(sel).flat
        sel_item0 = sel_flat[0]
        if len(sel_flat) == 1:
            sel = sel_item0

    nulls[coord] = False
    if util._is_null(sel):
        nulls[coord] = True
        if "x" in ds.coords or "y" in ds.coords or "idx" in ds.coords:
            # do not want to use _point_y_x for 1 million points
            sel = slice(ds[coord].values.min(), ds[coord].values.max())
        else:
            sel = np.unique(ds[coord].values).tolist()
    elif not isinstance(sel, (slice, list)):
        if is_irregular:
            if not isinstance(sel, (list, slice)):
                sel = [sel]
        else:
            sel = [sel]

    if isinstance(sel, slice):
        sel = _reorder_start_stop(sel)

    return sel


def _rstrip_index(sel):
    to_slice = False
    to_cross = False
    if isinstance(sel, list) and isinstance(sel[0], list):
        sel_list = sel[0]
        to_cross = True
    elif isinstance(sel, slice):
        sel_list = [sel.start, sel.stop, sel.step]
        to_slice = True
    elif isinstance(sel, Iterable) and not isinstance(sel, str):
        sel_list = np.array(sel).tolist()
    else:
        sel_list = sel.copy()

    if any("i" in str(val) for val in sel_list):
        sel = [
            int(ind.rstrip("i")) if isinstance(ind, str) else ind for ind in sel_list
        ]
        if to_slice:
            sel = slice(*sel)
        elif to_cross:
            sel = [sel]
    else:
        sel = []
    return sel


def magnitude_round(ds, sel, coord):
    diff = np.diff(ds[coord]).min()
    if diff == 0:
        diff = 1
    magnitude = np.floor(np.log10(diff))
    if magnitude >= 0 or np.isnan(magnitude):
        magnitude = 0
    else:
        magnitude = abs(int(magnitude))
    sel = pd.unique(sel.round(magnitude))
    return sel


def sel_y_x(
    ds,
    sel_y,
    sel_x,
    method="nearest",
    regrid_kwds=None,
    interp_kwds=None,
    var_y="y",
    var_x="x",
):
    """Subsets multidimensional lat/lon"""
    points = {}
    cross = {}
    slices = {}
    nulls = {}

    is_irregular = "idx" in ds.dims
    for coord, sel in {var_y: sel_y, var_x: sel_x}.items():
        sel = _preprocess_sel(ds, coord, sel, nulls, is_irregular)

        if coord == var_y:
            sel_y = sel
        else:
            sel_x = sel

        points[coord] = False
        cross[coord] = False
        slices[coord] = False
        if isinstance(sel, list) and isinstance(sel[0], list):
            cross[coord] = True
        elif isinstance(sel, slice):
            slices[coord] = True
        elif util._is_null(sel):
            nulls[coord] = True
        else:
            points[coord] = True
        C.LOG.debug(f"{coord} categorized!")

    is_rectilinear = all(coord in ds[coord].dims for coord in [var_y, var_x])
    C.LOG.debug(
        f"var_y: {var_y}, var_x: {var_x}, "
        f"rectilinear: {is_rectilinear}, points: {points}, "
        f"cross: {cross}, slices: {slices}, regrid_kwds: {regrid_kwds}"
    )

    pre_regrid = is_irregular and regrid_kwds
    if pre_regrid:
        interp_type = method if method in C.METHODS["interp"] else "nearest"
        if regrid_kwds is None:
            regrid_kwds = {"interp_type": interp_type, "hres": 0.5}
        C.LOG.debug(ds)

        try:
            if not nulls[var_y]:
                min_y = np.nanmin(np.ravel(sel_y)) - 10
                max_y = np.nanmax(np.ravel(sel_y)) + 10
                ds = _slice_sel(ds, slice(min_y, max_y), var_y)
        except Exception as e:
            C.LOG.warning(f"Unable to optimize pre-regrid for lat due to {e}")

        try:
            if not nulls[var_x]:
                min_x = np.min(np.ravel(sel_x)) - 10
                max_x = np.max(np.ravel(sel_x)) + 10
                ds = _slice_sel(ds, slice(min_x, max_x), var_x)
        except Exception as e:
            C.LOG.warning(f"Unable to optimize pre-regrid for lon due to {e}")

        ds, base_coords, var_coords = util._reset_var_coords(ds, var_y, var_x)
        interp_args = (var_y, var_x, regrid_kwds)
        try:
            if var_y == "lat" and var_x == "lon":
                ds = _interp_to_regular_grid_xesmf(ds, regrid_kwds)
            else:
                ds = ds.map(_interp_to_regular_grid_metpy, args=interp_args)
        except ImportError as e:
            C.LOG.warning(e)
            ds = ds.map(_interp_to_regular_grid_metpy, args=interp_args)
        ds = util._set_var_coords(ds, base_coords, var_coords)
        is_rectilinear = True

    if all(nulls.values()):
        return ds

    indices_y = _rstrip_index(sel_y)
    indices_x = _rstrip_index(sel_x)
    if not is_rectilinear and not is_irregular:
        sel_y = _index_to_value(ds, sel_y, var_y, indices_y, indices_2=indices_x)
        sel_x = _index_to_value(ds, sel_x, var_x, indices_y, indices_2=indices_x)
    else:
        sel_y = _index_to_value(ds, sel_y, var_y, indices_y)
        sel_x = _index_to_value(ds, sel_x, var_x, indices_x)

    if slices[var_y] and not isinstance(sel_y, slice):
        sel_y = slice(np.min(sel_y), np.max(sel_y))

    if slices[var_x] and not isinstance(sel_x, slice):
        sel_x = slice(np.min(sel_x), np.max(sel_x))

    C.LOG.debug(sel_y)
    C.LOG.debug(sel_x)

    if interp_kwds is None:
        interp_kwds = {"method": method}
    elif "interp_type" in interp_kwds:
        interp_kwds["method"] = interp_kwds.pop("interp_type")
    elif "method" not in interp_kwds:
        interp_kwds["method"] = method

    if cross[var_y] and cross[var_x]:
        ds = util.cross_section(
            ds, sel_y, sel_x, var_y=var_y, var_x=var_x, **interp_kwds
        )
    elif is_rectilinear:
        if points[var_y] and points[var_x]:
            sel_y, sel_x = _prod_y_x(sel_y, sel_x)
            sel_y = xr.DataArray(sel_y, dims="idx")
            sel_x = xr.DataArray(sel_x, dims="idx")
        ds = _sel_dims(ds, {var_y: sel_y, var_x: sel_x}, method)
    elif method not in C.METHODS["select"]:
        ds = _interp_y_x(ds, var_y, var_x, sel_y, sel_x, interp_kwds)
    elif slices[var_y] and slices[var_x]:
        ds = _slice_y_x(ds, var_y, var_x, sel_y, sel_x)
    elif points[var_y] and points[var_x]:
        ds = _point_y_x(ds, var_y, var_x, sel_y, sel_x, to_irregular=True)
    elif slices[var_y] and points[var_x]:
        sel_y = _slice_to_point(ds, var_y, var_x, sel_y, var_y)
        if nulls[var_y]:
            sel_y = magnitude_round(ds, sel_y, var_y)
        ds = _point_y_x(ds, var_y, var_x, sel_y, sel_x)
    elif slices[var_x] and points[var_y]:
        sel_x = _slice_to_point(ds, var_y, var_x, sel_x, var_x)
        if nulls[var_x]:
            sel_x = magnitude_round(ds, sel_x, var_x)
        ds = _point_y_x(ds, var_y, var_x, sel_y, sel_x)
    return ds


def sel_lat_lon(
    ds,
    sel_lat,
    sel_lon,
    method="nearest",
    regrid_kwds=None,
    interp_kwds=None,
    var_y="y",
    var_x="x",
):
    return sel_y_x(
        ds,
        sel_lat,
        sel_lon,
        method=method,
        regrid_kwds=regrid_kwds,
        interp_kwds=interp_kwds,
        var_y="lat",
        var_x="lon",
    )


def _sel_dims(ds, indexers_kwds, method):
    for dim, sel in indexers_kwds.items():
        if not isinstance(sel, (slice, Iterable)) or isinstance(sel, str):
            sel = [sel]
        elif isinstance(sel, list):  # handle [slice(0, 90)]
            if len(sel) == 1 and isinstance(sel[0], (slice, list)):
                sel = sel[0]
            if len(ds[dim]) == 1:
                sel = [sel[0]]

        indices = _rstrip_index(sel)
        sel = _index_to_value(ds, sel, dim, indices)

        if isinstance(sel, slice):
            method = None
            util._is_within(ds, dim, sel.start, "min")
            util._is_within(ds, dim, sel.stop, "max")
            if sel.step is not None:
                sel = _slice_to_range(sel)
        else:
            util._is_within(ds, dim, sel, "min")
            util._is_within(ds, dim, sel, "max")

        if dim == "tau" and not isinstance(sel, slice):
            sel = pd.to_timedelta(sel)
        indexers_kwds[dim] = sel

    if method in C.METHODS["select"] or method is None:
        ds_method = getattr(ds, "sel")
    elif method in C.METHODS["reindex"]:
        ds_method = getattr(ds, "reindex")
    elif method in C.METHODS["interp"]:
        ds_method = getattr(ds, "interp")
    else:
        raise NotImplementedError(f"{method} unavailable; select from: {C.METHODS}")

    try:
        ds = ds_method(**indexers_kwds, method=method)
    except TypeError:
        for dim, sel in indexers_kwds.items():
            ds = xr.concat((ds_method(**{dim: s}, method=method) for s in sel), dim)

    # remove duplicates
    if dim in ds.dims:
        _, index = np.unique(ds[dim], return_index=True)
        ds = ds.isel(**{dim: index})
    return ds


def sel_dims(ds, dim, sel, method="nearest"):
    if dim not in ds.dims or util._is_null(sel):
        return ds
    ds = _sel_dims(ds, {dim: sel}, method)
    return ds


def sel_level(ds, sel_lev, method="nearest"):
    nulls = {}
    sel_lev = _preprocess_sel(ds, "lev", sel_lev, nulls, False)
    if nulls["lev"]:
        return ds

    if "lev" in ds["lev"].dims:
        ds = sel_dims(ds, "lev", sel_lev, method=method)
        return ds
    elif len(np.atleast_1d(ds["lev"])) == 1:
        return ds

    indices = _rstrip_index(sel_lev)
    sel_lev = _index_to_value(ds, sel_lev, "lev", indices)
    util._is_within(ds, "lev", sel_lev, "min")
    util._is_within(ds, "lev", sel_lev, "max")
    if isinstance(sel_lev, slice):
        ds = _slice_sel(ds, sel_lev, "lev")
    else:
        lev_array = ds["lev"]
        icoord = sum(np.isclose(lev_array, lev) for lev in sel_lev) > 0
        if not np.any(icoord):
            # do this in two pass, find the nearest value first
            icoord = [(np.abs(lev_array - lev)).argmin() for lev in sel_lev]
            sel_lev = ds.isel(**{"idx": icoord})["lev"]
            # now find all the indices that match that value
            icoord = sum(np.isclose(lev_array, lev) for lev in sel_lev) > 0
        ds = ds.isel(**{"idx": icoord})
    return ds
