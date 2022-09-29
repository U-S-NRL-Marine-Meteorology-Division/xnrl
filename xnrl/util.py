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

import fnmatch
import glob
import os
from datetime import datetime

import numpy as np
import pandas as pd
import xarray as xr
import xnrl.constant as C
from packaging import version
from xnrl import packages


def _is_null(value):
    if value is None:
        return True
    elif isinstance(value, list):
        if "*" in value or None in value:
            return True
    elif isinstance(value, str):
        if value == "*":
            return True
    return False


def _expand_paths(paths):
    path_list = []
    for path in paths:
        if isinstance(path, list):
            path_list.extend(path)
        elif "*" in path:
            path_list.extend(sorted(glob.glob(path)))
        else:
            path_list.append(path)
    return path_list


def _wrangle_paths(paths):
    if isinstance(paths, str):
        paths = os.path.expandvars(os.path.expanduser(paths))

        if "*" in paths:
            return sorted(glob.glob(paths))
        else:
            return [paths]

    path_list = []
    path_list.extend(
        _expand_paths(
            _wrangle_paths(path) if isinstance(path, list) else path for path in paths
        )
    )
    return path_list


def sortby_coord(ds, coord_var):
    arr = ds[coord_var].values
    if arr.ndim == 1:
        if not np.all(arr[1:] >= arr[:-1]) and coord_var in ds.dims:
            ds = ds.reindex(**{coord_var: sorted(arr)})
    return ds


def _shift_lons(lon_arr, lons_range):
    if lon_arr is None or lons_range is None:
        return lon_arr

    if isinstance(lons_range, str):
        lons_range = int(lons_range)

    if isinstance(lon_arr, slice):
        start = _shift_lons(lon_arr.start, lons_range)
        stop = _shift_lons(lon_arr.stop, lons_range)
        step = lon_arr.step
        if start is not None and stop is not None:
            if start > stop:
                start, stop = stop, start
        if start == stop and start == -180:
            stop *= -1
        return slice(start, stop, step)
    elif not isinstance(lon_arr, xr.DataArray):
        lon_arr = np.array(lon_arr)

    if lons_range == 180:
        lon_arr = (lon_arr + 180) % 360 - 180
    elif lons_range == 360:
        lon_arr = (lon_arr - 360) % 360
    return lon_arr


def _detect_lons_range(ds):
    if isinstance(ds, (xr.Dataset, xr.DataArray)):
        if ds["lon"].min() < 0:
            lons_range_data = 180
        else:
            lons_range_data = 360
    else:
        if np.min(ds) < 0:
            lons_range_data = 180
        else:
            lons_range_data = 360
    return lons_range_data


def shift_lons(ds, lons_range):
    if lons_range is None:
        return ds

    if isinstance(lons_range, str):
        lons_range = int(lons_range)

    if not isinstance(ds, (xr.Dataset, xr.DataArray)):
        return _shift_lons(ds, lons_range)

    if lons_range not in [180, 360]:
        raise ValueError("lons_range only accepts either 180 or 360!")
    elif "lon" not in ds.coords:
        return ds

    lons_range_data = _detect_lons_range(ds)

    if lons_range_data == 360:
        ds["lon"] = xr.where(ds["lon"] == 360, 359.9999, ds["lon"])
    else:
        ds["lon"] = xr.where(ds["lon"] == 180, 179.9999, ds["lon"])

    if lons_range_data != lons_range:
        lons_attrs = ds["lon"].attrs
        ds["lon"] = _shift_lons(ds["lon"], lons_range)
        ds = sortby_coord(ds, "lon")
        ds["lon"].attrs = lons_attrs
    return ds


def _walk_dir(dir_, str_):
    desired_file = None
    if dir_ == "":
        dir_ = "."
    for file in os.listdir(dir_):
        if fnmatch.fnmatch(file, str_):
            desired_file = os.path.join(dir_, file)
            break
    return desired_file


def _locate_file(dir_, str_, error="ignore"):
    desired_file = _walk_dir(dir_, str_)
    if desired_file is None:
        up_dir_ = os.path.join(dir_, "..")
        desired_file = _walk_dir(up_dir_, str_)

    if desired_file is None and error in ["warn", "raise"]:
        abs_dir = os.path.abspath(".") if dir_ in ["", "."] else dir_
        exc_msg = (
            f"Could not find {str_}* file in {abs_dir}! "
            f"Ensure {str_} file is in the same directory (or one up)."
        )
        if error == "warn":
            C.LOG.warning(exc_msg)
        elif error == "raise":
            raise FileNotFoundError(exc_msg)
    return desired_file


def _decode(value):
    try:
        value = value.item()
    except (ValueError, AttributeError):
        pass

    if isinstance(value, (list, np.ndarray)):
        return [_decode(v) for v in value]

    if isinstance(value, bytes):
        return value.decode("utf-8")
    else:
        return value


def _explode_col(df, col):
    # https://stackoverflow.com/questions/32468402/
    # Flatten columns of lists
    col_flat = [
        item for sublist in np.atleast_1d(df[col]) for item in np.atleast_1d(sublist)
    ]
    # Row numbers to repeat
    scalars = (int, float, pd.Timedelta, pd.DateOffset)
    lens = [len(val) if not isinstance(val, scalars) else 1 for val in df[col]]
    vals = list(np.arange(df.shape[0]))
    ilocations = np.repeat(vals, lens)
    # Replicate rows and add flattened column of lists
    cols = [i for i, c in enumerate(df.columns) if c != col]
    new_df = df.iloc[ilocations, cols].copy()
    new_df[col] = col_flat
    return new_df


def _get_workers(num_paths, num_threads):
    return num_paths if num_threads > num_paths else num_threads


def strfdelta(tdelta, fmt):
    tdelta = pd.to_timedelta(tdelta)
    fmt_dict = {"days": tdelta.days}
    fmt_dict["h"], rem = divmod(tdelta.seconds, 3600)
    fmt_dict["m"], fmt_dict["s"] = divmod(rem, 60)
    return fmt.format(**fmt_dict)


def _parse_mbr(dir_, ind, offsets):
    try:
        return int(dir_[ind + offsets[0] : ind + offsets[1]])
    except ValueError:
        return C.MASK_VALUE


def _parse_exp(dir_):
    if not dir_:
        return C.UNTITLED

    labels = []
    for label in dir_.split(os.sep):
        if "_" in label:
            labels.extend(label.split("_"))
        else:
            labels.append(label)

    keywords = [
        os.environ["USER"],
        "users",
        "ftp",
        "receive",
        "work",
        "work1",
        "fs1",
        "scratch",
        "center",
        "home",
        "data",
    ]
    for keyword in keywords:
        if keyword in labels:
            labels = labels[labels.index(keyword) + 2 :]
            break

    this_century = int(str(datetime.utcnow().year)[:2])
    centuries = [str(this_century + i) for i in range(-1, 2)]
    seen = {"mbr": False, "dtg": False}
    labels_iter = labels.copy()
    for label in labels_iter:
        if not label:
            labels.remove(label)
            continue

        length = len(label)
        ends_with_digit = label[-1].isdigit()

        if not seen["mbr"]:
            is_mbr = (
                (label.startswith("m") and length == 4)
                or (label.startswith("mbr") and length == 6)
            ) and ends_with_digit
            if is_mbr:
                seen["mbr"] = True
                labels.remove(label)

        elif not seen["dtg"]:
            is_dtg = (
                any(label.startswith(century) for century in centuries) and length == 10
            ) and ends_with_digit
            if is_dtg:
                seen["dtg"] = True
                labels.remove(label)

    fillers = ["test", "data", "espc", "navgem", "fields", "output"]
    exp = "_".join(label for label in labels if label.lower() not in fillers)
    if exp == "":
        exp = C.UNTITLED
    return exp


def _format_number(num, fmt="03.2f"):
    try:
        if float(num).is_integer():
            int_fmt = fmt.split(".")[0]
            return f"{num:{int_fmt}.0f}"
        else:
            return f"{num:{fmt}}"
    except Exception:
        return num


def parse_iso8601_tau(tau):
    durations = {}
    # get all the alpha characters
    letters = [c for c in tau if not c.isdigit()]
    # the units are the suffix of the number
    # e.g. when iterating over PT00300S
    # T is not the unit of 00300, S is, so offset letters by 1
    units = {unit: letters[i + 1] for i, unit in enumerate(letters[:-1])}
    for c in tau:
        if c == letters[-1]:
            # reached the end; there are no numbers after
            break
        if not c.isdigit():
            unit = units[c]
            durations[unit] = ""  # create empty string to concat to
            continue
        durations[unit] += c  # concat the numbers as strings one by one
    # convert string duration to actual pandas object and sum all
    duration = pd.Series(
        [
            pd.to_timedelta(int(duration), unit=unit.replace("M", "min"))
            for unit, duration in durations.items()
            if duration != ""
        ]
    ).sum()
    return duration


def generate_grid(degrees, ds=None, lats_bounds=None, lons_bounds=None):
    if isinstance(lats_bounds, slice):
        lat_min = lats_bounds.start
        lat_max = lats_bounds.stop
    elif lats_bounds is not None:
        lat_min = lats_bounds[0]
        lat_max = lats_bounds[-1]
    elif ds is not None:
        lat_min = ds["lat"].min()
        lat_max = ds["lat"].max()
    else:
        lat_min = None
        lat_max = None

    if isinstance(lons_bounds, slice):
        lon_min = lons_bounds.start
        lon_max = lons_bounds.stop
    elif lons_bounds is not None:
        lon_min = lons_bounds[0]
        lon_max = lons_bounds[-1]
    elif ds is not None:
        lon_min = ds["lon"].min()
        lon_max = ds["lon"].max()
    else:
        lon_min = None
        lon_max = None

    if lat_min is not None and lat_max is not None:
        lat_range = np.arange(lat_min, lat_max + degrees, degrees)
    else:
        lat_range = np.arange(-90, 90 + degrees, degrees)

    if lon_min is not None and lon_max is not None:
        lons_range = np.arange(lon_min, lon_max + degrees, degrees)
    else:
        lons_range = np.arange(0, 360, degrees)

    ds2 = xr.Dataset({"lat": lat_range, "lon": lons_range})
    return ds2


def _check_coords(ds1, ds2):
    coord_bools = {}
    for coord in ["lat", "lon"]:
        coord_bools[coord] = {}
        coord1_len = len(ds1[coord])
        coord2_len = len(ds2[coord])
        coord1_min = ds1[coord].min()
        coord1_max = ds1[coord].max()
        coord2_min = ds2[coord].min()
        coord2_max = ds2[coord].max()
        coord_bools[coord]["len"] = coord1_len == coord2_len
        coord_bools[coord]["range"] = (
            coord1_min == coord2_min and coord1_max == coord2_max
        )
    return coord_bools


def _is_global(ds):
    lons_range = _detect_lons_range(ds)

    if lons_range != 180:
        if isinstance(ds, xr.DataArray):
            ds = ds.to_dataset()
        ds = shift_lons(ds.drop_vars(ds.data_vars), 180)

    step = np.diff(ds["lon"].values).max()
    low, hgh = -180 + step, 180 - step
    if ds["lon"].min() <= low and ds["lon"].max() >= hgh:
        periodic = True
    else:
        periodic = False

    return periodic


def _assign_cell_edges(ds):
    ds = ds.sortby("lat")
    ds = ds.sortby("lon")

    lat_res = ds["lat"].diff("lat").mean().values
    ds[f"lat_b"] = np.arange(
        ds["lat"].min() - lat_res / 2, ds["lat"].max() + lat_res, lat_res
    )
    lon_res = ds["lon"].diff("lon").mean().values
    ds[f"lon_b"] = np.arange(
        ds["lon"].min() - lon_res / 2, ds["lon"].max() + lon_res, lon_res
    )
    return ds


def _build_filename(ds1, ds2, method, peri, locstream_in, locstream_out):
    fmt = _detect_lons_range(ds1)
    if locstream_in and locstream_out:
        n_in = len(ds1["idx"])
        n_out = len(ds2["idx"])
        filename = f"{method}_{n_in}_{n_out}{peri}_{fmt}.nc"
    elif locstream_in:
        n_in = len(ds1["idx"])
        ny_out = len(ds2["lat"])
        nx_out = len(ds2["lon"])
        filename = f"{method}_{n_in}_{ny_out}x{nx_out}{peri}_{fmt}.nc"
    elif locstream_out:
        ny_in = len(ds1["lat"])
        nx_in = len(ds1["lon"])
        n_out = len(ds2["idx"])
        filename = f"{method}_{ny_in}x{nx_in}_{n_out}{peri}_{fmt}.nc"
    else:
        ny_in = len(ds1["lat"])
        nx_in = len(ds1["lon"])
        ny_out = len(ds2["lat"])
        nx_out = len(ds2["lon"])
        filename = f"{method}_{ny_in}x{nx_in}_{ny_out}x{nx_out}{peri}_{fmt}.nc"
    return filename


def regrid(
    ds1,
    ds2=None,
    degrees=1,
    method="bilinear",
    reuse_weights=True,
    periodic=None,
    filename=None,
    keep_attrs=True,
    check_nans=None,
):
    """Matches a dataset's grid to another. Adapted from xtools.

    Args:
        ds (xr.Dataset)
        ds2 (xr.Dataset)
        degrees (scalar): output resolution in degrees
        method (str): xESMF regridding method
        reuse_weights (bool): whether to reuse saved weights
        periodic (bool): whether the dataset is global
        filename (str): whether the dataset is global
        keep_attrs (bool): whether to keep attributes
        check_nans (bool): whether to check for NaNs

    Returns:
        xr.Dataset: dataset with shifted longitudes
    """
    xe = packages._import_xesmf()

    ds1 = ds1.copy()
    if ds2 is None:
        ds2 = generate_grid(degrees, ds=ds1)
    else:
        ds2 = ds2.copy()

    attrs = ds1.attrs
    coord_attrs = {coord_var: ds1[coord_var].attrs for coord_var in ds1.coords}

    coord_bools = _check_coords(ds1, ds2)
    if all(coord_bools["lat"].values()) and all(coord_bools["lon"].values()):
        return ds1

    if method == "conservative":
        ds1 = _assign_cell_edges(ds1)
        ds2 = _assign_cell_edges(ds2)

    if periodic is None and "lon" in ds1.dims and "lon" in ds2.dims:
        periodic = _is_global(ds1) and _is_global(ds2)
        peri = "_peri" if periodic else ""
    else:
        peri = ""

    locstream_in = not ds1["lat"].dims[0] == "lat" and "idx" in ds1.dims
    locstream_out = not ds2["lat"].dims[0] == "lat" and "idx" in ds2.dims

    ranges_match = coord_bools["lat"]["range"] and coord_bools["lon"]["range"]
    if check_nans or (check_nans is None and not ranges_match):
        offset = np.abs(ds1.min()) + 1
        ds1 = ds1 + offset
    else:
        offset = None

    if "idx" in ds1.dims:
        ds1 = ds1.dropna("idx")

    if "missing_dims" in xr.Dataset.transpose.__code__.co_varnames:
        # Xarray 0.16 ignored by default. New versions raise exception by default.
        ds1 = ds1.transpose(*[..., "lat", "lon"], missing_dims="ignore")
        ds2 = ds2.transpose(*[..., "lat", "lon"], missing_dims="ignore")
    else:
        # Backwards Compatible for transpose.
        ds1 = ds1.transpose(*[..., "lat", "lon"])
        ds2 = ds2.transpose(*[..., "lat", "lon"])

    if locstream_in and method in ["bilinear", "nearest"]:
        method = "nearest_s2d"

    if filename is None:
        filename = _build_filename(ds1, ds2, method, peri, locstream_in, locstream_out)

    filename_nc = None
    if version.parse(xe.__version__) > version.parse("0.3.0"):
        if not os.path.exists(filename):
            filename_nc = filename
            filename = None
            reuse_weights = False

    regridder = xe.Regridder(
        ds1,
        ds2,
        method=method,
        reuse_weights=reuse_weights,
        periodic=periodic,
        filename=filename,
        ignore_degenerate=True,
        locstream_in=locstream_in,
        locstream_out=locstream_out,
    )

    if filename_nc is not None:
        regridder.to_netcdf(filename_nc)

    if isinstance(ds1, xr.Dataset):
        ds1 = ds1.apply(regridder, keep_attrs=keep_attrs)
    else:
        ds1 = regridder(ds1, keep_attrs=keep_attrs)

    if offset is not None:
        ds1 = ds1.where(ds1 != 0) - offset

    if method == "conservative":
        ds1 = ds1.drop(["lat_b", "lon_b"])

    for coord_var in ds1.coords:
        ds1[coord_var].attrs.update(**coord_attrs[coord_var])

    ds1.attrs = attrs
    return ds1


def _is_within(ds, dim, sel, stat):
    try:
        if sel is not None:
            sel = np.array(sel)

            if dim == "lon":
                ds_lons_range = _detect_lons_range(ds)
                sel_lons_range = _detect_lons_range(sel)
                if ds_lons_range != sel_lons_range:
                    sel = _shift_lons(sel, ds_lons_range)
            bound = getattr(ds[dim], stat)().values
            check = "greater" if stat == "max" else "less"
            np_method = getattr(np, check)
            warn = np_method(sel, bound).any()
            if warn:
                if dim == "tau":
                    sel = pd.to_timedelta(sel)
                    bound = pd.to_timedelta(bound)
                sel = np.unique(sel[np_method(sel, bound)])
                C.LOG.warning(
                    f"Detected selections in {dim}, {sel}, "
                    f"is beyond the dataset's {stat}: "
                    f"{bound}!"
                )
    except Exception as e:
        C.LOG.debug(f"{e}")


def _reset_var_coords(ds, var_y, var_x):
    base_coords = ds.coords
    var_coords = (
        {coord for coord in ds.coords if ds[coord].ndim >= 2 or "idx" in ds[coord].dims}
        - set(ds.dims)
        - set([var_y, var_x, "time"])
    )

    ds = ds.reset_coords(var_coords)
    return ds, base_coords, var_coords


def _set_var_coords(ds, base_coords, var_coords):
    ds = ds.set_coords(var_coords)

    for key, val in base_coords.items():
        if key in ds.coords or key == "idx":
            continue
        elif "idx" in ds.dims and key in ["lat", "lon", "y", "x"]:
            continue
        if key not in ds.dims and key in C.BASE_DIMS:
            ds = ds.expand_dims(key)

        ds = ds.assign_coords(**{key: val.values})
    return ds


def _interp_prep(da, var_y, var_x):
    C.LOG.debug(f"Preparing to interpolate to regular grid for {da.name}!")
    if "x" in da.dims and "y" in da.dims and var_x != "x" and var_y != "y":
        da = da.stack({"idx": ["y", "x"]})
    elif "lon" in da.dims and "lat" in da.dims and var_x != "lon" and var_y != "lat":
        da = da.stack({"idx": ["lat", "lon"]})
    elif "idx" in da.dims:
        da = da.dropna("idx")

    dims = [dim for dim in da.dims if dim != "idx"]
    if len(dims) > 0:
        da_tmp = da.stack({"tmp_dim": dims})
    else:
        da_tmp = da.expand_dims("tmp_dim")

    coords = da_tmp.drop([var_y, var_x, "idx", "m2d"], errors="ignore").coords
    return da_tmp, dims, coords


def _interp_rebuild(da, var_y, var_x, data_list, dims, coords, to_irregular=False):
    if to_irregular:
        base_dims = ["tmp_dim", "idx"]
    else:
        base_dims = ["tmp_dim", var_y, var_x]

    da_interp = xr.DataArray(
        data_list,
        coords=coords,
        dims=base_dims,
        name=da.name,
        attrs=da.attrs,
    )
    if len(dims) > 0:
        if "missing_dims" in xr.Dataset.transpose.__code__.co_varnames:
            # Xarray 0.16 ignored by default. New versions raise exception by default.
            da_interp = da_interp.unstack("tmp_dim").transpose(
                *dims + base_dims[1:], missing_dims="ignore"
            )
        else:
            # Backwards Compatible for transpose.
            da_interp = da_interp.unstack("tmp_dim").transpose(*dims + base_dims[1:])
    else:
        da_interp = da_interp.squeeze("tmp_dim")
    return da_interp


def _cross_section_1d(da, var_y, var_x, ys, xs, method, num_steps):
    scipy = packages._import_scipy()
    interpolate = scipy.interpolate

    if method not in ["linear", "nearest"]:
        raise ValueError(
            f"For irregular grids, only nearest and linear supported, "
            f"but got {method}"
        )

    da_tmp, dims, coords = _interp_prep(da, var_y, var_x)

    if var_y == "lat" and var_x == "lon":
        cgeo = packages._import_geodesic()
        geod = cgeo.Geodesic()
        ini_coord, end_coord = np.concatenate((ys, xs))
        geod_inv = geod.inverse(ini_coord, end_coord)
        distance = geod_inv[0, 0]
        azimuth = geod_inv[0, 1]
        distances = np.linspace(0, distance, num=num_steps, endpoint=True)
        geod_dir = geod.direct(ini_coord, azimuth, distances)
        xs, ys = geod_dir[:, 0], geod_dir[:, 1]
    else:
        ys = np.linspace(*ys[0], num_steps)
        xs = np.linspace(*xs[0], num_steps)

    xy_coords = np.vstack((xs, ys)).T
    data_coords = np.vstack((da_tmp[var_x], da_tmp[var_y])).T
    tri = scipy.spatial.Delaunay(data_coords)

    data_list = []
    for tmp in da_tmp["tmp_dim"]:
        da_sub = da_tmp.sel(**{"tmp_dim": tmp})
        interp_args = tri, da_sub.values
        if method == "linear":
            interp_func = interpolate.LinearNDInterpolator(*interp_args)
        else:
            interp_func = interpolate.NearestNDInterpolator(*interp_args)
        data = interp_func(xy_coords)
        data_list.append(data)

    da_interp = _interp_rebuild(
        da, var_y, var_x, data_list, dims, coords, to_irregular=True
    )
    da_interp["idx"] = np.arange(len(da_interp["idx"]))
    da_interp[var_y] = "idx", ys
    da_interp[var_x] = "idx", xs
    return da_interp


def _cross_section_2d(ds, var_y, var_x, ys, xs, method, num_steps):
    interpolate = packages._import_interpolate()
    _, _ = packages._import_projection()  # needed to use metpy.parse_cf

    coords = np.concatenate((ys, xs))
    ds_cross = interpolate.cross_section(
        ds,
        coords[:, 0],
        coords[:, 1],
        steps=num_steps,
        interp_type=method,
    ).rename({"index": "idx"})
    remove_nan_list = [
        idx for idx in ds_cross["idx"].values if not np.isnan(ds_cross[var_y][idx])
    ]
    ds_cross = ds_cross.sel(idx=remove_nan_list)
    ds_cross["idx"] = np.arange(0, len(ds_cross["idx"]))
    lat_diff = ds_cross[var_y].diff("idx")
    lon_diff = ds_cross[var_x].diff("idx")
    idx_list = [
        idx + 1
        for idx in np.arange(len(lat_diff))
        if not (np.isclose(lat_diff[idx], 0) and np.isclose(lon_diff[idx], 0))
    ]
    if not idx_list:
        raise ValueError(f"No points available for lon, {xs}, or ys, {ys}!")

    if idx_list[0] == 1:
        # if first two coordinates are different
        # include 0th coordinate pair
        # this is necessary because
        # len(lat_diff) == len(ds_cross.lat("idx")) - 1
        idx_list.insert(0, 0)
    ds_cross = ds_cross.sel(idx=idx_list)
    ds_cross["idx"] = np.arange(0, len(ds_cross["idx"]))
    ds_cross = ds_cross.drop_vars("crs").assign_coords({"crs": ds["crs"]})
    return ds_cross


def cross_section(ds, ys, xs, var_y="lat", var_x="lon", method="linear", num_steps=100):
    var = None
    if isinstance(ds, xr.DataArray):
        var = ds.name
        ds = ds.to_dataset()

    ds = ds.metpy.parse_cf()
    ds, base_coords, var_coords = _reset_var_coords(ds, var_y, var_x)
    if not isinstance(ys[0], list):
        ys = [ys]

    if not isinstance(xs[0], list):
        xs = [xs]

    for dim, sel in {var_y: ys, var_x: xs}.items():
        _is_within(ds, dim, sel, "min")
        _is_within(ds, dim, sel, "max")

    ds_cross_list = []
    for y, x in zip(ys, xs):  # waypoints
        y = np.array(y).reshape(1, -1)
        x = np.array(x).reshape(1, -1)
        cross_args = (var_y, var_x, y, x, method, num_steps)
        if "idx" in ds.dims:
            ds_cross_part = ds.map(_cross_section_1d, args=cross_args)
        else:
            ds_cross_part = _cross_section_2d(ds, *cross_args)
        ds_cross_list.append(ds_cross_part)
    ds_cross = xr.concat(ds_cross_list, "idx")

    ds_cross = _set_var_coords(ds_cross, base_coords, var_coords)
    if var is not None:
        ds_cross = ds_cross[var]

    ds_cross.attrs = ds.attrs
    return ds_cross
