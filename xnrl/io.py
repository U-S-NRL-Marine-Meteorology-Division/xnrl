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

import numpy as np
import pandas as pd
import xarray as xr
import xnrl.constant as C
from xnrl import packages, util


def _assign_hdf5_attrs(hdf5_obj, attrs_dict):
    for key, value in attrs_dict.items():
        hdf5_obj.attrs[key] = str(value)
    return hdf5_obj


def export_hdf5(df, out_dir=C.WORK_DIR, out_fi=None, show=False):
    h5py = packages._import_h5py()
    ds_list = [df] if isinstance(df, xr.Dataset) else df["ds"].values

    ds_0 = ds_list[0]
    exp_str = ds_0["exp"].values[0]
    if "ini" in ds_0.coords:
        ini_dt = pd.to_datetime(ds_0["ini"].values[0])
    else:
        ini_dt = pd.to_datetime("2020-01-01")

    if "tau" in ds_0.coords:
        tau_str = util.strfdelta(ds_0["tau"].values[0], "PT{h:06d}H{m:02d}M")
    else:
        tau_str = "PT000000H00M"
    grid_dims = ["y", "x"] if "x" in ds_0.coords else C.GRID_DIMS.copy()

    global_attrs = ds_0.attrs
    grid_dim = global_attrs.pop("grid_dim")
    global_attrs["grid_dimensions"] = grid_dim
    global_attrs["experiment"] = exp_str
    global_attrs["initialization"] = ini_dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    global_attrs["tau"] = tau_str

    if out_fi is None:
        out_fi = C.FILE_FMTS[C.HDF5].format(
            exp=exp_str,
            grid_dim=grid_dim,
            ini=ini_dt.strftime("%Y%m%d%H%M"),
            tau=tau_str,
        )

    out_fp = os.path.join(out_dir, out_fi)
    if show:
        C.LOG.info(out_fp)
        C.LOG.info(ds_list)
        return out_fp

    with h5py.File(out_fp, "w") as out_hf:
        global_attrs_group = out_hf.create_group("global_attributes")
        global_attrs_group = _assign_hdf5_attrs(global_attrs_group, global_attrs)
        for coord in ["latitude", "longitude"]:
            coord_data = ds_0[coord[:3]].values.reshape((-1, 1))
            out_hf.create_dataset(coord, data=coord_data)

        for ds in ds_list:
            ds = ds.squeeze()
            if "lev" not in ds.dims:
                ds = ds.expand_dims("lev")

            if "idx" not in ds.dims:
                ds = ds.stack(**{"idx": grid_dims}).transpose("idx", "lev", ...)
            ds = ds.transpose("idx", "lev", ...)
            lev_type = ds.attrs.pop("lev_type")

            if lev_type != "untitled":
                group = out_hf.create_group(lev_type)
                hdf5_obj = group

                if lev_type == "native":
                    for native_coord in ["latitude", "longitude", "m2d"]:
                        native_data = ds[coord[:3]].values
                        if native_data.ndim != 2:
                            native_data = native_data.reshape(-1, 1)
                        group.create_dataset(native_coord, data=native_data)
            else:
                hdf5_obj = out_hf

            for var in ds.data_vars:
                da = ds[var]
                data = da.values
                if data.ndim != 2:
                    data = data.reshape(-1, 1)
                dataset = hdf5_obj.create_dataset(var, data=data)
                dataset = _assign_hdf5_attrs(dataset, da.attrs)
                if "lev" in da.coords:
                    dataset.attrs["Levels"] = da["lev"].values
                else:
                    dataset.attrs["Levels"] = np.array([0.0])
    return out_fp


def export_netcdf(df, out_dir=C.WORK_DIR, out_fi=None, show=False):
    ds_list = [df] if isinstance(df, xr.Dataset) else df["ds"].values

    auto_fi = out_fi is None
    out_fps = set([])
    for ds in ds_list:
        model = ds.attrs.get("model", C.UNTITLED)
        exps = "X".join(ds.get("exp", xr.DataArray([])).values)
        lev_type = ds.attrs.get("lev_type", C.UNTITLED)
        grid_dim = ds.attrs.get("grid_dim", C.UNTITLED)
        if auto_fi:
            out_fi = C.FILE_FMTS[C.NETCDF].format(
                model=model, lev_type=lev_type, grid_dim=grid_dim, exps=exps
            )
        out_fp = os.path.join(out_dir, out_fi)
        if not show:
            ds.to_netcdf(out_fp)
        else:
            C.LOG.info(ds)
        out_fps.add(out_fp)
    return out_fps


def _export_flatfile_datahd(ds, add_out_dir, metadata):
    if ds.attrs["lev_type"].startswith("pre"):
        lm = len(ds["lev"])
    else:
        lm = C.MASK_VALUE

    if ds.attrs["lev_type"].startswith("sig"):
        kk = len(ds["lev"])
    else:
        kk = C.MASK_VALUE

    nproj = 1
    phnt1, phnt2 = ds["crs"].attrs.get(
        "standard_parallel", [C.MASK_VALUE, C.MASK_VALUE]
    )
    alnnt = ds["crs"].attrs.get("longitude_of_central_meridian", C.MASK_VALUE)
    reflat = ds["crs"].attrs.get("latitude_of_projection_origin", C.MASK_VALUE)
    reflon = ds["crs"].attrs.get("central_longitude", C.MASK_VALUE)

    if "x" in ds.dims:
        delx = np.diff(ds["x"]).mean()
    else:
        delx = C.MASK_VALUE

    if "y" in ds.dims:
        dely = np.diff(ds["x"]).mean()
    else:
        dely = C.MASK_VALUE

    nnest = ds.attrs.get("grid_num", 1)
    if "x" in ds.dims:
        num_xs = len(ds["x"])
    elif "lon" in ds.dims:
        num_xs = len(ds["lon"])
    else:
        num_xs = C.MASK_VALUE

    if "y" in ds.dims:
        num_ys = len(ds["y"])
    elif "lat" in ds.dims:
        num_ys = len(ds["lat"])
    else:
        num_ys = C.MASK_VALUE

    lat_ll = ds.isel(x=0, y=0)["lat"].item()
    lon_ll = ds.isel(x=0, y=0)["lon"].item()
    lat_lr = ds.isel(x=-1, y=0)["lat"].item()
    lon_lr = ds.isel(x=-1, y=0)["lon"].item()
    lat_ul = ds.isel(x=0, y=-1)["lat"].item()
    lon_ul = ds.isel(x=0, y=-1)["lon"].item()
    lat_ur = ds.isel(x=-1, y=-1)["lat"].item()
    lon_ur = ds.isel(x=-1, y=-1)["lon"].item()

    if ds.attrs["lev_type"] == "sig":
        if "sig_m" in ds.coords:
            sigma_m = ds["sig_m"]
            sigma_w = ds["lev"]
        else:
            sigma_m = ds["lev"]
            sigma_w = ds["sig_w"]
    sigma_w = np.diff(sigma_w.values[::-1])[::-1].copy()
    sigma_w.resize((40, 5))
    sigma_m = sigma_m.values.copy()
    sigma_m.resize((40, 5))

    datahd_df = pd.DataFrame({i: np.repeat(0.0, 400) for i in np.arange(0, 5)})
    datahd_df.iloc[0] = [lm, kk, nproj, phnt1, phnt2]
    datahd_df.iloc[1] = [alnnt, reflat, reflon, delx, dely]
    datahd_df.iloc[2, 0] = nnest
    # datahd_df.iloc[3]
    # datahd_df.iloc[4]
    datahd_df.iloc[5, 4] = num_xs
    datahd_df.iloc[6, 0] = num_ys
    datahd_df.iloc[7, 1:] = dely, delx, lat_ll, lon_ll
    datahd_df.iloc[8] = [lat_lr, lon_lr, lat_ul, lon_ul, lat_ur]
    datahd_df.iloc[9, :2] = [lat_ur, nnest]
    datahd_df.iloc[100:140] = sigma_w
    datahd_df.iloc[160:200] = sigma_m

    datahd_metadata = metadata.copy()
    datahd_metadata["field"] = "datahd"
    datahd_metadata["lev"] = 0
    datahd_metadata["lev_type"] = "sfc"
    datahd_fi = C.FILE_FMTS[C.FLATFILE][C.COAMPS].format(**datahd_metadata)
    datahd_fi = datahd_fi.replace("fcstfld", "infofld")
    datahd_fp = os.path.join(add_out_dir, datahd_fi)
    np.savetxt(datahd_fp, datahd_df.values, fmt="%13e")
    return datahd_fp


def _export_flatfile_coord(coord, coord_data, add_out_dir, metadata):
    coord_metadata = metadata.copy()
    coord_metadata["field"] = coord
    coord_metadata["lev"] = 0
    coord_metadata["lev_type"] = "sfc"
    coord_fi = C.FILE_FMTS[C.FLATFILE][C.COAMPS].format(**coord_metadata)
    coord_fp = os.path.join(add_out_dir, coord_fi)
    coord_data = coord_data.ravel()
    coord_f = np.memmap(coord_fp, ">f", mode="w+", shape=len(coord_data))
    coord_f[:] = coord_data
    return coord_fp


def export_flatfile(df, out_dir=C.WORK_DIR, show=False, fmt=None, sigma=False):
    ds_list = [df] if isinstance(df, xr.Dataset) else df["ds"].values
    for ds in ds_list:
        encoding = ds.encoding.copy()
        if "idx" in ds.coords:
            C.LOG.warning("Cannot export irregular grid; skipping...")
            continue
        else:
            grid_dims = ["y", "x"] if "x" in ds.coords else C.GRID_DIMS.copy()
            model = ds.attrs.get("model", C.NAVGEM).lower()

        if fmt is None:
            if model in [C.NEPTUNE_LAM, C.COAMPS]:
                fmt = C.COAMPS
            else:
                fmt = C.NAVGEM
        else:
            fmt = fmt.lower()
            if fmt not in [C.NAVGEM, C.COAMPS]:
                raise ValueError(f"Choose NAVGEM or COAMPS for fmt; got {fmt}")

        stack_dims = C.BASE_DIMS[1:]
        expand_dims = list(set(stack_dims) - set(ds.dims))

        lev_upper = ds["lev"].max().item()
        is_pascal = ds["lev"].attrs.get("units", "") == "Pa"
        if is_pascal or int(lev_upper) > 9999 and fmt == C.NAVGEM:
            ds["lev"] = ds["lev"] / 100  # max 4 digits
            lev_upper = ds["lev"].max().item()

        if ds.attrs["lev_type"].startswith("sig") or sigma:
            stack_dims.remove("lev")
            if "lev" not in grid_dims:
                grid_dims.append("lev")
            lev = ds["lev"].attrs.get("sig_upper", lev_upper)
            ds = ds.sortby("lev", ascending=False)
            lev_lower = ds["lev"].min().item()
            C.LOG.warning(
                f"Exporting sigma levels as top to bottom; i.e. "
                f"i=0 {lev_upper} and i=-1 {lev_lower}!"
            )

        metadata = ds.attrs
        if "x" not in metadata.get("grid_dim", ""):
            if "y" in ds.dims:
                num_x = len(ds["y"])
                num_y = len(ds["x"])
            else:
                num_x = len(ds["lon"])
                num_y = len(ds["lat"])

            if fmt == C.COAMPS:
                grid_num = ds.attrs.get("grid_num", 1)
                metadata["grid_dim"] = f"{grid_num}a{num_x:04d}x{num_y:04d}"
            else:
                metadata["grid_dim"] = f"glob{num_x:03d}x{num_y:03d}"

        if "missing_dims" in xr.Dataset.transpose.__code__.co_varnames:
            # Xarray 0.16 ignored by default. New versions raise exception by default.
            ds_pre = (
                ds.expand_dims(expand_dims)
                .drop_vars(["sig_w", "sig_m"], errors="ignore")
                .stack(**{"dims_": stack_dims})
                .transpose("dims_", *grid_dims, missing_dims="ignore")
            )
        else:
            # Backwards Compatible for transpose.
            ds_pre = (
                ds.expand_dims(expand_dims)
                .drop_vars(["sig_w", "sig_m"], errors="ignore")
                .stack(**{"dims_": stack_dims})
                .transpose("dims_", *grid_dims)
            )

        out_fps = set([])
        missing_fps = set([])
        for field in ds_pre.data_vars:
            field_name = field.replace("_", "")
            metadata.update(**{"field": field_name})
            for slice_ in ds_pre[field]:
                if "missing_dims" in xr.Dataset.transpose.__code__.co_varnames:
                    # Xarray 0.16 ignored by default. New versions raise exception by default.
                    slice_ = slice_.transpose(
                        *[
                            dim
                            for dim in C.BASE_DIMS + C.GRID_DIMS + C.XY_DIMS
                            if dim in slice_.dims
                        ],
                        missing_dims="ignore",
                    )
                else:
                    # Backwards Compatible for transpose.
                    slice_ = slice_.transpose(
                        *[
                            dim
                            for dim in C.BASE_DIMS + C.GRID_DIMS + C.XY_DIMS
                            if dim in slice_.dims
                        ]
                    )

                try:
                    exp, mbr, ini, tau, lev = np.atleast_1d(slice_["dims_"].values)[0]
                except ValueError:
                    exp, mbr, ini, tau = np.atleast_1d(slice_["dims_"].values)[0]

                try:
                    seconds = tau.total_seconds()
                except AttributeError:
                    seconds = tau
                mm, ss = divmod(seconds, 60)
                hh, mm = divmod(mm, 60)
                mbr = mbr if mbr != C.MASK_VALUE and np.isnan(mbr) == False else 0
                metadata.update(
                    **{"mbr": mbr, "ini": ini, "hh": hh, "mm": mm, "ss": ss, "lev": lev}
                )
                add_out_dir = os.path.join(out_dir, os.path.join(exp, f"m{mbr:03d}"))
                os.makedirs(add_out_dir, exist_ok=True)

                out_fi = C.FILE_FMTS[C.FLATFILE][fmt].format(**metadata)
                out_fp = os.path.join(add_out_dir, out_fi)
                if out_fp in out_fps:
                    continue

                data = slice_.values.ravel()
                if all(np.isnan(data)):
                    missing_fps.add(out_fp)
                    continue
                else:
                    out_fps.add(out_fp)

                if not show:
                    f = np.memmap(out_fp, ">f", mode="w+", shape=len(data))
                    f[:] = data
                    del f

                    if fmt == C.COAMPS:
                        # cp datahd file if exists
                        datahd_path = encoding.get(metadata["grid_dim"], {}).get(
                            "datahd"
                        )
                        if datahd_path is not None:
                            os.system(f"cp {datahd_path} {add_out_dir}")
                            datahd_fi = os.path.basename(datahd_path)
                            out_fps.add(os.path.join(add_out_dir, datahd_fi))
                        else:
                            datahd_fp = _export_flatfile_datahd(
                                ds, add_out_dir, metadata
                            )
                            out_fps.add(datahd_fp)

                        # create latitu/longit files
                        lats = slice_["lat"].values
                        lons = slice_["lon"].values
                        if lats.ndim == 1:
                            lats, lons = np.meshgrid(lats, lons)
                        coords = {"latitu": lats, "longitu": lons}
                        for coord, coord_data in coords.items():
                            coord_fp = _export_flatfile_coord(
                                coord, coord_data, add_out_dir, metadata
                            )
                            out_fps.add(coord_fp)
                else:
                    C.LOG.info(out_fi)
                    C.LOG.info(slice_)

    if len(missing_fps) > 0:
        missing_fps = "\n".join(sorted(missing_fps))
        C.LOG.warning(f"All-NaNs detected in:\n{missing_fps}")

    return sorted(out_fps)
