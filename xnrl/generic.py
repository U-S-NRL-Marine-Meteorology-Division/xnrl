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
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import xarray as xr
from xnrl import constant as C
from xnrl import packages, select, util
from xnrl.internal import NRLFile


class GenericFile(NRLFile):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        packages._import_metpy()
        self._generic_ds = None

    def _extract_meta(self, path_df):
        if self.model.lower() in [C.RAOB, C.RADIANCE]:
            return path_df

        kwds = dict(combine="by_coords")
        if self._file_type == C.GRIB:
            if self._grib_name == "cfgrib":
                kwds.update(
                    dict(
                        engine="cfgrib",
                        backend_kwargs={"errors": "ignore", "indexpath": ""},
                    )
                )
            elif self._grib_name == "pynio":
                kwds.update(dict(engine="pynio"))
            else:
                raise ValueError(
                    "Cannot use pynio to open generic datasets! "
                    "please use cfgrib or pynio!"
                )

        ds_list = []
        paths = path_df["path"]
        for path in paths:
            for engine in ["netcdf4", "h5netcdf"]:
                kwds["engine"] = engine
                try:
                    try:
                        ds = xr.open_mfdataset(path, **kwds)
                    except ValueError:
                        ds = self._decode_time(
                            xr.open_mfdataset(path, decode_times=False, **kwds)
                        )
                except Exception:
                    pass
            if self.fields is not None:
                only_field = len(ds.data_vars) == 1 and len(self.fields) == 1
                from_field = list(ds.data_vars)[0]
                to_field = self.fields[0]
                if only_field and from_field != to_field and to_field != "*":
                    C.LOG.warning(f"The field {from_field} was renamed to {to_field}!")
                    ds = ds.rename({from_field: to_field})
            ds_list.append(ds)

        try:
            try:
                ds = xr.combine_by_coords(ds_list)
            except ValueError:
                ds = xr.combine_nested(ds_list, concat_dim="exp")
        except xr.MergeError as e:
            C.LOG.warning(f"Attempting to override coordinates due to {e}")
            combine_kwds = dict(compat="override", coords="minimal")
            try:
                ds = xr.combine_by_coords(ds_list, **combine_kwds)
            except ValueError:
                ds = xr.combine_nested(ds_list, concat_dim="exp", **combine_kwds)

        self._attrs.update({**ds.attrs})

        try:
            ds = self._assign_coord(
                ds,
                "ini",
                np.atleast_1d(ds["time"].values)[0],
                find_coords=["ini", "init", "init_time", "S", "initial_time"],
            )
            found_ini = True
        except KeyError:
            found_ini = False  # try after assigning time

        ds = self._assign_coord(
            ds,
            "time",
            datetime.utcnow(),
            axis="time",
            find_coords=["valid_time", "valid", "target", "time"],
        )

        if not found_ini:
            ds = self._assign_coord(
                ds,
                "ini",
                np.atleast_1d(ds["time"].values)[0],
                find_coords=["ini", "init", "init_time", "S", "initial_time"],
            )

        ds = self._assign_coord(ds, "lat", [C.MASK_VALUE], axis="y")
        ds = self._assign_coord(ds, "lon", [C.MASK_VALUE], axis="x")
        ds = self._assign_coord(
            ds,
            "tau",
            ds["time"] - ds["ini"],
            find_coords=["step", "tau", "fcst_hour", "L"],
        )
        ds = self._assign_coord(
            ds, "lev", [0], axis="vertical", find_coords=["surface"]
        )
        ds = self._assign_coord(
            ds,
            "mbr",
            C.MASK_VALUE,
            find_coords=["mbr", "member", "ens", "mem", "M", "ensmem"],
        )

        if self.temporal_dim == "tau":
            if "time" in ds.dims and len(np.atleast_1d(ds["ini"])) == 1:
                if "ini" in ds.dims:
                    ds = ds.squeeze("ini")
                ds = ds.swap_dims({"time": "tau"})
            elif "time" in ds.dims:
                ds = xr.concat(
                    (
                        ds.sel(**{"ini": ini}).swap_dims({"time": "tau"})
                        for ini in ds["ini"].values
                    ),
                    "ini",
                )

        for coord in ["ini", "mbr", "exp"]:
            if coord not in ds.dims:
                ds = ds.expand_dims(coord)

        if len(paths) == len(ds["exp"]):
            default_exps = [
                util._parse_exp(os.path.splitext(path)[0].replace("./", "")).strip("_")
                for path in paths
            ]
        else:
            default_exps = [f"{C.UNTITLED}_{i}" for i in range(len(ds["exp"]))]
        ds["exp"] = np.atleast_1d(ds.attrs.get("dataset_title", default_exps))

        ds = self._chunk_xarray(ds)

        for coord in ds.coords:
            if ds[coord].values.ndim == 1:
                ds = ds.sortby(coord)

        for dim in C.BASE_DIMS:
            if dim in ds.dims:
                ds = select.sel_dims(ds, dim, getattr(self, dim + "s"))

        if self.fields:
            if not isinstance(self.fields, list):
                self.fields = [self.fields]
            fields = [field for field in self.fields if field in ds.data_vars]
            if fields:
                ds = ds[fields]

        for base_dim in C.BASE_DIMS[1:]:
            if base_dim not in ds.coords:
                ds = ds.assign_coords(**{base_dim: C.MASK_VALUE})
            if base_dim not in ds.dims:
                ds = ds.expand_dims(base_dim)

        # reorder the dimensions
        ds = ds.transpose(*C.BASE_DIMS[1:] + list(set(ds.dims) - set(C.BASE_DIMS)))
        self._generic_ds = ds

        meta_df = path_df
        meta_df["ini"] = C.MASK_VALUE
        meta_df["tau"] = timedelta(0)
        meta_df["exp"] = C.UNTITLED
        meta_df["mbr"] = C.MASK_VALUE
        meta_df["lev"] = C.MASK_VALUE
        meta_df["lev_type"] = C.UNTITLED
        meta_df["grid_dim"] = C.UNTITLED
        meta_df["field"] = [f"C.UNTITLED_{i}" for i in range(len(path_df))]
        self._attrs[C.UNTITLED]["crs"] = {
            "grid_mapping_name": "latitude_longitude",
            "semi_major_axis": 6371000.0,
        }
        return meta_df

    def _generate_array(self, cube_df):
        pass

    def _reshape_array(self, field_df, dims, array):
        pass

    @staticmethod
    def _decode_time(ds):
        for coord in ds.coords:
            if "since" in ds[coord].attrs.get("units", ""):
                time_meta = ds[coord].attrs["units"].split(" since ")
                units = time_meta[0]
                start = pd.to_datetime(time_meta[-1])
                ds[coord] = [
                    start + pd.DateOffset(**{units: value})
                    for value in ds[coord].values
                ]
        return ds

    @staticmethod
    def _assign_coord(ds, coord, value, axis=None, find_coords=None):
        if coord in ds.data_vars:
            ds = ds.set_coords(coord)

        if find_coords is None:
            find_coords = []

        for coord_name in find_coords:
            if coord_name in ds.coords:
                ds = ds.rename({coord_name: coord})
                return ds

        if axis is not None:  # find by axis label rather than coord name
            try:
                da = ds[list(ds.data_vars)[0]]
                coord_name = da.metpy.find_axis_name(axis)
                ds = ds.rename({coord_name: coord})
                return ds
            except AttributeError as e:
                C.LOG.warning(
                    f"Unable to find {coord} using metpy axis {axis} due to {e}!"
                )

        ds.coords[coord] = value

        return ds

    def _create_dataset(self, cube_df):
        return self._generic_ds
