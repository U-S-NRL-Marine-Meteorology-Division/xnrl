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

import concurrent.futures as cf
import os
from collections import OrderedDict, defaultdict
from collections.abc import Iterable
from copy import deepcopy

import numpy as np
import pandas as pd
import xarray as xr
from xnrl import constant as C
from xnrl import packages, select, util

pd.set_option("display.max_colwidth", 100)
pd.set_option("display.max_columns", None)


class NRLFile(object):
    def __init__(self, **kwargs):
        self.model = kwargs["model"]
        self.lev_types = kwargs.get("lev_types")
        self.grid_dims = kwargs.get("grid_dims")
        self.fields = kwargs.get("fields")
        self.exps = kwargs.get("exps")
        self.mbrs = kwargs.get("mbrs")
        self.inis = kwargs.get("inis")
        self.taus = kwargs.get("taus")
        if isinstance(self.taus, (range, list)):
            try:
                self.taus = pd.to_timedelta(self.taus, unit="hour")
            except ValueError:
                pass
        self.times = kwargs.get("times")
        self.levs = kwargs.get("levs")
        self.lats = kwargs.get("lats")
        self.lons = kwargs.get("lons")
        self.ys = kwargs.get("ys")
        self.xs = kwargs.get("xs")
        self.exps_labels = kwargs.get("exps_labels")
        self.lons_range = kwargs.get("lons_range")
        self.wrap_df = kwargs.get("wrap_df")
        self.num_threads = kwargs.get("num_threads", C.NUM_THREADS)
        self.mask_value = kwargs.get("mask_value")
        self.leave_progress = kwargs.get("leave_progress")
        self.chunks = kwargs.get("chunks")
        self.only_meta = kwargs.get("only_meta", False)
        self.drop_inis = kwargs.get("drop_inis", False)
        self.max_files = kwargs.get("max_files", None)
        self.max_groups = kwargs.get("max_groups", None)
        self.error_out = kwargs.get("error_out", True)
        self.validate_size = kwargs.get("validate_size", True)
        self.method = kwargs.get("method", "nearest")
        self.regrid_kwds = kwargs.get("regrid_kwds")
        self.interp_kwds = kwargs.get("interp_kwds")
        self.temporal_dim = kwargs.get("temporal_dim", C.TEMPORAL_DIM)
        self.merge_lev_types = kwargs.get("merge_lev_types", False)
        self.stationary_coords = kwargs.get("stationary_coords", True)
        self.datahd_path = kwargs.get("datahd_path", None)

        self._base_dims = C.BASE_DIMS.copy()
        self._other_cols = []
        if self.temporal_dim == "time":
            tau_index = self._base_dims.index(C.TEMPORAL_DIM)
            self._base_dims[tau_index] = self.temporal_dim
        elif self.temporal_dim != "tau":
            raise NotImplementedError(
                f'"{self.temporal_dim}" temporal_dim is not supported! '
                f'Select either "tau" or "time" instead.'
            )

        self._attrs = defaultdict(lambda: {})
        self._coords = defaultdict(lambda: {})
        self._coords_nonstationary = defaultdict(lambda: {})
        self._encodings = defaultdict(lambda: {})
        self._grid_dim = None
        self._lev_type = None
        self._field = None
        if self.chunks is not None:
            self._dask = packages._import_dask()
        else:
            self._dask = None
        self._cartopy_crs = None
        self._cartopy_projection = None
        self._tqdm = packages._import_tqdm()

    def _ignore_meta(self, df):
        # placeholder for engines like COAMPS GrADS
        return df

    def _map_jobs(self, func, iterable, *args, **kwds):
        desc = kwds.pop("desc", "")
        unit = kwds.pop("unit", "")
        disable = kwds.pop("disable", None)
        warn = kwds.pop("warn", True)
        num_iterable = len(iterable)
        if disable is None:
            disable = num_iterable < 10
        num_workers = util._get_workers(num_iterable, self.num_threads)

        if num_iterable > 500 and self.num_threads == 1 and warn:
            C.LOG.warning(
                f"Currently, self.num_threads={self.num_threads} and "
                f"there are {num_iterable} iterables to process so you may want "
                f"to set num_threads to a larger number to process faster!"
            )

        with cf.ThreadPoolExecutor(max_workers=num_workers) as exc:
            output = list(
                self._tqdm(
                    exc.map(func, iterable, *args, **kwds),
                    desc=desc,
                    total=num_iterable,
                    unit=unit,
                    leave=self.leave_progress,
                    disable=disable,
                )
            )
        return output

    def _split_paths(self, paths):
        if isinstance(paths, str):
            paths = [paths]

        path_df = pd.DataFrame(
            (list(os.path.split(path)) + [path] for path in paths if path),
            columns=["directory", "file", "path"],
        )
        path_df = self._ignore_meta(path_df)

        if self.max_files is not None:
            C.LOG.debug(f"Limiting to {self.max_files} paths to open!")
            path_df = path_df.iloc[: self.max_files]

        if self.validate_size:
            paths = path_df["path"]
            path_df["size"] = self._map_jobs(
                os.path.getsize, paths, desc="validate size ", unit=" files"
            )
            is_empty = path_df["size"] == 0
            if is_empty.any():
                empty_paths = path_df.loc[is_empty, "path"].tolist()
                path_df = path_df.loc[~is_empty]
                num_empty = len(empty_paths)
                if num_empty > 10:
                    empty_paths = empty_paths[:5] + ["..."] + empty_paths[-5:]
                empty_str = "\n".join(empty_paths)
                C.LOG.warning(
                    f"Found {num_empty} files with 0 bytes; "
                    f"dropping them:\n{empty_str}"
                )
                if len(path_df) == 0:
                    raise ValueError(f"All input files are corrupted!")
            path_df = path_df.drop(columns=["size"])

        return path_df

    @staticmethod
    def _extract_mbrs(path_df):
        dirs = path_df["directory"]
        if f"{os.sep}mbr" in dirs.iloc[0]:
            inds = dirs.str.find(f"{os.sep}mbr")
            offsets = (4, 7)
        else:
            inds = dirs.str.find(f"{os.sep}m")
            offsets = (2, 5)
        mbrs = [util._parse_mbr(dir_, ind, offsets) for dir_, ind in zip(dirs, inds)]
        return mbrs

    @staticmethod
    def _extract_exps(path_df):
        dirs = path_df["directory"]
        replacements = {dir_: util._parse_exp(dir_) for dir_ in dirs.unique()}
        exps = dirs.map(replacements).fillna(C.UNTITLED)
        return exps

    def _process_prefix(self, meta_df, coord_prefix):
        if coord_prefix in self._model_meta["meta_name"][:2]:
            meta_df = meta_df.loc[~meta_df["field"].str.startswith(coord_prefix)]
            coord_name = coord_prefix[:3]
            coord_error = "warn"
        else:
            coord_name = coord_prefix
            coord_error = "ignore"
        coord_name = coord_name.lower()
        return coord_name, coord_error

    def _extract_meta(self, path_df):
        raise NotImplementedError("Do not call NRLFile directly!")

    @staticmethod
    def _trigger_warning(values):
        try:
            if isinstance(values, list):
                flat_values = np.array(values).flat[0]
                if isinstance(flat_values, str):
                    if flat_values == "*" or "i" in flat_values:
                        return False
            elif isinstance(values, str):
                if values == "*" or "i" in values:
                    return False
            elif isinstance(values, slice):
                if values.start == "*" or "i" in values.start:
                    return False
        except Exception:
            pass
        return True

    def _apply_select(self, meta_df, col, values):
        if isinstance(values, (Iterable, range)) and not isinstance(values, str):
            inds = meta_df[col].isin(values)
        elif isinstance(values, slice):
            start = values.start
            stop = values.stop
            step = values.step
            if col == "tau":
                if isinstance(start, int):
                    start = f"{start} hours"
                if isinstance(stop, int):
                    stop = f"{stop} hours"
                if isinstance(step, int):
                    step = f"{step} hours"
            if start and stop and step:
                selection = np.arange(start, stop + step, step)
                inds = meta_df[col].isin(selection)
            elif start and stop:
                inds = (meta_df[col] >= start) & (meta_df[col] <= stop)
            elif start:
                inds = meta_df[col] >= start
            elif stop:
                inds = meta_df[col] <= stop
        elif isinstance(values, (int, float, str, pd.Timestamp, pd.Timedelta)):
            if values == "*":
                return meta_df
            if col == "tau" and isinstance(values, int):
                values = f"{values} hours"
            if col == "lev_type" and self._file_type == C.FLATFILE:
                values = values[:3]
            inds = meta_df[col] == values
        else:
            return meta_df

        if col == "lev":
            single_lev_fields = [
                field
                for field, field_df in meta_df.groupby("field")
                if len(field_df["lev"].dropna().unique()) == 1
            ]
            inds |= meta_df["field"].isin(single_lev_fields)

        if inds.sum() == 0:
            if self._file_type != C.NETCDF and self._trigger_warning(values):
                C.LOG.warning(
                    f"Unable to locate any of the input values in {col} "
                    f"during the first pass of selecting! May succeed in "
                    f"the second pass of selecting as a xr.Dataset. "
                    f"Initially ignored input: {values}"
                )
            return meta_df
        else:
            return meta_df.loc[inds]

    def _pre_check(self, meta_df):
        paths = meta_df["tau"]
        num_paths = len(paths)
        if num_paths == 0 or all(pd.isnull(paths)):
            raise FileNotFoundError(f"Error with trying to read {paths}!")
        return meta_df

    @staticmethod
    def _drop_inis(meta_df):
        exp_inis = {
            exp: meta_df.loc[meta_df["exp"] == exp, "ini"].unique()
            for exp in meta_df["exp"].unique()
        }

        for i, vals in enumerate(exp_inis.values()):
            if i == 0:
                unique_inis = set(vals)
            else:
                unique_inis &= set(vals)

        meta_df = meta_df.loc[meta_df["ini"].isin(unique_inis)]
        return meta_df

    def _select_meta(self, meta_df):
        selections = {col: getattr(self, col + "s") for col in C.COLS_DIMS}
        for col, values in selections.items():
            if col in meta_df.columns:
                try:
                    meta_df = self._apply_select(meta_df, col, values)
                except Exception as e:
                    if self._trigger_warning(values):
                        C.LOG.warning(
                            f"Unable to subset {values} on {col} in "
                            f"first pass due to {e}!"
                        )
        if self.drop_inis:
            meta_df = self._drop_inis(meta_df)
        meta_df = self._pre_check(meta_df)
        return meta_df

    def _validate_hypercube(self, group_df):
        cols = ["path", "lev_type"] + self._base_dims + self._other_cols
        cols += [col for col in ["lev_idx", "group"] if col in group_df.columns]
        cube_df = pd.concat(
            (
                field_df[cols]
                .dropna()
                .drop_duplicates(self._base_dims)
                .set_index(self._base_dims)
                .to_xarray()
                .to_dataframe()
            )
            for _, field_df in group_df.groupby("field")
        ).reset_index()
        cube_df = cube_df.sort_values(self._base_dims[1:])
        return cube_df

    def _populate_metadata(self, ds):
        if self._grid_dim != C.UNTITLED:
            grid_dim = self._grid_dim
        elif "lon" in ds and "lat" in ds:
            num_xs = len(ds["lon"])
            num_ys = len(ds["lat"])
            grid_dim = f"{num_xs}x{num_ys}"
        else:
            grid_dim = C.UNTITLED

        ds.attrs.update(
            {"model": self.model, "lev_type": self._lev_type, "grid_dim": grid_dim}
        )

        for field in ds.data_vars:
            field_attrs = {
                "long_name": self._model_meta.get("long_name", {}).get(field, field),
                "standard_name": self._model_meta.get("standard_name", {}).get(
                    field, ""
                ),
                "short_name": self._model_meta.get("short_name", {}).get(field, ""),
                "units": self._model_meta.get("units", {}).get(field, ""),
                "grid_mapping": "crs",
            }
            field_attrs.update(**ds[field].attrs)
            ds[field].attrs.update(
                {
                    key: value
                    for key, value in field_attrs.items()
                    if value is not None and value != ""
                }
            )

            if "time" not in ds.coords:
                try:
                    ds.coords["time"] = ds["ini"] + ds["tau"]
                except Exception as e:
                    ds.coords["time"] = (
                        ("ini", "tau"),
                        [
                            [ini + pd.to_timedelta(tau.values) for tau in ds["tau"]]
                            for ini in xr.CFTimeIndex(ds["ini"].values)
                        ],
                    )
            elif "tau" not in ds.coords:
                ds.coords["tau"] = ds["time"] - ds["ini"]

        if self.exps_labels is not None:
            if isinstance(self.exps_labels, str):
                self.exps_labels = [self.exps_labels]
            try:
                ds.coords["exp"] = self.exps_labels
            except ValueError:
                num_exps = len(ds["exp"])
                num_labels = len(self.exps_labels)
                sub_labels = self.exps_labels[:num_exps]
                ds.coords["exp"] = sub_labels
                C.LOG.warning(
                    f"Found {num_exps} experiments, but passed {num_labels}; "
                    f"using only the first two labels: {sub_labels}"
                )
        else:
            ds.coords["exp"] = ds.coords["exp"].str.replace(os.sep, "_")
        ds.coords["crs"] = 0

        if "crs" in self._attrs.get(self._grid_dim, {}):
            crs = self._attrs.get(self._grid_dim)["crs"]
            crs.update(C.COORD_ATTR["crs"])
        else:
            crs = None

        not_lat_lon_dims = "lat" not in ds.dims or "lon" not in ds.dims
        if not_lat_lon_dims and "idx" not in ds.dims:
            ccrs, cf_proj = packages._import_projection()
            has_ccrs = ccrs is not None
            has_cfprojection = cf_proj is not None
            has_crs = crs is not None
            has_lat_lon = "lon" in ds and "lat" in ds
            if has_ccrs and has_cfprojection and has_crs and has_lat_lon:
                cf_projection = cf_proj(crs)
                cartopy_crs = cf_projection.to_cartopy()
                cartopy_projection = ccrs.Geodetic(cf_projection.cartopy_globe)
                try:
                    xs, ys = select.transform_points(
                        cartopy_crs,
                        cartopy_projection,
                        ds["lon"].values,
                        ds["lat"].values,
                    )
                    ds.coords["x"] = xs
                    ds.coords["y"] = ys
                except Exception as e:
                    C.LOG.warning(
                        f"Unable to generate x/y coordinates "
                        f"from lat/lon due to {e}!"
                    )

        for coord in ds.coords:
            if coord == "crs":
                if crs is not None:
                    ds[coord].attrs = crs
                continue
            elif "lev" in coord:
                if "sig_w" in ds.coords:
                    key = "sig_m"
                elif "sig_m" in ds.coords:
                    key = "sig_w"
                elif "_" in self._lev_type:
                    key = self._lev_type.split("_")[0]
                else:
                    key = self._lev_type
                if "lev_upper" in self._attrs.get(self._grid_dim, {}):
                    ds[coord].attrs["sig_upper"] = self._attrs[self._grid_dim][
                        "lev_upper"
                    ]
            else:
                key = coord
            coord_attrs = C.COORD_ATTR.get(key, {"long_name": key})
            coord_attrs.update(**ds[coord].attrs)
            ds[coord].attrs.update(**coord_attrs)

        coords_nonstationary = self._coords_nonstationary[grid_dim]
        if len(coords_nonstationary) > 1:
            ds.coords["iy"] = "lat", np.arange(len(ds["lat"]))
            ds.coords["ix"] = "lon", np.arange(len(ds["lon"]))
            ds = ds.swap_dims({"lat": "iy", "lon": "ix"})
            for coord, coord_array in coords_nonstationary.items():
                coord_origin = ds.get(coord)
                if coord_origin is None:
                    continue
                coord_dims = tuple([self.temporal_dim] + list(coord_origin.dims))
                try:
                    coord_array = coord_array.reshape(
                        coord_array.shape[0], *coord_origin.shape
                    )
                    ds[coord] = coord_dims, coord_array
                except Exception:
                    coord_array = np.array([pd.unique(arr) for arr in coord_array])
                    coord_array = coord_array.reshape(
                        coord_array.shape[0], *coord_origin.shape
                    )
                    ds[coord] = coord_dims, coord_array

        ds.encoding.update(**self._encodings)
        return ds

    def _postprocess_coords(self, field, field_df):
        dims = OrderedDict({dim: field_df[dim].unique() for dim in self._base_dims[1:]})
        coords = deepcopy(self._coords[self._grid_dim])

        if "lev_sig" in coords and "sig" in self._lev_type:
            attrs = self._attrs[self._grid_dim]
            if field == "wwwind":
                dims["lev"] = coords.pop("lev_sig_w")
                coords["sig_m"] = coords.pop("lev_sig")
            else:
                dims["lev"] = coords.pop("lev_sig")
                coords["sig_w"] = coords.pop("lev_sig_w")
            # sig formula for COAMPS: σ = H(z-zs) / (H-zs)
            # H top of atmos; zs surface / terrain
            sig_top = attrs["lev_upper"]
            sig_sfc = coords.get("terrht", ("lev", 0))
            sig_dim, sig_sfc = sig_sfc
            if isinstance(sig_sfc, int):
                sig_sfc = np.array([0])
                sig_lev = dims["lev"]
                sig_dim = "lev"
            else:
                sig_lev = (
                    np.tile(dims["lev"], sig_sfc.shape)
                    .reshape(*sig_sfc.shape[::-1], *dims["lev"].shape)
                    .T
                )
                sig_dim = ("lev", *sig_dim)
            coords["zht"] = ((sig_lev * (sig_top - sig_sfc)) / sig_top) + sig_sfc
            coords["zht"] = (sig_dim, coords["zht"])
        elif "lev_sig" in coords:
            coords.pop("lev_sig")
            coords.pop("lev_sig_w")

        if "y" in coords:
            if "lat" in coords and "lon" in coords:
                if isinstance(coords["lat"], tuple):
                    lats = pd.unique(coords["lat"][-1].ravel())
                    lons = pd.unique(coords["lon"][-1].ravel())
                else:
                    lats = pd.unique(coords["lat"].ravel())
                    lons = pd.unique(coords["lon"].ravel())

                if len(lats) == len(coords["y"]):
                    dims["lat"] = lats
                    dims["lon"] = lons
                    coords.pop("y")
                    coords.pop("x")
                    coords = {
                        key: (
                            [
                                v.replace("y", "lat").replace("x", "lon")
                                for v in value[0]
                            ],
                            value[1],
                        )
                        if isinstance(value, tuple) and value[1].ndim >= 2
                        else value
                        for key, value in coords.items()
                    }
            if "lat" not in dims and "lon" not in dims:
                dims["y"] = coords["y"]
                dims["x"] = coords["x"]
        elif "idx" not in coords and "lat" in coords and "lon" in coords:
            dims["lat"] = coords["lat"]
            dims["lon"] = coords["lon"]

        return dims, coords

    @staticmethod
    def _correct_shape(array, shape, dims, dim=None):
        if dim is None:
            index = shape.index(None)
            dim = list(dims)[index]
        else:
            index = C.BASE_DIMS.index(dim) - 1

        size = np.prod(shape[:index] + shape[index + 1 :])
        shape[index] = int(np.ceil(array.size / size))
        dims[dim] = np.arange(shape[index])
        return shape, dims

    def _reshape_array(self, field_df, dims, array):
        shape = [
            len(value) if value is not None else None for key, value in dims.items()
        ]

        if shape.count(None) == 1:
            shape, dims = self._correct_shape(array, shape, dims)

        if array.size != np.prod(shape):
            shape, dims = self._correct_shape(array, shape, dims, dim="lev")

        try:
            array = array.reshape(shape)
        except ValueError:
            field = field_df.reset_index()["field"][0]
            raise ValueError(
                f"Could not reshape {self._lev_type} {field} "
                f"{array.shape} into: {dict(zip(dims, shape))}"
            )
        return array

    def _stack_array(self, arrays, size, chunks):
        if chunks:
            array = self._dask.array.stack(
                [
                    self._dask.array.from_delayed(array, dtype=">f", shape=(size,))
                    for array in arrays
                ]
            )
        else:
            array = np.stack(arrays)
        if array.shape[-1] == 0:
            raise ValueError(f"Corrupted data found!")
        return array

    def _try_stack(self, arrays, sizes, chunks=False):
        size = sizes[0]
        try:
            array = self._stack_array(arrays, size, chunks)
        except ValueError as e:
            desc = f"{self._lev_type} {self._grid_dim} {self._field}"
            C.LOG.warning(f"Found some bad data for {desc}; attempting to resolve...")
            arrays = [
                np.full(size, np.nan) if array.size != size else array
                for array in arrays
            ]
            array = self._stack_array(arrays, size, chunks)
            if array.shape[0] == 1:
                raise ValueError(
                    f"All bad data found for {desc}; cannot continue due to {e}!"
                )
        return array

    def _get_sizes(self, num_paths):
        size = self._attrs[self._grid_dim]["size"]
        if "sig" in self._lev_type:
            if "lev_sig" in self._coords[self._grid_dim]:
                repeats = len(self._coords[self._grid_dim]["lev_sig"])
            else:
                repeats = 1

            if self._field == "wwwind":
                repeats += 1
            sizes = np.repeat(size * repeats, num_paths)
        else:
            sizes = np.repeat(size, num_paths)
        return sizes

    def _generate_array(self, field_df):
        paths = field_df["path"].values

        if self.chunks:
            read_data = self._dask.delayed(self._read_data, pure=True)
        else:
            read_data = self._read_data

        num_paths = len(paths)
        if num_paths == 0:
            return

        sizes = self._get_sizes(num_paths)
        desc = f"{self._lev_type} {self._grid_dim} {self._field}"
        arrays = self._map_jobs(read_data, paths, sizes, desc=desc, unit=" files read")
        array = self._try_stack(arrays, sizes, chunks=self.chunks)
        return array

    def _chunk_xarray(self, da):
        if self.chunks is None:
            return da

        if self.chunks == "auto":
            da = da.chunk({dim: "auto" for dim in da.dims})
        elif not isinstance(self.chunks, bool):
            da = da.chunk(self.chunks)
        return da

    def _create_dataset(self, cube_df):
        ds_dict = {}
        coord_vars = set([])
        coord_etcs = {}
        for i, (field, field_df) in enumerate(cube_df.groupby("field")):
            self._field = field
            dims, coords = self._postprocess_coords(field, field_df)

            array = self._generate_array(field_df)
            array = self._reshape_array(field_df, dims, array)
            coords.update(**dims)

            for coord, values in coords.copy().items():
                if coord not in dims and not isinstance(values, tuple):
                    coord_etcs[coord] = coords.pop(coord)

            da = xr.DataArray(array, coords=coords, dims=dims.keys(), name=field)
            da = self._chunk_xarray(da)
            if da.name in da.coords:
                coord_vars.add(da.name)
            if self.mask_value is not None:
                da = da.where(~np.isclose(da, self.mask_value))
                if "idx" in da.dims:
                    da = da.dropna("idx")
            ds_dict[field] = da

        try:
            for field, da in ds_dict.items():
                if len(da["lev"]) == 1 and 0 in da["lev"]:
                    ds_dict[field] = da.squeeze("lev")
        except KeyError:
            pass

        ds = xr.merge([da.drop_vars(coord_vars) for da in ds_dict.values()])
        if self.mask_value is not None:
            for var in list(ds.coords) + list(ds.data_vars):
                if var in ["tau", "ini", "time", "exp"]:
                    continue
                try:
                    ds[var] = ds[var].where(~np.isclose(ds[var], self.mask_value))
                except Exception as e:
                    C.LOG.warning(
                        f"Unable to apply mask {self.mask_value} to "
                        f"{self._lev_type} {self._grid_dim} {var} "
                        f"due to {e}!"
                    )

        try:
            ds = ds.assign_coords(**coord_etcs)
        except Exception:
            pass

        return ds

    def _sub_sel(self, ds):
        sel_map = {
            "exp": self.exps,
            "mbr": self.mbrs,
            "ini": self.inis,
            "tau": self.taus,
            "time": self.times,
        }
        if self.method is not None and self.method != "*":
            method = self.method
        else:
            method = "nearest"

        for dim, sel in sel_map.items():
            try:
                ds = select.sel_dims(ds, dim, sel, method=method)
            except Exception as e:
                C.LOG.warning(f"Error indexing {dim} with {sel} due to {e}")

        if "lev" in ds:
            lev_sel = getattr(self, "levs")
            ds = select.sel_level(ds, lev_sel, method=method)

        sub_lats = not util._is_null(self.lats)
        sub_lons = not util._is_null(self.lons)
        has_lat_lon = "lat" in ds and "lon" in ds

        sub_ys = not util._is_null(self.ys)
        sub_xs = not util._is_null(self.xs)
        has_y_x = "y" in ds and "x" in ds

        regrid_kwds = (self.regrid_kwds or {}).copy()
        if regrid_kwds:
            regrid_vars = regrid_kwds.pop("vars", "latlon")
            if has_y_x or regrid_vars == "xy":
                regrid = "y_x"
            else:
                regrid = "lat_lon"
        else:
            regrid = False

        if (sub_lats or sub_lons or regrid == "lat_lon") and has_lat_lon:
            lat_sel = self.lats
            try:
                lons_range = self.lons_range
                if lons_range is None:
                    lons_range = util._detect_lons_range(ds)
                lon_sel = util._shift_lons(self.lons, lons_range)
            except Exception as e:
                lon_sel = self.lons
                if not util._is_null(lon_sel):
                    C.LOG.warning(
                        f"Unable to shift lons selection: " f"{lon_sel} due to {e}"
                    )

            regrid_lat_lon_kwds = regrid_kwds if regrid == "lat_lon" else None
            ds = select.sel_lat_lon(
                ds,
                lat_sel,
                lon_sel,
                method=method,
                regrid_kwds=regrid_kwds,
                interp_kwds=self.interp_kwds,
            )

        if (sub_ys or sub_xs or regrid == "y_x") and has_y_x:
            y_sel = self.ys
            x_sel = self.xs
            regrid_y_x_kwds = regrid_kwds if regrid == "y_x" else None
            ds = select.sel_y_x(
                ds,
                y_sel,
                x_sel,
                method=method,
                regrid_kwds=regrid_y_x_kwds,
                interp_kwds=self.interp_kwds,
            )

        if self.fields:  # reorder in user input
            if isinstance(self.fields, str):
                self.fields = [self.fields]

            if len(ds.data_vars) == 1 and len(self.fields) == 1:
                from_field = list(ds.data_vars)[0]
                to_field = self.fields[0]
                if from_field != to_field and to_field != "*":
                    C.LOG.warning(f"The field {from_field} was renamed to {to_field}!")
                    ds = ds.rename({from_field: self._field})
            elif "*" not in self.fields:
                missing_fields = set(self.fields) - set(ds.data_vars)
                dataset_fields = list(ds.data_vars)
                num_missing = len(missing_fields)
                if num_missing == len(self.fields):
                    raise ValueError(
                        f"Could not find any of the following "
                        f"fields in dataset under all {self.lev_types} "
                        f"{missing_fields}; select from {dataset_fields}"
                    )
                elif num_missing:
                    C.LOG.warning(
                        f"Could not find some of the following "
                        f"fields in dataset under {self._lev_type} {missing_fields}; "
                        f"select from {dataset_fields}"
                    )
                all_fields = self.fields + [
                    field for field in ds.data_vars if field not in self.fields
                ]
                ds = xr.merge(
                    ds[field] for field in all_fields if field in dataset_fields
                ).assign_attrs(**ds.attrs)

        for dim in ds.dims:
            if len(ds[dim]) == 0:
                raise ValueError(
                    f"{dim}s subset returned an empty dataset! Ensure the subset "
                    f"matches the sort order of the dataset!"
                )

        return ds

    def _create_row(self, ds, lev_type=None, grid_dim=None, field=None):
        field_str = field or C.ROW_DLM.join(ds.data_vars)
        exp_str = C.ROW_DLM.join(np.atleast_1d(ds["exp"].values))
        mbr_str = C.ROW_DLM.join(np.atleast_1d(ds["mbr"].astype(str).values))
        lev_str = C.ROW_DLM.join(
            (util._format_number(lev) for lev in np.atleast_1d(ds["lev"].values))
        )
        try:
            ini_str = C.ROW_DLM.join(
                np.atleast_1d(ds["ini"].dt.strftime("%Y%m%d%H").values)
            )
        except TypeError:
            ini_str = C.ROW_DLM.join(np.atleast_1d(ds["ini"].astype(str).values))

        if self.temporal_dim == "tau":
            temporal_str = C.ROW_DLM.join(
                (
                    util._format_number(tau)
                    for tau in np.atleast_1d(
                        ds["tau"].astype(float).values / 3600 / 1e9
                    )
                )
            )
        else:
            try:
                temporal_str = C.ROW_DLM.join(
                    np.atleast_1d(ds["time"].dt.strftime("%Y%m%d%H").values)
                )
            except TypeError:
                temporal_str = C.ROW_DLM.join(
                    np.atleast_1d(ds["time"].astype(str).values)
                )

        row = {
            "lev_type": lev_type or self._lev_type,
            "grid_dim": grid_dim or self._grid_dim,
            "field": field_str,
            "exp": exp_str,
            "mbr": mbr_str,
            "ini": ini_str,
            self.temporal_dim: temporal_str,
            "lev": lev_str,
            "ds": ds,
            "model": self.model,
        }

        if field:
            row.pop("ds")

        return row

    def _replace_metadata(self, selc_df):
        selc_df["units"] = selc_df["field"].replace(self._model_meta.get("units", {}))
        selc_df["long_name"] = selc_df["field"].replace(
            self._model_meta.get("long_name", {})
        )
        return selc_df

    def _list_metadata(self, selc_df):
        selc_df = selc_df.copy()
        selc_df["model"] = self.model
        selc_df = self._replace_metadata(selc_df)
        return selc_df

    def _summarize_metadata(self, selc_df):
        selc_df = selc_df.drop(columns=["directory", "file", "path"]).copy()
        groupby_cols = C.BASE_COLS + ["field"]
        selc_df = pd.DataFrame(
            self._create_row(
                group_df.set_index(list(group_df.columns)).to_xarray(),
                lev_type=group[0],
                grid_dim=group[1],
                field=group[2],
            )
            for group, group_df in selc_df.groupby(groupby_cols)
        )
        selc_df = self._replace_metadata(selc_df)
        return selc_df

    def open_dataset(self, paths):
        path_df = self._split_paths(paths)
        C.LOG.debug("Split paths successfully!")
        meta_df = self._extract_meta(path_df)
        C.LOG.debug("Extracted meta successfully!")
        selc_df = self._select_meta(meta_df)
        C.LOG.debug("Selected meta successfully!")

        if self.only_meta and self.wrap_df is not None:
            C.LOG.warning("wrap_df is ignored when only_meta is set!")

        if self.only_meta == "list":
            return self._list_metadata(selc_df)
        elif self.only_meta:
            return self._summarize_metadata(selc_df)

        rows = []
        selc_groupby = selc_df.groupby(C.BASE_COLS)
        for i, (group, group_df) in enumerate(selc_groupby):
            try:
                self._lev_type, self._grid_dim = group[:2]
                cube_df = self._validate_hypercube(group_df)
                C.LOG.debug(
                    f"Validated hypercube for {self._lev_type} "
                    f"{self._grid_dim} successfully!"
                )
                ds = self._create_dataset(cube_df)
                C.LOG.debug(
                    f"Created dataset for {self._lev_type} "
                    f"{self._grid_dim} successfully!"
                )
                if len(ds.data_vars) == 0:
                    continue
                ds = self._populate_metadata(ds)
                C.LOG.debug(
                    f"Populated metadata for {self._lev_type} "
                    f"{self._grid_dim} successfully!"
                )
                ds = util.shift_lons(ds, self.lons_range)
                C.LOG.debug(
                    f"Shifted lons for {self._lev_type} "
                    f"{self._grid_dim} successfully!"
                )
                ds = self._sub_sel(ds)
                C.LOG.debug(
                    f"Sub-selected for {self._lev_type} "
                    f"{self._grid_dim} successfully!"
                )
                if "missing_dims" in xr.Dataset.transpose.__code__.co_varnames:
                    # Xarray 0.16 ignored by default. New versions raise exception by default.
                    ds = ds.transpose(*C.BASE_DIMS + [...], missing_dims="ignore")
                else:
                    # Backwards Compatible for transpose.
                    ds = ds.transpose(*C.BASE_DIMS + [...])
                num_groups = selc_groupby.ngroups
                if self.max_groups is not None and self.max_groups < num_groups:
                    num_groups = self.max_groups
                multi_rows = self.wrap_df is False and not self.merge_lev_types
                if multi_rows and num_groups > 1 and i == 0:
                    C.LOG.warning(
                        f"Must wrap xr.Datasets into a pd.DataFrame "
                        f"because there are {num_groups} groups!"
                    )
                elif not self.wrap_df and num_groups == 1:
                    return ds
                row = self._create_row(ds)
                C.LOG.debug(
                    f"Created row for {self._lev_type} "
                    f"{self._grid_dim} successfully!"
                )
            except Exception as e:
                msg = f"Unable to process {group}!"
                if not self.error_out:
                    C.LOG.warning(f"{msg} due to {e}")
                else:
                    msg = (
                        f"{msg} due to {e}; either fix bad data or set error_out=False!"
                    )
                    raise RuntimeError(msg)
                continue
            rows.append(row)
            if self.max_groups is not None and i + 1 >= self.max_groups:
                break

        df = pd.DataFrame(rows)
        if self.merge_lev_types:
            lev_type_label = "+".join(df["lev_type"].values)
            ds = xr.merge(df["ds"], compat="override", combine_attrs="override")
            ds.attrs["lev_type"] = lev_type_label
            if self.wrap_df:
                row = self._create_row(ds, lev_type=lev_type_label)
                df = pd.DataFrame([row])
                return df
            else:
                return ds
        else:
            return df


class StackedFile(NRLFile):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @staticmethod
    def _read_data(path, size, offset):
        try:
            return np.fromfile(path, dtype=">f", count=size, offset=offset)
        except Exception as e:
            if str(path) != "nan":
                C.LOG.warning(f"Could not open {path} due to {e}!")
            return np.full(size, np.nan)

    def _generate_array(self, field_df):
        field = field_df["field"].iloc[0]
        paths = field_df["path"].values
        offsets = field_df["offset"].fillna(0).astype(int).values

        if self.chunks:
            read_data = self._dask.delayed(self._read_data, pure=True)
        else:
            read_data = self._read_data

        num_paths = len(paths)
        if num_paths == 0:
            return

        sizes = np.repeat(self._attrs[self._grid_dim]["size"], num_paths)
        desc = f"{self._lev_type} {self._grid_dim} {field}"
        arrays = self._map_jobs(
            read_data, paths, sizes, offsets, desc=desc, unit=" datasets read"
        )
        array = self._try_stack(arrays, sizes, chunks=self.chunks)
        return array
