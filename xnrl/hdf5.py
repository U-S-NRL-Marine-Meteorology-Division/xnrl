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
from xnrl import constant as C
from xnrl import packages, util
from xnrl.flatfile import FlatFile
from xnrl.internal import NRLFile


class HDF5File(NRLFile):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._file_type = C.HDF5
        self._h5py = packages._import_h5py()
        self._data_mapping = {}
        self._placeholder = [{}] if not self.only_meta else ""

    def _unpack_dict(self, df):
        df = df.where(df.notna(), self._placeholder)
        unpack_df = pd.concat(
            {
                col: pd.DataFrame(df[col].values.tolist(), index=df.index)
                for col in df.columns
            },
            axis=1,
        )
        if any(unpack_df.iloc[:, 0].apply(isinstance, args=(dict,))):
            unpack_df = self._unpack_dict(unpack_df)
            return unpack_df
        return df

    def _wrap_dask(self, hds, field):
        if self.chunks:
            chunks = "auto" if isinstance(self.chunks, bool) else self.chunks
            data = self._dask.array.from_array(hds[field], chunks=chunks)
        else:
            data = hds[field][:]
        return data

    def _extract_meta(self, path_df):
        meta_dict = {}
        for i, path in enumerate(path_df["path"]):
            hf = self._h5py.File(path, mode="r")
            meta_dict = self._load_meta(i, hf, meta_dict)
            if not self.chunks:
                hf.close()

        meta_df = (
            pd.DataFrame(meta_dict)
            .transpose()
            .rename(columns=self._model_meta["varying_attrs"])
        )
        meta_df = meta_df.where(meta_df != "", self._placeholder)
        meta_df = meta_df.join(path_df)

        lev_types = list(
            set(meta_df.columns)
            - set(C.META_COLS)
            - set(C.COLS_DIMS)
            - set(self._model_meta["varying_attrs"].values())
        )

        field_df = (
            self._unpack_dict(meta_df[lev_types])
            .unstack()
            .rename_axis(["lev_type", "field", "index"])
            .reset_index()
            .set_index("index")
        )

        field_df = pd.concat(
            [
                field_df.drop(columns=0),
                pd.DataFrame(field_df[0].values.tolist(), index=field_df.index),
            ],
            axis=1,
        )

        meta_df = meta_df.drop(columns=lev_types).join(field_df, rsuffix="_")
        meta_df["time"] = meta_df["ini"] + meta_df["tau"]
        meta_df["mbr"] = C.MASK_VALUE
        if "long_name" not in meta_df.columns:
            meta_df["long_name"] = meta_df["field"]
        else:
            meta_df["long_name"] = meta_df["long_name"].fillna(meta_df["field"])

        if "units" not in meta_df.columns:
            meta_df["units"] = ""
        else:
            meta_df["units"] = meta_df["units"].fillna("")

        self._model_meta["long_name"] = dict(
            zip(meta_df["field"], meta_df["long_name"].fillna(""))
        )
        self._model_meta["units"] = dict(zip(meta_df["field"], meta_df["units"]))

        if "standard_name" in meta_df.columns:
            self._model_meta["standard_name"] = dict(
                zip(meta_df["field"], meta_df["standard_name"].fillna(""))
            )

        if "short_name" in meta_df.columns:
            self._model_meta["short_name"] = dict(
                zip(meta_df["field"], meta_df["short_name"].fillna(""))
            )

        meta_df["lev_idx"] = range(len(meta_df))
        meta_df = util._explode_col(meta_df, "levels")
        meta_df = meta_df.rename(columns={"levels": "lev"})
        meta_df["lev_idx"] = meta_df.groupby("lev_idx").cumcount()
        return meta_df

    def _pre_check(self, meta_df):
        meta_df = super()._pre_check(meta_df)
        for _ in range(0, 100):  # try up to 100 times or no more dups
            meta_duplicated = meta_df.duplicated(C.BASE_COLS + self._base_dims)
            if not meta_duplicated.any():
                break
            # if any duplicate exp names for a subset, suffix the dups with +1
            meta_df.loc[meta_duplicated, "exp"] += "+1"
        return meta_df

    def _read_data(self, path, lev_type, field):
        try:
            hf = self._h5py.File(path, "r")
            hds = hf if lev_type == C.UNTITLED else hf[lev_type]
            data = self._wrap_dask(hds, field)
            if not self.chunks:
                hf.close()
            return data
        except Exception as e:
            C.LOG.warning(f"Unable to read {path} {lev_type} {field} due to {e}!")
            return np.nan

    @staticmethod
    def _get_levels(shape, level):
        if not pd.isnull(np.atleast_1d(level)).any():
            return level
        elif len(shape) == 2:
            return range(shape[0])
        else:
            return [0]

    def _make_mapping(self, cube_df):
        unique_df = cube_df.drop_duplicates(subset=["path", "lev_type", "field"])
        paths = unique_df["path"]
        lev_types = unique_df["lev_type"]
        fields = unique_df["field"]

        num_paths = len(paths)
        if num_paths == 0:
            return

        desc = f"{self._lev_type} {self._grid_dim}"
        unique_arrays = self._map_jobs(
            self._read_data, paths, lev_types, fields, desc=desc, unit=" datasets read"
        )

        unique_keys = tuple(zip(paths, lev_types, fields))
        self._data_mapping = dict(zip(unique_keys, unique_arrays))

    def _create_dataset(self, cube_df):
        self._make_mapping(cube_df)
        ds = super()._create_dataset(cube_df)
        return ds

    def _populate_metadata(self, ds):
        ds = super()._populate_metadata(ds)
        ignore_keys = list(self._model_meta["varying_attrs"]) + [
            "is_restart",
            self._grid_dim,
        ]
        ds.attrs.update(
            **{
                key: value
                for key, value in self._attrs.items()
                if key not in ignore_keys and not isinstance(value, dict)
            }
        )
        return ds


class NEPTUNEHDF5File(HDF5File):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._file_type = C.HDF5
        self._model_meta = C.MODEL_META.get(
            self.model.lower(), C.MODEL_META[C.NEPTUNE]
        )[self._file_type].copy()
        self._lev_units = {}
        self._native_grids = {}

    @staticmethod
    def _backward_compatibility(i, hf, meta_dict):
        # older versions don't have these in attrs, gotta parse
        fp = os.path.basename(hf.filename)
        time_strs = fp.split("/")[-1].split(".")[0].split("_")

        init_time = pd.to_datetime(time_strs[-2], format=C.DTG_FMT)
        meta_dict[i]["initialization"] = init_time

        tau = time_strs[-1]
        meta_dict[i]["tau"] = tau

        ftype_idx = time_strs.index("output")
        experiment = "_".join(time_strs[:ftype_idx])
        meta_dict[i]["experiment"] = experiment
        meta_dict[i]["grid_dimensions"] = time_strs[ftype_idx + 1]
        return meta_dict

    def _load_meta(self, i, hf, meta_dict):
        if "global_attributes" in hf:
            global_attr_str = "global_attributes"
        else:
            global_attr_str = "global attributes"

        if global_attr_str in hf:
            self._attrs.update(
                {
                    key: util._decode(value)
                    for key, value in hf[global_attr_str].attrs.items()
                    if key not in ["valid"]
                }
            )

        if "missing" in self._attrs:
            self.mask_value = self._attrs["missing"]

        meta_dict[i] = {
            key: self._attrs.get(key, "")
            for key in self._model_meta["varying_attrs"].keys()
        }
        if meta_dict[i]["initialization"] == "":
            meta_dict = self._backward_compatibility(i, hf, meta_dict)

        if meta_dict[i]["tau"] == "":
            meta_dict[i]["tau"] = os.path.splitext(
                os.path.basename(hf.filename).split("_")[-1]
            )[0]

        meta_dict[i]["initialization"] = pd.to_datetime(meta_dict[i]["initialization"])
        try:
            meta_dict[i]["tau"] = util.parse_iso8601_tau(meta_dict[i]["tau"])
        except Exception as e:
            original_tau = meta_dict[i]["tau"]
            C.LOG.warning(
                f"Unable to parse {original_tau} as a tau due to {e}; "
                f"defaulting to tau=0!"
            )
            meta_dict[i]["tau"] = pd.Timedelta(hours=0)

        grid_dim = meta_dict[i]["grid_dimensions"]
        self._attrs[grid_dim]["crs"] = {
            "grid_mapping_name": "latitude_longitude",
            "semi_major_axis": 6371000.0,
        }

        coords = self._coords[grid_dim]
        for lev_type in hf:
            hds = hf[lev_type]

            if lev_type in self._model_meta["meta_lev_types"]:
                continue
            elif isinstance(hds, self._h5py.Dataset):
                lev_type = C.UNTITLED
                hds = hf

            if lev_type not in coords.keys():
                coords[lev_type] = {
                    coord[:3]: hf[coord][:].squeeze()
                    for coord in ["latitude", "longitude", "x", "y"]
                    if coord in hf
                }

            restart_fn = "restart" in hf.filename or lev_type == "native"
            native_attr = self._attrs.get("output_grid", "") == "native"
            if restart_fn or native_attr:
                self._attrs["irregular"][lev_type] = True

            if "level_units" in hds.attrs:
                self._lev_units[lev_type] = util._decode(hds.attrs["level_units"])

            meta_dict[i][lev_type] = {}
            for field in hds:
                is_not_sel = field not in (self.fields or [])
                if field == self._model_meta["meta_lev_types"][0]:
                    continue
                elif field in self._model_meta["meta_coords"] and is_not_sel:
                    # add coordinates
                    if field in self._model_meta["meta_coords"][:2]:
                        label = field[:3]
                    else:
                        label = field
                    if field == "terrain":
                        coord_array = self._wrap_dask(hds, field).squeeze()
                    else:
                        coord_array = hds[field][:].squeeze()
                    coords[lev_type][label] = coord_array
                    if self._attrs["irregular"].get(lev_type, False):
                        coords[lev_type]["idx"] = range(coord_array.shape[0])
                else:
                    # add data var attrs
                    meta_dict[i][lev_type][field] = {
                        key.lower(): util._decode(value)
                        for key, value in hds[field].attrs.items()
                    }

                    no_levels = "levels" not in meta_dict[i][lev_type][field]
                    has_shape = hasattr(hds[field], "shape")
                    if no_levels and has_shape:
                        meta_dict[i][lev_type][field]["levels"] = range(
                            hds[field].shape[1]
                        )

        # add terrain to every lev_type
        if C.UNTITLED in coords:
            terrain = coords[C.UNTITLED].get("terrain")
            sigma_z = coords[C.UNTITLED].get("sigma_z")
            z_height = coords[C.UNTITLED].get("z_height")

            for lev_type in coords:
                if terrain is not None:
                    coords[lev_type]["terrain"] = terrain
                if z_height is not None:
                    coords[lev_type]["z_height"] = z_height

                if lev_type not in meta_dict[i]:
                    continue
                for field in meta_dict[i][lev_type]:
                    if "levels" not in meta_dict[i][lev_type][field]:
                        continue
                    if sigma_z is not None:
                        meta_dict[i][lev_type][field]["levels"] = sigma_z
                    coords[lev_type]["lev"] = meta_dict[i][lev_type][field]["levels"]
        return meta_dict

    def _generate_array(self, field_df):
        stack = self._dask.array.stack if self.chunks else np.stack
        cols = ["path", "lev_type", "field", "lev_idx"]
        zip_ = zip(*[field_df[col] for col in cols])

        data_list = []
        for path, lev_type, field, lev_idx in zip_:
            if np.isnan(lev_idx):
                # this nan was generated from self._validate_hypercube
                # for now, use None as a placeholder since we don't know
                # the shape of the dataset
                data = None
            else:
                # actually has data
                lev_idx = int(lev_idx)
                try:
                    data = self._data_mapping[(path, lev_type, field)][:, lev_idx]
                except IndexError:
                    pass
                shape = data.shape
            data_list.append(data)

        # scan through the list again to replace None with a
        # grid of NaNs, matching the shape of the other data
        empty = np.full(shape, np.nan)
        for i, data in enumerate(data_list):
            if data is None:
                data_list[i] = empty

        array = stack(data_list)
        return array

    def _postprocess_coords(self, field, field_df):
        dims, coords = super()._postprocess_coords(field, field_df)

        coords = coords[self._lev_type]
        is_irregular = self._lev_type in self._attrs.get("irregular", {})
        has_zht = False
        if is_irregular:
            dims["idx"] = coords.get("idx")

            for coord in ["lat", "lon", "x", "y"]:
                if coord in coords:
                    coords[coord] = "idx", coords[coord]
            if "m2d" in coords:
                m2d_array = coords["m2d"]
                if coords["m2d"].ndim == 2:
                    if len(dims["lev"]) == 1:
                        m2d_array = np.swapaxes(m2d_array, 0, 1)[0]
                        m2d_dims = "idx"
                    else:
                        m2d_array = np.swapaxes(m2d_array, 0, 1)
                        m2d_dims = "lev", "idx"
                else:
                    m2d_dims = "idx"
                coords["m2d"] = m2d_dims, m2d_array
            if "terrain" in coords:
                coords["terrain"] = "idx", coords["terrain"]
            if "z_height" in coords:
                zht_array = coords["z_height"]
                num_levs = len(dims["lev"])
                if coords["z_height"].ndim == 2:
                    if num_levs == 1:
                        zht_array = np.swapaxes(zht_array, 0, 1)[0]
                        zht_dims = "idx"
                    else:
                        zht_array = np.swapaxes(zht_array, 0, 1)
                        zht_dims = "lev", "idx"
                else:
                    m2d_dims = "idx"
                if zht_array.shape[0] == num_levs:
                    coords["zht"] = zht_dims, zht_array
                    has_zht = True
        else:
            for coord in ["lat", "lon", "x", "y"]:
                if coord in coords:
                    dims[coord] = pd.unique(coords[coord].ravel())
            if "terrain" in coords:
                try:
                    terrain = coords["terrain"].reshape(
                        len(dims["lat"]), len(dims["lon"])
                    )
                    coords["terrain"] = (("lat", "lon"), terrain)
                except Exception:
                    pass

        if "sigma_z" in coords:
            coords["sigma_z"] = "lev", coords["sigma_z"]

        if self._lev_type in ["modlev", "model_level"] or is_irregular and not has_zht:
            # For both cases z = sigma + A*terrain
            # where sigma is the modlev value, terrain is the terrain height
            # at that point, and A is defined below. Both modlev and A are
            # independent of the horizontal coordinate.
            # For ivertcoord == 0 (terrain-following sigma coordinate)
            #   A = 1 – sigma/ztop
            # For ivertcoord == 1 (terrain-following hybrid coordinate)
            #   A = cos(pi/2*sigma/zhybrid) if sigma < zhybrid
            #   A = 0 otherwise
            try:
                attrs = self._attrs
                coords = self._create_zht(dims, coords, attrs)
            except Exception as e:
                C.LOG.warning(f"Could not create zht due to {e}")

        return dims, coords

    @staticmethod
    def _create_zht(dims, coords, attrs):
        levels = np.array(dims["lev"])
        if len(levels) > 1:
            zhybrid = attrs["zhybrid"]
            num_levs = len(levels)
            terrain = coords["terrain"][1]
            ndims = terrain.ndim
            terrain = np.broadcast_to(terrain, (num_levs, *terrain.shape)).transpose()
            ivertcoord = attrs.get("ivertcoord", 1)
            if ivertcoord == 0:
                a_const = 1 - levels / attrs["ztop"]
            elif ivertcoord == 1:
                a_const = np.cos(np.pi / 2 * levels / zhybrid)
                indices = np.where(levels > zhybrid)
                a_const[indices] = 0
            elif ivertcoord == 2:
                a_const = 1 - levels / zhybrid
                indices = np.where(levels > zhybrid)
                a_const[indices] = 0
            zheight = levels + a_const * terrain
            if ndims == 0:
                coords["zht"] = ("lev", zheight.transpose())
            elif ndims == 1:
                coords["zht"] = (("lev", "idx"), zheight.transpose())
            else:
                if "x" in coords and "y" in coords:
                    coords["zht"] = (("lev", "y", "x"), zheight.transpose())
                else:
                    coords["zht"] = (("lev", "lat", "lon"), zheight.transpose())
        return coords

    def _populate_metadata(self, ds):
        ds = super()._populate_metadata(ds)
        if self._lev_type in self._lev_units:
            ds["lev"].attrs["units"] = self._lev_units.pop(self._lev_type)
        return ds


class NAVGEMHDF5File(HDF5File):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._file_type = C.HDF5
        self._model_meta = C.MODEL_META.get(self.model.lower(), C.MODEL_META[C.NAVGEM])[
            self._file_type
        ].copy()

    def _load_meta(self, i, hf, meta_dict):
        if "Geometry" in hf:
            self._attrs.update(
                {
                    key: util._decode(value)
                    for key, value in hf["Geometry"].attrs.items()
                }
            )
            geometry = {key: value[:] for key, value in hf["Geometry"].items()}
        else:
            geometry = None

        meta_dict[i] = {
            key: self._attrs.get(key, "")
            for key in self._model_meta["varying_attrs"].keys()
        }
        fp_attrs = os.path.basename(hf.filename).split("_")
        try:
            meta_dict[i]["ini"] = pd.to_datetime(fp_attrs[5], format="%Y%m%d%H")
            meta_dict[i]["exp"] = "_".join(fp_attrs[3:5])
        except IndexError:  # Fei Lu's version
            meta_dict[i]["ini"] = pd.to_datetime("2020-01-01")
            meta_dict[i]["exp"] = "testing"
        meta_dict[i]["tau"] = pd.to_timedelta(
            int(os.path.splitext(fp_attrs[-1])[0]), unit="H"
        )
        meta_dict[i]["grid_dim"] = fp_attrs[2]

        grid_dim = meta_dict[i]["grid_dim"]
        self._attrs[grid_dim]["crs"] = {
            "grid_mapping_name": "latitude_longitude",
            "semi_major_axis": 6371000.0,
        }
        coords = self._coords[grid_dim]

        if geometry is not None:
            base_lats = geometry["Latitudes"][:].squeeze()
            asig = geometry["Asig"][0]
            bsig = geometry["Bsig"][0]

            points_per_lat = geometry["Points_per_lat"][:].squeeze()
            coords["lat"] = np.hstack(
                [np.repeat(base_lats[i], pts) for i, pts in enumerate(points_per_lat)]
            )
            coords["lon"] = np.hstack(
                [np.linspace(0, 360, pts) for pts in points_per_lat]
            )
            coords["idx"] = range(self._attrs.pop("NumPts"))

            points_at_equator = self._attrs["NumLon"]
            weights = np.cos(np.deg2rad(base_lats)) * (
                points_at_equator / points_per_lat
            )
            coords["m2d"] = np.hstack(
                [np.repeat(weights[i], pts) for i, pts in enumerate(points_per_lat)]
            )

            if "Gaussian" in hf:
                nz = len(asig) - 1
                p_s = hf["Gaussian"]["ps"]
                sig = np.zeros((nz, p_s.shape[1]))
                for k in np.arange(nz):
                    sig[k] = (0.5 * (asig[k] + asig[k + 1])) + (
                        0.5 * (bsig[k] + bsig[k + 1]) * p_s[:, :]
                    )
                coords["sig"] = sig

        for lev_type in hf:
            hds = hf[lev_type]
            if lev_type in self._model_meta["meta_lev_types"]:
                continue

            meta_dict[i][lev_type] = {}
            for field in hds:
                meta_dict[i][lev_type][field] = {
                    key.lower(): util._decode(value)
                    for key, value in hds[field].attrs.items()
                    if util._decode(value)
                }

                no_levels = "levels" not in meta_dict[i][lev_type][field]
                has_shape = hasattr(hds[field], "shape")
                if no_levels and has_shape:
                    meta_dict[i][lev_type][field]["levels"] = range(hds[field].shape[0])

        return meta_dict

    def _generate_array(self, field_df):
        stack = self._dask.array.stack if self.chunks else np.stack
        cols = ["path", "lev_type", "field", "lev_idx"]
        zip_ = zip(*[field_df[col] for col in cols])

        data_list = []
        for path, lev_type, field, lev_idx in zip_:
            if np.isnan(lev_idx):
                # this nan was generated from self._validate_hypercube
                # for now, use None as a placeholder since we don't know
                # the shape of the dataset
                data = None
            else:
                # actually has data
                lev_idx = int(lev_idx)
                data = self._data_mapping[(path, lev_type, field)][lev_idx]
                shape = data.shape
            data_list.append(data)

        # scan through the list again to replace None with a
        # grid of NaNs, matching the shape of the other data
        empty = np.full(shape, np.nan)
        for i, data in enumerate(data_list):
            if data is None:
                data_list[i] = empty

        array = stack(data_list)
        return array

    def _postprocess_coords(self, field, field_df):
        dims, coords = super()._postprocess_coords(field, field_df)
        if "idx" in coords:
            dims["idx"] = coords.get("idx")

            coords["lat"] = "idx", coords["lat"]
            coords["lon"] = "idx", coords["lon"]
            coords["m2d"] = "idx", coords["m2d"]
            if "sig" in coords:
                if len(dims["lev"]) == len(coords["sig"]):
                    coords["sig"] = ("lev", "idx"), coords["sig"]
                else:
                    coords.pop("sig")
        return dims, coords


class COAMPSHDF5FlatFile(FlatFile):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._file_type = C.HDF5
        self._h5py = packages._import_h5py()
        self._data_mapping = {}
        self._model_meta = C.MODEL_META.get(self.model.lower(), C.MODEL_META[C.COAMPS])[
            C.FLATFILE
        ].copy()

    def _split_paths(self, paths):
        path_df = super()._split_paths(paths)

        # files here are actually HDF5 datasets
        files = []
        for path in path_df["path"]:
            with self._h5py.File(path, mode="r") as hf:
                files.append(list(hf))
        path_df["file"] = files
        path_df = util._explode_col(path_df, "file")
        path_df.index = range(len(path_df))
        path_df = path_df.sort_values("file")
        return path_df

    def _extract_meta(self, path_df):
        meta_df = super()._extract_meta(path_df)
        grid_dim_mapping = {}
        for grid_dim, grid_dim_df in meta_df.groupby("grid_dim")[["path", "file"]]:
            grid_dim_row = grid_dim_df.iloc[0]
            if grid_dim_row["file"].endswith("infofld"):
                continue
            with self._h5py.File(grid_dim_row["path"], "r") as hf:
                shape = hf[grid_dim_row["file"]].shape
                num_ys, num_xs = shape[-2:]
                grid_dim_mapping[grid_dim] = shape[-1]

            datahd_df = pd.DataFrame(
                self._read_data(
                    *meta_df.loc[
                        meta_df["file"].str.startswith("datahd"), ["path", "file"]
                    ].iloc[0],
                    chunks=False,
                ).reshape(400, 5)
            )
            self._parse_datahd(datahd_df, grid_dim)

            self._coords[grid_dim]["y"] = range(num_ys)
            self._coords[grid_dim]["x"] = range(num_xs)

            for coord_prefix in self._model_meta["meta_name"]:
                coord_name, coord_error = self._process_prefix(
                    grid_dim_df, coord_prefix
                )
                coord_glob = f"{coord_prefix}_sfc_000000_000000_{grid_dim}"
                coord_df = grid_dim_df.loc[
                    grid_dim_df["file"].str.startswith(coord_glob)
                ]

                if len(coord_df) > 0:
                    path = coord_df["path"].iloc[0]
                    file = coord_df["file"].iloc[0]
                    coord_array = (
                        self._read_data(path, file, chunks=False)
                        .reshape(num_ys, num_xs)
                        .astype(float)
                    )
                    self._coords[grid_dim][coord_name] = (("y", "x"), coord_array)

            if "sig" in grid_dim_row["lev_type"]:
                lev_upper = self._coords[grid_dim]["lev_sig_w"].max()
            else:
                lev_upper = grid_dim_row["lev"]
            self._attrs[grid_dim]["lev_upper"] = max(
                self._attrs[grid_dim].get("lev_upper", 0), lev_upper
            )
            self._attrs[grid_dim]["lev_lower"] = float(grid_dim_row["lev_lower"])

        # some files have shapes of 18576x0040 and it doesn't fit in
        # the typical format of '1a****x0041' so extract the shape manually
        meta_df["num_xs"] = meta_df["grid_dim"].map(grid_dim_mapping)

        meta_df = self._ignore_meta(meta_df)
        meta_df["group"] = meta_df["file"]
        return meta_df

    def _read_data(self, path, group, chunks=None):
        try:
            hf = self._h5py.File(path, "r")
            if self.chunks:
                if chunks is None:
                    chunks = "auto" if isinstance(self.chunks, bool) else self.chunks
                data = self._dask.array.from_array(hf[group], chunks=chunks).squeeze()
            else:
                data = hf[group][:].load().squeeze()
                hf.close()  # only close file if data is already loaded
            return data
        except Exception as e:
            C.LOG.warning(f"Unable to read {path} {group} due to {e}!")
            return np.nan

    def _make_mapping(self, cube_df):
        unique_df = cube_df.drop_duplicates(subset=["path", "group"])
        paths = unique_df["path"]
        groups = unique_df["group"]

        num_paths = len(paths)
        if num_paths == 0:
            return

        desc = f"{self._lev_type} {self._grid_dim}"
        unique_arrays = self._map_jobs(
            self._read_data, paths, paths, groups, desc=desc, unit=" datasets read"
        )

        unique_keys = tuple(zip(paths, groups))
        self._data_mapping = dict(zip(unique_keys, unique_arrays))

    def _generate_array(self, field_df):
        stack = self._dask.array.stack if self.chunks else np.stack
        cols = ["path", "group"]
        zip_ = zip(*[field_df[col] for col in cols])
        array = stack([self._data_mapping[(path, group)] for path, group in zip_])
        return array

    def _create_dataset(self, cube_df):
        self._make_mapping(cube_df)
        ds = super()._create_dataset(cube_df)
        return ds
