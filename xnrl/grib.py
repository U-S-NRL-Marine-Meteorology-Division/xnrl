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

import numpy as np
import pandas as pd
import xarray as xr
from xnrl import constant as C
from xnrl import packages, util
from xnrl.generic import GenericFile
from xnrl.internal import NRLFile


class GRIBFlatFile(NRLFile):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._file_type = C.GRIB
        self._grib_name, self._grib_lib = packages._import_grib()

    def _extract_meta(self, path_df):
        meta_df = pd.concat(
            (
                path_df["file"].str[inds].rename(key)
                for key, inds in self._model_meta["indices"].items()
            ),
            axis=1,
        )
        meta_df = meta_df.join(path_df)
        meta_df["exp"] = self._extract_exps(path_df)
        meta_df["mbr"] = C.MASK_VALUE
        meta_df["ini"] = pd.to_datetime(meta_df["ini"], format=C.DTG_FMT)
        meta_df["tau"] = pd.to_timedelta(meta_df["tau"].astype(int), unit="H")
        meta_df["lev"] = meta_df["lev"].astype(int)
        meta_df["time"] = meta_df["ini"] + meta_df["tau"]

        grid_dim_df = meta_df.drop_duplicates(subset="grid_dim")
        grid_dim_shapes = {}
        grid_dim_path = zip(grid_dim_df["grid_dim"], grid_dim_df["path"])
        for grid_dim, path in grid_dim_path:
            grid_dim_shapes[grid_dim] = self._read_data(path, None, to_1d=False).shape
        meta_df["shape"] = meta_df["grid_dim"].map(grid_dim_shapes)
        meta_df[["num_ys", "num_xs"]] = pd.DataFrame(
            meta_df["shape"].tolist(), index=meta_df.index
        )
        meta_df = meta_df.drop(columns=["shape"])
        meta_df["lev_type"] = (
            meta_df["lev_type"].astype(int).replace(self._model_meta["lev_type"])
        )
        return meta_df

    def _read_data(self, path, size, to_1d=True):
        try:
            if self._grib_name == "pygrib":
                # grab the first value, 1 because pygrib
                # indexes start at 1 strangely
                with self._grib_lib.open(path) as grbs:
                    data = grbs[1].values
            else:
                if self._grib_name == "cfgrib":
                    kwds = dict(
                        engine="cfgrib",
                        backend_kwargs={"errors": "ignore", "indexpath": ""},
                    )
                else:
                    kwds = dict(engine="pynio")
                with xr.open_dataset(path, **kwds) as ds:
                    var = list(ds.data_vars)[0]
                    data = ds[var].values
            if to_1d:
                data = data.ravel()
            return data
        except Exception as e:
            C.LOG.warning(f"Unable to read {path} due to {e}!")
            return np.full(size, np.nan)


class NAVGEMGRIBFlatFile(GRIBFlatFile):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._model_meta = C.MODEL_META.get(self.model.lower(), C.MODEL_META[C.NAVGEM])[
            self._file_type
        ].copy()

    def _extract_meta(self, path_df):
        meta_df = super()._extract_meta(path_df)
        meta_df["grid_dim"] = (
            meta_df["num_xs"].astype(str).str.zfill(4)
            + "x"
            + meta_df["num_ys"].astype(str).str.zfill(4)
        )

        for grid_dim, grid_dim_df in meta_df.groupby("grid_dim"):
            num_ys, num_xs = (
                grid_dim_df.iloc[0]
                .astype(str)
                .str.lstrip("b")
                .str.lstrip("x")[["num_ys", "num_xs"]]
                .astype(int)
            )
            self._attrs[grid_dim]["size"] = num_ys * num_xs
            self._coords[grid_dim]["lat"] = np.linspace(-90, 90, num_ys)
            self._coords[grid_dim]["lon"] = np.linspace(0, 360, num_xs, endpoint=False)
            self._attrs[grid_dim]["crs"] = {
                "grid_mapping_name": "latitude_longitude",
                "semi_major_axis": 6371000.0,
            }
        return meta_df


class COAMPSGRIBFlatFile(GRIBFlatFile):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._model_meta = C.MODEL_META.get(self.model.lower(), C.MODEL_META[C.COAMPS])[
            self._file_type
        ].copy()

    def _split_paths(self, paths):
        path_df = super(GRIBFlatFile, self)._split_paths(paths)
        path_df = path_df.loc[~(path_df["file"].str.endswith(".idx"))]
        if len(path_df) > 1:
            temp_df = path_df.loc[
                ~path_df["file"].str.endswith(tuple(self._model_meta["meta_name"]))
            ]
            if len(temp_df) > 0:
                path_df = temp_df
        return path_df

    def _extract_meta(self, path_df):
        meta_df = super()._extract_meta(path_df)
        meta_df["grid_dim"] = (
            meta_df["nest_num"].astype(str)
            + "a"
            + meta_df["num_xs"].astype(str).str.zfill(4)
            + "x"
            + meta_df["num_ys"].astype(str).str.zfill(4)
        )

        for grid_dim, grid_dim_df in meta_df.groupby("grid_dim"):
            grid_dim_row = grid_dim_df.iloc[0]
            directory = grid_dim_row["directory"]
            nest_num = int(grid_dim_row["nest_num"])
            num_ys = int(grid_dim_row["num_ys"])
            num_xs = int(grid_dim_row["num_xs"])
            self._coords[grid_dim]["y"] = range(num_ys)
            self._coords[grid_dim]["x"] = range(num_xs)
            size = num_ys * num_xs
            self._attrs[grid_dim]["size"] = size

            for coord_suffix in self._model_meta["meta_name"]:
                coord_glob = f"*-n{nest_num}*{coord_suffix}"
                if coord_suffix in self._model_meta["meta_name"][:2]:
                    coord_name = coord_suffix[:3]
                    coord_error = "warn"
                else:
                    coord_name = coord_suffix
                    coord_error = "ignore"
                coord_path = util._locate_file(directory, coord_glob, error=coord_error)
                if coord_path is not None:
                    coord_array = (
                        self._read_data(coord_path, size)
                        .reshape(num_ys, num_xs)
                        .astype(float)
                    )
                    self._coords[grid_dim][coord_name] = coord_array

            self._attrs[grid_dim]["crs"] = {
                "grid_mapping_name": (self._model_meta["proj"][nest_num])
            }
        return meta_df


class GenericGRIBFile(GenericFile):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._file_type = C.GRIB
        self._grib_name, self._grib_lib = packages._import_grib()
        self._model_meta = C.MODEL_META.get(
            self.model.lower(), C.MODEL_META[C.GENERIC]
        )[self._file_type].copy()
