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
import tarfile

import numpy as np
import pandas as pd
from xnrl import constant as C
from xnrl import util
from xnrl.internal import NRLFile


class FlatFile(NRLFile):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._file_type = C.FLATFILE

    @staticmethod
    def _append_types(files, grid_dims):
        file_types = files.str.split("_", expand=True).iloc[:, -1]
        file_types = file_types.loc[file_types != "infofld"]
        if len(file_types.unique()) > 1:
            grid_dims += "_" + file_types
        return grid_dims

    def _extract_meta(self, path_df):
        meta_df = pd.concat(
            (
                path_df["file"].str[inds].rename(key)
                for key, inds in self._model_meta["indices"].items()
            ),
            axis=1,
        )
        bad_files = meta_df["ini"] == ""
        if any(bad_files):
            num_bad_files = sum(bad_files)
            C.LOG.debug(
                f"Found {num_bad_files} bad files; excluding the following!\n"
                f"{path_df.loc[bad_files, 'path']}"
            )
            meta_df = meta_df.loc[~bad_files]
            path_df = path_df.loc[~bad_files]

        meta_df["exp"] = self._extract_exps(path_df)
        meta_df["mbr"] = self._extract_mbrs(path_df)
        meta_df["ini"] = pd.to_datetime(meta_df["ini"], format=C.DTG_FMT)
        meta_df["tau"] = pd.to_timedelta(meta_df["tau"].astype(int), unit="H")
        meta_df["lev"] = meta_df["lev"].astype(float)
        meta_df = meta_df.join(path_df)
        meta_df["lev_type"] = self._append_types(meta_df["file"], meta_df["lev_type"])
        meta_df["time"] = meta_df["ini"] + meta_df["tau"]
        meta_df["field"] = meta_df["field"].str.rstrip("*")
        return meta_df

    @staticmethod
    def _read_data(path, size):
        try:
            return np.fromfile(path, dtype=">f")
        except Exception as e:
            if str(path) != "nan":
                C.LOG.warning(f"Could not open {path} due to {e}!")
            return np.full(size, np.nan)

    def _parse_datahd(self, datahd_path, grid_dim):
        try:
            datahd_df = pd.read_fwf(datahd_path, widths=[13] * 5, header=None)
        except UnicodeDecodeError:
            # sometimes in binary format instead of plain ASCII
            datahd_df = pd.DataFrame(
                np.fromfile(datahd_path, dtype=">f").reshape(400, 5)
            )

        num_sigs = int(datahd_df.loc[0, 1])
        self._coords[grid_dim]["lev_sig"] = (datahd_df.iloc[160:200].stack().unique())[
            :num_sigs
        ]
        self._coords[grid_dim]["lev_sig_w"] = (
            np.cumsum(datahd_df[100:140].values.ravel()[::-1])[::-1]
        )[: num_sigs + 1]

        self._attrs[grid_dim]["crs"] = {
            "grid_mapping_name": (self._model_meta["proj"][int(datahd_df.loc[0, 2])]),
            "standard_parallel": [int(datahd_df.loc[0, 4]), int(datahd_df.loc[0, 3])],
            "central_longitude": datahd_df.loc[1, 2],
            "longitude_of_central_meridian": datahd_df.loc[1, 0],
            "latitude_of_projection_origin": datahd_df.loc[1, 1],
            "earth_radius": 6370997,
        }
        if self._attrs[grid_dim]["crs"]["grid_mapping_name"] == "mercator":
            self._attrs[grid_dim]["crs"].pop("latitude_of_projection_origin")


class NAVGEMFlatFile(FlatFile):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._model_meta = C.MODEL_META.get(self.model.lower(), C.MODEL_META[C.NAVGEM])[
            self._file_type
        ].copy()

    def _extract_meta(self, path_df):
        meta_df = super()._extract_meta(path_df)
        meta_df = meta_df.drop(columns="nest_num")
        for grid_dim, grid_dim_df in meta_df.groupby("grid_dim"):
            num_ys, num_xs = (
                grid_dim_df.iloc[0]
                .astype(str)
                .str.lstrip("b")
                .str.lstrip("x")[["num_ys", "num_xs"]]
                .astype(int)
            )
            self._coords[grid_dim]["lat"] = np.linspace(-90, 90, num_ys)
            self._coords[grid_dim]["lon"] = np.linspace(0, 360, num_xs, endpoint=False)
            self._attrs[grid_dim]["crs"] = {
                "grid_mapping_name": "latitude_longitude",
                "semi_major_axis": 6371000.0,
            }
            self._attrs[grid_dim]["size"] = num_ys * num_xs
        return meta_df


class NAVGEMTGZFlatFile(NAVGEMFlatFile):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._model_meta = C.MODEL_META.get(self.model.lower(), C.MODEL_META[C.NAVGEM])[
            self._file_type
        ].copy()
        self._other_cols = ["key", "name"]
        self._data_mapping = {}

        if self.chunks:
            C.LOG.warning(
                f"Note! Lazy loading is not supported for compressed files! "
                f"However, the data will be chunked."
            )

    def _extract_meta(self, path_df):
        tar_dfs = []
        for path in path_df["path"]:
            tar_obj = tarfile.open(path, mode="r")
            base_name = os.path.splitext(os.path.basename(path))[0]
            names = pd.Series(tar_obj.getnames())
            names = names.loc[names != base_name]
            files = names.str.split("/", expand=True)[1]
            keys = path + "//TGZ//" + names
            tar_dfs.append(
                pd.DataFrame(
                    {
                        "path": path,  # path to tar.gz
                        "file": files,  # flatfile name without directory
                        "name": names,  # flatfile as named within tar.gz
                        "key": keys,  # data mapping UNIQUE keys
                    }
                )
            )
            self._data_mapping[path] = tar_obj
        tar_df = pd.concat(tar_dfs)
        path_df = pd.merge(path_df.drop("file", axis=1), tar_df, on="path")
        meta_df = super()._extract_meta(path_df)
        return meta_df

    def _make_mapping(self, tar_df):
        path = tar_df["path"].iloc[0]
        tar_obj = self._data_mapping.pop(path)
        for key, name in zip(tar_df["key"], tar_df["name"]):
            data = tar_obj.extractfile(name).read()
            if data:  # skip 0 sized data
                self._data_mapping[key] = data
        tar_obj.close()

    def _select_meta(self, meta_df):
        meta_df = super()._select_meta(meta_df)
        tar_dfs = [tar_df for path, tar_df in meta_df.groupby("path")]
        self._map_jobs(self._make_mapping, tar_dfs, desc="pre-load", unit=" paths read")
        return meta_df

    def _read_data(self, key, size):
        try:
            extracted = self._data_mapping[key]
            return np.frombuffer(extracted, dtype=">f")
        except Exception as e:
            if str(key) != "nan":
                C.LOG.warning(f"Could not open {key} due to corrupted data!")
            return np.full(size, np.nan)

    def _generate_array(self, field_df):
        keys = field_df["key"].values

        if self.chunks:
            read_data = self._dask.delayed(self._read_data, pure=True)
        else:
            read_data = self._read_data

        num_keys = len(keys)
        if num_keys == 0:
            return

        if "sig" in self._lev_type:
            repeats = len(self._coords[self._grid_dim]["lev_sig"])
            if self._field == "wwwind":
                repeats += 1
            sizes = np.repeat(self._attrs[self._grid_dim]["size"] * repeats, num_keys)
        else:
            sizes = np.repeat(self._attrs[self._grid_dim]["size"], num_keys)

        desc = f"{self._lev_type} {self._grid_dim} {self._field}"
        arrays = self._map_jobs(
            read_data, keys, sizes, desc=desc, unit=" datasets read", warn=False
        )
        array = self._try_stack(arrays, sizes, chunks=self.chunks)
        return array


class COAMPSFlatFile(FlatFile):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._model_meta = C.MODEL_META.get(self.model.lower(), C.MODEL_META[C.COAMPS])[
            self._file_type
        ].copy()

    def _ignore_meta(self, df):
        df = df.loc[~df["file"].str.endswith("infofld")]
        if len(df) > 1:
            temp_df = df.loc[
                ~df["file"].str.startswith(tuple(self._model_meta["meta_name"]))
            ]
            if len(temp_df) > 0:
                df = temp_df
        return df

    def _extract_meta(self, path_df):
        meta_df = super()._extract_meta(path_df)

        for grid_dim, grid_dim_df in meta_df.groupby("grid_dim"):
            grid_dim_row = grid_dim_df.iloc[0]
            num_ys, num_xs = grid_dim_row[["num_ys", "num_xs"]].astype(int)
            size = num_ys * num_xs
            coord_dir = grid_dim_row["directory"]
            lev_type = grid_dim_row["lev_type"]

            if self.datahd_path is not None:
                if os.path.isdir(self.datahd_path):
                    datahd_path = util._locate_file(
                        self.datahd_path, "datahd_sfc*", error="warn"
                    )
                elif "*" in self.datahd_path:
                    datahd_path = util._locate_file(
                        *os.path.split(self.datahd_path), error="warn"
                    )
                else:
                    datahd_path = self.datahd_path
            else:
                datahd_path = util._locate_file(coord_dir, "datahd_sfc*", error="warn")

            if datahd_path is not None:
                self._parse_datahd(datahd_path, grid_dim)
                self._encodings[grid_dim]["datahd"] = datahd_path

            self._coords[grid_dim]["y"] = np.arange(num_ys)
            self._coords[grid_dim]["x"] = np.arange(num_xs)
            self._attrs[grid_dim]["size"] = size

            for coord_prefix in self._model_meta["meta_name"]:
                if coord_prefix == "datahd":
                    continue

                coord_name, coord_error = self._process_prefix(
                    grid_dim_df, coord_prefix
                )

                coord_glob = f"{coord_prefix}_sfc_000000_000000_{grid_dim}*"
                coord_path = util._locate_file(coord_dir, coord_glob, error=coord_error)
                if coord_path is not None:
                    coord_array = (
                        self._read_data(coord_path, size)
                        .reshape(num_ys, num_xs)
                        .astype(float)
                    )
                    self._coords[grid_dim][coord_name] = (("y", "x"), coord_array)

            try:
                if "sig" in lev_type:
                    lev_upper = self._coords[grid_dim]["lev_sig_w"].max()
                else:
                    lev_upper = grid_dim_row["lev"]
            except KeyError:
                lev_upper = C.MASK_VALUE

            self._attrs[grid_dim]["lev_upper"] = lev_upper
            self._attrs[grid_dim]["lev_lower"] = float(grid_dim_row["lev_lower"])

        return meta_df

    def _select_meta(self, meta_df):
        meta_df = super()._select_meta(meta_df)
        if self.stationary_coords:
            return meta_df

        for grid_dim, grid_dim_df in meta_df.groupby("grid_dim"):
            grid_dim_row = grid_dim_df.iloc[0]
            lev_type = grid_dim_row["lev_type"]

            for coord_prefix in self._model_meta["meta_name"]:
                if coord_prefix == "datahd":
                    continue
                coord_name, coord_error = self._process_prefix(
                    grid_dim_df, coord_prefix
                )
                coord_label = f"{coord_prefix}_sfc_000000_000000_"
                # extract 1a0361x0191_2020082812_00000000_fcstfld from
                # uuwind_zht_000010_000000_1a0361x0191_2020082812_00030000_fcstfld
                file_suffix = grid_dim_df["file"].str[25:]
                coord_df = grid_dim_df.loc[
                    grid_dim_df["field"] == grid_dim_df["field"].unique()[0]
                ].copy()
                coord_df["file"] = coord_label + file_suffix
                coord_df["path"] = [
                    path if os.path.exists(path) else None
                    for path in coord_df["directory"].str.cat(
                        coord_df["file"], sep=os.sep
                    )
                ]
                coord_df["path"] = coord_df["path"].ffill()
                self._grid_dim = grid_dim
                self._lev_type = lev_type
                self._field = coord_prefix
                coord_array = self._generate_array(coord_df)
                self._coords_nonstationary[grid_dim][coord_name] = coord_array

        return meta_df


class GFSFlatFile(FlatFile):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._model_meta = C.MODEL_META.get(self.model.lower(), C.MODEL_META[C.GFS])[
            self._file_type
        ].copy()

    @staticmethod
    def _append_types(files, grid_dims):
        return grid_dims

    @staticmethod
    def _read_data(path, size):
        try:
            return np.fromfile(path, offset=8, dtype=">f")
        except Exception as e:
            if str(path) != "nan":
                C.LOG.warning(f"Could not open {path} due to {e}!")
            return np.full(size, np.nan)

    def _extract_meta(self, path_df):
        meta_df = super()._extract_meta(path_df)
        meta_df = meta_df.drop(columns="nest_num")
        meta_df["num_ys"] = 361
        meta_df["num_xs"] = 720
        meta_df["grid_dim"] = "glob720x361"
        for grid_dim, grid_dim_df in meta_df.groupby("grid_dim"):
            num_ys, num_xs = (
                grid_dim_df.iloc[0]
                .astype(str)
                .str.lstrip("b")
                .str.lstrip("x")[["num_ys", "num_xs"]]
                .astype(int)
            )
            self._coords[grid_dim]["lat"] = np.linspace(-90, 90, num_ys)
            self._coords[grid_dim]["lon"] = np.linspace(0, 360, num_xs, endpoint=False)
            self._attrs[grid_dim]["crs"] = {
                "grid_mapping_name": "latitude_longitude",
                "semi_major_axis": 6371000.0,
            }
            self._attrs[grid_dim]["size"] = num_ys * num_xs
        return meta_df
