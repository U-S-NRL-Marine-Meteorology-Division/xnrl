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
from xnrl import constant as C
from xnrl import util
from xnrl.internal import StackedFile
from xnrl.packages import _import_cftime


class GrADSFile(StackedFile):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._file_type = C.GRADS
        self._other_cols += ["offset"]

    @staticmethod
    def _separate_ctl(path_df):
        # separate ctl files from dat files
        is_ctl = path_df["path"].str.endswith(".ctl")
        ctl_df = path_df.loc[is_ctl, "path"].rename("ctl_path").to_frame()
        path_df = path_df.loc[~is_ctl].copy()
        return path_df, ctl_df

    @staticmethod
    def _select_ctl(path_df, ctl_df):
        # generate expected ctl file names
        path_df["ctl_expected"] = (
            path_df["path"]
            .str.split(".", expand=True)
            .iloc[:, :-1]
            .agg(".".join, axis=1)
            + ".ctl"
        )

        if len(ctl_df) > 0:
            # if user passed in some ctl files in the list
            ctl_df["ctl_expected"] = ctl_df["ctl_path"]
            path_df = path_df.merge(ctl_df, how="left", on="ctl_expected")

            # validate there's a corresponding ctl for each dat file
            null_ctl = pd.isnull(path_df["ctl_path"])
            if null_ctl.all():
                num_ctls = len(ctl_df)
                ctl_path = ctl_df["ctl_path"].iloc[0]
                if num_ctls >= 1:
                    # if there's no match, but user passed a ctl file, use it
                    if num_ctls != 1:
                        C.LOG.warning(
                            f"Unable to match any .ctl paths with .dat paths; "
                            f"using the first .ctl path: {ctl_path}"
                        )
                    path_df["ctl_path"] = ctl_path
            elif null_ctl.any():
                # if there are some matches, but missing some, forward fill
                missing_ctls = "\n".join(path_df.loc[null_ctl]["path"])
                C.LOG.warning(
                    f"Missing some corresponding .ctl paths; "
                    f"will forward fill .ctl paths for:\n{missing_ctls}"
                )
                path_df["ctl_path"] = path_df["ctl_path"].ffill()
        else:
            # if user did not pass any ctl files
            path_df["ctl_path"] = path_df["ctl_expected"]
        return path_df

    def _find_ctl(self, path_df):
        ctl_paths = []
        for ctl_path in path_df["ctl_path"]:
            if not os.path.exists(ctl_path):
                ctl_dir, ctl_file = os.path.split(ctl_path)
                ctl_expected = self._replace_ctl(ctl_file)
                ctl_paths.append(ctl_expected)
                C.LOG.warning(
                    f"Could not find {ctl_path}; "
                    f"replacing with {ctl_expected} instead!"
                )
            else:
                ctl_paths.append(ctl_path)

        path_df["ctl_path"] = ctl_paths
        path_df["file_size"] = [os.path.getsize(path) for path in path_df["path"]]
        return path_df

    def _extract_meta(self, path_df):
        path_df, ctl_df = self._separate_ctl(path_df)
        path_df = self._select_ctl(path_df, ctl_df)
        path_df = self._find_ctl(path_df)

        group_dfs = []
        sfc_fields = set([])
        attrs = {}
        for ctl_path, group_df in path_df.groupby("ctl_path"):
            fields = []  # order matters
            ctl_dir, ctl_file = os.path.split(ctl_path)
            if "*" in ctl_file:
                ctl_path = util._locate_file(ctl_dir, ctl_path, error="raise")

            are_fields = False
            with open(ctl_path, "r") as f:
                contents = f.read()
                # if zdef levs are on separate lines; make it into a single line
                contents = contents.replace("\n       ", " ")
                for line in contents.splitlines():
                    if "endvars" in line.lower():
                        break
                    line = line.replace("fcst_ops", "")
                    key, val = line.split(maxsplit=1)
                    key = key.lower()
                    if key == "vars":
                        are_fields = True
                    elif not are_fields:
                        attrs[key] = val
                    else:
                        num_levs, _, label = val.split(maxsplit=2)
                        num_levs = int(num_levs)
                        long_name, units = label.rsplit(maxsplit=1)
                        self._model_meta[key] = {}
                        fields.append(key)
                        if num_levs == 0:
                            sfc_fields.add(key)
                        self._model_meta["long_name"][key] = long_name.split()[0]
                        self._model_meta["units"][key] = units.lstrip("[").rstrip("]")

            attrs = {
                key: val.strip() if isinstance(val, str) else val
                for key, val in attrs.items()
            }

            if self.mask_value is None:
                mask_value = attrs.get("undef")
                if mask_value is not None:
                    self.mask_value = float(mask_value)

            num_xs, _, lon_start, lon_step = [
                float(val) if val[-1].isnumeric() else val
                for val in attrs["xdef"].split()
            ]
            num_ys, _, lat_start, lat_step = [
                float(val) if val[-1].isnumeric() else val
                for val in attrs["ydef"].split()
            ]

            num_xs = int(num_xs)
            num_ys = int(num_ys)

            grid_dim = f"glob{num_xs:03d}x{num_ys:03d}"
            attrs["size"] = int(num_ys * num_xs)
            group_df["size"] = attrs["size"]

            lons = np.arange(lon_start, num_xs * lon_step - abs(lon_start), lon_step)
            lats = np.arange(lat_start, num_ys * lat_step - abs(lat_start), lat_step)
            self._coords[grid_dim]["lon"] = lons
            self._coords[grid_dim]["lat"] = lats

            num_taus, _, ini, tau = attrs["tdef"].split()
            num_taus = int(num_taus)

            ini = self._extract_ini(group_df, ini)
            taus = self._extract_tau(group_df, tau, num_taus)
            total_levs, _, levs = attrs["zdef"].split(maxsplit=2)
            levs = [float(lev) for lev in levs.split(" ")]

            group_df = group_df.assign(
                **{
                    "field": [fields] * len(group_df),
                    "lev_type": "pre",
                    "lev": [levs] * len(group_df),
                    "num_xs": num_xs,
                    "num_ys": num_ys,
                    "grid_dim": grid_dim,
                    "ini": ini,
                    "tau": taus,
                }
            )
            group_df = util._explode_col(group_df, "field")
            group_df["field"] = pd.Categorical(
                group_df["field"], categories=fields, ordered=True
            )
            is_sfc_field = group_df["field"].isin(sfc_fields)
            group_df.loc[is_sfc_field, "lev"] = 0
            group_df.loc[is_sfc_field, "lev_type"] = "sfc"
            group_df = util._explode_col(group_df, "lev")
            group_df = util._explode_col(group_df, "tau")
            group_df = group_df.drop_duplicates()

            # must be in the order of tau, field, lev or else offset is wrong
            # taus within file
            sort_cols = ["file", "tau", "field", "lev"]
            ascending = [True, True, True, False]
            group_df = group_df.sort_values(sort_cols, ascending=ascending)
            group_df["field"] = group_df["field"].astype(str)

            # multiple levels and taus in a file;
            # if we are going to parallelize,
            # we need to know where to look
            group_df["offset"] = (
                group_df.groupby("file")["size"].cumsum() - group_df["size"]
            ) * 4
            group_df = group_df.loc[group_df["offset"] < group_df["file_size"]]

            exp = self._extract_exps(group_df)
            if (exp == C.UNTITLED).all():
                group_df["exp"] = ctl_file.lstrip(os.sep)
            else:
                group_df["exp"] = exp
            self._attrs[grid_dim] = attrs
            group_dfs.append(group_df)

        meta_df = pd.concat(group_dfs)
        meta_df = meta_df.drop(columns=["ctl_path", "ctl_expected", "size"])
        meta_df["mbr"] = self._extract_mbrs(meta_df)
        meta_df["time"] = meta_df["ini"] + meta_df["tau"]
        meta_df.index = np.arange(len(meta_df))
        return meta_df


class NAVGEMGrADSFile(GrADSFile):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._model_meta = C.MODEL_META.get(self.model.lower(), C.MODEL_META[C.NAVGEM])[
            self._file_type
        ].copy()

    @staticmethod
    def _replace_ctl(ctl_file):
        ctl_expected = ctl_file.split(".")[0] + ".*.ctl"
        return ctl_expected

    def _extract_ini(self, group_df, ini):
        try:
            ini = pd.to_datetime(
                group_df["file"].str.split(".", expand=True).iloc[:, -2],
                format="%Y%m%d%H",
            )
        except Exception:
            ini = pd.to_datetime(ini, format="%HZ%d%b%Y")
        return ini

    @staticmethod
    def _extract_tau(group_df, tau, num_taus):
        try:
            taus = group_df["file"].str.split(".dat", expand=True).iloc[:, -1]
            taus = pd.to_timedelta(taus.astype(float), unit="H")
            taus = taus if len(taus) == len(group_df) else [taus] * len(group_df)
        except Exception:
            tau = pd.to_timedelta(tau)
            taus = pd.to_timedelta(
                np.linspace(0, num_taus * tau.total_seconds() / 3600, num_taus),
                unit="H",
            )
        return taus


class CM1GrADSFile(GrADSFile):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._model_meta = C.MODEL_META.get(self.model.lower(), C.MODEL_META[C.NAVGEM])[
            self._file_type
        ].copy()
        self._cftime = _import_cftime()

    @staticmethod
    def _replace_ctl(ctl_file):
        ctl_expected = ctl_file.split("_")[0] + "*.ctl"
        return ctl_expected

    def _extract_ini(self, group_df, ini):
        hour = int(ini[0:2])
        minute = int(ini[3:5])
        day = int(ini[6:8])
        month = pd.to_datetime(ini[8:11], format="%b").month
        year = int(ini[11:15])
        ini = self._cftime.DatetimeProlepticGregorian(year, month, day, hour, minute)
        return ini

    @staticmethod
    def _extract_tau(group_df, tau, num_taus):
        taus = group_df["file"].str.split("_", expand=True).iloc[:, 1]
        taus = pd.to_timedelta(taus.astype(float) * 365, unit="D")
        taus = taus if len(taus) == len(group_df) else [taus] * len(group_df)
        return taus

    def _populate_metadata(self, ds):
        ds["ini"] = xr.CFTimeIndex(ds["ini"].values)
        ds = super()._populate_metadata(ds)
        return ds

    @staticmethod
    def _read_data(path, size, offset):
        try:
            return np.fromfile(path, dtype="float32", count=size, offset=offset)
        except Exception as e:
            if str(path) != "nan":
                C.LOG.warning(f"Could not open {path} due to {e}!")
            return np.full(size, np.nan)
