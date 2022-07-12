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


class VisualFile(StackedFile):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._file_type = C.VISUAL
        self._other_cols += ["offset"]

    def _parse_metadata(self, meta_path):
        with open(meta_path, "r") as f:
            contents = f.read()
            is_pres = False
            attrs = {}
            for line in contents.splitlines():
                if line.startswith("$"):
                    continue
                line_value = line.split("!")[0].strip()
                if not line_value:
                    continue

                if "=" in line_value:
                    key, val = line_value.split("=", maxsplit=1)
                else:
                    val = line_value

                key = key.lower().strip()
                val = val.strip()

                if key == "pres":
                    if key not in attrs:
                        attrs[key] = []
                    attrs[key].append(float(val.rstrip(",")))
                else:
                    if val.rstrip(".")[-1].isnumeric():
                        val = float(val)
                    attrs[key] = val
        return attrs

    def _compute_grid(self, grid_dim, attrs):
        reflat = attrs["alat"]
        reflon = attrs["alon"]
        stdlon = attrs["alnnt"]
        iref = attrs["ipole"]
        jref = attrs["jpole"]
        stdlt1 = attrs["stdlat1"]
        delx = attrs["xgrid1"] * 1000
        dely = attrs["ygrid1"] * 1000
        nx = attrs["nx"]
        ny = attrs["ny"]

        pi2 = np.pi / 2.0
        pi4 = np.pi / 4.0
        d2r = np.pi / 180.0
        r2d = 180.0 / np.pi
        radius = 6371229.0
        omega4 = 4.0 * np.pi / 86400.0
        onedeg = radius * 2.0 * np.pi / 360.0

        gcon = 1.0
        ogcon = 1.0 / gcon
        ihem = round(abs(stdlt1) / stdlt1)
        deg = (90.0 - abs(stdlt1)) * d2r
        cn1 = np.sin(deg)
        cn2 = radius * cn1 * ogcon
        deg = deg * 0.5
        cn3 = np.tan(deg)
        deg = (90.0 - abs(reflat)) * 0.5 * d2r
        cn4 = np.tan(deg)
        rih = cn2 * (cn4 / cn3) * gcon
        deg = (reflon - stdlon) * d2r * gcon
        xih = rih * np.sin(deg)
        yih = -rih * np.cos(deg) * ihem

        ii, jj = np.meshgrid(np.arange(1, nx + 1), np.arange(1, ny + 1))
        x = xih + (ii - iref) * delx
        y = yih + (jj - jref) * dely

        rr = np.sqrt(x * x + y * y)
        grdlat = r2d * (pi2 - 2.0 * np.arctan(cn3 * (rr / cn2) * ogcon)) * ihem
        xx = x
        yy = -y * ihem
        angle = np.arctan2(xx, yy) * r2d
        angle[np.where((yy == 0) & (xx <= 0))] = -90
        angle[np.where((yy == 0) & (xx > 0))] = 90
        grdlon = stdlon + angle * ogcon

        self._coords[grid_dim]["lat"] = ("y", "x"), grdlat
        self._coords[grid_dim]["lon"] = ("y", "x"), grdlon

    def _extract_meta(self, path_df):
        path_df = path_df.loc[~path_df["file"].str.startswith("visual")]
        coord_dir = path_df["directory"].iloc[0]
        meta_path = util._locate_file(coord_dir, "visual.dat*")
        attrs = self._parse_metadata(meta_path)
        grid_num = attrs["gridnum"]
        num_ys = int(attrs["ny"])
        num_xs = int(attrs["nx"])
        size = num_ys * num_xs
        grid_dim = f"{grid_num:.0f}a{num_ys:.0f}x{num_xs:.0f}"
        meta_df = pd.DataFrame(
            {
                "time": pd.to_datetime(path_df["file"].str[:11], format="%y%m%d%H%MZ"),
                "field": path_df["file"]
                .str[11:]
                .str.split(".", expand=True)[0]
                .str.lower(),
                "exp": self._extract_exps(path_df),
                "mbr": self._extract_mbrs(path_df),
                "ini": pd.to_datetime(attrs["init"], format="%y%m%d/%H%MZ"),
                "lev": [attrs["pres"]] * len(path_df),
                "lev_type": "sig",
                "grid_dim": grid_dim,
                "file_size": [os.path.getsize(path) for path in path_df["path"]],
            }
        )
        meta_df["tau"] = meta_df["time"] - meta_df["ini"]
        meta_df = util._explode_col(meta_df, "lev")
        meta_df = meta_df.join(path_df)
        meta_df = meta_df.assign(**{"dump": 8, "offset": size})
        meta_df["dump"] = meta_df.groupby("file")["dump"].cumsum() - 8
        meta_df["offset"] = meta_df.groupby("file")["offset"].cumsum()
        meta_df["offset"] = (meta_df["offset"] - size) * 4 + meta_df["dump"]
        is_single_lev = meta_df["offset"] < (meta_df["file_size"])
        meta_df.loc[is_single_lev, "lev_type"] = "sfc"
        meta_df = meta_df.loc[is_single_lev]
        meta_df.index = np.arange(len(meta_df))

        self._coords[grid_dim]["y"] = np.arange(num_ys)
        self._coords[grid_dim]["x"] = np.arange(num_xs)
        self._compute_grid(grid_dim, attrs)

        self._attrs[grid_dim]["size"] = size
        self._attrs[grid_dim]["crs"] = {
            "grid_mapping_name": "polar_stereographic",
            "standard_parallel": [attrs["stdlat1"], attrs["stdlat2"]],
            "central_longitude": attrs["alnnt"],
            "longitude_of_central_meridian": attrs["alon"],
            "latitude_of_projection_origin": attrs["alat"],
        }
        return meta_df


class COAMPSVisualFile(VisualFile):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._file_type = C.VISUAL
        self._model_meta = C.MODEL_META.get(self.model.lower(), C.MODEL_META[C.COAMPS])[
            self._file_type
        ].copy()
