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

from datetime import timedelta

import numpy as np
import pandas as pd
import xarray as xr
from xnrl import constant as C
from xnrl import packages, util
from xnrl.generic import GenericFile


class NetCDFFile(GenericFile):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._file_type = C.NETCDF
        self._model_meta = C.MODEL_META.get(
            self.model.lower(), C.MODEL_META[C.GENERIC]
        )[self._file_type].copy()


class GenericNetCDFFile(NetCDFFile):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class RAOBNetCDFFile(NetCDFFile):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        packages._import_cf_xarray()

    def _extract_meta(self, path_df):
        meta_df = super()._extract_meta(path_df)

        try:
            ds = xr.open_mfdataset(path_df["path"])
        except ValueError:
            ds = self._decode_time(
                xr.open_mfdataset(path_df["path"], decode_times=False)
            )
        except OSError:
            ds = xr.open_mfdataset(path_df["path"], engine="h5netcdf")
        self._attrs.update({**ds.attrs})

        coords = {
            coord: coord.replace("@MetaData", "")
            for coord in ds.data_vars
            if "@MetaData" in coord
        }
        ds = ds.rename(coords).set_coords(list(coords.values()))
        for coord in ds.coords:
            ds.coords[coord].attrs["long_name"] = coord

        ds = self._parse_cf_coords(ds)
        ds["datetime"] = np.min(
            pd.to_datetime(
                util._decode(ds["datetime"].values), format="%Y-%m-%dT%H:%M:%SZ",
            )
        )
        ds = ds.rename({"datetime": "ini", "nlocs": "idx"}).assign_coords(
            {
                "tau": timedelta(0),
                "mbr": C.MASK_VALUE,
                "exp": self._extract_exps(path_df)[0],
                "idx": np.arange(len(ds["nlocs"])),
            }
        )

        if self.fields:
            if not isinstance(self.fields, list):
                self.fields = [self.fields]
            fields = [field for field in self.fields if field in ds.data_vars]
            ds = ds[fields]
        try:
            ds = ds.where(ds < 9e36)
        except TypeError:
            pass

        ds = self._chunk_xarray(ds)
        self._generic_ds = ds

        meta_dims = C.BASE_DIMS.copy()
        meta_dims.remove("field")
        meta_df = ds[list(ds.coords)].to_dataframe()[meta_dims]
        meta_df["lev_type"] = "pressure"
        meta_df["grid_dim"] = C.UNTITLED
        meta_df = meta_df.assign(key=1).merge(
            pd.DataFrame({"field": ds.data_vars}).assign(key=1), on="key"
        )
        meta_df = (
            path_df.assign(key=1)
            .merge(meta_df.assign(key=1), on="key")
            .drop("key", axis=1)
        )
        self._attrs[C.UNTITLED]["crs"] = {
            "grid_mapping_name": "latitude_longitude",
            "semi_major_axis": 6371000.0,
        }
        meta_df = meta_df.drop_duplicates()
        meta_df.index = np.arange(len(meta_df))
        return meta_df

    @staticmethod
    def _parse_cf_coords(ds):
        coords = ds.cf.coordinates
        ds = ds.cf.rename(
            {
                coords["latitude"][0]: "lat",
                coords["longitude"][0]: "lon",
                coords["vertical"][0]: "lev",
            }
        )
        return ds


class RadianceNetCDFFile(RAOBNetCDFFile):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _extract_meta(self, path_df):
        meta_df = super()._extract_meta(path_df)
        meta_df["lev_type"] = "height_above_mean_sea_level"
        return meta_df
