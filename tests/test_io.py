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

import xnrl


def test_export_flatfile_sigma():
    ds = xnrl.tutorial.open_dataset(
        "NEPTUNE",
        label="restart",
        lev_types="dynamics",
        fields="rho",
        regrid_kwds={"hres": 0.5},
    )
    ds.attrs["lev_type"] = "sig"
    fp_list = xnrl.export_flatfile(ds)
    fp_list = [
        fp
        for fp in fp_list
        if not any(var in fp for var in ["latitu", "longit", "datahd"])
    ]

    for i, fp in enumerate(fp_list):
        ds2 = xnrl.open_dataset(fp, model="NEPTUNE")
        ds2 = ds2.sortby("lev", ascending=False)
        data_var = list(ds2.data_vars)[0]
        data1 = ds[data_var].isel(tau=i).squeeze(drop=True).values
        data2 = ds2[data_var].squeeze(drop=True).values
        assert np.isclose(data1, data2).all()


def test_export_flatfile_coamps():
    ds1 = xnrl.tutorial.open_dataset("COAMPS")
    fp = xnrl.export_flatfile(ds1)
    ds2 = xnrl.open_dataset(fp)
    for var in ["slpres", "lat", "lon"]:
        (ds1[var].values == ds2[var].values).all()
