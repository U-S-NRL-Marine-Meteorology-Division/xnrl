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

import glob
import os

import xnrl
import xnrl.constant as C


def _get_paths(model, file_type, label):
    glob_path = os.path.join(C.TEST_DIR, model, file_type, label, "*")
    paths = sorted(glob.iglob(glob_path))
    if not paths:
        raise FileNotFoundError(f"Failed to retrieve data using {glob_path}!")
    return paths


def open_dataset(model=C.NAVGEM, file_type=C.FLATFILE, label="basic", **kwds):
    if model.lower() == C.NEPTUNE and file_type == C.FLATFILE:
        file_type = C.HDF5
    paths = _get_paths(model.lower(), file_type, label)
    df_ds = xnrl.open_dataset(paths, model=model, **kwds)
    C.LOG.debug(f"Opened {paths}")
    return df_ds
