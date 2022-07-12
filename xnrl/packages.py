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

import importlib

import xnrl.constant as C


def _null_import(*args, **kwargs):
    return args[0]


def _import_tqdm():
    try:
        from tqdm import tqdm
    except (ImportError, ModuleNotFoundError):
        tqdm = _null_import
        C.LOG.warning(
            "Could not import tqdm; enter `pip install tqdm` in the terminal "
            "to view progress bars!"
        )
    return tqdm


def _import_h5py():
    try:
        import h5py
    except (ImportError, ModuleNotFoundError):
        raise ImportError(
            "Could not import h5py; enter `pip install h5py` in the terminal "
            "to read HDF5 files!"
        )
    return h5py


def _import_metpy():
    try:
        import metpy
    except (ImportError, ModuleNotFoundError):
        raise ImportError(
            "Could not import metpy; enter "
            "`pip install metpy` in the terminal "
            "to auto detect coordinates!"
        )
    return metpy


def _import_cf_xarray():
    try:
        import cf_xarray
    except (ImportError, ModuleNotFoundError):
        raise ImportError(
            "Could not import cf_xarray; enter "
            "`pip install cf_xarray` in the terminal "
            "to auto detect coordinates!"
        )
    return cf_xarray


def _import_scipy():
    try:
        import scipy.spatial
        import scipy.interpolate
    except (ImportError, ModuleNotFoundError):
        raise ImportError(
            "Could not import scipy; enter "
            "`pip install scipy` in the terminal "
            "to improve runtimes of nearest neighbor!"
        )
    return scipy


def _import_projection():
    try:
        import cartopy.crs as ccrs
        from metpy.plots.mapping import CFProjection as cf_proj
    except (ImportError, ModuleNotFoundError):
        C.LOG.warning(
            "Could not import cartopy / metpy; enter "
            "`conda install -c conda-forge cartopy` and "
            "`pip install metpy` in the terminal "
            "to assign projections!"
        )
        ccrs = None
        cf_proj = None
    return ccrs, cf_proj


def _import_geodesic():
    try:
        import cartopy.geodesic as cgeo
    except (ImportError, ModuleNotFoundError):
        C.LOG.warning(
            "Could not import cartopy; enter "
            "`conda install -c conda-forge cartopy"
            "to assign projections!"
        )
        cgeo = None
    return cgeo


def _import_interpolate():
    try:
        from metpy import interpolate
    except (ImportError, ModuleNotFoundError):
        C.LOG.warning(
            "Could not import metpy; enter `conda install -c conda-forge metpy"
        )
        interpolate = None
    return interpolate


def _import_grib():
    grib_lib = None
    for grib_name in ["pygrib", "cfgrib", "Nio"]:
        try:
            grib_lib = importlib.import_module(grib_name)
            break
        except (ImportError, ModuleNotFoundError):
            pass

    if grib_lib is None:
        raise ImportError(
            "Could not import pygrib, cfgrib, and Nio; enter "
            "`conda install -c conda-forge pygrib`, "
            "`conda install -c conda-forge cfgrib`, or"
            "`conda install -c conda-forge Nio`, "
            "to read grib files!"
        )
    return grib_name, grib_lib


def _import_dask():
    try:
        import dask
        import dask.delayed
        import dask.diagnostics
    except (ImportError, ModuleNotFoundError):
        dask = _null_import
        C.LOG.warning(
            "Could not import dask; enter `pip install dask[delayed]` "
            "in the terminal to utilize lazy reads!"
        )
    return dask


def _import_xskillscore():
    try:
        import xskillscore as xs
    except (ImportError, ModuleNotFoundError):
        xs = _null_import
        C.LOG.warning(
            "Could not import xskillscore; enter `pip install xskillscore` "
            "in the terminal to utilize xskillscore!"
        )
    return xs


def _import_xesmf():
    try:
        import xesmf as xe
    except (ImportError, ModuleNotFoundError):
        xe = _null_import
        raise ImportError(
            "Could not import xesmf; enter `conda install -c conda-forge xesmf` "
            "in the terminal to utilize xesmf!"
        )
    return xe


def _import_cftime():
    try:
        import cftime
    except (ImportError, ModuleNotFoundError):
        cftime = _null_import
        raise ImportError(
            "Could not import cftime; enter `pip install cftime` "
            "in the terminal to utilize cftime!"
        )
    return cftime
