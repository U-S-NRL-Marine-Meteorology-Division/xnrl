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

import logging
import os
import shutil
import traceback
from collections import defaultdict
from datetime import datetime
from itertools import zip_longest

import pandas as pd
import xarray as xr
from xnrl import constant as C
from xnrl import util
from xnrl.flatfile import COAMPSFlatFile, NAVGEMFlatFile, NAVGEMTGZFlatFile, GFSFlatFile
from xnrl.grads import CM1GrADSFile, NAVGEMGrADSFile
from xnrl.grib import COAMPSGRIBFlatFile, GenericGRIBFile, NAVGEMGRIBFlatFile
from xnrl.hdf5 import COAMPSHDF5FlatFile, NAVGEMHDF5File, NEPTUNEHDF5File
from xnrl.netcdf import GenericNetCDFFile, RadianceNetCDFFile, RAOBNetCDFFile
from xnrl.visual import COAMPSVisualFile

ALL_ENGINES_MAPPING = {
    C.RADIANCE: {C.NETCDF: RadianceNetCDFFile},
    C.RAOB: {C.NETCDF: RAOBNetCDFFile},
    C.NAVGEM: {
        C.FLATFILE: NAVGEMFlatFile,
        C.GRIB: NAVGEMGRIBFlatFile,
        C.TGZ: NAVGEMTGZFlatFile,
        C.HDF5: NAVGEMHDF5File,
        C.GRADS: NAVGEMGrADSFile,
        C.NETCDF: GenericNetCDFFile,
    },
    C.GFS: {C.FLATFILE: GFSFlatFile, C.NETCDF: GenericNetCDFFile,},
    C.CM1: {C.FLATFILE: NAVGEMFlatFile, C.GRADS: CM1GrADSFile},
    C.COAMPS: {
        C.FLATFILE: COAMPSFlatFile,
        C.GRIB: COAMPSGRIBFlatFile,
        C.HDF5: COAMPSHDF5FlatFile,
        C.NETCDF: GenericNetCDFFile,
        C.VISUAL: COAMPSVisualFile,
    },
    C.NEPTUNE: {
        C.FLATFILE: NAVGEMFlatFile,
        C.HDF5: NEPTUNEHDF5File,
        C.NETCDF: GenericNetCDFFile,
    },
    C.NEPTUNE_LAM: {
        C.FLATFILE: COAMPSFlatFile,
        C.HDF5: NEPTUNEHDF5File,
        C.NETCDF: GenericNetCDFFile,
    },
    C.UNTITLED: {
        C.FLATFILE: NAVGEMFlatFile,
        C.GRIB: GenericGRIBFile,
        C.NETCDF: GenericNetCDFFile,
    },
}


def _get_file_types(path):
    path = path.lower()
    grib_ends_with = path.endswith(("gr2", "grb", "grib", "grib2"))
    grib_starts_with = os.path.basename(path).startswith("US")
    if path.endswith("fld"):
        file_type = C.FLATFILE
    elif path.endswith("grd1"):
        file_type = C.VISUAL
    elif path.endswith(("tgz", "tar", "gz")):
        file_type = C.TGZ
    elif ".dat" in path or ".ctl" in path:
        file_type = C.GRADS
    elif path.endswith((".hdf", ".hdf5", "h5")):
        file_type = C.HDF5
    elif grib_ends_with or grib_starts_with:
        file_type = C.GRIB
    elif path.endswith(("nc", "nc4")):
        file_type = C.NETCDF
    else:
        file_type = None

    if os.path.isdir(path):
        C.LOG.warning(f"{path} is a directory, not a file!")

    if file_type is None:
        C.LOG.debug(
            f"Unable to determine file type of {path}! "
            f"Will iterate through all available file types!"
        )
        file_types = C.ALL_FILE_TYPES
    else:
        file_types = [file_type]

    return file_types


def _attempt_model(model, wrangled_paths, attempted_engines, kwargs):
    if isinstance(model, list):
        path_dict = defaultdict(list)
        for path in wrangled_paths:
            key = os.path.split(path)[0] + os.path.splitext(path)[-1]
            path_dict[key].append(path)
        wrap_df = kwargs["wrap_df"]

        paths = path_dict.values()
        dfs = []
        for mod, ps in zip_longest(model, paths, fillvalue=model[-1]):
            kwargs.update(**{"model": model, "wrap_df": True})
            file_types = _get_file_types(ps[0])
            df, attempted_engines = _attempt_model(mod, ps, attempted_engines, kwargs)
            dfs.append(df)
        df = pd.concat(dfs)

        merged_attrs = {}
        for ds in df["ds"]:
            for attr in ds.attrs:
                if attr not in merged_attrs.keys():
                    merged_attrs[attr] = ds.attrs[attr]
                else:
                    merged_attrs[attr] += f" & {ds.attrs[attr]}"
        ds = xr.combine_by_coords(df["ds"]).assign_attrs(**merged_attrs)

        if wrap_df is None or not wrap_df:
            dataset = ds
        else:
            dataset = df.iloc[[0]].assign(**{"ds": [ds], "model": ds.attrs["model"]})
        return dataset, attempted_engines
    else:
        file_types = _get_file_types(wrangled_paths[0])

    engines = ALL_ENGINES_MAPPING.get(model.lower(), [])
    kwargs["model"] = model.upper()
    for file_type in file_types:
        if len(engines) == 0:
            if file_type in [C.NETCDF, C.GRIB]:
                engines = ALL_ENGINES_MAPPING[C.UNTITLED]
            else:
                engines = ALL_ENGINES_MAPPING[C.NAVGEM]

        if file_type not in engines:
            continue
        else:
            engine = engines[file_type]
            attempted_engines.append(engine)

        terminal_width = int(shutil.get_terminal_size((80, 20))[0] / 3)
        divider = (
            "\n" + "#" * terminal_width + f" {engine.__name__} " + "#" * terminal_width
        )

        C.LOG.debug(divider)
        try:
            dataset = engine(**kwargs).open_dataset(wrangled_paths)
            return dataset, attempted_engines
        except Exception:
            exc = traceback.format_exc()
            exc_split = (
                "During handling of the above exception, another exception occurred:"
            )
            if exc_split in exc:
                exc = "\n".join(exc.split(exc_split)).strip()
            dataset = f"{engine.__name__} engine failed to read {model}!\n\n{exc}"
            return dataset, attempted_engines

    return None, attempted_engines


def _try_model_engine(model, wrangled_paths, kwargs):
    if model is None or model == "*":
        C.LOG.warning(
            "Model was not specified so iterating through "
            "available ones! Specify model for a more "
            "optimized run!"
        )
        models = ALL_ENGINES_MAPPING.keys()
    else:
        models = [model]

    dataset = None
    attempted_engines = []
    for model in models:
        ds_or_err, attempted_engines = _attempt_model(
            model, wrangled_paths, attempted_engines, kwargs
        )
        if isinstance(ds_or_err, str):
            dataset = ds_or_err
        elif ds_or_err is not None:
            return ds_or_err

    if len(attempted_engines) == 0:
        raise RuntimeError(f"No engines available to read {model} files!")
    elif dataset is None or isinstance(dataset, str):
        if len(wrangled_paths) > 6:
            joined_paths = "\n".join(
                wrangled_paths[0:3] + ["..."] + wrangled_paths[-4:-1]
            )
        else:
            joined_paths = "\n".join(wrangled_paths)
        attempted_engines = ", ".join(engine.__name__ for engine in attempted_engines)
        raise ValueError(
            f"Failed to read the following paths with "
            f"these engines ({attempted_engines}); "
            f"the last attempt's traceback is below. "
            f"Ensure that the specified model is correct, "
            f'and set log_lev="debug" to see more info!\n\n'
            f"Paths:\n{joined_paths}\n\n{dataset}"
        )


def open_dataset(
    paths,
    model=None,
    lev_types=None,
    grid_dims=None,
    fields=None,
    exps=None,
    mbrs=None,
    inis=None,
    taus=None,
    times=None,
    levs=None,
    lats=None,
    lons=None,
    ys=None,
    xs=None,
    exps_labels=None,
    lons_range=None,
    wrap_df=None,
    num_threads=C.NUM_THREADS,
    mask_value=None,
    log_lev=C.LOG_LEV,
    leave_progress=None,
    chunks=None,
    only_meta=False,
    drop_inis=False,
    max_files=None,
    max_groups=None,
    error_out=True,
    validate_size=True,
    method=C.METHOD,
    regrid_kwds=None,
    interp_kwds=None,
    temporal_dim=C.TEMPORAL_DIM,
    merge_lev_types=False,
    stationary_coords=True,
    datahd_path=None,
):
    """
    Open and decode a dataset from a US Naval Research Laboratory file.
    File types include binary flatfiles, GRIB, HDF5, and netCDF.

    Args:
        paths (str / list): wildcard strings, list, or nested wildcard
        model (str / list): the model label
        lev_types (str / list): level types or groups to subset from files
        grid_dims (str / list): grid dimensions to subset from files
        fields (str / list): fields to subset from files
        exps (str / list): experiment labels to subset from files
        inis (str / list): datetime groups (%Y%m%d%H) to subset from files
        taus (str / list / slice): forecast hours to subset from files
        times (str / list / slice): target times to subset from files
        levs (str / list / slice): vertical levels to subset from files
        lats (str / list / slice): latitudes to subset from files
        lons (str / list / slice): longitudes to subset from files
        ys (str / list / slice): generic ys to subset from files
        xs (str / list / slice): generic xs to subset from files
        exps_labels (str, list): new experiment labels
        lons_range (int): the range of longitudes; select...
            180 for longitudes to range from -180 to 180
            360 (default) for longitudes to range from 0 to 360
        wrap_df (bool): to wrap the xr.Dataset with a pd.DataFrame; select...
            True to ensure a pd.DataFrame is always returned,
            False or None (default) to return an xr.Dataset if possible
        num_threads (int): number of threads to utilize when opening files;
            only recommended to increment with flatfiles and grib files.
        mask_value (float): the value to mask; defaults to -9999
        log_lev (str): the log level; select...
            DEBUG for debugging
            INFO for typical info
            WARNING for only warnings
        leave_progress (bool): to hide the progress bar after completion;
            requires tqdm to be installed
        chunks (bool / str / dict): to lazily open dataset if provided;
            requires dask to be installed
        only_meta (bool): to only return parsed metadata as a pd.DataFrame
        drop_inis (bool): drop inis that do not exist across all exps
        max_files (int): maximum number of files to read
        max_groups (int): maximum number of lev_types + grid_dims to read
        error_out (bool): whether to raise exception
        validate_size (bool): whether to check input file size
        method (str): method for selection / interpolation; select...
            nearest to get nearest coordinate
            ffill / pad to forward fill previous coordinate
            bfill / backfill to backward fill following coordinate
            linear / cubic for interpolation
        regrid_kwds (dict): regridding keywords; only valid for native grids
        interp_kwds (dict): interpolation keywords
        temporal_dim (str): the unlimited temporal dimension; select...
            tau for forecast hour
            time for valid time
        merge_lev_types (bool): forcibly merge level types;
            may overwrite variables with identical names
        stationary_coords (bool): whether to only read a single coord file;
            if not, uses a unique coord file for each separate tau,
            but will forward fill the previous coord if non-existent;
            only valid for COAMPS IEEE flatfiles
        datahd_path (str): path to COAMPS IEEE datahd flatfiles;
            applicable if datahd is not in the same directory or one up;
            only valid for COAMPS IEEE flatfiles
    Returns:
        df (pd.DataFrame) - nests an xarray dataset within each row, where
            each row contains a unique level type and grid dimension
    """
    ini_dt = datetime.utcnow()

    if isinstance(log_lev, str):
        log_lev = log_lev.upper()
        if log_lev not in C.LOG_LEVS:
            C.LOG.warning(f"Please select a supported log level: {C.LOG_LEVS}")
            log_lev = "INFO"
        C.LOG.setLevel(getattr(logging, log_lev))
    elif isinstance(log_lev, int):
        C.LOG.setLevel(log_lev)

    kwargs = dict(
        model=model,
        lev_types=lev_types,
        grid_dims=grid_dims,
        fields=fields,
        exps=exps,
        mbrs=mbrs,
        inis=inis,
        taus=taus,
        times=times,
        levs=levs,
        lats=lats,
        lons=lons,
        ys=ys,
        xs=xs,
        exps_labels=exps_labels,
        lons_range=lons_range,
        wrap_df=wrap_df,
        num_threads=num_threads,
        mask_value=mask_value,
        log_lev=log_lev,
        leave_progress=leave_progress or log_lev == "debug",
        chunks=chunks,
        only_meta=only_meta,
        drop_inis=drop_inis,
        max_files=max_files,
        max_groups=max_groups,
        error_out=error_out,
        validate_size=validate_size,
        method=method,
        regrid_kwds=regrid_kwds,
        interp_kwds=interp_kwds,
        temporal_dim=temporal_dim,
        merge_lev_types=merge_lev_types,
        stationary_coords=stationary_coords,
        datahd_path=datahd_path,
    )

    if "*" in paths:
        C.LOG.info(f"Searching for paths to open using {paths}...")
    else:
        C.LOG.debug(f"Searching for paths to open using {paths}...")

    wrangled_paths = util._wrangle_paths(paths)
    num_paths = len(wrangled_paths)

    try:
        if num_paths == 0:
            raise FileNotFoundError(f"No files found for paths={paths}")

        C.LOG.debug(f"Found {num_paths} paths to open!")

        dataset = _try_model_engine(model, wrangled_paths, kwargs)
        end_dt = datetime.utcnow()
        elapsed_td = end_dt - ini_dt
        C.LOG.debug(f"xnrl finished successfully in {elapsed_td}!")
    except Exception as e:
        if error_out:
            raise e
        else:
            C.LOG.warning(f"xnrl failed due to {e}; returning empty dataset.")
            dataset = pd.DataFrame() if wrap_df else xr.Dataset()

    return dataset
