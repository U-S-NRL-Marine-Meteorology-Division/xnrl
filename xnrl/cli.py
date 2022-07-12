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

import argparse
import datetime

import pandas as pd
import xnrl.constant as C
from xnrl import compute, io, main, plot

pd.set_option("display.max_rows", 500)
pd.set_option("display.max_columns", 20)
pd.set_option("display.width", 1000)


def _str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def _is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def _str2value(v):
    if isinstance(v, (int, float)):
        return v
    elif isinstance(v, str):
        if "," in v or ":" in v:
            key = "," if "," in v else ":"
            v = v.lstrip("(").rstrip(")").replace(" ", "")
            v = slice(*map(float, v.split(key)))
        elif _is_number(v):
            v = float(v)
    return v


def _log_elapsed(ini):
    end = datetime.datetime.utcnow()
    C.LOG.info(f"finished in {end - ini}")


def _create_parser(add_ops=False):
    parser = argparse.ArgumentParser(
        description="Show an overview of input paths, similar to ncdump -h."
    )
    parser.add_argument("paths", type=str, nargs="+", help="paths to read")
    parser.add_argument("-mo", "--model", type=str, nargs="?", help="file model")
    parser.add_argument(
        "-lt", "--lev_types", type=str, nargs="+", help="level types to keep"
    )
    parser.add_argument(
        "-gd", "--grid_dims", type=str, nargs="+", help="grid dimensions to keep"
    )
    parser.add_argument("-f", "--fields", type=str, nargs="+", help="fields to keep")
    parser.add_argument("-e", "--exps", type=str, nargs="+", help="experiments to keep")
    parser.add_argument("-mb", "--mbrs", type=str, nargs="+", help="members to keep")
    parser.add_argument(
        "-i",
        "--inis",
        type=_str2value,
        nargs="+",
        help="initialization DTGs YYYYMMDDHH to keep",
    )
    parser.add_argument(
        "-t", "--taus", type=_str2value, nargs="+", help="forecast hours to keep"
    )
    parser.add_argument(
        "-le", "--levs", type=_str2value, nargs="+", help="levels to keep"
    )
    parser.add_argument(
        "-la", "--lats", type=_str2value, nargs="+", help="latitudes to keep"
    )
    parser.add_argument(
        "-lo", "--lons", type=_str2value, nargs="+", help="longitudes to keep"
    )
    parser.add_argument(
        "-lr",
        "--lons_range",
        type=int,
        nargs="?",
        default=None,
        help="longitude range",
    )
    parser.add_argument(
        "-wd",
        "--wrap_df",
        type=_str2bool,
        nargs="?",
        const=True,
        default=None,
        help="wrap dataframe",
    )
    parser.add_argument(
        "-nt",
        "--num_threads",
        type=int,
        nargs="?",
        default=C.NUM_THREADS,
        help="number of threads to use",
    )
    parser.add_argument(
        "-mv",
        "--mask_value",
        type=float,
        nargs="?",
        default=C.MASK_VALUE,
        help="mask value",
    )
    parser.add_argument(
        "-ll",
        "--log_lev",
        type=str,
        nargs="?",
        const="debug",
        default=C.LOG_LEV,
        help="log level",
    )
    parser.add_argument(
        "-ch",
        "--chunks",
        type=_str2bool,
        nargs="?",
        const=True,
        default=None,
        help="whether to lazily load data",
    )
    parser.add_argument(
        "-op",
        "--ops",
        type=str,
        default="",
        help=("mathematical / statistical operations; e.g. `mean.lat,lon...diff.exp`"),
    )
    parser.add_argument(
        "-om",
        "--only_meta",
        type=_str2bool,
        nargs="?",
        const=True,
        default=None,
        help="whether to only show metadata",
    )
    parser.add_argument(
        "-el", "--exps_labels", type=str, nargs="+", help="relabel experiments",
    )

    if add_ops:
        parser.add_argument(
            "-od",
            "--out_dir",
            type=str,
            default=C.WORK_DIR,
            help="the output directory",
        )
        parser.add_argument(
            "-of", "--out_fi", type=str, default=None, help="the output file"
        )
        parser.add_argument(
            "-sh",
            "--show",
            default=False,
            action="store_true",
            help="whether to show without saving",
        )
        parser.add_argument(
            "-d",
            "--display",
            action="store_true",
            help="whether to display DataArray for viewing",
        )

    return parser


def xdump():
    ini = datetime.datetime.utcnow()
    parser = _create_parser()
    parser.add_argument(
        "-v", "--variables", type=str, nargs="+", help="variables to examine"
    )
    kwargs = vars(parser.parse_args())
    if not kwargs["ops"]:
        if kwargs["only_meta"] is None:
            kwargs["only_meta"] = True
        if kwargs["chunks"] is None:
            kwargs["chunks"] = True
    ops = kwargs.pop("ops")
    variables = kwargs.pop("variables")
    ds_df = main.open_dataset(**kwargs)

    if ops:
        ds_df = compute.evaluate(ds_df, ops)

    print("\n")
    if variables is not None:
        for var in variables:
            print(var)
            print(ds_df[var].values)
            if not isinstance(ds_df, pd.DataFrame):
                print(f"attrs: {ds_df[var].attrs}")
            print("\n")
    else:
        print(ds_df)
        print("\n")

    _log_elapsed(ini)


def xoutf():
    ini = datetime.datetime.utcnow()
    parser = _create_parser(add_ops=True)
    parser.add_argument(
        "-ft",
        "--file_type",
        choices=[C.HDF5, C.FLATFILE, C.NETCDF],
        default=C.HDF5,
        help="export file type",
    )
    kwargs = vars(parser.parse_args())
    if not kwargs["ops"]:
        if kwargs["only_meta"] is None:
            kwargs["only_meta"] = False
        if kwargs["chunks"] is None:
            kwargs["chunks"] = True
    kwargs["wrap_df"] = True if kwargs["only_meta"] else False
    ops = kwargs.pop("ops")
    display_ = kwargs.pop("display")
    out_dir = kwargs.pop("out_dir")
    out_fi = kwargs.pop("out_fi")
    show = kwargs.pop("show")
    file_type = kwargs.pop("file_type")
    df = main.open_dataset(**kwargs)
    C.LOG.debug(df)

    if ops:
        df = compute.evaluate(df, ops)

    if display_:
        print(df)
        _log_elapsed(ini)
        return

    if file_type == C.HDF5:
        out_fp = io.export_hdf5(df, out_dir=out_dir, out_fi=out_fi, show=show)
    elif file_type == C.FLATFILE:
        out_fp = io.export_flatfile(df, out_dir=out_dir, show=show)
    elif file_type == C.NETCDF:
        out_fp = io.export_netcdf(df, out_dir=out_dir, out_fi=out_fi, show=show)
    else:
        raise NotImplementedError(
            f"{file_type} unsupported; select from "
            f"{C.HDF5}, {C.FLATFILE}, or {C.NETCDF}"
        )

    C.LOG.info(f"Saved to {out_fp}!")
    _log_elapsed(ini)


def xplot():
    ini = datetime.datetime.utcnow()
    parser = _create_parser(add_ops=True)
    parser.add_argument(
        "-x", "--x", type=str, default="lon", help="dimension to plot on the x-axis"
    )
    parser.add_argument(
        "-y", "--y", type=str, default="lat", help="dimension to plot on the y-axis"
    )
    parser.add_argument(
        "-c", "--col", type=str, default="", help="dimension to plot as columns"
    )
    parser.add_argument(
        "-o", "--ovl", type=str, default="", help="dimension to plot as overlays"
    )
    parser.add_argument(
        "-s", "--sep", type=str, default="", help="dimension to plot as separate files"
    )
    parser.add_argument(
        "-nc",
        "--num_cols",
        type=int,
        default=None,
        help="number of columns before wrapping",
    )

    kwargs = vars(parser.parse_args())
    plot_kwargs = {key: kwargs.pop(key) for key in plot.PLOT_KWDS}
    ds = main.open_dataset(**kwargs)
    plot.visualize(ds, **plot_kwargs)
    _log_elapsed(ini)


if __name__ == "__main__":
    main()
