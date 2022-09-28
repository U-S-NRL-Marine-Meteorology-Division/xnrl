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
import re
import warnings

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xnrl.constant as C
import xarray as xr
from mpl_toolkits.axes_grid1 import make_axes_locatable
from xnrl import compute, util

PLOT_KWDS = [
    "x",
    "y",
    "col",
    "sep",
    "ovl",
    "ops",
    "display",
    "num_cols",
    "out_dir",
    "out_fi",
    "show",
]


def _iterate_over(da, key):
    if key:
        if key in da.dims:
            return [da.sel(**{key: k}) for k in da[key]]
        else:
            return [da]
    else:
        return [da]


def _order_of_magnitude(x):
    if x == 0:
        return 0
    else:
        return np.floor(np.log10(np.abs(x)))


def _base_round(value, base=None, method="up"):
    if base is None:
        oom = _order_of_magnitude(value)
        scale = 10 ** oom
        if oom > 0:
            scale = np.log10(scale)
        base = scale * 5

    if method == "up":
        return np.ceil(value / base) * base
    elif method == "down":
        return np.floor(value / base) * base
    elif method == "nearest":
        return np.round(value / base) * base
    else:
        raise ValueError("Method is only valid for up, down, nearest!")


def _add_colorbar(fig, ax, im, colorbar_label):
    divider = make_axes_locatable(ax)
    ax_cb = divider.append_axes("right", size="1.35%", pad=0.1, axes_class=plt.Axes)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fig.add_axes(ax_cb)
    colorbar = plt.colorbar(im, cax=ax_cb, ax=ax)
    colorbar.set_label(colorbar_label)
    return colorbar


def _parse_attrs(da, key):
    if key == "data":
        return da.name
    try:
        attrs = da[key].attrs
        lname = attrs.get("long_name", key)
        units = attrs.get("units", "")
        label = f"{lname} [{units}]" if units else lname
        return label[:20]
    except KeyError:
        return key


def _sanitize_file_label(label):
    return (
        re.sub("[\(\[].*?[\)\]]", "", label)  # noqa
        .strip()
        .replace("  ", "_")
        .replace(" ", "_")
        .replace(",", ".")
    )


def _plot(
    ds,
    da,
    x,
    y,
    col,
    ovl,
    ops,
    num_cols,
    sep="",
    val="",
    sharez=True,
    symmetric=None,
    out_dir=".",
    out_fi=None,
    show=False,
):
    if not show:
        matplotlib.use("agg")
    da = da.load()

    is_geo = (x == "lon" and y == "lat") or (x == "x" and y == "y")
    if is_geo:
        try:
            import cartopy.crs as ccrs
        except ImportError:
            C.LOG.warning("Install cartopy to draw coastlines!")
            ccrs = None
            is_geo = False
        try:
            import metpy
        except ImportError:
            metpy = None
    else:
        ccrs = metpy = None

    if ccrs and is_geo:
        if metpy and x == "x" and y == "y":
            da.attrs.update({"grid_mapping": "crs"})
            crs = da.to_dataset().metpy.parse_cf()["crs"].item().to_cartopy()
            transform = crs
            projection = crs
        else:
            transform = ccrs.PlateCarree()
            projection = ccrs.PlateCarree(central_longitude=180)
        da = da.drop("crs")
    else:
        transform = None
        projection = None

    if col in da.dims:
        len_cols = len(da[col])
        if num_cols is None:
            num_cols = int(np.ceil(np.sqrt(len_cols / 1.25)))
        num_cols = min(num_cols, len_cols)
        num_rows = int(np.ceil(len_cols / num_cols))
        figsize = (4.25 * num_cols, 2 + 2 * num_rows)
        single = False
    else:
        num_cols = 1
        num_rows = 1
        figsize = (8, 5.5) if not is_geo else (6, 4)
        single = True

    legend_proxy = plt.Rectangle((0, 0), 1, 1, facecolor="none")
    fig, axes = plt.subplots(
        num_rows,
        num_cols,
        sharex=True,
        sharey=True,
        figsize=figsize,
        subplot_kw=dict(projection=projection),
    )

    if sharez:
        vmin = _base_round(da.quantile(0.01), method="down")
        vmax = _base_round(da.quantile(0.99), method="up")
        if symmetric or vmin < 0 and vmax > 0:
            base = max(abs(vmin), abs(vmax))
            vmin = -base
            vmax = base
    else:
        vmin = vmax = None

    for p, da_col in enumerate(_iterate_over(da, col)):
        ir = int(p / num_cols)
        ic = p % num_cols
        if num_rows > 1:
            ax = axes[ir, ic]
        elif num_cols > 1:
            ax = axes[ic]
        else:
            ax = axes

        legend_labels = []
        for i, da_ovl in enumerate(_iterate_over(da_col, ovl)):
            da_ovl = da_ovl.squeeze()
            name_ovl = str(da_ovl.name)
            if x == "data" or y == "data":
                im = None
                if ovl:
                    legend_label = util._format_number(da_ovl[ovl].values)
                else:
                    legend_label = ""
                if x == "data":
                    ax.plot(da_ovl.values, da_ovl[y], label=legend_label)
                else:
                    ax.plot(da_ovl[x], da_ovl.values, label=legend_label)
                ax.grid(color="gray", alpha=0.3)
            else:
                if y in da_ovl.dims and x in da_ovl.dims:
                    if("missing_dims" in xr.Dataset.transpose.__code__.co_varnames):
                        # Xarray 0.16 ignored by default. New versions raise exception by default.
                        da_ovl = da_ovl.transpose(y, x, missing_dims="ignore")
                    else:
                        #Backwards Compatible for transpose.
                        da_ovl = da_ovl.transpose(y, x)

                plot_args = (da_ovl[x], da_ovl[y], da_ovl.values)
                plot_kwds = dict(transform=transform, shading="auto")

                if sharez:
                    plot_kwds["vmin"] = vmin
                    plot_kwds["vmax"] = vmax

                if ovl:
                    legend_label = f"{ovl}={util._format_number(da_ovl[ovl].values)}"
                else:
                    legend_label = ""

                if "idx" in da_ovl.dims:
                    im = ax.scatter(
                        *plot_args[:2],
                        s=10,
                        c=da_ovl.values,
                        cmap="RdBu_r",
                        **plot_kwds,
                    )
                    if ovl:
                        colorbar_label = _parse_attrs(ds, ovl)
                    elif sep == "var":
                        colorbar_label = _parse_attrs(ds, val)
                    else:
                        colorbar_label = ""
                    if not sharez or single:
                        colorbar = _add_colorbar(fig, ax, im, colorbar_label)
                elif i == 0:
                    if legend_label:
                        legend_labels.append(f"{legend_label} fill")
                    im = ax.pcolormesh(*plot_args, cmap="RdBu_r", **plot_kwds)
                    if ovl:
                        colorbar_label = _parse_attrs(ds, ovl)
                    elif sep == "var":
                        colorbar_label = _parse_attrs(ds, val)
                    else:
                        colorbar_label = ""
                    if not sharez or single:
                        colorbar = _add_colorbar(fig, ax, im, colorbar_label)
                else:
                    if legend_label:
                        legend_labels.append(f"{legend_label} line")
                    ax.contour(*plot_args, **plot_kwds)

        if i > 0 and legend_labels:
            legend = plt.legend(
                [legend_proxy, legend_proxy],
                legend_labels,
                handlelength=0.5,
                handletextpad=0.5,
                frameon=False,
                bbox_to_anchor=(0.0, 1.02, 1.0, 0.102),
                loc="lower left",
                ncol=2,
                mode="expand",
                borderaxespad=0.0,
                fontsize=6.5,
            )
        elif i > 0:
            legend = ax.legend(ncol=4)
        else:
            legend = None

        if not is_geo:
            if ir == num_rows - 1:
                if x != "data":
                    xlabel = _parse_attrs(ds, x)
                elif col == "var":
                    xlabel = x
                else:
                    xlabel = name_ovl
                ax.set_xlabel(xlabel)
            if ic == 0:
                if y != "data":
                    ylabel = _parse_attrs(ds, y)
                elif col == "var":
                    ylabel = y
                else:
                    ylabel = name_ovl
                ax.set_ylabel(ylabel)
        elif ccrs:
            ax.coastlines()
            gl = ax.gridlines(draw_labels=True, color="gray", linestyle=(0, (5, 8)))
            gl.top_labels = False
            if ir != num_rows - 1:
                gl.bottom_labels = False
            if ic != 0:
                gl.left_labels = False
            gl.right_labels = False
            gl.top_labels = False

        col_label = f"{col}={da_col[col].values}" if col else ""
        title_y = 1.2 if legend else 1
        ax.set_title(col_label, size=10, y=title_y)

    if not is_geo:
        for ax in np.ravel(axes):
            ax.set(frame_on=False)
            ax.tick_params(axis="both", which="both", length=0)

    if im and sharez and not single:
        colorbar = fig.colorbar(im, ax=axes, shrink=0.8, aspect=35)
        colorbar.set_label(colorbar_label)
    coord_label = " ".join(
        [
            f"{key}={value.values}"
            for key, value in da_col.coords.items()
            if key not in [x, y, col, ovl, sep, "var"]
            and len(np.atleast_1d(da_col[key])) == 1
        ]
    )
    sep_label = f"{sep}={val}" if sep else ""
    sup_title_label = f"{coord_label} {sep_label} {ops}".strip()
    if not col:
        ax.set_title(sup_title_label)
    else:
        fig.suptitle(sup_title_label, y=0.98 - num_rows * 0.0075)

    if x == "time":
        fig.autofmt_xdate()
    if not is_geo:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fig.tight_layout(rect=[0, 0.03, 1, 0.985])

    # remove units
    name_fmtd = _sanitize_file_label(da.name if sep != "var" else val)
    col_fmtd = _sanitize_file_label(f"X{col}" if col else "")
    ops_fmtd = _sanitize_file_label(f"_{ops}" if ops else "")
    sep_fmtd = _sanitize_file_label(f"_{sep}.{val}" if sep and sep != "var" else "")

    if not show:
        if out_fi is None:
            out_fi = f"{name_fmtd}_{x}X{y}{col_fmtd}{ops_fmtd}{sep_fmtd}.png"
        out_fp = os.path.join(out_dir, out_fi)
        save_kwds = dict(fname=out_fp, bbox_inches="tight")
        plt.savefig(**save_kwds)
        C.LOG.info(f"Saved to {out_fp}!")
        return ax
    else:
        C.LOG.info(f"Showing to screen!")
        plt.show()


def visualize(
    ds,
    x="lon",
    y="lat",
    col="",
    sep="",
    ovl="",
    ops="",
    num_cols=None,
    out_dir=".",
    out_fi=None,
    show=False,
    display=False,
):
    # remove dimensions that are not attached to any variables
    if isinstance(ds, pd.DataFrame):
        raise ValueError(
            f"Found multiple grid_dims / lev_types; select one to plot!\n"
            f'grid_dims: {np.unique(ds["grid_dim"])}\n'
            f'lev_types: {np.unique(ds["lev_type"])}'
        )

    var_dims = set([])
    for var in ds.data_vars:
        var_dims |= set(ds[var].dims)

    for dim in ds.dims:
        if dim not in var_dims:
            ds = ds.drop(dim)

    # turn data variables into "coordinates"
    labels = list(ds.data_vars)
    da = ds.to_array(dim="var").rename("X".join(labels))
    if len(ds.data_vars) == 1:
        da.attrs.update(**ds[list(ds.data_vars)[0]].attrs)

    # drop these coordinates if they are null
    for coord in ["exp", "mbr"]:
        try:
            coord_val = da[coord].values[0]
            is_mask_untitled = coord_val in [C.MASK_VALUE, C.UNTITLED]
            if len(da[coord]) == 1 and is_mask_untitled:
                da = da.drop(coord)
        except TypeError:
            pass

    # format the coordinates to look better
    plot_dims = {"x": x, "y": y, "col": col, "sep": sep, "ovl": ovl}
    plot_dims_subset = [dim for kwd, dim in plot_dims.items() if kwd != "sep"]
    # if it's not used in x/y because matplotlib handles datetimes
    if "ini" in da.coords and "ini" not in plot_dims_subset:
        da["ini"] = da["ini"].dt.strftime("%Y%m%d%HZ")
    if "time" in da.coords and "time" not in plot_dims_subset:
        da["time"] = da["time"].dt.strftime("%Y%m%d%HZ")
    if "tau" in da.coords:
        tau_unit = "H"
        da["tau"] = da["tau"].astype(float).values / 1e9 / 3600
        if da["tau"].max() > 168:
            da["tau"] = da["tau"] / 24
            tau_unit = "D"
        if "tau" not in plot_dims_subset:
            taus_fmtd = [
                util._format_number(tau) + tau_unit
                for tau in np.atleast_1d(da["tau"].values)
            ]
            da["tau"] = taus_fmtd if "tau" in da.dims else taus_fmtd[0]
        da["tau"].attrs["units"] = tau_unit
    if "lev" in da.coords:
        levs_fmtd = [
            util._format_number(lev, "04.2f") for lev in np.atleast_1d(da["lev"].values)
        ]
        da["lev"] = levs_fmtd if "lev" in da.dims else levs_fmtd[0]
    if "mbr" in da.dims:
        da["mbr"] = [
            util._format_number(mbr, "03.0f") for mbr in np.atleast_1d(da["mbr"].values)
        ]

    # evaluate operations
    da, ops_dims = compute.evaluate(da, ops, return_dims=True)

    # if user wants to plot time, make sure it's in the dims
    if "time" in x or "time" in y or "time" in ops:
        da = da.swap_dims({"tau": "time"}).drop("tau")
    elif "time" in da.coords:
        da = da.drop("time")

    if x not in da.coords:
        x = "data"
    if y not in da.coords:
        y = "data"
    if x == "data" and y == "data":
        y = list(da.squeeze().dims)[0]

    plot_dims["x"] = x
    plot_dims["y"] = y

    da = da.squeeze()
    data_dims = list(da.dims)
    miss_dims = sorted(set(data_dims) - set(plot_dims.values()))
    sub_dims = set()  # e.g. lat (x, y) so sub_dims = [x, y]
    for dim in list(plot_dims.values()) + ops_dims:
        if dim in da.coords:
            sub_dims |= set(da[dim].dims)

    if miss_dims:
        for kwd, dim in plot_dims.items():
            if not dim and len(miss_dims) > 0:
                for miss_dim in miss_dims:
                    not_sub_dim = miss_dim not in sub_dims
                    not_plot_dim = miss_dim not in plot_dims.values()
                    if not_sub_dim and not_plot_dim:
                        plot_dims[kwd] = miss_dim
                        break

        remain_dims = set(miss_dims) - set(plot_dims.values()) - sub_dims
        if len(remain_dims) > 0:
            print(
                f"Unable to account for {remain_dims}; "
                f"selecting the first value if not a subset dim!"
            )
            da = da.isel(**{dim: 0 for dim in remain_dims})

    col = plot_dims["col"] if plot_dims["col"] in da.dims else ""
    sep = plot_dims["sep"] if plot_dims["sep"] in da.dims else ""
    ovl = plot_dims["ovl"] if plot_dims["ovl"] in da.dims else ""
    C.LOG.info(f"\n\tx={x}\n\ty={y}\n\tcol={col}\n\tsep={sep}\n\tovl={ovl}")

    if display:
        print(da)
        return

    plot_args = [x, y, col, ovl, ops, num_cols]
    plot_kwds = dict(out_dir=out_dir, out_fi=out_fi, show=show)
    if sep:
        for val in da[sep].values:
            return _plot(
                ds, da.sel(**{sep: val}), *plot_args, sep=sep, val=val, **plot_kwds
            )
    else:
        return _plot(ds, da, *plot_args, **plot_kwds)