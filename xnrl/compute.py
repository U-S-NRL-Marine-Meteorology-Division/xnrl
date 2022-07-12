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
import pandas as pd
import xarray as xr
import xnrl.constant as C
from xnrl import packages


def get_weights(ds, dim="lat"):
    if "m2d" in ds:
        return ds["m2d"]

    weights = np.abs(np.cos(np.deg2rad(ds[dim]))).clip(0.0, 1.0)

    _, weights = xr.broadcast(ds, weights)
    weights.coords[dim] = ds[dim]
    return weights.rename("weights")


def apply_ops(da, ops, dims, weights=None, **kwds):
    if weights is None and ("lat" in dims or "idx" in dims):
        C.LOG.debug(
            "No weights were passed, but detected lat in dims; "
            "automatically applying weighted operations!"
        )
        weights = get_weights(da)

    if hasattr(da, ops):
        if weights is None:
            return getattr(da, ops)(dims, **kwds)
        else:
            try:
                return getattr(da.weighted(weights), ops)(dims, **kwds)
            except AttributeError:
                C.LOG.warning(
                    f"Could not apply weighted {ops} on {dims}; "
                    f"falling back to non-weighted {ops}!"
                )
                return getattr(da, ops)(dims, **kwds)
    elif hasattr(np, ops):
        try:
            return getattr(np, ops)(da, dims).assign_attrs(da.attrs)
        except TypeError:
            return getattr(np, ops)(da).assign_attrs(da.attrs)
    else:
        xs = packages._import_xskillscore()
        if hasattr(xs, ops):
            return getattr(xs, ops)(
                *[da.sel(**{"exp": exp}) for exp in da["exp"]],
                dims,
                weights=weights,
                **kwds,
            )
        else:
            raise NotImplementedError(
                f"Could not apply {ops} on {dims} for {da.coords}"
            )


def evaluate(da, ops, return_dims=False):
    dims_list = []

    if not ops:
        if return_dims:
            return da, dims_list
        else:
            return da

    if "var" in ops:
        associated_dims = set()
        for var in da.data_vars:
            associated_dims |= set(da[var].dims)
        unassociated_dims = {
            dim for dim in da.dims if set(da[dim].dims) - associated_dims
        }
        da = da.drop(list(unassociated_dims)).to_array(name="var")

    if isinstance(da, pd.DataFrame):
        da["ds"] = [evaluate(d, ops) for d in da["ds"]]
        if return_dims:
            return da, dims_list
        else:
            return da

    ops = ops.replace("+", ",")
    ops_list = ops.replace(" ", "").split("...")
    for op_str in ops_list:
        try:
            op, dims_str = op_str.split(".")
            dims = dims_str.split(",")
        except ValueError:  # absolute value
            op = op_str
            dims = []

        for dim in dims:
            if dim not in da.dims and dim not in da.coords:
                raise ValueError(f"{dim} not found in dataset:\n{da.coords}")
            elif dim == "lon" and "x" in da.dims:
                dims[dims.index(dim)] = "x"
            elif dim == "lat" and "y" in da.dims:
                dims[dims.index(dim)] = "y"
            elif dim == "lon" and "idx" in da.dims:
                dims[dims.index(dim)] = "idx"
            elif dim == "lat" and "idx" in da.dims:
                dims[dims.index(dim)] = "idx"
            elif dim not in da.dims:
                da = da.expand_dims(dim)
            dims_list.append(dim)

        if len(dims) == 1:
            dims = dims[0]

        try:
            da = apply_ops(da, op, dims, skipna=True, keep_attrs=True)
        except TypeError:
            da = apply_ops(da, op, dims)

    if return_dims:
        return da, dims_list
    else:
        return da
