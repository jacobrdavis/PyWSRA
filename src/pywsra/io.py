"""
Input/output methods.

TODO:
- consider replacing with xr.open_mfdataset
    https://docs.xarray.dev/en/stable/generated/xarray.open_mfdataset.html
- update forward slash compatibility for windows OS
- save as nc
"""

__all__ = [
    "read_wsra_directory",
    "read_wsra_file"
]

import glob
import os
from typing import List

import pandas as pd
import numpy as np
import xarray as xr


ATTR_KEYS = ['title', 'history', 'flight_id', 'mission_id', 'storm_id',
             'date_created', 'time_coverage_start', 'time_coverage_end']


def read_wsra_file(filepath: str, index_by_time: bool = True):
    """
    Read Level 4 WSRA data as an xarray Dataset.

    Args:
        filepath (str): path to WSRA file.
        index_by_time (bool, optional): if True, use time as the primary index.
            Otherwise use the default `trajectory`. Defaults to True.

    Returns:
        xr.Dataset: WSRA dataset.
    """
    wsra_ds = xr.open_dataset(filepath)

    if index_by_time:
        wsra_ds = _replace_coord_with_var(wsra_ds, 'trajectory', 'time')
        wsra_ds = wsra_ds.sortby('time')

    return wsra_ds

#TODO: use xr.open_mfdataset() instead with concat kw arguments
def read_wsra_directory(
    directory: str,
    file_type: str = 'nc',
    index_by_time: bool = True,
    **concat_kwargs
) -> xr.Dataset:
    """
    Read a directory of Level 4 WSRA data files and concatenate into a Dataset.

    Args:
        directory (str): directory containing WSRA files
        file_type (str, optional): WSRA data file type. Defaults to '.nc'.
        index_by_time (bool, optional): if True, use time as the primary index.
            Otherwise use the default `trajectory`. Defaults to True.
        concat_kwargs (optional): additional keyword arguments to be
            passed to the xr.concat method.

    Raises:
        FileNotFoundError: If no files of type `file_type` are found
            inside of `directory`.

    Returns:
        xr.Dataset: all WSRA files concatenated into a single Dataset.
    """
    wsra_files = glob.glob(os.path.join(directory, '*' + file_type))
    wsra_files.sort()

    wsra = {}
    if not wsra_files:
        raise FileNotFoundError(
            f'No files of type "{file_type}" found in "{directory}". '
            'Please double check the directory and file_type.')

    for file in wsra_files:
        wsra_ds = read_wsra_file(file, index_by_time)
        # TODO: update forward slash compatibility for windows OS:
        key = file.split('/')[-1].split('.')[0]
        wsra[key] = wsra_ds

    if index_by_time:
        concat_dim = 'time'
    else:
        concat_dim = 'trajectory'

    wsra_ds_concat = xr.concat(list(wsra.values()),
                               dim=concat_dim,
                               combine_attrs=_combine_attrs,
                               **concat_kwargs)

    return wsra_ds_concat


def _replace_coord_with_var(
    ds: xr.Dataset,
    coord: str,
    var: str
) -> xr.Dataset:
    """Replace a Dataset coordinate with another variable of the same shape.

    Note: `coord` and `var` must have the same shape.  The original coord is
    dropped in this process.

    Args:
        ds (xr.Dataset): The xarray Dataset to operate on.
        coord (str): Coordinate to be replaced.
        var (str): Variable to replace it with.

    Returns:
        xr.Dataset: The xarray Dataset with coord replaced by var.
    """
    ds.coords[coord] = ds[var]
    dropped = ds.drop_vars([var])
    renamed = dropped.rename({coord: var})
    return renamed


def _combine_attrs(variable_attrs: List, context=None) -> dict:
    """ WSRA attribute handler passed to xr.concat.

    If `variable_attrs` contains metadata, concatenate the attributes
    accordingly.  Otherwise, if `variable_attrs` contains variable
    descriptions, pass back the first set of attributes.

    Args:
        variable_attrs (List): Attribute dictionaries to combine.
        TODO: context (_type_, optional): _description_. Defaults to None.

    Returns:
        dict: Combined attributes.
    """
    # TODO: check if any of ATTR_KEYS in keys?
    if 'title' in variable_attrs[0].keys():    #  all(key in variable_attrs[0].keys() for key in ATTR_KEYS)
        attrs = _concat_attrs(variable_attrs)
    else:
        attrs = variable_attrs[0]
    return attrs


def _concat_attrs(variable_attrs: List):
    """Concatenate WSFA metadata attributes.

    Handle attributes during concatenation of WSRA Datasets.  Assumes all
    attribute dictionaries in `variable_attrs` contain the same keys but with
    possibly different values.  Where possible, unique values are taken.
    Otherwise, values are aggregated into a list.

    Args:
        variable_attrs (List): Attribute dictionaries to combine.

    Raises:
        KeyError: if `variable_attrs.keys()` contains a key not in `ATTR_KEYS`.

    Returns:
        dict: Combined attributes.
    """
    attrs = {k: [] for k in ATTR_KEYS}
    for key in ATTR_KEYS:
        if key == 'title':
            attrs[key] = _get_unique_attrs(variable_attrs, key)
        elif key == 'history':
            attrs[key] = _get_unique_attrs(variable_attrs, key)
        elif key == 'flight_id':
            attrs[key] = _aggregate_attrs(variable_attrs, key)
        elif key == 'mission_id':
            attrs[key] = _aggregate_attrs(variable_attrs, key)
        elif key == 'storm_id':
            attrs[key] = _get_unique_attrs(variable_attrs, key)[0]
        elif key == 'date_created':
            attrs[key] = _aggregate_attrs(variable_attrs, key)
        elif key == 'time_coverage_start':
            attrs[key] = np.sort(_attrs_to_datetime(variable_attrs, key))[0]
        elif key == 'time_coverage_end':
            attrs[key] = np.sort(_attrs_to_datetime(variable_attrs, key))[-1]
        else:  # TODO: probably don't need to raise an exception. Warning?
            raise KeyError(f'Key `{key}` not a valid attribute: {ATTR_KEYS}.')
    return attrs


def _get_unique_attrs(variable_attrs, key) -> List:
    """ Return unique values from a set of attributes """
    all_attrs = _aggregate_attrs(variable_attrs, key)
    return list(np.unique(all_attrs))  # TODO: try replacing with built-in


def _aggregate_attrs(variable_attrs, key) -> List:
    """ Aggregate all attributes into a list """
    return [attrs[key] for attrs in variable_attrs]


def _attrs_to_datetime(variable_attrs, key) -> List:
    """ Convert date-like attributes to datetimes """
    all_attrs = _aggregate_attrs(variable_attrs, key)
    return list(pd.to_datetime(all_attrs))

