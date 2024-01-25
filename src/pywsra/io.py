"""
Input/output methods.

TODO:
- rename to prevent overlap with built-in io module...
- consider replacing with xr.open_mfdataset
    https://docs.xarray.dev/en/stable/generated/xarray.open_mfdataset.html
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

    wsra_ds.attrs['pywsra_file'] = os.path.basename(filepath)

    if index_by_time:  #TODO: make wsra ds .wsra method
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
        key = os.path.basename(file).rsplit('.')[0]
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
    #TODO:  UserWarning: rename 'trajectory' to 'time' does not create an
    # index anymore. Try using swap_dims instead or use set_index after rename
    # to create an indexed coordinate.
    renamed = dropped.rename({coord: var})
    return renamed


def _combine_attrs(variable_attrs: List, context=None) -> dict:
    """ WSRA attribute handler passed to xr.concat.

    If `variable_attrs` contains metadata at the Dataset level, concatenate the
    attributes accordingly.  Otherwise, if `variable_attrs` contains variable
    descriptions at the DataArray level, pass back the first set of attributes.

    Args:
        variable_attrs (List): Attribute dictionaries to combine.
        TODO: context (_type_, optional): _description_. Defaults to None.

    Returns:
        dict: Combined attributes.
    """
    # Check if the keys are at the Dataset level. If so, concatenate them.
    # Otherwise, they are DataArray attributes and only the first is taken.
    if 'title' in variable_attrs[0].keys():  #  all(key in variable_attrs[0].keys() for key in ATTR_KEYS)
        # TODO: check if any of ATTR_KEYS in keys?
        attrs = _concat_attrs(variable_attrs)
    else:
        attrs = variable_attrs[0]
    return attrs


def _concat_attrs(variable_attrs: List):
    """Concatenate WSFA metadata attributes.

    Handle attributes during concatenation of WSRA Datasets.  Explicit
    handling is defined for standard WSRA attributes.  Where possible, unique
    values are taken. Otherwise, values are aggregated into a list.

    For all non-standard WSRA attributes, only unique values are taken.

    Args:
        variable_attrs (List): Attribute dictionaries to combine.

    Raises:
        KeyError: if `variable_attrs.keys()` contains a key not in `ATTR_KEYS`.

    Returns:
        dict: Combined attributes.
    """
    attr_keys = _get_unique_keys(variable_attrs)
    attrs = {k: [] for k in attr_keys}
    for key in attr_keys:
        if key == 'title':
            attrs[key] = _get_unique_attrs(variable_attrs, key)
        elif key == 'history':
            attrs[key] = _get_unique_attrs(variable_attrs, key)
        elif key == 'flight_id':
            attrs[key] = _aggregate_attrs(variable_attrs, key)
        elif key == 'mission_id':
            attrs[key] = _aggregate_attrs(variable_attrs, key)
        elif key == 'storm_id':
            # TODO: this can be misleading and should be fixed to return a
            # single value if len=1 and a list otherwise.
            attrs[key] = _get_unique_attrs(variable_attrs, key)[0]
        elif key == 'date_created':
            attrs[key] = _aggregate_attrs(variable_attrs, key)
        elif key == 'time_coverage_start':
            attrs[key] = _attrs_to_datetime(variable_attrs, key)[0].isoformat()
        elif key == 'time_coverage_end':
            attrs[key] = _attrs_to_datetime(variable_attrs, key)[-1].isoformat()
        else:
            attrs[key] = _get_unique_attrs(variable_attrs, key)
    return attrs


def _get_unique_keys(variable_attrs):
    """ Return unique keys from a set of attributes """
    # return [key for key in {key:None for attrs in variable_attrs for key in attrs}]
    return list({key: None for attrs in variable_attrs for key in attrs})


def _get_unique_attrs(variable_attrs, key) -> List:
    """ Return unique values from a set of attributes """
    all_attrs = _aggregate_attrs(variable_attrs, key)
    return list(np.unique(all_attrs))  # TODO: try replacing with built-in set


def _aggregate_attrs(variable_attrs, key) -> List:
    """ Aggregate all attributes into a list """
    return [attrs[key] for attrs in variable_attrs if key in attrs.keys()]


def _attrs_to_datetime(variable_attrs, key) -> List:
    """ Convert date-like attributes to datetimes """
    all_attrs = _aggregate_attrs(variable_attrs, key)
    attrs_as_datetimes = np.sort(pd.to_datetime(all_attrs))
    return list(attrs_as_datetimes)
