"""
Input/output methods.

TODO:
- consider replacing with xr.open_mfdataset
    https://docs.xarray.dev/en/stable/generated/xarray.open_mfdataset.html
- update forward slash compatibility for windows OS

"""

__all__ = [
    "read_wsra_directory",
    "read_wsra_file"
]

import glob
import os
import warnings
from typing import Hashable

import xarray as xr


def read_wsra_file(filepath: str, index_by_time: bool = True):
    """
    Read Level 4 WSRA data as an xarray Dataset.

    Args:
        filepath (str): path to WSRA file.

    Returns:
        xr.Dataset: WSRA dataset.
    """
    wsra_ds = xr.open_dataset(filepath)

    if index_by_time:
        wsra_ds = _replace_coord_with_var(wsra_ds, 'trajectory', 'time')

    return wsra_ds


#TODO: replace with xr open dir?
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
        index_by_time
        concat_kwargs (optional): additional keyword arguments to be
            passed to the xr.concat method.

    Raises:
        FileNotFoundError: If no files of type `file_type` are found
            inside of `directory`.

    Returns:
        dict[xr.Dataset]: _description_
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
        key = file.split('/')[-1].split('.')[0]
        wsra[key] = wsra_ds

    #TODO:
    # def combine_attrs(variable_attrs, context=None):
    #     print(variable_attrs)
        # print(context)

    if index_by_time:
        concat_dim = 'time'
    else:
        concat_dim = 'trajectory'

    wsra_ds_concat = xr.concat(list(wsra.values()),
                               dim=concat_dim,
                            #    combine_attrs=combine_attrs,
                            #    combine_attrs='no_conflicts',  #TODO: callable?
                               **concat_kwargs)

    return wsra_ds_concat


def _replace_coord_with_var(ds, coord, var):
    #TODO: docstr: note coord and var should have same shape
    ds.coords[coord] = ds[var]
    dropped = ds.drop_vars([var])
    renamed = dropped.rename({coord: var})
    return renamed

