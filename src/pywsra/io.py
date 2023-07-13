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


def read_wsra_file(filepath):
    """
    Read Level 4 WSRA data as an xarray Dataset.

    Args:
        filepath (str): path to WSRA file.

    Returns:
        xr.Dataset: WSRA dataset.
    """
    wsra_ds = xr.open_dataset(filepath)
    return wsra_ds


def read_wsra_directory(
    directory: str,
    file_type: str = 'nc',
    concat: bool = False,
    concat_dim: Hashable = "trajectory",
    **concat_kwargs
) -> dict[xr.Dataset]:
    """
    Read all Level 4 WSRA data files in a directory into a dictionary of
    xarray Datasets or concatenated into a single Dataset.

    Args:
        directory (str): directory containing WSRA files
        file_type (str, optional): WSRA data file type. Defaults to '.nc'.
        concat (bool, optional): Concatenate all WSRA Datasets into a
            single Dataset. Defaults to False.
        concat_dim (Hashable, optional): Name of the dimension to
            concatenate along. See xr.concat for more information.
            Defaults to "trajectory".
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
        wsra_ds = read_wsra_file(file)
        key = file.split('/')[-1].split('.')[0]
        wsra[key] = wsra_ds

    if concat:
        wsra = xr.concat(list(wsra.values()),
                         concat_dim,
                         **concat_kwargs)

    return wsra