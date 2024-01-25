"""
Functions for working with P-3 meterological datasets.
"""

__all__ = [
    "read_met_directory",
    "read_met_file",
    "merge_met_vars",
]

import glob
import os
from typing import List, Callable, Union

import numpy as np
import pandas as pd
import xarray as xr

#TODO: incorporate tail radar data https://www.aoml.noaa.gov/ftp/pub/hrd/data/radar/level3/
#TODO: could automate reading of met data using WSRA filename (or date) which
# has the airplane identifier on it.
def read_met_directory(
    directory: str,
    file_type: str = 'nc',
    data_vars: Union[str, List] = 'all',
    **concat_kwargs,
) -> xr.Dataset:
    """
    Read a directory of P-3 met data files and concatenate into a Dataset.

    P-3 met data are available at: seb.noaa.gov/pub/acdata/{year}/MET/ where
    {year} is the year which corresponds to the season of data collection.
    The met data file date and mission identifier should match those of the
    WSRA filenames (e.g. 20220927H1).

    Note: P-3 met datasets contain many variables (>600).  Providing a subset
    of desired variables to `data_vars` can speed up the read-in process,
    since these variables are extracted prior to passing the dataset operations
    which require reading in the dataset into memory.

    Args:
        directory (str): Directory containing the met data files.
        file_type (str, optional): Met data file type. Defaults to '.nc'.
        data_vars (str or List): Variables to load into memory. Acceptable
            values include a list of variables or 'all'.  Defaults to 'all'.
        concat_kwargs (optional): Additional keyword arguments are passed to
            the xarray.concat.

    Raises:
        FileNotFoundError: If no files of type `file_type` are found
            inside of `directory`.

    Returns:
        xr.Dataset: all met data files concatenated into a single Dataset.
    """
    met_files = glob.glob(os.path.join(directory, '*' + file_type))
    met_files.sort()

    met = {}
    if not met_files:
        raise FileNotFoundError(
            f'No files of type "{file_type}" found in "{directory}". '
            'Please double check the directory and file_type.')

    for file in met_files:
        met_ds = read_met_file(file, data_vars)
        # TODO: update forward slash compatibility for windows OS:
        key = file.split('/')[-1].split('.')[0]
        met[key] = met_ds

    concat_dim = 'Time'

    wsra_ds_concat = xr.concat(list(met.values()),
                               dim=concat_dim,
                            #    combine_attrs=_combine_attrs,  #TODO: implement this!
                               **concat_kwargs)

    return wsra_ds_concat


def read_met_file(
    filepath: str,
    data_vars: Union[str, List] = 'all',
) -> xr.Dataset:
    """
    Read P-3 met data as an xarray Dataset.

    Note: P-3 met datasets contain many variables (>600).  Providing a subset
    of desired variables to `data_vars` can speed up the read-in process,
    since these variables are extracted prior to passing the dataset operations
    which require reading in the dataset into memory.

    Args:
        filepath (str): Path to met data file.
        data_vars (str or List): Variables to load into memory. Acceptable
            values include a list of variables or 'all'.  Defaults to 'all'.
    Returns:
        xr.Dataset: P-3 met dataset.
    """
    met_ds = xr.open_dataset(filepath, decode_times=False)
    if data_vars != 'all':
        met_ds = met_ds[data_vars]
    else:
        pass
    met_ds = _met_time_to_datetime(met_ds)
    return met_ds


def merge_met_vars(
    wsra_ds: xr.Dataset,
    met_ds: xr.Dataset,
    data_vars: List,
    resample_method: Callable = np.nanmean,
    rename_dict: Union[dict, None] = None,
    **merge_kwargs,
) -> xr.Dataset:
    """ Merge met data variables into a WSRA Dataset.

    Prior to merging, met data are resampled from their original frequency of
    1 Hz onto the WSRA frequency of 0.02 Hz (50 s) using `resample_method`. For
    example, if `resample_method = np.nanmean`, the preceeding and proceeding
    25 s, offset from each WSRA observation time, of met data in `data_vars`
    are averaged.

    Args:
        wsra_ds (xr.Dataset): WSRA dataset.
        met_ds (xr.Dataset): P-3 met dataset.
        data_vars (List): Variable names from `met_ds` to resample and merge
            into `met_ds`.
        resample_method (Callable, optional): Method used to resample
            `data_vars` every 50 s. Defaults to np.nanmean.
        rename_dict (dict, optional): Dictionary of current met variable names
            as keys with desired names as values. Passed to Dataset.rename().
        merge_kwargs (optional): additional keyword arguments are passed
            to the xarray.Dataset.merge method.

    Returns:
        xr.Dataset: Original WSRA dataset with met variables merged in.
    """
    wsra_times = wsra_ds['time'].sel(time=slice(met_ds['Time'][0],
                                                met_ds['Time'][-1]))
    met_resampled_ds = _resample_met_vars(met_ds,
                                          data_vars,
                                          wsra_times.values,
                                          resample_method)
    if rename_dict:
        met_resampled_ds = met_resampled_ds.rename(rename_dict)

    return wsra_ds.merge(met_resampled_ds, **merge_kwargs)


def _resample_met_vars(
    met_ds: xr.Dataset,
    data_vars: List,
    resample_times: np.ndarray[np.datetime64],
    resample_method: Callable
) -> xr.Dataset:

    # Resample each variable onto `resample_times`.  Aggregate observations
    # in a 50 s window centered on each time using `resample_method`.
    var_dict = {var: [] for var in data_vars}
    for t in resample_times:
        t_start = t - pd.Timedelta(25, 'S')
        t_end = t + pd.Timedelta(25, 'S')
        met_in_window = met_ds.sel(Time=slice(t_start, t_end))
        for var, values in var_dict.items():
            values.append(resample_method(met_in_window[var].values))

    # Construct a new Dataset using the resampled variables and new times
    met_resampled_ds = xr.Dataset(
        data_vars={
            var: (['time'], values) for var, values in var_dict.items()
        },
        coords=dict(
            time=resample_times,
        ),
    )

    # Copy attributes from the original DataArray(s)
    for var in var_dict.keys():
        met_resampled_ds[var].attrs = met_ds[var].attrs

    return met_resampled_ds


def _met_time_to_datetime(met_ds: xr.Dataset) -> xr.Dataset:
    """Convert the met data time coordinate to datetimes.

    Convert the P-3 met data time coordinate, which are provided as
    seconds since the start of the flight, to a datetime array.

    Args:
        met_ds (xr.Dataset): P-3 met dataset with original time coordinate.

    Returns:
        xr.Dataset: P-3 met dataset with datetime coordinate.
    """
    # Drop NaNs and sort the Dataset by time (seconds from start of flight).
    met_ds = (met_ds
              .dropna(dim='Time', how='all', subset=['Time'])
              .sortby('Time'))

    # Convert seconds from start of flight to datetimes using the POSIX
    # timestamp stored in attributes.  Assign it as the new coordinate.
    start_datetime_posix = met_ds.attrs['StartTime']
    datetime_posix = start_datetime_posix + met_ds['Time']
    datetime = pd.to_datetime(datetime_posix, unit='s', origin='unix')
    met_ds['Time'] = datetime
    return met_ds
