from typing import Tuple

import numpy as np
import scipy
import xarray as xr


def colocate_with_grid(
    wsra_ds: xr.Dataset,
    grid_ds: xr.Dataset,
    grid_vars: Tuple,
    wsra_vars: Tuple = ('time', 'latitude', 'longitude'),
    temporal_tolerance: np.timedelta64 = np.timedelta64(30, 'm'),
) -> np.ndarray:
    """
    Match WSRA observations with gridded data (e.g., a model) using
    linear interpolation in time and bilinear interpolation in space.

    Note:
        `grid_vars` and `wsra_vars` are tuples specifying the names of the
        coordinates and fields to interpolate. The names must be ordered as:
        time, latitude, longitude, and field where field is the variable in
        `grid_ds` to be in colocated onto the WSRA dataset (and thus is only
        provided only for `grid_vars`).

        For instance, if the gridded dataset coordinates are labeled as 'time',
        'lat', and 'lon' and 'wind_speed' is the field to be colocated onto the
        WSRA dataset, `grid_vars` should be:
        >>> grid_vars = ('time', 'lat', 'lon', 'wind_speed)

        `wsra_vars` defaults to the standard dataset names, though these should
        be provided if the defaults have been modified.

        Out-of-bound points are replaced by NaNs.

    Args:
        wsra_ds (xr.Dataset): WSRA observations
        grid_ds (xr.Dataset): gridded data with a field variable to be
            interpolated onto the WSRA observations
        temporal_tolerance (np.timedelta64, optional): max allowable time delta
            between model and grid times. Defaults to np.timedelta64(30, 'm').

    Returns:
        np.ndarray: field variable values interpolated onto the WSRA time and
            spatial coordinates.
    """

    wsra_time = wsra_ds[wsra_vars[0]].values
    wsra_latitude = wsra_ds[wsra_vars[1]].values
    wsra_longitude = wsra_ds[wsra_vars[2]].values

    grid_time = grid_ds[grid_vars[0]].values
    grid_latitude = grid_ds[grid_vars[1]].values
    grid_longitude = grid_ds[grid_vars[2]].values
    grid_field = grid_ds[grid_vars[3]].values

    t_sort_indices = np.searchsorted(grid_time, wsra_time)

    field_matches = []

    grid_points = (grid_latitude, grid_longitude)
    for i, j in enumerate(t_sort_indices):

        if j < len(grid_time):
            time_difference = np.abs(wsra_time[i] - grid_time[j])
        else:
            time_difference = None

        if not time_difference or time_difference > temporal_tolerance:
            value_at_wsra = np.nan
        else:
            x_i = (wsra_latitude[i], wsra_longitude[i])

            bilinear_value_jm1 = scipy.interpolate.interpn(grid_points,
                                                           grid_field[j-1],
                                                           x_i,
                                                           method='linear',
                                                           bounds_error=False,
                                                           fill_value=np.NaN)

            bilinear_value_j = scipy.interpolate.interpn(grid_points,
                                                         grid_field[j],
                                                         x_i,
                                                         method='linear',
                                                         bounds_error=False,
                                                         fill_value=np.NaN)

            adjacent_bilinear_values = np.concatenate([bilinear_value_jm1,
                                                       bilinear_value_j])

            adjacent_grid_times = np.array([grid_time[j-1],
                                            grid_time[j]])

            value_at_wsra = np.interp(wsra_time[i].astype("float"),
                                      adjacent_grid_times.astype("float"),
                                      adjacent_bilinear_values)

        field_matches.append(value_at_wsra)


    # TODO:
    # ian_ds[grid_vars[3]] = (('trajectory'), y_eye_rot)

    return np.array(field_matches)
