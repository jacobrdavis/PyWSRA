from typing import List, Tuple

import numpy as np
import scipy
import xarray as xr

__all__ = [
    "colocate_with_path",
    "colocate_with_grid",
]



#TODO: should be method of wsra Dataset
def colocate_with_path_ds(
    wsra_ds: xr.Dataset,
    path_ds: xr.Dataset,
    path_vars: List,
    path_coords: Tuple,
    wsra_coords: Tuple = ('time', 'longitude', 'latitude'),
    temporal_tolerance: np.timedelta64 = np.timedelta64(30, 'm'),
    spatial_tolerance: float = 50.0,  # km
    prefix: str | None = None,
) -> xr.Dataset:
    """
    Find matching WSRA observations with data along another path (i.e. a
    drifting buoy) based on temporal and spatial tolerances.

    The `path_vars` argument is a list of variable names along the coordinates
    specified in `path_coords`.  These variables will be appended to `wsra_ds`.
    For instance, if `significant_wave_height` and `mean_square_slope` are
    variables in `path_ds` which are to be matched with `wsra_ds`, then
    `path_vars` should be:

    >>> path_vars = ['significant_wave_height', 'mean_square_slope']

    Inputs `path_coords` and `wsra_coords` are tuples specifying the names of
    the coordinates to match on. The names must be ordered as:
    (time, longitude, latitude). For instance, if the path dataset coordinates
    are labeled as 'time', 'lon', and 'lat', then `path_coords`
    should be:

    >>> path_coords = ('time', 'lon', 'lat')

    `wsra_coords` defaults to the standard dataset names, though these should
    be provided if the defaults have been modified.

    Note: Empty matches (no colocated data) are still appended to `wsra_ds`.

    Args:
        wsra_ds (xr.Dataset): WSRA observations
        path_ds (xr.Dataset): path data
        path_vars (List): path variable names (see above)
        path_coords (Tuple): path coordinate names (see above)
        wsra_coords (Tuple): WSRA coordinate names (see above)
        temporal_tolerance (np.timedelta64, optional): max allowable time delta
            between WSRA and path times. Defaults to np.timedelta64(30, 'm').
        spatial_tolerance (float, optional): max allowable distance
            between WSRA and path times. Defaults to 5.0 km.

    Returns:
        xr.Dataset: A new Dataset created by merging `wsra_ds` with matched
            `path_vars`.
    """
    # Get the indices where the WSRA and path Datasets are colocated and then
    # use them to select the colocated portion of each Dataset.
    wsra_indices, path_indices, distance, time_difference = colocate_with_path(
        wsra_time=wsra_ds[wsra_coords[0]].values,
        wsra_longitude=wsra_ds[wsra_coords[1]].values,
        wsra_latitude=wsra_ds[wsra_coords[2]].values,
        path_time=path_ds[path_coords[0]].values,
        path_longitude=path_ds[path_coords[1]].values,
        path_latitude=path_ds[path_coords[2]].values,
        temporal_tolerance=temporal_tolerance,
        spatial_tolerance=spatial_tolerance,
    )
    path_colocated_ds = path_ds.isel(time=path_indices)
    wsra_colocated_ds = wsra_ds.isel(time=wsra_indices)

    # Extract only `path_coords` and `path_vars` from the path Dataset and
    # rename the coordinates for later merging.
    path_subset_ds = path_colocated_ds[list(path_coords) + path_vars]
    path_coord_name_dict = {path_coords[0]: 'time',
                            path_coords[1]: 'longitude',
                            path_coords[2]: 'latitude'}
    path_subset_ds = path_subset_ds.rename(path_coord_name_dict)

    # Assign the WSRA times and add the distance and time difference as vars.
    path_subset_ds['time'] = wsra_colocated_ds['time']
    path_subset_ds['distance'] = ('time', distance)
    path_subset_ds['time_difference'] = ('time', time_difference)

    if prefix:
        var_names = path_subset_ds.data_vars.keys()
        name_dict = {name: prefix + '_' + name for name in var_names}
        path_subset_ds = path_subset_ds.rename(name_dict)

    wsra_merged_ds = xr.merge([wsra_ds, path_subset_ds])
    return wsra_merged_ds


def colocate_with_path(
    wsra_time: np.ndarray,
    wsra_longitude: np.ndarray,
    wsra_latitude: np.ndarray,
    path_time: np.ndarray,
    path_longitude: np.ndarray,
    path_latitude: np.ndarray,
    temporal_tolerance: np.timedelta64 = np.timedelta64(30, 'm'),
    spatial_tolerance: float = 50.0,  # km
) -> Tuple[np.ndarray, np.ndarray,  np.ndarray,  np.ndarray]:
    """
    Find where WSRA observations match with data along another path (i.e. a
    drifting buoy) based on temporal and spatial tolerances.

    For a WSRA Dataset with a time array of length `t` and a path Dataset with
    a time array of length `p`, the matching WSRA and path indices will be
    returned with shape `(m,)` where `m` is the number of matches (`m` <= `t`).

    Args:
        wsra_time (np.ndarray): WSRA datetimes with shape `(t,)`
        wsra_longitude (np.ndarray): WSRA longitudes with shape `(t,)`
        wsra_latitude (np.ndarray): WSRA latitudes with shape `(t,)`
        path_time (np.ndarray): path datetimes with shape `(p,)`
        path_longitude (np.ndarray): path longitude with shape `(p,)`
        path_latitude (np.ndarray):  path latitudes with shape `(p,)`
        temporal_tolerance (np.timedelta64, optional): max allowable time delta
            between WSRA and path times. Defaults to np.timedelta64(30, 'm').
        spatial_tolerance (float, optional): max allowable distance
            between WSRA and path times. Defaults to 50.0 km.

    Returns:
        Tuple containing
        np.ndarray: matching WSRA indices with shape `(m,)`
        np.ndarray: matching path indices with shape `(m,)`
        np.ndarray: great circle distance at each match with shape `(m,)`
        np.ndarray: time difference at each match with shape `(m,)`
    """
    # Get the indices where the WSRA times fit within the path times.
    t_sort_indices = np.searchsorted(path_time, wsra_time)
    t_sort_indices[t_sort_indices >= len(path_time)] = len(path_time)-1

    # Determine the time difference between WSRA and the path.
    time_difference = np.abs(wsra_time - path_time[t_sort_indices])

    # Determine the distance between WSRA and the path.
    distance, bearing = great_circle_pairwise(
        longitude_a=wsra_longitude,
        latitude_a=wsra_latitude,
        longitude_b=path_longitude[t_sort_indices],
        latitude_b=path_latitude[t_sort_indices]
    )

    # Matches are where time and distance constraints are satisfied.
    in_time = time_difference < temporal_tolerance
    in_range = distance < spatial_tolerance
    in_time_and_range = np.logical_and(in_time, in_range)

    # Get matching WSRA and path indices and the distance and time difference
    # at each match.
    wsra_indices = np.where(in_time_and_range)[0]
    path_indices = t_sort_indices[in_time_and_range]
    distances = distance[in_time_and_range]
    time_differences = time_difference[in_time_and_range]

    return wsra_indices, path_indices, distances, time_differences


#TODO:  will want to extract from grid via pointwise indexing
# https://gist.github.com/abkfenris/9386ad183d110578fe761c2cc3b25874
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


def great_circle_pairwise(
    longitude_a: np.ndarray,
    latitude_a: np.ndarray,
    longitude_b: np.ndarray,
    latitude_b: np.ndarray,
    earth_radius: float = 6378.137,
    mod_bearing: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes the great circle distance (km) and true fore bearing (deg) between
    pairs of observations in input arrays `longitude_a` and `longitude_b` and
    `latitude_a` and `latitude_b`.

    For two longitude and latitude pairs, the great circle distance is the
    shortest distance between the two points along the Earth's surface. This
    distance is calculated using the Haversine formula. The instances in
    `longitude_a` and `latitude_a` are designated as point `a`; the instances
    in `longitude_b` and `latitude_b` then form point `b`. The true fore
    bearing is the bearing, measured from true north, of `b` as seen from `a`.

    Args:
        longitude_a (np.array): of shape (n,) in units of decimal degrees
        latitude (np.array): of shape (n,) in units of decimal degrees
        earth_radius (float, optional): earth's radius in units of km. Defaults to 6378.137 km (WGS-84)
        mod_bearing (bool, optional): return bearings modulo 360 deg. Defaults to True.

    Returns:
        Tuple[np.array, np.array]: great circle distances (in km) and true fore
        bearings between adjacent longitude and latitude pairs; shape (n,)
    """
    # Convert decimal degrees to radians
    longitude_a_rad, latitude_a_rad = map(np.radians, [longitude_a, latitude_a])
    longitude_b_rad, latitude_b_rad = map(np.radians, [longitude_b, latitude_b])

    # Difference longitude and latitude
    longitude_difference = longitude_b_rad - longitude_a_rad
    latitude_difference = latitude_b_rad - latitude_a_rad

    # Haversine formula
    a_1 = np.sin(latitude_difference / 2) ** 2
    a_2 = np.cos(latitude_a_rad)
    a_3 = np.cos(latitude_b_rad)
    a_4 = np.sin(longitude_difference / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a_1 + a_2 * a_3 * a_4))
    distance_km = earth_radius * c

    # True bearing
    bearing_num = np.cos(latitude_b_rad) * np.sin(-longitude_difference)
    bearing_den_1 = np.cos(latitude_a_rad) * np.sin(latitude_b_rad)
    bearing_den_2 = - np.sin(latitude_a_rad) * np.cos(latitude_b_rad) * np.cos(longitude_difference)
    bearing_deg = -np.degrees(np.arctan2(bearing_num, bearing_den_1 + bearing_den_2))

    if mod_bearing:
        bearing_deg = bearing_deg % 360

    return distance_km, bearing_deg
