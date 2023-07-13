"""
TODO:

"""

__all__ = [
    "WsraDatasetAccessor",
    "WsraDataArrayAccessor",
]

from typing import Hashable, Tuple

import numpy as np
import xarray as xr
from .ibtracs import get_storm_track


@xr.register_dataset_accessor("wsra")
class WsraDatasetAccessor:
    """Extend xarray Dataset objects with WSRA-specific functionality.

    #TODO: document specific additions:

    """
    def __init__(self, xarray_obj):
        self._obj = xarray_obj
        self._center = None

    @property
    def center(self):
        """Return the geographic center point of this dataset."""
        if self._center is None:
            # we can use a cache on our accessor objects, because accessors
            # themselves are cached on instances that access them.
            lon = self._obj.latitude
            lat = self._obj.longitude
            self._center = (float(lon.mean()), float(lat.mean()))
        return self._center

    def _rotate_xy(self, x: np.array, y: np.array, theta: np.array) -> Tuple:
        """
        Perform 2D coordinate rotation of points (x, y).

        Args:
            x (np.array): x coordinate to rotate
            y (np.array): y coordinate to rotate
            theta (np.array): rotation angles in [rad]

        Returns:
            Tuple[np.array, np.array]: rotated x and y coordinates
        """
        x_rot = x*np.cos(theta) - y*np.sin(theta)
        y_rot = x*np.sin(theta) + y*np.cos(theta)

        return x_rot, y_rot

    def to_storm_coord(self) -> None:
        """
        Rotate x and y coordinates into the hurricane reference frame.

        Rotates observation coordinates, expressed in distances to the
        hurricane center, from a geographic coordinate system (N-S, E-W)
        into a coordinate system aligned with the hurricane's direction of
        motion. Each observation's distance to the hurricane center is
        provided directly in the Level 4 WSRA data. The storm direction is
        determined from IBTrACS database and is interpolated onto the
        WSRA observation times to construct the coordinate rotation matrix.

        Args:
            wsra_ds (xr.Dataset): WSRA dataset
            storm_name (str): name of the storm/hurricane
        """
        X_VAR_NAME = 'hurricane_eye_distance_east'
        Y_VAR_NAME = 'hurricane_eye_distance_north'
        LONG_NAME_UPDATE = ', rotated into the hurricane coordinate system'

        y_eye = self._obj[Y_VAR_NAME].values
        x_eye = self._obj[X_VAR_NAME].values

        best_track = get_storm_track(storm_name=self._obj.attrs['storm_id'])
        storm_datetime = best_track.index
        storm_direction = best_track['STORM_DIR']
        wsra_datetime = self._obj['time'].values

        interp_storm_direction = np.interp(wsra_datetime.astype("float"),
                                           storm_datetime.values.astype("float"),
                                           storm_direction.astype("float"))

        theta = np.deg2rad(interp_storm_direction)
        x_eye_rot, y_eye_rot = self._rotate_xy(x_eye, y_eye, theta)

        self._obj[X_VAR_NAME + '_storm_coord'] = (('trajectory'), x_eye_rot)
        self._obj[Y_VAR_NAME + '_storm_coord'] = (('trajectory'), y_eye_rot)

        for var_name in [X_VAR_NAME, Y_VAR_NAME]:
            new_var_name = var_name + '_storm_coord'
            new_long_name = self._obj[var_name].attrs['long_name'] + LONG_NAME_UPDATE
            self._obj[new_var_name].attrs['long_name'] = new_long_name
            self._obj[new_var_name].attrs['units'] = self._obj[var_name].attrs['units']

    def create_trajectory_mask(
        self,
        mask_dict=None,
        roll_limit=None,
        altitude_limit=None,
        speed_limit=None,
    ):
        #TODO: document

        if not mask_dict:
            mask_dict = {}
            if roll_limit:
                mask_dict['wsra_computed_roll'] = (-1*roll_limit, roll_limit)
            if altitude_limit:
                mask_dict['platform_radar_altitude'] = (0, altitude_limit)
            if speed_limit:
                mask_dict['platform_speed_wrt_ground'] = (0, speed_limit)

        masks = []
        for variable, bounds in mask_dict.items():
            if 'obs' in self._obj[variable].dims:
                da = np.median(self._obj[variable], axis=1)
            else:
                da = self._obj[variable]
            masks.append(create_mask(da, bounds))

        trajectory_mask = np.logical_and.reduce(masks)

        self._obj.coords['trajectory_mask'] = (('trajectory'), trajectory_mask)

        masked_values = np.sum(~self._obj.trajectory_mask.values)
        self._obj['trajectory_mask'].attrs['masked_values'] = masked_values

        for variable, bounds in mask_dict.items():
            attr_name = variable + '_bounds'
            self._obj['trajectory_mask'].attrs[attr_name] = bounds


@xr.register_dataarray_accessor("wsra")
class WsraDataArrayAccessor:
    """Extend xarray DataArray objects with WSRA-specific functionality.

    #TODO: document specific additions:
    - Adds dimension mask

    """
    def __init__(self, xarray_obj):
        self._obj = xarray_obj
        self._center = None

    def mask(self, dim=0) -> xr.DataArray:
        """Filter elements from this object according to a prestablished
        dimension mask.

        Args:
            dim (int, optional): index of `dims` along which the mask is
                defined. Defaults to 0.

        Returns:
            xr.DataArray: original DataArray masked along a dimension
        """
        try:
            mask = self._obj.dims[dim] + '_mask'
            return self._obj.where(self._obj.coords[mask])
        except AttributeError as error:
            print(f"{error}.\n"
                  f"To create a mask, use the: "
                  f"`<Dataset>.wsra.create_<dim>_mask(...)` method.")
            return self._obj
        except KeyError as error:
            print(f"Mask {error} does not exist in coordinates.\n"
                  f"To create a mask, use the: "
                  f"`<Dataset>.wsra.create_<dim>_mask(...)`method.")
            return self._obj


def create_mask(da, bounds):
    return np.logical_and(da >= bounds[0],
                          da <= bounds[-1])