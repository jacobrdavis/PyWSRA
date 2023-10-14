"""
TODO:

"""

__all__ = [
    "WsraDatasetAccessor",
    "WsraDataArrayAccessor",
]

from typing import Hashable, Tuple
from urllib.error import URLError

import cartopy
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from matplotlib.axes import Axes
from .best_track import BestTrack
from .plot import WsraChart
from .operations import rotate_xy, calculate_mean_spectral_area


@xr.register_dataset_accessor("wsra")
class WsraDatasetAccessor:
    """Extend xarray Dataset objects with WSRA-specific functionality.

    #TODO: document specific additions:

    """
    def __init__(self, xarray_obj) -> xr.Dataset:
        self._obj = xarray_obj
        self._center = None
        self._best_track = None
        self._chart = None
        self._trajectory_dim = None
        # self._mean_spectral_area = None
        #ˇTODO: save filenames as attrs?

    @property
    def trajectory_dim(self):  # TODO: this will break when dims are cycled
        if self._trajectory_dim is None:
            dim_names = list(self._obj.dims.keys())
            self._trajectory_dim = dim_names[0]
        return self._trajectory_dim

    @trajectory_dim.setter
    def trajectory_dim(self, value):
        self._trajectory_dim = value

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
    
    # @property
    # def mean_spectral_area(self):
    #     """Return the mean spectral area.
        
    #     Return spectral area (rad^2/m^2) as the mean area of all
    #     spectral squares formed by the wavenumber arrays.
    #     """
    #     if self._mean_spectral_area is None:

    #         self._mean_spectral_area = calculate_mean_spectral_area(self._obj['wavenumber_east'].values,
    #                                      self._obj['wavenumber_north'].values)

    #     return self._mean_spectral_area

    #TODO:
    @property
    def best_track(self):
        #TODO: replace
        if self._best_track is None:
            try:
                storm_name = self._obj.attrs['storm_id']
            except AttributeError as e:
                #TODO: raise or try from filename?
                print('{e}, please set the `storm_id` attr of the dataset and try again.')
            except URLError as e:
                print('{e}, unable to load the best track database.')
            self._best_track = BestTrack(storm_name)

        return self._best_track

    @property
    def chart(self):
        if self._chart is None:
            self._chart = WsraChart(self._obj)
        return self._chart

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

        # best_track = get_storm_track(storm_name=self._obj.attrs['storm_id'])
        storm_datetime = self.best_track.df.index
        storm_direction = self.best_track.df['STORM_DIR']
        wsra_datetime = self._obj['time'].values

        interp_storm_direction = np.interp(wsra_datetime.astype("float"),
                                           storm_datetime.values.astype("float"),
                                           storm_direction.astype("float"))

        theta = np.deg2rad(interp_storm_direction)
        x_eye_rot, y_eye_rot = rotate_xy(x_eye, y_eye, theta)

        self._obj[X_VAR_NAME + '_storm_coord'] = ((self.trajectory_dim), x_eye_rot)
        self._obj[Y_VAR_NAME + '_storm_coord'] = ((self.trajectory_dim), y_eye_rot)

        for var_name in [X_VAR_NAME, Y_VAR_NAME]:
            new_var_name = var_name + '_storm_coord'
            new_long_name = self._obj[var_name].attrs['long_name'] + LONG_NAME_UPDATE
            self._obj[new_var_name].attrs['long_name'] = new_long_name
            self._obj[new_var_name].attrs['units'] = self._obj[var_name].attrs['units']

    def create_trajectory_mask(
        self,
        mask_dict=None,
        roll_limit=3,
        altitude_limits=(500, 4000),
        speed_limits=None,
        #TODO: minimum PSV limit
    ):
        #TODO: document
        #TODO: default values are those specified in Pincus et al. (2021)

        if not mask_dict:
            mask_dict = {}
            if roll_limit:
                mask_dict['wsra_computed_roll'] = (-1*roll_limit, roll_limit)
            if altitude_limits:
                mask_dict['platform_radar_altitude'] = altitude_limits
            if speed_limits:
                mask_dict['platform_speed_wrt_ground'] = speed_limits

        masks = []
        for variable, bounds in mask_dict.items():
            if 'obs' in self._obj[variable].dims:
                da = np.median(self._obj[variable], axis=1)
            else:
                da = self._obj[variable]
            masks.append(create_mask(da, bounds))

        trajectory_mask = np.logical_and.reduce(masks)
        mask_name = f'{self.trajectory_dim}_mask'

        self._obj.coords[mask_name] = ((self.trajectory_dim), trajectory_mask)

        num_masked_values = np.sum(~self._obj[mask_name].values)
        self._obj[mask_name].attrs['num_masked_values'] = num_masked_values

        for variable, bounds in mask_dict.items():
            attr_name = variable + '_bounds'
            self._obj[mask_name].attrs[attr_name] = bounds

    def mask(self, dim=0, **kwargs) -> xr.Dataset:
        """Filter elements from this object according to a prestablished
        dimension mask.

        See xarray.Dataset.where.

        Args:
            dim (int, optional): index of `dims` along which the mask is
                defined. Defaults to 0.
            kwargs (optional): Additional keyword arguments for Dataset.where

        Returns:
            xr.Dataset: original Dataset masked along a dimension
        """
        try:
            dim_names = list(self._obj.dims.keys())
            mask = dim_names[dim] + '_mask'
            return self._obj.where(self._obj.coords[mask], **kwargs)
        except KeyError as error:
            print(f"Mask {error} does not exist in coordinates.\n"
                  f"To create a mask, use the: "
                  f"`<Dataset>.wsra.create_<dim>_mask(...)`method.")
            return self._obj

    def plot(
        self,
        ax=None,
        extent=None,
        plot_best_track=True,
        **plt_kwargs
    ) -> Axes:
        if ax is None:
            fig = plt.figure(figsize=(5, 5))  # plt.gcf()
            proj = cartopy.crs.PlateCarree()
            ax = fig.add_subplot(1, 1, 1, projection=proj)

        self.chart.extent = extent
        self.chart.plot_best_track = plot_best_track
        self.chart.plot(ax, **plt_kwargs)
        return ax



@xr.register_dataarray_accessor("wsra")
class WsraDataArrayAccessor:
    """Extend xarray DataArray objects with WSRA-specific functionality.

    #TODO: document specific additions:
    - Adds dimension mask

    """
    def __init__(self, xarray_obj):
        self._obj = xarray_obj
        self._center = None

    def mask(self, dim=0, **kwargs) -> xr.DataArray:
        """Filter elements from this object according to a prestablished
        dimension mask.

        See xarray.DataArray.where.

        Args:
            dim (int, optional): index of `dims` along which the mask is
                defined. Defaults to 0.
            kwargs (optional): Additional keyword arguments for DataArray.where

        Returns:
            xr.DataArray: original DataArray masked along a dimension
        """
        try:
            mask = self._obj.dims[dim] + '_mask'
            return self._obj.where(self._obj.coords[mask], **kwargs)
        except KeyError as error:
            print(f"Mask {error} does not exist in coordinates.\n"
                  f"To create a mask, use the: "
                  f"`<Dataset>.wsra.create_<dim>_mask(...)`method.")
            return self._obj


def create_mask(da, bounds):
    return np.logical_and(da >= bounds[0],
                          da <= bounds[-1])