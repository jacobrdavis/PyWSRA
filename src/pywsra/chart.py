"""
TODO:

"""

__all__ = [
    "WsraChart",
]

from typing import Hashable, Tuple

import cartopy
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import xarray as xr


class WsraChart:
    #TODO:
    def __init__(
        self,
        # ax,
        wsra_ds,
        extent=None,
        # **plt_kwargs
    ):
        # self.ax = ax
        self.wsra_ds = wsra_ds
        self._extent = extent

        # self.plt_kwargs = plt_kwargs
        self.ocean_color = 'lightblue'
        self.land_color = 'tan'
        self.coastline_color = 'black'

        self.crs = "EPSG:4326"
        self.buffer_percent = 0.1
        self.plot_bathy = False  # TODO:
        self.bathy_plt_kwargs = {}

        self.plot_best_track = True
        self.best_track_plt_kwargs = {}

        self._gdf = None
        self._gridlines = None
        self._ocean = None

    @property
    def gdf(self):
        # if self._gdf is None:
        dim_to_drop = ["wavenumber_east", "wavenumber_north", "wavelength"]
        df = self.wsra_ds.isel(obs=2).drop_dims(dim_to_drop).to_dataframe()
        df.reset_index(inplace=True)

        points = gpd.points_from_xy(df.longitude, df.latitude)
        self._gdf = gpd.GeoDataFrame(df, geometry=points, crs=self.crs)
        return self._gdf

    # @property
    # def gridlines(self):
    #     if self._gridlines is None:
    #         self._gridlines = self.ax.gridlines(draw_labels=True,
    #                                             dms=False,
    #                                             x_inline=False,
    #                                             y_inline=False,
    #                                             top_labels=True)
    #         # self._gridlines.top_labels = False
    #         # self._gridlines.left_labels = False
    #         # self._gridlines.right_labels = True

    #     return self._gridlines

    @property
    def extent(self):
        if self._extent is None:
            extent = np.array([np.min(self.wsra_ds['longitude']),  # (x0, x1, y0, y1)
                               np.max(self.wsra_ds['longitude']),
                               np.min(self.wsra_ds['latitude']),
                               np.max(self.wsra_ds['latitude'])])

            buffer_norm = np.array([-1, 1]) * self.buffer_percent
            longitude_buffer = buffer_norm * (extent[1] - extent[0])
            latitude_buffer = buffer_norm * (extent[3] - extent[2])
            buffer = np.hstack([longitude_buffer, latitude_buffer])

            self._extent = extent + buffer

        return self._extent

    @extent.setter
    def extent(self, value):
        self._extent = value
    

    # @property
    # def ocean(self):
    #     if self._ocean is None:
    #         self._ocean = self.ax.add_feature(cartopy.feature.OCEAN,
    #                                           color=self.ocean_color)
    #     return self._ocean


    # @property
    # def wsra_plot

    def plot_gridlines(self, ax):
        self._gridlines = ax.gridlines(draw_labels=True,
                                       dms=False,
                                       x_inline=False,
                                       y_inline=False)
        self._gridlines.top_labels = False
        self._gridlines.left_labels = False
        self._gridlines.right_labels = True

    def plot_base_chart(self, ax):
        states_kwargs = {'category': 'cultural',
                         'name': 'admin_1_states_provinces_lines',
                         'scale': '50m',
                         'facecolor': 'none'}
        ax.add_feature(cartopy.feature.OCEAN, color=self.ocean_color)
        ax.add_feature(cartopy.feature.LAND, color=self.land_color)
        ax.add_feature(cartopy.feature.COASTLINE, color=self.coastline_color)
        states = cartopy.feature.NaturalEarthFeature(**states_kwargs)
        ax.add_feature(states, edgecolor='black')

    def plot(self, ax, **plt_kwargs):
        #  TODO: should this return ax?
        ax.set_extent(self.extent)
        # self.gridlines
        # self.ocean
        self.plot_gridlines(ax)
        self.plot_base_chart(ax)

        if self.plot_best_track:
            try:
                self.wsra_ds.wsra.best_track.plot(
                    ax,
                    **self.best_track_plt_kwargs,
                )
            except ValueError:  # as e
                print('Best track unavailable.')

        #TODO: save children to attrs?

        if 'color' not in plt_kwargs and 'column' not in plt_kwargs:
            plt_kwargs['color'] = 'k'

        if 'markersize' not in plt_kwargs:
            plt_kwargs['markersize'] = 5

        wsra_plot = self.gdf.plot(
            ax=ax,
            # transform=cartopy.crs.PlateCarree(),
            **plt_kwargs,
        )
        wsra_plot.set_zorder(6)

