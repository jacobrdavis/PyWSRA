"""
TODO:
- make this plot.py? and add a class for dir spectra

"""

__all__ = [
    "WsraChart",
    "plot_wavenumber_spectrum",
]

from typing import Hashable, Tuple

import cartopy
import cmocean
import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import xarray as xr
from mpl_toolkits.axes_grid1 import make_axes_locatable
from .operations import calculate_mean_spectral_area


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
        self.buffer_percent = 0.1  #TODO: fraction?
        self.plot_bathy = False  # TODO:
        self.bathy_plt_kwargs = {}

        self.plot_best_track = True
        self.best_track_plt_kwargs = {}

        self._gdf = None
        self.gridlines = None #TODO: need to improve gridliner
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
        self.gridlines = ax.gridlines(draw_labels=True,
                                       dms=False,
                                       x_inline=False,
                                       y_inline=False,
                                       zorder=0)
        self.gridlines.top_labels = False
        self.gridlines.left_labels = True
        self.gridlines.right_labels = False
        # self.gridlines.xlines = False
        # self.gridlines.ylines = False

    def plot_base_chart(self, ax):
        states_kwargs = {'category': 'cultural',
                         'name': 'admin_1_states_provinces_lines',
                         'scale': '50m',
                         'facecolor': 'none'}
        ax.add_feature(cartopy.feature.OCEAN, facecolor=self.ocean_color)
        ax.add_feature(cartopy.feature.LAND, facecolor=self.land_color)
        ax.add_feature(cartopy.feature.COASTLINE, edgecolor=self.coastline_color)
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


def plot_wavenumber_spectrum(
    energy,
    wavenumber_east,
    wavenumber_north,
    # depth: float = None,  #TODO:
    normalize: bool = False,
    density: bool = True,
    # as_frequency_spectrum: bool = True,
    ax = None, #TODO:
    **pcm_kwargs,
):
    if density:
        #TODO: replace with function
        mean_area = calculate_mean_spectral_area(wavenumber_east,
                                                 wavenumber_north)  # rad^2/m^2
        energy_plot = energy / mean_area  # m^4/rad^2
        cbar_label = 'energy density ($\mathrm{m^4 / rad^2}$)'
    else:
        energy_plot = energy.copy()  # m^2
        cbar_label = 'energy ($\mathrm{m^2}$)'

    if normalize:
        energy_plot = energy / energy.max()
        cbar_label = 'normalized energy (-)'

    # #TODO:
    # if as_frequency_spectrum:
    # see WSRA/backup/wsra_hurricane_ian

    # energy_density_fq_noregrid, direction_noregrid, frequency_noregrid = pywsra.wn_spectrum_to_fq_spectrum(
    # energy=np.squeeze(ian_end_slice['directional_wave_spectrum']).values,
    # wavenumber_east=ian_end_slice['wavenumber_east'].values,
    # wavenumber_north=ian_end_slice['wavenumber_north'].values,
    # depth=1000,
    # regrid=False,
    # )

    # fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    # cmap = cmocean.cm.amp
    # cmap.set_under('white')
    # pcm = ax.pcolormesh(direction_noregrid, # (-direction_noregrid + np.pi/2) % (2 * np.pi),
    #                     frequency_noregrid/(2*np.pi),
    #                     energy_density_fq_noregrid,
    #                     shading='gouraud',
    #                     # norm=mpl.colors.LogNorm(vmin=1*10**(-2), vmax=1*10**(1)))
    #                     cmap=cmap)
    # ax.set_theta_zero_location("N")  # theta=0 at the top  #TODO: need to reconvert convention when plotting
    # ax.set_theta_direction(-1)  # +CW
    # cb = fig.colorbar(pcm)
    # cb.set_label('energy density (m$^2$/Hz)')


    # else:
    # #TODO:

    cmap = cmocean.cm.amp
    cmap.set_under('white')

    pcm = ax.pcolormesh(
        wavenumber_east,
        wavenumber_north,
        energy_plot,
        cmap=cmap,  #'magma',
        # vmin=0.1,
        shading='gouraud',
        **pcm_kwargs,
    )
    ax.set_aspect('equal')
    ax.spines['left'].set_position(('data', 0.0))
    ax.spines['bottom'].set_position(('data', 0.0))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # ax.spines.bottom.set_bounds(ticks.min(), ticks.max())
    # ax.spines.left.set_bounds(ticks.min(), ticks.max())

    ticks = np.linspace(-0.08, 0.08, 9)
    ticks = ticks[np.abs(ticks) > 0]
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xlim([ticks.min(), ticks.max()])
    ax.set_ylim([ticks.min(), ticks.max()])

    ax.xaxis.set_tick_params(labelsize=8)
    ax.yaxis.set_tick_params(labelsize=8)

    #TODO: could be clever and use the region opposite of the psv

    # ax.set_xlabel('east wavenumber (rad/m)', ha='right', va='top', rotation='horizontal')
    # ax.xaxis.set_label_coords(1, 0)
    ax.set_xlabel('east wavenumber (rad/m)', ha='right', va='bottom', rotation=0, fontsize=9)
    ax.xaxis.set_label_coords(1.0, 0.5)  # (1, 0)

    # ax.set_ylabel('north wavenumber (rad/m)', ha='right', va='bottom', rotation=180)
    # ax.yaxis.set_label_coords(0, 1)
    ax.set_ylabel('north wavenumber (rad/m)', ha='right', va='top', rotation=90, fontsize=9)
    ax.yaxis.set_label_coords(0.505, 1.0)

    circle_radians = np.deg2rad(np.linspace(0, 360, 100))
    circle_x = np.sin(circle_radians)
    circle_y = np.cos(circle_radians)
    circle_wavenumbers = np.array([0.02, 0.04, 0.06, 0.08])
    circles_xx = np.outer(circle_x, circle_wavenumbers)
    circles_yy = np.outer(circle_y, circle_wavenumbers)
    circles = ax.plot(
        circles_xx,
        circles_yy,
        color='k',
        alpha=0.25,
        linewidth=1,
        linestyle=':',
    )

    spoke_angles = np.array([45, 135])
    spoke_angles = np.deg2rad(np.linspace(0, 360, 17))

    spoke_x = 0.08 * np.sin(spoke_angles)[None, :]
    spoke_y = 0.08 * np.cos(spoke_angles)[None, :]

    spoke_x_line = np.vstack([np.zeros(spoke_x.shape), spoke_x])
    spoke_y_line = np.vstack([np.zeros(spoke_y.shape), spoke_y])

    spokes = ax.plot(
        spoke_x_line, spoke_y_line,
        color='k',
        alpha=0.25,
        linewidth=1,
        linestyle='-',
    )

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("left", size="5%", pad="5%", axes_class=mpl.axes.Axes)
    cbar = plt.colorbar(pcm, cax=cax, location="left")
    cbar.set_label(cbar_label)

    #  Border
    # ax.patch.set_edgecolor('black')
    # ax.patch.set_linewidth(1)

    return pcm #TODO: Cbar?

# def plot_scalar_spectrum():
    # fq_spectrum_var = pywsra.calculate_fq_spectrum_variance(energy_density_fq,
    #                                                 direction[:, 0],
    #                                                 frequency[0])


    # scalar_energy_density = np.trapz(energy_density_fq, direction, axis=0)
    # variance_frequency_spectrum = np.trapz(energy_density_f, angular_frequency[0])
    # print(variance_frequency_spectrum)

    # energy_density_wn = energy_wn / pywsra.calculate_mean_spectral_area(wavenumber_east, wavenumber_north)
    # in_circle = angular_frequency_noregrid >= angular_frequency.max()  # Mask outside polar region (to account for variance)
    # energy_density_wn[in_circle] = 0

    # energy_density_int_x = np.trapz(energy_density_wn, x=wavenumber_east, axis=0)
    # variance_wavenumber_spectrum = np.trapz(energy_density_int_x, x=wavenumber_north, axis=0)
    # print(variance_wavenumber_spectrum)

    # fig, ax = plt.subplots()
    # ax.plot(angular_frequency[0]/(2*np.pi), energy_density_f, label='WSRA')
    # ax.set_yscale('log')
    # ax.set_ylabel('energy density (m^2/Hz)')
    # ax.set_xlabel('frequency (Hz)')

    # np.trapz(energy_density_fq, direction, axis=0)