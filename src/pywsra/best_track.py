"""

#TODO: fetch directly from IBTRACS site

"""

__all__ = [
    "BestTrack",
]


import re
import pickle
import urllib.request
from datetime import datetime, timezone

import cartopy
import geopandas as gpd
import pandas as pd
import shapely

IBTRACS_BASE_URL = ('https://www.ncei.noaa.gov/data/international-best-'
                    'track-archive-for-climate-stewardship-ibtracs/'
                    'v04r00/access/csv/')


class BestTrack:
    #TODO:
    def __init__(
        self,
        storm_name: str,
        filepath=None,
        database_subset='last3years'

    ):
        self._storm_name = storm_name.upper()
        self._filepath = filepath

        self._database_subset = database_subset
        self._database = None
        self._database_storm_names = None
        self._database_storm_ids = None
        self._set_best_track_database()

        self._df = None
        self._gdf = None
        self._set_storm_df()

    @property
    def storm_name(self):
        return self._storm_name

    @storm_name.setter
    def storm_name(self, value):
        self._storm_name = value.upper()
        self._set_storm_df()

    @property
    def df(self):
        return self._df

    @property
    def gdf(self):
        if self._gdf is None:
            self._gdf = gpd.GeoDataFrame(
                self.df,
                geometry=gpd.points_from_xy(self.df['LON'], self.df['LAT']),
                crs="EPSG:4326"
            )
        return self._gdf

    def _set_best_track_database(self):
        if self._filepath is not None:
            ibtracs_path = self._filepath
        else:
            ibtracs_csv = f'ibtracs.{self._database_subset}.list.v04r00.csv'
            ibtracs_path = IBTRACS_BASE_URL + ibtracs_csv

        ibtracs_df = pd.read_csv(ibtracs_path, low_memory=False)  #, dtype=object)

        self._database = ibtracs_df
        self._database_storm_names = ibtracs_df['NAME'].unique()
        self._database_storm_ids = ibtracs_df['USA_ATCF_ID'].unique()


    # with open(self._filepath, 'rb') as handle:
    #     ibtracs_df = pickle.load(handle)

    # ibtracs_df = self._fetch_best_track_database()

    # def _fetch_best_track_database(self):
    #     ibtracs_csv = f'ibtracs.{self._database_subset}.list.v04r00.csv'
    #     ibtracs_df = pd.read_csv(IBTRACS_BASE_URL + ibtracs_csv,
    #                              dtype=object)
    #     # urllib.request.urlretrieve(IBTRACS_BASE_URL + IBTRACS_CSV_FILENAME,
    #     #                            IBTRACS_CSV_FILENAME)
    #     return ibtracs_df

    def _set_storm_df(self):
        #TODO: need to support name and year
        is_storm_id = re.match(r'[A-Z]+[0-9]+', self._storm_name)
        if is_storm_id:
            storm_df = self._get_storm_df_by_id(is_storm_id.group())
        else:
            storm_df = self._get_storm_df_by_name(self._storm_name)

        storm_df["ISO_TIME"] = pd.to_datetime(storm_df['ISO_TIME'],
                                              format='%Y-%m-%d %H:%M:%S',
                                              utc=True)
        storm_df.set_index('ISO_TIME', inplace=True)
        self._df = storm_df

    def _get_storm_df_by_id(self, storm_id):
        if storm_id not in self._database_storm_ids:
            raise ValueError(
                f'"{storm_id}" not found in IBTrACS database. Please check '
                f'spelling.\n Valid IDs are:\n {self._database_storm_ids}')
        in_storm_id = self._database['USA_ATCF_ID'] == storm_id
        return self._database.loc[in_storm_id].reset_index()

    def _get_storm_df_by_name(self, storm_name):
        if storm_name not in self._database_storm_names:
            raise ValueError(
                f'"{storm_name}" not found in IBTrACS database. Please check '
                f'spelling.\n Valid names are:\n {self._database_storm_names}')
        in_storm_name = self._database['NAME'] == self._storm_name
        return self._database.loc[in_storm_name].reset_index()

    def plot(self, ax, **plt_kwargs):

        # cmap = mpl.cm.get_cmap('YlOrRd', 6)  # discrete colors
        # TODO: plot USA_SSHS

        track_line = shapely.geometry.LineString(self.gdf.geometry.values)
        line = ax.add_geometries(
            track_line,
            cartopy.crs.PlateCarree(),  #TODO:
            edgecolor='k',
            facecolor='none',
        )
        

        if 'color' not in plt_kwargs and 'column' not in plt_kwargs:
            plt_kwargs['color'] = 'k'

        if 'markersize' not in plt_kwargs:
            plt_kwargs['markersize'] = 10

    
        self.gdf.plot(
            ax=ax,
            zorder=line.get_zorder(),
            **plt_kwargs,
        )

