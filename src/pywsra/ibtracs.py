#TODO: fetch directly from IBTRACS site
import re

import pickle
from datetime import datetime, timezone

import pandas as pd

IBTRACS_PATH = '/Users/jacob/Dropbox/Projects/NHCI/data/IBTrACS/IBTrACS.pickle'

with open(IBTRACS_PATH, 'rb') as handle:
    ibtracs = pickle.load(handle)

ibtracs = ibtracs[ibtracs.index > datetime(2018, 1, 2, tzinfo=timezone.utc)]

STORM_NAMES = ibtracs['NAME'].unique()
STORM_IDS = ibtracs['USA_ATCF_ID'].unique()


def get_storm_track(storm_name: str) -> pd.DataFrame:
    """
    Return a pandas DataFrame for a storm.

    Args:
        storm_name (str): Name of the storm (in all caps)

    Raises:
        ValueError: If storm name does not match any valid names in the
            IBTrACS database.

    Returns:
        pd.DataFrame: Subset of the IBTrACS DataFrame
    """
    storm_name_upper = storm_name.upper()

    is_storm_id = re.match(r'[A-Z]+[0-9]+', storm_name_upper)

    if is_storm_id:
        storm_id = is_storm_id.group()
        if storm_id not in STORM_IDS:
            raise ValueError(
                f'"{storm_id}" not found in IBTrACS database. Please check '
                f'the spelling.\n Valid IDs are:\n {STORM_IDS}')

        storm_track = ibtracs[ibtracs['USA_ATCF_ID'] == storm_id]

    else:
        if storm_name_upper not in STORM_NAMES:
            raise ValueError(
                f'"{storm_name}" not found in IBTrACS database. Please check '
                f'the spelling.\n Valid names are:\n {STORM_NAMES}')

        storm_track = ibtracs[ibtracs['NAME'] == storm_name_upper]

    return storm_track