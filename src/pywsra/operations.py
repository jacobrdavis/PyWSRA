"""
TODO:
"""

__all__ = [
    "rotate_xy",
    "wn_spectrum_to_fq_dir_spectrum",
    "calculate_mean_spectral_area",
    "calculate_wn_mag_and_dir",
]

from typing import Tuple

import numpy as np
import xarray as xr
from scipy.interpolate import RegularGridInterpolator

from . import waves


def rotate_xy(
    x: np.ndarray,
    y: np.ndarray,
    theta: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform 2D coordinate rotation of points (x, y).

    Args:
        x (np.ndarray): x coordinate to rotate
        y (np.ndarray): y coordinate to rotate
        theta (np.ndarray): rotation angles in [rad]

    Returns:
        Tuple[np.ndarray, np.ndarray]: rotated x and y coordinates
    """
    x_rot = x*np.cos(theta) - y*np.sin(theta)
    y_rot = x*np.sin(theta) + y*np.cos(theta)

    return x_rot, y_rot


def wn_spectrum_to_fq_dir_spectrum(  # spectrum_wavenumber_to_frequency
    energy: xr.DataArray,
    wavenumber_east: xr.DataArray,
    wavenumber_north: xr.DataArray,
    depth: float = 1000.0,  #TODO: np.inf?
    regrid: bool = True,
    directional_resolution: float = 1,  # deg
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    if 'time' in energy.dims:
        energy = energy.transpose('wavenumber_east', 'wavenumber_north', 'time').values
    else:
        energy = energy.values[:, :, None]

    spectral_area = calculate_mean_spectral_area(wavenumber_east,  # rad^2/m^2
                                                 wavenumber_north)
    energy_density_wn = energy / spectral_area  # m^4/rad^2

    if regrid:
        fq_dir_spectrum = _wn_spectrum_to_fq_dir_spectrum_regrid(
            energy_density_wn=energy_density_wn,  #TODO: will need to move inside
            wavenumber_east=wavenumber_east.values,
            wavenumber_north=wavenumber_north.values,
            depth=depth,
            directional_resolution=directional_resolution,
        )

    else:  #TODO: need to test new implementation
        fq_dir_spectrum = _wn_spectrum_to_fq_dir_spectrum_no_regrid(
            energy_density_wn=energy_density_wn,
            wavenumber_east=wavenumber_east,
            wavenumber_north=wavenumber_north,
            depth=depth,
        )

    energy_density_fq_dir, direction, frequency = fq_dir_spectrum

    xr.

    return energy_density_fq_dir, direction, frequency


def _wn_spectrum_to_fq_dir_spectrum_regrid(
    energy_density_wn: np.ndarray,
    wavenumber_east: np.ndarray,
    wavenumber_north: np.ndarray,
    depth: float,
    directional_resolution: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    positive_wavenumber = wavenumber_north[wavenumber_north > 0]
    angular_frequency_1d = waves.intrinsic_dispersion(positive_wavenumber)

    direction_1d = np.deg2rad(np.arange(0, 360 + directional_resolution, directional_resolution))

    angular_frequency, direction = np.meshgrid(angular_frequency_1d,
                                               direction_1d)
    frequency = angular_frequency / (2 * np.pi)

    wavenumber = waves.dispersion_solver(angular_frequency/(2*np.pi),
                                         depth)
    wavenumber_direction_x = wavenumber * np.cos(direction)
    wavenumber_direction_y = wavenumber * np.sin(direction)

    original_points = (wavenumber_east, wavenumber_north)
    interpolation_points = (wavenumber_direction_x, wavenumber_direction_y)
    interpolator = RegularGridInterpolator(original_points,
                                           energy_density_wn)
    energy_density_intp = interpolator(interpolation_points)
    energy_density_fq_dir = waves.wn_energy_to_fq_energy(energy_density_intp,
                                                         wavenumber,
                                                         depth)

    #TODO: need to re-transpose outputs
    fq_dir_spectrum_var = _calculate_fq_dir_spectrum_var(energy_density_fq_dir,
                                                         direction[:, 0],
                                                         frequency[0])

    wn_spectrum_var = _calculate_wn_spectrum_var(energy_density_wn,
                                                 wavenumber_east,
                                                 wavenumber_north,
                                                 blank_corners=True)

    if not np.allclose(fq_dir_spectrum_var, wn_spectrum_var, rtol=0.01):
        raise ValueError(
            f'Variance mismatch:'
            f'Frequency spectrum variance is {fq_dir_spectrum_var} m^2.'
            f'Wavenumber spectrum variance is {wn_spectrum_var} m^2.'
        )
    return energy_density_fq_dir, direction, frequency


def _wn_spectrum_to_fq_dir_spectrum_no_regrid(
    energy_density_wn: np.ndarray,
    wavenumber_east: np.ndarray,
    wavenumber_north: np.ndarray,
    depth: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    #TODO:
    wavenumber, direction = calculate_wn_mag_and_dir(wavenumber_east,
                                                     wavenumber_north)
    direction = waves.trig_to_met(direction)
    angular_frequency = waves.intrinsic_dispersion(wavenumber, depth)
    frequency = angular_frequency / (2 * np.pi)
    energy_density_fq_dir = waves.wn_energy_to_fq_energy(energy_density_wn,
                                                         wavenumber,
                                                         depth)
    return energy_density_fq_dir, direction, frequency


def _calculate_fq_dir_spectrum_var(
    energy_density,
    direction,
    frequency,
):
    scalar_energy_density = np.trapz(energy_density, direction, axis=0)
    variance = np.trapz(scalar_energy_density, frequency, axis=0)
    return variance


def _calculate_wn_spectrum_var(
    energy_density,
    wavenumber_east,
    wavenumber_north,
    blank_corners: bool = False,
):
    if blank_corners:
        energy_density = _blank_wn_spectrum_corners(energy_density,
                                                    wavenumber_east,
                                                    wavenumber_north)

    energy_density_north = np.trapz(energy_density, wavenumber_east, axis=0)
    variance = np.trapz(energy_density_north, wavenumber_north, axis=0)
    return variance


def _blank_wn_spectrum_corners(
    energy_density,
    wavenumber_east,
    wavenumber_north,
):
    #ASSUMES square!
    # Blank the variance in the corners which are not regridded
    wavenumber = calculate_wn_mag_and_dir(wavenumber_east, wavenumber_north)[0]
    in_circle = wavenumber >= wavenumber_north.max()
    energy_density_no_corners = energy_density.copy()
    energy_density_no_corners[in_circle] = 0
    return energy_density_no_corners


def calculate_mean_spectral_area(
    wavenumber_east: np.ndarray,
    wavenumber_north: np.ndarray,
) -> float:
    east_spacing = np.diff(wavenumber_east)
    north_spacing = np.diff(wavenumber_north)
    areas = np.outer(east_spacing, north_spacing)
    return areas.mean() # rad^2/m^2


def calculate_wn_mag_and_dir(
    wavenumber_east: np.ndarray,
    wavenumber_north: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    wavenumber_east_grid, wavenumber_north_grid = np.meshgrid(wavenumber_east,
                                                              wavenumber_north,
                                                              indexing='xy')
    magnitude = np.sqrt(wavenumber_east_grid**2 + wavenumber_north_grid**2)
    direction = np.arctan2(wavenumber_north_grid, wavenumber_east_grid)  # TODO: swap for met conv?
    return magnitude, direction


#TODO: use Chris table as test
def correct_mss_for_rain(mss_0, rain_rate, altitude):
    """Correct WSRA mean square slopes for rain attenuation.

    WSRA mean square slope is calculated from the slope of the backscattered
    power versus the tangent squared of the off-nadir angle.  Since scans
    further from nadir have a slightly longer path length, they experience more
    rain attenuation.  This function accounts for this effect by adjusting the
    original mean square slope, `mss_0`, by an attenuation factor, DN:

    mss = (1/mss_0 - DN/const)^(-1)

    where const = 20log10(e).  The attenuation DN is a function of rainfall
    rate and radar altitude.

    Args:
        mss_0 (np.ndarray): original WSRA mean square slope with shape (n,)
        rain_rate (np.ndarray): rainfall rate in mm/hr with shape (n,)
        altitude (np.ndarray): radar altitude in m with shape (n,)

    Returns:
        np.ndarray: mean square slope corrected for rain attenuation with
            shape (n,).
    """
    altitude_km = altitude * 10**(-3)
    atten = calculate_rain_attenuation(rain_rate, altitude_km)
    const = 20 * np.log10(np.e)
    return (1/mss_0 - atten/const)**(-1)


def calculate_rain_attenuation(
    rain_rate: np.ndarray,
    altitude_km: np.ndarray,
) -> np.ndarray:
    """Calculate rain attenuation based on rainfall rate and radar altitude.

    The attenuation DN, expressed in dB, is:

    DN = alpha * rain_rate * altitude_km

    where `alpha` = 0.16 dBZ/km/(mm/hr) is a 2-way attenuation coefficient,
    `rain_rate` is the rainfall rate in mm/hr, and `altitude_km` is the radar
    altitude in km.

    Args:
        rain_rate (np.ndarray): rainfall rate in mm/hr with shape (n,)
        altitude_km (np.ndarray): radar altitude in km with shape (n,)

    Returns:
        np.ndarray: attenuation, in dB, at each rain rate and altitude with
            shape (n,).
    """
    alpha = 0.16  # 2-way attenuation coeff in dBZ/km/(mm/hr)
    return alpha * rain_rate * altitude_km
