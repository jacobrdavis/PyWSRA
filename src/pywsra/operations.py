"""
TODO:
"""

__all__ = [
    "rotate_xy",
    "wn_spectrum_to_fq_spectrum",
    "calculate_mean_spectral_area",
    "calculate_wn_mag_and_dir",
]

from typing import Tuple

import numpy as np
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


def wn_spectrum_to_fq_spectrum(  # spectrum_wavenumber_to_frequency
    energy: np.ndarray,
    wavenumber_east: np.ndarray,
    wavenumber_north: np.ndarray,
    depth: float = np.inf,
    regrid: bool = True,
    directional_resolution: float = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    spectral_area = calculate_mean_spectral_area(wavenumber_east,  # rad^2/m^2
                                                 wavenumber_north)
    energy_density_wn = energy / spectral_area  # m^4/rad^2

    if regrid:
        positive_wavenumber = wavenumber_north[wavenumber_north > 0]
        angular_frequency_1d = waves.intrinsic_dispersion(positive_wavenumber)

        direction_1d = np.deg2rad(np.arange(0, 360 + directional_resolution, directional_resolution))

        angular_frequency, direction = np.meshgrid(angular_frequency_1d,
                                                   direction_1d)
        frequency = angular_frequency / (2 * np.pi)

        wavenumber = waves.dispersion_solver(angular_frequency/(2*np.pi),
                                             depth=1.0*10**3)
        wavenumber_direction_x = wavenumber * np.cos(direction)
        wavenumber_direction_y = wavenumber * np.sin(direction)

        original_points = (wavenumber_east, wavenumber_north)
        interpolation_points = (wavenumber_direction_x, wavenumber_direction_y)
        interpolator = RegularGridInterpolator(original_points,
                                               energy_density_wn)
        energy_density_intp = interpolator(interpolation_points)
        energy_density_fq = waves.wn_energy_to_fq_energy(energy_density_intp,
                                                         wavenumber,
                                                         depth)

        fq_spectrum_var = _calculate_fq_spectrum_variance(energy_density_fq,
                                                          direction[:, 0],
                                                          frequency[0])

        wn_spectrum_var = _calculate_wn_spectrum_variance(energy_density_wn,
                                                          wavenumber_east,
                                                          wavenumber_north,
                                                          blank_corners=True)

        if not np.isclose(fq_spectrum_var, wn_spectrum_var, rtol=0.01):
            raise ValueError(
                f'Variance mismatch:'
                f'Frequency spectrum variance is {fq_spectrum_var} m^2.'
                f'Wavenumber spectrum variance is {wn_spectrum_var} m^2.'
            )

    else:
        wavenumber, direction = calculate_wn_mag_and_dir(wavenumber_east,
                                                         wavenumber_north)
        direction = waves.trig_to_met(direction)
        angular_frequency = waves.intrinsic_dispersion(wavenumber)
        frequency = angular_frequency / (2 * np.pi)
        energy_density_fq = waves.wn_energy_to_fq_energy(energy_density_wn,
                                                         wavenumber,
                                                         depth)

    return energy_density_fq, direction, frequency


def _calculate_fq_spectrum_variance(
    energy_density,
    direction,
    frequency,
):
    scalar_energy_density = np.trapz(energy_density, direction, axis=0)
    variance = np.trapz(scalar_energy_density, frequency)
    return variance


def _calculate_wn_spectrum_variance(
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
