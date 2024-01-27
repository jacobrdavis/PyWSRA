"""
TODO:
"""

__all__ = [
    "intrinsic_dispersion",
    "phase_velocity",
    "group_to_phase_ratio",
    "intrinsic_group_velocity",
    "dispersion_solver",
    "frequency_to_angular_frequency",
    "deep_water_dispersion",
    "trig_to_met",
    "wn_energy_to_fq_energy",
]

from typing import Union, Tuple

import numpy as np
from scipy.optimize import newton

GRAVITY = 9.81


def intrinsic_dispersion(wavenumber, depth=np.inf):
    GRAVITY = 9.81
    gk = GRAVITY * wavenumber
    kh = wavenumber * depth
    return np.sqrt(gk * np.tanh(kh))  # angular frequency


def phase_velocity(wavenumber, depth=np.inf):
    return intrinsic_dispersion(wavenumber, depth) / wavenumber


def group_to_phase_ratio(
    wavenumber: np.ndarray,
    depth: float = np.inf,
) -> np.ndarray:
    """ Compute the ratio of group velocity to phase velocity.

    Note: to prevent overflows in `np.sinh`, the product of wavenumber and
    depth (relative depth) are used to assign ratios at deep or shallow limits:

        shallow:  Cg = 1.0 if kh < np.pi/10 (h < L/20)
           deep:  Cg = 0.5 if kh > np.pi    (h > L/2)

    Args:
        wavenumber (np.ndarray): of shape (k,) containing wavenumbers
        depth (float, optional): positive water depth. Defaults to np.inf.

    Returns:
        np.ndarray: of shape (k,) containing ratio at each wavenumber.
    """
    kh = wavenumber * depth
    in_deep, in_shallow, in_intermd = depth_regime(kh)
    ratio = np.empty(kh.shape)
    ratio[in_deep] = 0.5
    ratio[in_shallow] = 1.0
    ratio[in_intermd] = 0.5 + kh[in_intermd] / np.sinh(2 * kh[in_intermd])
    return ratio


def depth_regime(kh: np.ndarray) -> Tuple:
    """ Classify depth regime based on relative depth.

    Classify depth regime based on relative depth (product of wavenumber
    and depth) using the shallow and deep limits:

        shallow:  kh < np.pi/10 (h < L/20)
           deep:  kh > np.pi    (h > L/2)

    The depth regime is classified as intermediate if not at the deep or
    shallow limits.

    Args:
        kh (np.ndarray): relative depth of shape (k, )

    Returns:
        np.ndarray[bool]: true where kh is deep, false otherwise
        np.ndarray[bool]: true where kh is shallow, false otherwise
        np.ndarray[bool]: true where kh is intermediate, false otherwise
    """
    in_deep = kh > np.pi
    in_shallow = kh < np.pi/10
    in_intermd = np.logical_and(~in_deep, ~in_shallow)
    return in_deep, in_shallow, in_intermd

def intrinsic_group_velocity(wavenumber, depth=np.inf):
    ratio = group_to_phase_ratio(wavenumber, depth)
    return ratio * phase_velocity(wavenumber, depth)


def dispersion_solver(  #TODO: inverse dispersion?
    frequency: np.ndarray,
    depth: Union[float, np.ndarray],
) -> np.ndarray:
    r"""Solve the linear dispersion relationship.

    Solves the linear dispersion relationship w^2 = gk tanh(kh) using a
    Scipy Newton-Raphson root-finding implementation.

    Note:
        Expects input as numpy.ndarrays of shape (d,f) where f is the number
        of frequencies and d is the number of depths. The input `frequency` is
        the frequency in Hz and NOT the angular frequency, omega or w.

    Args:
        frequency (np.ndarray): of shape (d,f) containing frequencies in [Hz].
        depth (np.ndarray): of shape (d,f) containing water depths.

    Returns:
        np.ndarray: of shape (d,f) containing wavenumbers.
    """

    angular_frequency = frequency_to_angular_frequency(frequency)

    wavenumber_deep = deep_water_dispersion(frequency)

    wavenumber = newton(func=_dispersion_root,
                        x0=wavenumber_deep,
                        args=(angular_frequency, depth),
                        fprime=_dispersion_derivative)
    return np.asarray(wavenumber)


def _dispersion_root(wavenumber, angular_frequency, depth):
    gk = GRAVITY * wavenumber
    kh = wavenumber * depth
    return  gk * np.tanh(kh) - angular_frequency**2


def _dispersion_derivative(wavenumber, angular_frequency, depth):
    gk = GRAVITY * wavenumber
    kh = wavenumber * depth
    return GRAVITY * np.tanh(kh) + gk * depth * (1 - np.tanh(kh)**2)


def frequency_to_angular_frequency(frequency):
    """Helper function to convert frequency (f) to angular frequency (omega)"""
    return 2 * np.pi * frequency


def deep_water_dispersion(frequency):
    """Computes wavenumber from the deep water linear dispersion relationship.

    Given frequencies (in Hz) solve the linear dispersion relationship in the
    deep water limit for the corresponding wavenumbers, k. The linear
    dispersion relationship in the deep water limit, tanh(kh) -> 1, has the
    closed form solution k = omega^2 / g and is (approximately) valid for
    kh > np.pi (h > L/2).

    Args:
        frequency (np.ndarray): of any shape containing frequencies
            in [Hz]. NOT the angular frequency, omega or w.

    Returns:
        np.ndarray: (of shape equal to the input shape) containing wavenumbers.
    """
    angular_frequency = frequency_to_angular_frequency(frequency)
    return angular_frequency**2 / GRAVITY


def trig_to_met(  #TODO: broken!
    angle_trig: np.ndarray,
    degrees=False
) -> np.ndarray:
    """Convert an angle from the trigonometric to meterological convention.

    Convert from a trignometric angle convention, defined as counterclockwise
    positive with 0 aligned with the Cartesian x-axis, to the meterological
    convention, clockwise positive with 0 aligned with the Cartesian y-axis
    (which represents North).

    Args:
        angle_trig (np.ndarray): angles in the trigonometric convention.
        degrees (bool, optional): True if input angle is in degrees. Defaults
            to False.

    Returns:
        np.ndarray: angle in meterological convention with equivalent units
            to the input `angle_trig`.
    """
    if degrees:
        offset = 90
        modulus = 360
        # angle_met = (-angle_trig + 90) % 360
    else:
        offset = np.pi/2
        modulus = 2*np.pi
        # angle_met = (-angle_trig + np.pi/2) % (2*np.pi)
    return (-angle_trig + offset) % modulus


def wn_energy_to_fq_energy(
    energy_density_wavenumber: np.ndarray,
    wavenumber: np.ndarray,
    depth: float = np.inf,
) -> np.ndarray:
    """ Transform energy density from wavenumber space to frequency space.

    Transform energy density, defined on a 2-D wavenumber grid, to energy
    density on a frequency-direction grid using the appropriate Jacobian.
    The calculation follows that of the WaMoS II processing in [1]:

    E(w, theta) = E(kx, ky) k dk/dw

    and

    E(f, theta) = E(w, theta) * 2 pi

    Where E(w, theta) is the energy density as a function of angular frequency
    (w) and direction (theta), E(kx, ky) is the energy density as a function of
    the east wavenumber (kx) and north wavenumber (kx), k is the
    scalar wavenumber computed as magnitude{kx, ky}, and k dk/dw comprise the
    Jacobian with dk/dw equal to the inverse of the group velocity
    (i.e., 1/Cg). The final result is converted from angular frequency to
    frequency (f) such that E(f, theta) is returned. This transformation
    assumes linear wave dispersion.

    References:
        1. Stephen F. Barstow, Jean-Raymond Bidlot, Sofia Caires, Mark A.
        Donelan, William M. Drennan, et al.. Measuring and Analysing the
        directional spectrum of ocean waves. D. Hauser, K. Kahma, H. Krogstad,
        S. Monbaliu, S. Lehner et L. Wyatt. COST Office, pp.465, 2005,
        COST 714; EUR 21367. ffhal-00529755f

    Args:
        energy_density_wavenumber (np.ndarray): Energy density in 2-D
            wavenumber space with shape (x, y).
        wavenumber (np.ndarray): Wavenumber magnitudes with shape (x, y).
        depth (float, optional): Positive depth. Defaults to np.inf.

    Returns:
        np.ndarray: energy density in frequency-direction space, E(f, theta),
            with shape (x, y).
    """
    dk_dw = 1 / intrinsic_group_velocity(wavenumber, depth)
    jacobian = wavenumber * dk_dw
    return energy_density_wavenumber * jacobian[:, :, None] * 2*np.pi  # [:, :, None]


# For testing:
# w = intrinsic_dispersion(1, 1000)
# cp = phase_velocity(1, 1000)
# cg = intrinsic_group_velocity(1, 1000)
# print(w)
# print(cp)
# print(cg)

# wavenumber = 0.5
# group_velocity = intrinsic_group_velocity(1, 1000)
# jacobian = 1 / group_velocity