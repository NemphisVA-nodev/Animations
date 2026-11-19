"""Newtonian tidal acceleration calculations for spaghettification.

All calculations use SI units and are independent of animation frame rate.
The engine is Newtonian: it does not model general relativistic effects near
or inside a black hole event horizon. That limitation is documented and must
be disclosed whenever results approach the Schwarzschild radius.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

GRAVITATIONAL_CONSTANT = 6.67430e-11  # m^3 kg^-1 s^-2 (CODATA 2022)


@dataclass(frozen=True)
class TidalParameters:
    """Inputs for a tidal acceleration evaluation.

    All fields are SI units.
    """

    attracting_mass_kg: float
    object_length_m: float
    center_distance_m: float


def radial_acceleration(mass_kg: float, distance_m: float) -> float:
    """Newtonian gravitational acceleration magnitude toward the mass.

    Parameters
    ----------
    mass_kg:
        Attracting central mass in kilograms (must be positive).
    distance_m:
        Radial distance from the mass centre in metres (must be positive).

    Returns
    -------
    float
        Acceleration magnitude in m/s^2 directed toward the mass.
    """
    if mass_kg <= 0.0:
        raise ValueError("attracting mass must be positive")
    if distance_m <= 0.0:
        raise ValueError("distance must be positive")
    return float(GRAVITATIONAL_CONSTANT * mass_kg / distance_m**2)


def tidal_difference_exact(params: TidalParameters) -> float:
    """Exact Newtonian near-minus-far tidal acceleration magnitude.

    The near end (closer to the mass) is pulled more strongly than the far
    end, producing radial stretching. The result is the difference between
    the acceleration magnitude at the near end and at the far end of an
    object of length ``object_length_m`` centred at ``center_distance_m``.

    Parameters
    ----------
    params:
        TidalParameters with all SI values.

    Returns
    -------
    float
        Exact tidal acceleration difference in m/s^2.

    Raises
    ------
    ValueError
        If the centre distance does not exceed half the object length (the
        near end would reach or cross the mass centre).
    """
    half = params.object_length_m / 2.0
    if params.center_distance_m <= half:
        raise ValueError(
            "center distance must exceed half the object length "
            f"({half} m); near end would reach the mass centre"
        )
    near_distance = params.center_distance_m - half
    far_distance = params.center_distance_m + half
    near_accel = GRAVITATIONAL_CONSTANT * params.attracting_mass_kg / near_distance**2
    far_accel = GRAVITATIONAL_CONSTANT * params.attracting_mass_kg / far_distance**2
    return float(near_accel - far_accel)


def tidal_difference_approximate(params: TidalParameters) -> float:
    """First-order Newtonian tidal approximation 2 G M L / r^3.

    Valid when the object length is much smaller than the centre distance
    (L << r). This is the inverse-cube scaling used for the educational
    approximation.

    Parameters
    ----------
    params:
        TidalParameters with all SI values.

    Returns
    -------
    float
        Approximate tidal acceleration in m/s^2.
    """
    if params.attracting_mass_kg <= 0.0:
        raise ValueError("attracting mass must be positive")
    if params.object_length_m <= 0.0:
        raise ValueError("object length must be positive")
    if params.center_distance_m <= 0.0:
        raise ValueError("distance must be positive")
    return float(
        2.0
        * GRAVITATIONAL_CONSTANT
        * params.attracting_mass_kg
        * params.object_length_m
        / params.center_distance_m**3
    )


def transverse_compression(params: TidalParameters) -> float:
    """Approximate transverse tidal compression.

    Perpendicular to the radial direction the tidal field compresses with
    magnitude G M L / r^3, opposite in sign to the radial stretch. The
    returned value is negative to denote compression.

    Parameters
    ----------
    params:
        TidalParameters with all SI values.

    Returns
    -------
    float
        Transverse tidal acceleration in m/s^2 (negative = compression).
    """
    return -0.5 * tidal_difference_approximate(params)


def particle_positions(
    params: TidalParameters,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return centre, near-end, and far-end positions relative to the mass.

    Positions are radial distances in metres along the outward radial axis
    (positive away from the mass). Gravitational acceleration points toward
    the mass (negative radial direction).

    Returns
    -------
    tuple of np.ndarray
        (centre, near, far) positions in metres.
    """
    half = params.object_length_m / 2.0
    centre = np.array([params.center_distance_m], dtype=np.float64)
    near = np.array([params.center_distance_m - half], dtype=np.float64)
    far = np.array([params.center_distance_m + half], dtype=np.float64)
    return centre, near, far
