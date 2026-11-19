"""Idealized nonrotating (Schwarzschild) black hole model for Newtonian test-particle integration.

The Schwarzschild radius is metadata and a simulation boundary only; it does
not correct the Newtonian acceleration. This model does not simulate general
relativistic motion near or inside the event horizon.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.constants import c as speed_of_light
from scipy.constants import G as gravitational_constant


@dataclass(frozen=True)
class BlackHole:
    """An idealized nonrotating black hole for Newtonian test-particle integration.

    Attributes
    ----------
    mass_kg:
        Mass of the black hole in kilograms (must be positive and finite).
    position:
        Cartesian position of the black hole centre, shape ``(2,)`` in metres.
    cutoff_radius_m:
        Integration cutoff radius in metres. Particles stop being simulated
        when they reach this radius. Must be positive, finite, and strictly
        greater than the Schwarzschild radius.
    """

    mass_kg: float
    position: np.ndarray
    cutoff_radius_m: float

    def __post_init__(self) -> None:
        if not np.isfinite(self.mass_kg) or self.mass_kg <= 0.0:
            raise ValueError("mass_kg must be a positive finite value")
        pos = np.array(self.position, dtype=np.float64, copy=True)
        if pos.shape != (2,):
            raise ValueError("position must have shape (2,)")
        if not np.all(np.isfinite(pos)):
            raise ValueError("position must contain only finite values")
        if not np.isfinite(self.cutoff_radius_m) or self.cutoff_radius_m <= 0.0:
            raise ValueError("cutoff_radius_m must be a positive finite value")
        if self.cutoff_radius_m <= self.schwarzschild_radius_m:
            raise ValueError(
                "cutoff_radius_m must be greater than the Schwarzschild radius "
                f"({self.schwarzschild_radius_m:.6e} m)"
            )
        pos.setflags(write=False)
        object.__setattr__(self, "position", pos)

    @property
    def schwarzschild_radius_m(self) -> float:
        """Schwarzschild radius ``r_s = 2 G M / c^2`` in metres.

        Metadata and simulation boundary only; not a relativistic correction
        to the Newtonian acceleration.
        """
        return float(2.0 * gravitational_constant * self.mass_kg / speed_of_light**2)

    def _displacement(self, position: np.ndarray) -> np.ndarray:
        pos = np.asarray(position, dtype=np.float64)
        if pos.shape != (2,):
            raise ValueError("position must have shape (2,)")
        if not np.all(np.isfinite(pos)):
            raise ValueError("position must contain only finite values")
        return pos - self.position

    def distance(self, position: np.ndarray) -> float:
        """Distance from the black hole centre to ``position`` in metres."""
        return float(np.linalg.norm(self._displacement(position)))

    def gravitational_acceleration(self, position: np.ndarray) -> np.ndarray:
        """Newtonian gravitational acceleration vector at ``position``.

        ``a(r) = -G M (r - r_BH) / |r - r_BH|^3``

        Returns acceleration in m/s^2, shape ``(2,)``, pointing toward the
        black hole centre. Raises ``ValueError`` if ``position`` coincides
        with the black hole centre (singularity).
        """
        displacement = self._displacement(position)
        distance_sq = float(np.dot(displacement, displacement))
        if distance_sq <= 0.0:
            raise ValueError("position coincides with the black hole centre (singularity)")
        inverse_cubed = distance_sq ** -1.5
        return -gravitational_constant * self.mass_kg * displacement * inverse_cubed
