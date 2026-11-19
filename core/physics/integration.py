"""Numerical trajectory integration for independent test particles.

Uses SciPy ``solve_ivp`` (DOP853 by default) to integrate each test particle
independently. Each particle is stopped by an event when it reaches the
black hole's configured cutoff radius, so particles reaching the cutoff at
different times are handled correctly. Results are interpolated onto a
common, frame-rate-independent time grid with an explicit validity mask.

The solver does not add gravitational softening; trajectories that approach
the black hole centre are singular and are bounded by the cutoff.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp

from .black_hole import BlackHole
from .particles import ParticleCloud

_DEFAULT_METHOD = "DOP853"


@dataclass(frozen=True)
class TrajectoryResult:
    """Structured result of a multi-particle trajectory integration.

    Attributes
    ----------
    timestamps:
        Common simulation time grid, shape ``(T,)`` in seconds.
    positions:
        Particle positions, shape ``(T, N, 2)`` in metres. Invalid entries
        (after a particle's termination) are NaN.
    velocities:
        Particle velocities, shape ``(T, N, 2)`` in m/s. Invalid entries are
        NaN.
    validity_mask:
        Boolean mask, shape ``(T, N)``. True where the particle was still
        being simulated at that timestamp.
    particle_ids:
        Stable particle identifiers, shape ``(N,)``.
    termination_times:
        Time at which each particle reached the cutoff radius, or ``None`` if
        it never did, in seconds.
    black_hole:
        The black hole model used.
    solver_success:
        Whether every particle's solver reached its end condition (normal
        completion or expected terminal cutoff event) without failure.
    solver_message:
        Aggregated solver status message.
    """

    timestamps: np.ndarray
    positions: np.ndarray
    velocities: np.ndarray
    validity_mask: np.ndarray
    particle_ids: list[str]
    termination_times: list[float | None]
    black_hole: BlackHole
    solver_success: bool
    solver_message: str
    solver_method: str = _DEFAULT_METHOD
    solver_rtol: float = 1e-9
    solver_atol: float = 1e-12


def _make_rhs(black_hole: BlackHole):
    def rhs(_t: float, state: np.ndarray) -> np.ndarray:
        position = state[:2]
        velocity = state[2:]
        acceleration = black_hole.gravitational_acceleration(position)
        return np.array(
            [velocity[0], velocity[1], acceleration[0], acceleration[1]],
            dtype=np.float64,
        )

    return rhs


def _make_cutoff_event(black_hole: BlackHole):
    def event(_t: float, state: np.ndarray) -> float:
        position = state[:2]
        return black_hole.distance(position) - black_hole.cutoff_radius_m

    event.terminal = True
    event.direction = -1
    return event


def integrate_particles(
    black_hole: BlackHole,
    cloud: ParticleCloud,
    duration_s: float,
    num_steps: int,
    *,
    rtol: float = 1e-9,
    atol: float = 1e-12,
    method: str = _DEFAULT_METHOD,
) -> TrajectoryResult:
    """Integrate independent test-particle trajectories around a black hole.

    Each particle is integrated independently with event-based cutoff
    detection. Results are interpolated onto a common time grid; particles
    that reached the cutoff are marked invalid after their termination time
    and their trailing entries are NaN.

    Parameters
    ----------
    black_hole:
        The black hole model defining the gravitational field and cutoff.
    cloud:
        The particles to integrate.
    duration_s:
        Total integration duration in seconds (must be positive).
    num_steps:
        Number of samples on the common time grid (must be >= 2).
    rtol:
        Relative tolerance for the solver (must be positive).
    atol:
        Absolute tolerance for the solver (must be positive).
    method:
        SciPy ``solve_ivp`` integration method.

    Returns
    -------
    TrajectoryResult
        Structured integration result.
    """
    if not np.isfinite(duration_s) or duration_s <= 0.0:
        raise ValueError("duration_s must be positive")
    if (
        isinstance(num_steps, (bool, np.bool_))
        or not isinstance(num_steps, (int, np.integer))
        or num_steps < 2
    ):
        raise ValueError("num_steps must be an integer of at least 2")
    if not np.isfinite(rtol) or rtol <= 0.0:
        raise ValueError("rtol must be a positive finite value")
    if not np.isfinite(atol) or atol <= 0.0:
        raise ValueError("atol must be a positive finite value")

    if cloud.count == 0:
        raise ValueError("particle cloud must not be empty")
    for particle in cloud.particles:
        if black_hole.distance(particle.position) <= black_hole.cutoff_radius_m:
            raise ValueError(
                f"particle {particle.id!r} starts at or inside the cutoff radius"
            )

    timestamps = np.linspace(0.0, duration_s, num_steps)
    count = cloud.count
    particle_ids = cloud.ids

    positions = np.full((num_steps, count, 2), np.nan, dtype=np.float64)
    velocities = np.full((num_steps, count, 2), np.nan, dtype=np.float64)
    validity_mask = np.zeros((num_steps, count), dtype=bool)
    termination_times: list[float | None] = []
    statuses: list[int] = []
    messages: list[str] = []

    rhs = _make_rhs(black_hole)
    cutoff_event = _make_cutoff_event(black_hole)

    for i in range(count):
        particle = cloud.particles[i]
        state0 = np.array(
            [
                particle.position[0],
                particle.position[1],
                particle.velocity[0],
                particle.velocity[1],
            ],
            dtype=np.float64,
        )

        sol = solve_ivp(
            rhs,
            (0.0, duration_s),
            state0,
            method=method,
            events=cutoff_event,
            dense_output=True,
            rtol=rtol,
            atol=atol,
        )

        if not sol.success or sol.status not in (0, 1) or sol.sol is None:
            raise RuntimeError(
                f"integration failed for particle {particle.id!r}: {sol.message}"
            )
        statuses.append(int(sol.status))
        messages.append(str(sol.message))

        t_term: float | None = None
        if sol.t_events[0].size > 0:
            t_term = float(sol.t_events[0][0])
        termination_times.append(t_term)

        if t_term is None:
            valid_mask = np.ones(num_steps, dtype=bool)
        else:
            valid_mask = timestamps <= t_term

        if np.any(valid_mask):
            eval_times = timestamps[valid_mask]
            sampled = sol.sol(eval_times)
            positions[valid_mask, i, 0] = sampled[0]
            positions[valid_mask, i, 1] = sampled[1]
            velocities[valid_mask, i, 0] = sampled[2]
            velocities[valid_mask, i, 1] = sampled[3]
            validity_mask[valid_mask, i] = True

    # Status 0 = reached end of tspan; 1 = terminal cutoff event (expected).
    # Any other status is a solver failure.
    solver_success = all(s in (0, 1) for s in statuses)
    solver_message = "; ".join(messages)

    return TrajectoryResult(
        timestamps=timestamps,
        positions=positions,
        velocities=velocities,
        validity_mask=validity_mask,
        particle_ids=particle_ids,
        termination_times=termination_times,
        black_hole=black_hole,
        solver_success=solver_success,
        solver_message=solver_message,
        solver_method=method,
        solver_rtol=rtol,
        solver_atol=atol,
    )
