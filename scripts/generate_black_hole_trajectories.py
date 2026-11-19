"""Command-line generator for black hole test-particle trajectory datasets.

Produces deterministic, renderer-independent Newtonian trajectory datasets
(see ``docs/simulation_data_format.md``) for two demonstration scenarios:

Scenario A -- Radial Infall
    Particles fall radially toward a black hole, showing increasing
    acceleration and differential near/far acceleration.

Scenario B -- Tidal Particle Cloud
    A grid of particles falls toward a black hole, showing radial tidal
    stretching and transverse tidal compression.

Output is a JSON metadata sidecar plus a compressed NumPy NPZ archive. The
files can be read with only the Python standard library and NumPy.

These are Newtonian educational examples; they are not general-relativistic
solutions and do not represent real material deformation.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.physics.black_hole import BlackHole  # noqa: E402
from core.physics.integration import integrate_particles  # noqa: E402
from core.physics.measurements import (  # noqa: E402
    radial_tidal_acceleration,
    transverse_tidal_acceleration,
)
from core.physics.particles import (  # noqa: E402
    Particle,
    ParticleCloud,
    grid_particles,
)
from core.simulation.dataset import (  # noqa: E402
    NONFINITE_POLICY_KEY,
    NONFINITE_POLICY_MASKED,
    SimulationDataset,
    SimulationMetadata,
    save_dataset,
    trajectory_to_dataset,
)

SOLAR_MASS_KG = 1.9885e30

# Scenario defaults.
DEFAULT_BH_MASS_KG = 10.0 * SOLAR_MASS_KG  # 10 solar masses
DEFAULT_CUTOFF_M = 1.0e7  # 10,000 km (~339 Schwarzschild radii)
DEFAULT_INITIAL_RADIUS_M = 1.0e8  # 100,000 km
DEFAULT_DURATION_S = 50.0
DEFAULT_NUM_STEPS = 500

MODEL_NAME = "newtonian_test_particle"
MODEL_LIMITATIONS = (
    "Newtonian point-mass test-particle integration. Does not model general "
    "relativistic effects near or inside the Schwarzschild radius. The cutoff "
    "radius is placed outside the Schwarzschild radius; trajectories are not "
    "simulated at or inside the horizon."
)


def _build_black_hole(mass_kg: float, cutoff_radius_m: float) -> BlackHole:
    return BlackHole(
        mass_kg=mass_kg,
        position=np.array([0.0, 0.0], dtype=np.float64),
        cutoff_radius_m=cutoff_radius_m,
    )


def _run_scenario(
    black_hole: BlackHole,
    cloud: ParticleCloud,
    duration_s: float,
    num_steps: int,
    simulation_id: str,
    output_stem: Path,
    object_length_m: float,
) -> None:
    result = integrate_particles(
        black_hole,
        cloud,
        duration_s=duration_s,
        num_steps=num_steps,
    )

    dataset = trajectory_to_dataset(
        result,
        simulation_id=simulation_id,
        notes=(
            "Newtonian test-particle trajectories around an idealized "
            "nonrotating black hole. Terminated particles are marked invalid "
            "and trailing samples are NaN; no data is fabricated after "
            "termination."
        ),
    )
    dataset.metadata[NONFINITE_POLICY_KEY] = NONFINITE_POLICY_MASKED

    meta_path, data_path = save_dataset(dataset, output_stem)

    print(f"scenario: {simulation_id}")
    print(f"  wrote {meta_path}")
    print(f"  wrote {data_path}")
    print(
        f"  particles={cloud.count} duration={duration_s} s "
        f"steps={num_steps} cutoff={black_hole.cutoff_radius_m:.3e} m"
    )
    terminated = sum(1 for t in result.termination_times if t is not None)
    print(f"  terminated={terminated}/{cloud.count}")
    print(f"  solver_success={result.solver_success}")
    radial = radial_tidal_acceleration(result, object_length_m)
    print(
        f"  radial tidal range: "
        f"{np.nanmin(radial):.3e} to {np.nanmax(radial):.3e} m/s^2"
    )


def scenario_a_radial_infall(args: argparse.Namespace) -> None:
    """A small collection of particles falls radially toward a black hole."""
    black_hole = _build_black_hole(args.mass, args.cutoff)
    # Three particles on the inward radial line at slightly different radii,
    # all starting from rest. The nearest particle accelerates most strongly,
    # so radial separation grows.
    particles = []
    for i in range(3):
        radius = args.initial_radius - i * args.spacing
        if radius <= black_hole.cutoff_radius_m:
            raise ValueError(
                "initial radius minus particle spacing must stay outside the "
                "cutoff radius"
            )
        particles.append(
            Particle(
                id=f"radial_{i:04d}",
                position=np.array([radius, 0.0], dtype=np.float64),
                velocity=np.array([0.0, 0.0], dtype=np.float64),
            )
        )
    cloud = ParticleCloud(particles=particles)

    output_stem = Path(args.output).with_name(
        Path(args.output).stem + "_radial_infall"
    )
    _run_scenario(
        black_hole,
        cloud,
        duration_s=args.duration,
        num_steps=args.steps,
        simulation_id="black_hole_radial_infall",
        output_stem=output_stem,
        object_length_m=args.length,
    )


def scenario_b_tidal_cloud(args: argparse.Namespace) -> None:
    """A grid of particles falls toward a black hole (tidal stretch/compress)."""
    black_hole = _build_black_hole(args.mass, args.cutoff)
    cloud = grid_particles(
        center=(args.initial_radius, 0.0),
        nx=args.nx,
        ny=args.ny,
        spacing_m=args.spacing,
        common_velocity=(0.0, 0.0),
        id_prefix="grid",
    )
    output_stem = Path(args.output).with_name(
        Path(args.output).stem + "_tidal_cloud"
    )
    _run_scenario(
        black_hole,
        cloud,
        duration_s=args.duration,
        num_steps=args.steps,
        simulation_id="black_hole_tidal_cloud",
        output_stem=output_stem,
        object_length_m=args.length,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate deterministic black hole trajectory datasets."
    )
    sub = parser.add_subparsers(dest="scenario", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--mass", type=float, default=DEFAULT_BH_MASS_KG,
        help="Black hole mass in kg (default: 10 solar masses).",
    )
    common.add_argument(
        "--cutoff", type=float, default=DEFAULT_CUTOFF_M,
        dest="cutoff",
        help="Integration cutoff radius in metres (default: 1.0e7).",
    )
    common.add_argument(
        "--initial-radius", type=float, default=DEFAULT_INITIAL_RADIUS_M,
        dest="initial_radius",
        help="Initial distance from the black hole in metres (default: 1.0e8).",
    )
    common.add_argument(
        "--duration", type=float, default=DEFAULT_DURATION_S,
        dest="duration",
        help="Integration duration in seconds (default: 50.0).",
    )
    common.add_argument(
        "--steps", type=int, default=DEFAULT_NUM_STEPS,
        dest="steps",
        help="Number of time samples (default: 500).",
    )
    common.add_argument(
        "--length", type=float, default=1.0e6,
        dest="length",
        help="Characteristic object length for tidal estimates in metres "
        "(default: 1.0e6).",
    )
    common.add_argument(
        "--output", required=True,
        help="Output path stem (produces .meta.json and .data.npz).",
    )

    p_a = sub.add_parser(
        "radial-infall", parents=[common], help="Scenario A: radial infall."
    )
    p_a.add_argument(
        "--spacing", type=float, default=1.0e6,
        help="Radial spacing between particles in metres (default: 1.0e6).",
    )
    p_a.set_defaults(func=scenario_a_radial_infall)

    p_b = sub.add_parser(
        "tidal-cloud", parents=[common], help="Scenario B: tidal particle cloud."
    )
    p_b.add_argument(
        "--nx", type=int, default=5,
        help="Grid columns (default: 5).",
    )
    p_b.add_argument(
        "--ny", type=int, default=5,
        help="Grid rows (default: 5).",
    )
    p_b.add_argument(
        "--spacing", type=float, default=2.0e5,
        help="Grid spacing in metres (default: 2.0e5).",
    )
    p_b.set_defaults(func=scenario_b_tidal_cloud)

    args = parser.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
