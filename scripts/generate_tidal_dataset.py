"""Command-line generator for the radial tidal acceleration dataset.

Produces a deterministic, renderer-independent dataset showing how Newtonian
tidal acceleration changes for a fixed object length as it approaches a
gravitating mass. Output is a JSON metadata sidecar plus a compressed NumPy
NPZ array archive (see ``docs/simulation_data_format.md``).

The generator depends only on the Python standard library and NumPy, so the
resulting files can be read by any consumer with NumPy (including a future
Blender bundled Python) without installing the full scientific dependency set.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# Allow execution both as ``python scripts/generate_tidal_dataset.py`` and
# as an installed module by ensuring the repository root is importable.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.physics.tidal import (  # noqa: E402
    GRAVITATIONAL_CONSTANT,
    TidalParameters,
    radial_acceleration,
    tidal_difference_approximate,
    tidal_difference_exact,
    transverse_compression,
)
from core.simulation.dataset import (  # noqa: E402
    SimulationDataset,
    SimulationMetadata,
    save_dataset,
    validate_dataset,
)

SCHEMA_VERSION = "1.0.0"
MODEL_NAME = "newtonian_point_mass_tidal"
MODEL_LIMITATIONS = (
    "Newtonian point-mass approximation. Does not model general "
    "relativistic effects near or inside a black hole event horizon. "
    "Valid only where the near end of the object remains outside the "
    "attracting mass centre (centre distance > object length / 2) and far "
    "from the Schwarzschild radius."
)
REFERENCE_FRAME = {
    "origin": "centre of the attracting mass",
    "radial_axis": "positive outward from the mass",
    "gravity_direction": "toward the mass (negative radial)",
    "units": "SI",
}

DEFAULT_MASS_KG = 5.972e24  # Earth mass
DEFAULT_LENGTH_M = 10.0
DEFAULT_R_MIN_M = 2.0e7
DEFAULT_R_MAX_M = 1.0e8
DEFAULT_POINTS = 200


def build_dataset(
    mass_kg: float,
    length_m: float,
    r_min_m: float,
    r_max_m: float,
    num_points: int,
) -> SimulationDataset:
    """Build the radial tidal acceleration dataset.

    Parameters
    ----------
    mass_kg:
        Attracting central mass in kilograms.
    length_m:
        Fixed object length in metres.
    r_min_m:
        Minimum centre distance in metres (must exceed length_m / 2).
    r_max_m:
        Maximum centre distance in metres (> r_min_m).
    num_points:
        Number of sampled distances (>= 2).

    Returns
    -------
    SimulationDataset
        Deterministic dataset with SI units and strictly increasing
        timestamps.
    """
    if mass_kg <= 0.0:
        raise ValueError("attracting mass must be positive")
    if length_m <= 0.0:
        raise ValueError("object length must be positive")
    if r_min_m <= length_m / 2.0:
        raise ValueError(
            "r_min_m must exceed half the object length "
            f"({length_m / 2.0} m) so no invalid calculation point lies "
            "within the configured boundary"
        )
    if r_max_m <= r_min_m:
        raise ValueError("r_max_m must be greater than r_min_m")
    if num_points < 2:
        raise ValueError("num_points must be at least 2")

    distances = np.linspace(r_min_m, r_max_m, num_points).astype(np.float64)
    timestamps = np.arange(num_points, dtype=np.float64)  # 1 s cadence

    near_accel = np.empty(num_points, dtype=np.float64)
    far_accel = np.empty(num_points, dtype=np.float64)
    exact = np.empty(num_points, dtype=np.float64)
    approx = np.empty(num_points, dtype=np.float64)
    transverse = np.empty(num_points, dtype=np.float64)

    for i, r in enumerate(distances):
        params = TidalParameters(
            attracting_mass_kg=mass_kg,
            object_length_m=length_m,
            center_distance_m=float(r),
        )
        half = length_m / 2.0
        near_accel[i] = radial_acceleration(mass_kg, r - half)
        far_accel[i] = radial_acceleration(mass_kg, r + half)
        exact[i] = tidal_difference_exact(params)
        approx[i] = tidal_difference_approximate(params)
        transverse[i] = transverse_compression(params)

    arrays = {
        "timestamps": timestamps,
        "distances": distances,
        "near_acceleration": near_accel,
        "far_acceleration": far_accel,
        "tidal_difference_exact": exact,
        "tidal_difference_approximate": approx,
        "transverse_compression": transverse,
    }

    metadata: SimulationMetadata = SimulationMetadata(
        schema_version=SCHEMA_VERSION,
        simulation_id="tidal_radial_newtonian",
        simulation_type="tidal_acceleration",
        model=MODEL_NAME,
        model_limitations=MODEL_LIMITATIONS,
        units={
            "time": "s",
            "position": "m",
            "acceleration": "m/s^2",
            "mass": "kg",
            "length": "m",
        },
        reference_frame=REFERENCE_FRAME,
        physical_parameters={
            "attracting_mass_kg": mass_kg,
            "object_length_m": length_m,
            "r_min_m": r_min_m,
            "r_max_m": r_max_m,
            "num_points": num_points,
            "gravitational_constant": GRAVITATIONAL_CONSTANT,
        },
        created_at=datetime.now(timezone.utc).isoformat(),
        notes=(
            "Deterministic Newtonian tidal acceleration versus centre "
            "distance for a fixed object length. Exact near/far difference "
            "plus the 2 G M L / r^3 approximation and transverse "
            "compression."
        ),
    )

    return SimulationDataset(metadata=metadata, arrays=arrays)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate a deterministic radial tidal acceleration dataset."
    )
    parser.add_argument(
        "--mass", type=float, default=DEFAULT_MASS_KG,
        help="Attracting central mass in kg (default: Earth mass).",
    )
    parser.add_argument(
        "--length", type=float, default=DEFAULT_LENGTH_M,
        help="Fixed object length in metres (default: 10.0).",
    )
    parser.add_argument(
        "--r-min", type=float, default=DEFAULT_R_MIN_M,
        dest="r_min",
        help="Minimum centre distance in metres (default: 2.0e7).",
    )
    parser.add_argument(
        "--r-max", type=float, default=DEFAULT_R_MAX_M,
        dest="r_max",
        help="Maximum centre distance in metres (default: 1.0e8).",
    )
    parser.add_argument(
        "--points", type=int, default=DEFAULT_POINTS,
        help="Number of sampled distances (default: 200).",
    )
    parser.add_argument(
        "--output", required=True,
        help="Output path stem (produces .meta.json and .data.npz).",
    )
    args = parser.parse_args(argv)

    dataset = build_dataset(
        mass_kg=args.mass,
        length_m=args.length,
        r_min_m=args.r_min,
        r_max_m=args.r_max,
        num_points=args.points,
    )
    validate_dataset(dataset)
    meta_path, data_path = save_dataset(dataset, args.output)

    print(f"wrote {meta_path}")
    print(f"wrote {data_path}")
    print(
        f"points={args.points} mass={args.mass:.6e} kg length={args.length} m "
        f"r=[{args.r_min:.6e}, {args.r_max:.6e}] m"
    )
    print(
        f"tidal acceleration range: "
        f"{dataset.arrays['tidal_difference_exact'].min():.6e} to "
        f"{dataset.arrays['tidal_difference_exact'].max():.6e} m/s^2"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
