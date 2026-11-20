"""Renderer-independent helpers for the Step 5C cinematic Blender preview.

The physics samples are never changed. This module only decides how much of an
already-calculated trajectory to display and how to frame it with a camera.
"""

from __future__ import annotations

import numpy as np


def trail_reveal_windows(
    validity: np.ndarray,
    *,
    tail_frames: int,
    active_frames: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return start/end curve factors and visibility for each presentation frame.

    Curve points correspond to active frames with valid particle positions.
    No part of a future trajectory is revealed. The trailing window retains
    at most ``tail_frames`` preceding line segments and freezes during holds.
    Terminated particles never reappear.
    """
    mask = np.asarray(validity)
    if mask.ndim != 1 or mask.dtype.kind != "b":
        raise ValueError("validity must be a one-dimensional boolean array")
    if mask.size == 0:
        raise ValueError("validity must have at least one frame")
    if isinstance(tail_frames, (bool, np.bool_)) or not isinstance(
        tail_frames, (int, np.integer)
    ) or tail_frames < 1:
        raise ValueError("tail_frames must be a positive integer")
    if isinstance(active_frames, (bool, np.bool_)) or not isinstance(
        active_frames, (int, np.integer)
    ) or not 1 <= active_frames <= mask.size:
        raise ValueError("active_frames must be between 1 and the total frame count")
    if np.any(~mask[:-1] & mask[1:]):
        raise ValueError("terminated particles must not become valid again")

    valid_active = int(np.count_nonzero(mask[:active_frames]))
    starts = np.zeros(mask.size, dtype=np.float64)
    ends = np.zeros(mask.size, dtype=np.float64)
    visible = np.zeros(mask.size, dtype=bool)
    if valid_active < 2:
        return starts, ends, visible

    last_index = valid_active - 1
    for frame in range(mask.size):
        if not mask[frame]:
            continue
        index = min(frame, last_index)
        if index == 0:
            continue
        starts[frame] = max(0, index - tail_frames) / last_index
        ends[frame] = index / last_index
        visible[frame] = True

    return starts, ends, visible


def camera_plan(
    scaled_positions_xyz: np.ndarray,
    validity_mask: np.ndarray,
    *,
    min_half_extent: float = 2.5,
) -> tuple[float, float, float]:
    """Return (centre_x, centre_y, half_extent) of valid XY positions and origin.

    The origin matters because the cinematic black hole sits at (0, 0).
    ``min_half_extent`` keeps the schematic accretion rings in the camera view.
    """
    positions = np.asarray(scaled_positions_xyz, dtype=np.float64)
    valid = np.asarray(validity_mask)
    if positions.ndim != 3 or positions.shape[-1] != 3:
        raise ValueError("scaled_positions_xyz must have shape (T, N, 3)")
    if valid.shape != positions.shape[:2] or valid.dtype.kind != "b":
        raise ValueError("validity_mask must be boolean with shape (T, N)")
    if not np.isfinite(min_half_extent) or min_half_extent <= 0:
        raise ValueError("min_half_extent must be positive and finite")

    active = positions[valid]
    if active.size == 0:
        raise ValueError("no valid sampled positions available for camera")
    if not np.all(np.isfinite(active)):
        raise ValueError("valid positions must be finite")

    min_x = min(0.0, float(np.min(active[:, 0])))
    max_x = max(0.0, float(np.max(active[:, 0])))
    min_y = min(0.0, float(np.min(active[:, 1])))
    max_y = max(0.0, float(np.max(active[:, 1])))
    center_x = (min_x + max_x) / 2.0
    center_y = (min_y + max_y) / 2.0
    half_extent = max(
        (max_x - min_x) / 2.0, (max_y - min_y) / 2.0,
        float(min_half_extent),
    )
    return center_x, center_y, half_extent


def select_still_frame(
    scaled_positions_xyz: np.ndarray,
    validity_mask: np.ndarray,
    *,
    active_frames: int,
    core_radius: float = 0.62,
    particle_radius: float = 0.425,
) -> tuple[int, int]:
    """Choose a 1-based still frame and count particles outside the core.

    The selection only changes *which existing simulated frame* is shown.
    It does not adjust positions, validity or simulation timing.

    Frames with the most visible particle centres outside the schematic core
    are preferred. Among ties, choose the one closest to 30% into the active
    sequence, giving trails some time to develop. If all particle centres are
    too close to the core, choose among frames with the most valid particles
    and return a zero visible count; callers can warn about the scale.
    """
    positions = np.asarray(scaled_positions_xyz, dtype=np.float64)
    mask = np.asarray(validity_mask)
    if positions.ndim != 3 or positions.shape[-1] != 3:
        raise ValueError("scaled_positions_xyz must have shape (T, N, 3)")
    if mask.shape != positions.shape[:2] or mask.dtype.kind != "b":
        raise ValueError("validity_mask must be boolean with shape (T, N)")
    if isinstance(active_frames, (bool, np.bool_)) or not isinstance(
        active_frames, (int, np.integer)
    ) or not 1 <= active_frames <= positions.shape[0]:
        raise ValueError("active_frames must be between 1 and the total frame count")
    if any(not np.isfinite(x) or x < 0 for x in (core_radius, particle_radius)):
        raise ValueError("core_radius and particle_radius must be finite and nonnegative")

    valid_pos = positions[mask]
    if valid_pos.size == 0:
        raise ValueError("no valid particle positions available for still preview")
    if not np.all(np.isfinite(valid_pos)):
        raise ValueError("valid particle positions must be finite")

    # Ignore invalid NaN positions so terminated particles never score.
    positions_active = np.where(
        mask[:active_frames, :, None], positions[:active_frames], 0.0
    )
    distance = np.linalg.norm(positions_active[:, :, :2], axis=2)
    outside = mask[:active_frames] & (distance > core_radius + particle_radius)
    visible_count = np.count_nonzero(outside, axis=1)
    max_visible = int(np.max(visible_count))
    scores = visible_count
    if max_visible == 0:
        # The caller can warn that visual scale is hiding every particle.
        scores = np.count_nonzero(mask[:active_frames], axis=1)
    if np.max(scores) == 0:
        raise ValueError("no active frame contains a valid particle")

    preferred_index = round((active_frames - 1) * 0.30)
    candidates = np.flatnonzero(scores == np.max(scores))
    winner = int(min(candidates, key=lambda i: (abs(int(i) - preferred_index), i)))
    return winner + 1, max_visible
