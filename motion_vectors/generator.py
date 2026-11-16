"""Per-pixel screen-space motion vector generation.

The generator reprojects the surface point seen at every current-frame pixel
back into the previous frame and returns the displacement in pixels together
with a per-pixel validity mask.

Coordinate conventions
----------------------
* NDC is OpenGL style: ``x, y in [-1, 1]`` with y pointing up and
  ``z in [-1, 1]`` where ``-1`` is the near plane and ``+1`` is the far plane.
* A clip-space point ``(cx, cy, cz, cw)`` maps to NDC by dividing by ``cw``.
* Pixel coordinates use the upper-left pixel as origin. Pixel ``(col, row)``
  has its center at ``x = (2*col + 1) / width - 1`` and
  ``y = 1 - (2*row + 1) / height``.
* Motion vectors are expressed in pixels and point from the current sample
  position to the previous-frame sample position, i.e. the direction a
  temporal / motion-blur pass walks when fetching history. Geometry is
  reprojected at unjittered pixel centers, and the sub-pixel jitter delta
  ``jitter_prev - jitter_curr`` is added explicitly.
* Sub-pixel jitter is supplied in pixels, measured around each pixel center
  with y positive downward.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

# ---------------------------------------------------------------------------
# Validity mask flags.
# ---------------------------------------------------------------------------
INVALID_NONE = 0
"""Pixel is fully valid."""

INVALID_EMPTY = 1 << 0
"""Depth buffer contains no surface (NaN, infinity or the empty sentinel)."""

INVALID_BEHIND_CAMERA = 1 << 1
"""Reprojected point lands behind the previous camera or is clipped by its
near/far planes, so no previous-frame screen position exists."""

INVALID_DEPTH_BOUNDARY = 1 << 2
"""Linear eye-space depth jumps against a 4-connected neighbour, i.e. the
pixel straddles a foreground / background depth edge."""

INVALID_JITTER = 1 << 3
"""The vector carries a non-zero sub-pixel jitter delta between frames."""

INVALID_OUT_OF_SCREEN = 1 << 4
"""The previous-frame sample falls outside the viewport."""

INVALID_SKY = 1 << 5
"""Depth sits on the far plane; the sky has no reconstructable position and
is forced to a zero-length vector instead of an unbounded one."""

_EPS = 1e-6
_MIN_W = 1e-6


@dataclass(frozen=True)
class DynamicObject:
    """A rigid dynamic instance.

    ``model_prev`` / ``model_curr`` are object-to-world 4x4 transforms for the
    previous and current frames. Pixels whose ``object_id`` matches
    ``object_id`` are first moved into current object space, then re-expressed
    in the previous frame with ``model_prev`` before camera reprojection.
    """

    object_id: int
    model_prev: np.ndarray
    model_curr: np.ndarray


@dataclass(frozen=True)
class MotionVectors:
    """Result of :func:`generate_motion_vectors`."""

    motion: np.ndarray
    """``(height, width, 2)`` float64 displacement in pixels, ``(dx, dy)``
    with ``dy`` positive downward. Always finite; invalid pixels are zero."""

    mask: np.ndarray
    """``(height, width)`` uint8 bitfield of the ``INVALID_*`` flags."""

    linear_depth: np.ndarray
    """``(height, width)`` float64 eye-space depth (camera ``w``). Zero for
    pixels without a reconstructable surface."""


def _as_matrix4(name: str, value: Sequence[float]) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4):
        raise ValueError(f"{name} must have shape (4, 4), got {matrix.shape}")
    return matrix


def _pixel_to_ndc(width: int, height: int,
                  jitter_x: float, jitter_y: float) -> tuple[np.ndarray, np.ndarray]:
    """Return the jittered NDC sample position of every pixel."""
    cols = np.arange(width, dtype=np.float64)
    rows = np.arange(height, dtype=np.float64)
    grid_x, grid_y = np.meshgrid(cols, rows)
    ndc_x = (2.0 * (grid_x + 0.5 + jitter_x) / width) - 1.0
    # Pixel rows grow downward while NDC y grows upward.
    ndc_y = 1.0 - (2.0 * (grid_y + 0.5 + jitter_y) / height)
    return ndc_x, ndc_y


def _depth_boundary_mask(linear_depth: np.ndarray,
                         surface_mask: np.ndarray,
                         threshold: float) -> np.ndarray:
    """Flag depth discontinuities against the four connected neighbours."""
    boundary = np.zeros(linear_depth.shape, dtype=bool)
    if not surface_mask.any():
        return boundary

    depth = np.where(surface_mask, linear_depth, 0.0)
    denominator = np.maximum(depth, _EPS)
    shifts = ((0, 1), (0, -1), (1, 0), (-1, 0))
    for shift_y, shift_x in shifts:
        neighbour = np.roll(depth, shift=(shift_y, shift_x), axis=(0, 1))
        neighbour_valid = np.roll(surface_mask,
                                  shift=(shift_y, shift_x), axis=(0, 1))
        # A missing neighbour (empty / sky) is itself a discontinuity.
        gap = np.abs(depth - neighbour) / denominator
        boundary |= surface_mask & (~neighbour_valid | (gap > threshold))
    return boundary


def generate_motion_vectors(
    depth: np.ndarray,
    view_proj: np.ndarray,
    view_proj_prev: np.ndarray,
    resolution: tuple[int, int],
    jitter_curr: tuple[float, float] = (0.0, 0.0),
    jitter_prev: tuple[float, float] = (0.0, 0.0),
    dynamic_objects: Iterable[DynamicObject] = (),
    object_ids: np.ndarray | None = None,
    empty_value: float = np.nan,
    depth_boundary_threshold: float = 0.1,
) -> MotionVectors:
    """Generate per-pixel screen motion vectors.

    Parameters
    ----------
    depth:
        ``(height, width)`` current-frame depth in NDC z (OpenGL convention,
        ``-1`` near, ``+1`` far). NaN, +/-inf and values equal to
        ``empty_value`` mark empty pixels; values at the far plane mark sky.
    view_proj, view_proj_prev:
        Current and previous frame view-projection matrices (4x4).
    resolution:
        ``(width, height)`` of the framebuffer; must match ``depth``.
    jitter_curr, jitter_prev:
        Sub-pixel sample offsets in pixels, ``(x, y)`` with y downward.
    dynamic_objects:
        Rigid instances carrying per-frame object-to-world transforms.
    object_ids:
        ``(height, width)`` integer instance id per pixel, ``-1`` for static
        geometry. Required when ``dynamic_objects`` is non-empty.
    empty_value:
        Sentinel used in ``depth`` for pixels without a surface.
    depth_boundary_threshold:
        Relative eye-space depth gap that counts as a depth boundary.
    """

    width, height = resolution
    depth = np.asarray(depth, dtype=np.float64)
    if depth.shape != (height, width):
        raise ValueError(
            f"depth shape {depth.shape} does not match resolution "
            f"({width}, {height})"
        )
    if width <= 0 or height <= 0:
        raise ValueError("resolution components must be positive")

    vp = _as_matrix4("view_proj", view_proj)
    vp_prev = _as_matrix4("view_proj_prev", view_proj_prev)
    inv_vp = np.linalg.inv(vp)

    jitter_x_curr, jitter_y_curr = float(jitter_curr[0]), float(jitter_curr[1])
    jitter_x_prev, jitter_y_prev = float(jitter_prev[0]), float(jitter_prev[1])

    objects = {item.object_id: item for item in dynamic_objects}
    if object_ids is not None:
        object_ids = np.asarray(object_ids)
        if object_ids.shape != (height, width):
            raise ValueError(
                f"object_ids shape {object_ids.shape} does not match resolution"
            )
        unknown = set(np.unique(object_ids).tolist()) - set(objects) - {-1}
        if unknown:
            raise ValueError(
                f"missing DynamicObject transforms for ids {sorted(unknown)}"
            )
    elif objects:
        raise ValueError("object_ids is required when dynamic_objects are provided")

    # ------------------------------------------------------------------
    # Classify pixels.
    # ------------------------------------------------------------------
    with np.errstate(invalid="ignore"):
        empty_mask = (~np.isfinite(depth)) | (depth == empty_value)
    sky_mask = (~empty_mask) & (depth >= 1.0 - _EPS)
    surface_mask = (~empty_mask) & (~sky_mask)

    mask = np.zeros((height, width), dtype=np.uint8)
    mask[empty_mask] |= INVALID_EMPTY
    mask[sky_mask] |= INVALID_SKY

    jitter_delta_x = jitter_x_curr - jitter_x_prev
    jitter_delta_y = jitter_y_curr - jitter_y_prev
    if jitter_delta_x != 0.0 or jitter_delta_y != 0.0:
        mask[surface_mask] |= INVALID_JITTER

    motion = np.zeros((height, width, 2), dtype=np.float64)
    linear_depth = np.zeros((height, width), dtype=np.float64)

    if not surface_mask.any():
        return MotionVectors(motion=motion, mask=mask, linear_depth=linear_depth)

    # ------------------------------------------------------------------
    # Reconstruct current-frame world positions from NDC sample points.
    # An NDC point (x, y, z, 1) unprojects to a homogeneous world position
    # whose w normalisation yields the actual surface point.
    # ------------------------------------------------------------------
    # Geometry is reconstructed at unjittered pixel centers; jitter is folded
    # into the vector afterwards so its contribution stays explicit.
    ndc_x, ndc_y = _pixel_to_ndc(width, height, 0.0, 0.0)
    ndc = np.stack([ndc_x, ndc_y, depth, np.ones_like(depth)], axis=-1)
    ndc = np.nan_to_num(ndc, nan=0.0, posinf=0.0, neginf=0.0)
    world_h = ndc @ inv_vp.T
    world_w = np.where(np.abs(world_h[..., 3:4]) > _EPS, world_h[..., 3:4], 1.0)
    world = world_h[..., :3] / world_w

    # Current camera w = eye-space depth, used for the boundary test.
    clip_cur_h = ndc @ vp.T
    eye_depth = clip_cur_h[..., 3] / world_h[..., 3]
    eye_depth = np.where(surface_mask & np.isfinite(eye_depth) & (eye_depth > 0.0),
                         eye_depth, 0.0)
    linear_depth[surface_mask] = eye_depth[surface_mask]

    # ------------------------------------------------------------------
    # Move dynamic pixels from current object space to previous object space.
    # ------------------------------------------------------------------
    world_prev = world.copy()
    if objects:
        for obj_id, item in objects.items():
            member = object_ids == obj_id
            if not member.any():
                continue
            model_cur = _as_matrix4("model_curr", item.model_curr)
            model_pre = _as_matrix4("model_prev", item.model_prev)
            local = np.linalg.inv(model_cur)
            points = np.concatenate(
                [world[member], np.ones((member.sum(), 1), dtype=np.float64)],
                axis=1,
            )
            reexpressed = (model_pre @ (local @ points.T)).T
            world_prev[member] = reexpressed[:, :3] / reexpressed[:, 3:4]

    # ------------------------------------------------------------------
    # Project into the previous frame.
    # ------------------------------------------------------------------
    points_prev = np.concatenate(
        [world_prev, np.ones((height, width, 1), dtype=np.float64)], axis=-1
    )
    clip_prev = points_prev @ vp_prev.T
    w_prev = clip_prev[..., 3]

    behind = surface_mask & (w_prev <= _MIN_W)
    safe_w = np.where(behind, 1.0, w_prev)
    prev_ndc_x = clip_prev[..., 0] / safe_w
    prev_ndc_y = clip_prev[..., 1] / safe_w
    prev_ndc_z = clip_prev[..., 2] / safe_w

    # Near/far plane clipping still rejects points with valid w.
    clipped = surface_mask & ~behind & (
        (prev_ndc_z < -1.0 - _EPS) | (prev_ndc_z > 1.0 + _EPS)
    )
    behind |= clipped
    mask[behind] |= INVALID_BEHIND_CAMERA

    # Previous sample in pixel-grid coordinates (origin at upper-left centre).
    prev_grid_x = (prev_ndc_x + 1.0) * 0.5 * width
    prev_grid_y = (1.0 - prev_ndc_y) * 0.5 * height

    curr_grid_x = np.broadcast_to(
        np.arange(width, dtype=np.float64) + 0.5,
        (height, width),
    )
    curr_grid_y = np.broadcast_to(
        np.arange(height, dtype=np.float64)[:, None] + 0.5,
        (height, width),
    )

    delta_x = prev_grid_x - curr_grid_x
    delta_y = prev_grid_y - curr_grid_y

    # Jitter contribution: the current sample is offset by jitter_curr and
    # the historical one by jitter_prev, so their difference shifts the vector.
    delta_x += jitter_x_prev - jitter_x_curr
    delta_y += jitter_y_prev - jitter_y_curr

    out_of_screen = surface_mask & ~behind & (
        (prev_grid_x < 0.0) | (prev_grid_x > width)
        | (prev_grid_y < 0.0) | (prev_grid_y > height)
    )
    mask[out_of_screen] |= INVALID_OUT_OF_SCREEN

    valid_motion = surface_mask & ~behind
    motion[valid_motion, 0] = delta_x[valid_motion]
    motion[valid_motion, 1] = delta_y[valid_motion]

    # ------------------------------------------------------------------
    # Depth discontinuities.
    # ------------------------------------------------------------------
    boundary = _depth_boundary_mask(linear_depth, surface_mask,
                                    depth_boundary_threshold)
    mask[boundary] |= INVALID_DEPTH_BOUNDARY

    # Defensive guarantee: callers always receive finite numbers.
    finite = np.isfinite(motion).all(axis=-1)
    motion[~finite] = 0.0

    return MotionVectors(motion=motion, mask=mask, linear_depth=linear_depth)
