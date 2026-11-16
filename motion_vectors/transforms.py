"""Small 4x4 transform helpers used by the library and its tests.

Matrices are row-vector friendly only in the sense that the generator
multiplies row vectors as ``p @ M.T``; the matrices themselves follow the
standard column-vector conventions of OpenGL.
"""

from __future__ import annotations

import numpy as np


def identity() -> np.ndarray:
    return np.eye(4, dtype=np.float64)


def translation(x: float, y: float, z: float) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    matrix[0, 3] = x
    matrix[1, 3] = y
    matrix[2, 3] = z
    return matrix


def rotation_y(angle_rad: float) -> np.ndarray:
    """Right-handed rotation about the +y (up) axis."""
    cosine = np.cos(angle_rad)
    sine = np.sin(angle_rad)
    matrix = np.eye(4, dtype=np.float64)
    matrix[0, 0] = cosine
    matrix[0, 2] = sine
    matrix[2, 0] = -sine
    matrix[2, 2] = cosine
    return matrix


def perspective(fov_y_rad: float, aspect: float,
                near: float, far: float) -> np.ndarray:
    """OpenGL-style perspective projection (NDC z in ``[-1, 1]``)."""
    focal = 1.0 / np.tan(fov_y_rad * 0.5)
    matrix = np.zeros((4, 4), dtype=np.float64)
    matrix[0, 0] = focal / aspect
    matrix[1, 1] = focal
    # OpenGL NDC z: near plane -> -1, far plane -> +1.
    matrix[2, 2] = (far + near) / (near - far)
    matrix[2, 3] = (2.0 * far * near) / (near - far)
    matrix[3, 2] = -1.0
    return matrix


def look_at(eye: np.ndarray, target: np.ndarray,
            up: np.ndarray = np.array((0.0, 1.0, 0.0))) -> np.ndarray:
    """View matrix for a camera positioned at ``eye`` looking at ``target``."""
    eye = np.asarray(eye, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    up = np.asarray(up, dtype=np.float64)

    forward = target - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    true_up = np.cross(right, forward)

    matrix = np.eye(4, dtype=np.float64)
    matrix[0, :3] = right
    matrix[1, :3] = true_up
    matrix[2, :3] = -forward
    matrix[0, 3] = -np.dot(right, eye)
    matrix[1, 3] = -np.dot(true_up, eye)
    matrix[2, 3] = np.dot(forward, eye)
    return matrix


def view_projection(eye: np.ndarray, target: np.ndarray,
                    fov_y_rad: float, aspect: float,
                    near: float, far: float) -> np.ndarray:
    return perspective(fov_y_rad, aspect, near, far) @ look_at(eye, target)


def render_plane_depth(plane_z: float,
                       view_proj: np.ndarray,
                       resolution: tuple[int, int]) -> np.ndarray:
    """Analytically render a fronto-parallel world plane ``z = plane_z``.

    Every pixel ray is intersected with the plane, projected with
    ``view_proj`` and the projected NDC z is stored. Rays whose intersection
    lies behind the camera receive NaN, matching an empty depth buffer.
    """
    width, height = resolution
    inv_vp = np.linalg.inv(view_proj)

    cols = np.arange(width, dtype=np.float64)
    rows = np.arange(height, dtype=np.float64)
    grid_x, grid_y = np.meshgrid(cols, rows)
    ndc_x = (2.0 * (grid_x + 0.5) / width) - 1.0
    ndc_y = 1.0 - (2.0 * (grid_y + 0.5) / height)

    far_points = np.stack(
        [ndc_x, ndc_y, np.ones_like(ndc_x), np.ones_like(ndc_x)], axis=-1
    )
    near_points = np.stack(
        [ndc_x, ndc_y, -np.ones_like(ndc_x), np.ones_like(ndc_x)], axis=-1
    )
    world_far = far_points @ inv_vp.T
    world_near = near_points @ inv_vp.T
    world_far = world_far[..., :3] / world_far[..., 3:4]
    world_near = world_near[..., :3] / world_near[..., 3:4]

    direction = world_far - world_near
    with np.errstate(divide="ignore", invalid="ignore"):
        travel = (plane_z - world_near[..., 2]) / direction[..., 2]
    surface = world_near + direction * travel[..., None]
    behind = travel < 0.0

    homogeneous = np.concatenate(
        [surface, np.ones((*surface.shape[:2], 1), dtype=np.float64)], axis=-1
    )
    clip = homogeneous @ view_proj.T
    depth = clip[..., 2] / clip[..., 3]
    depth[behind] = np.nan
    return depth
