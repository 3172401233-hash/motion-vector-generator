import unittest

import numpy as np

from motion_vectors import (
    INVALID_BEHIND_CAMERA,
    INVALID_DEPTH_BOUNDARY,
    INVALID_EMPTY,
    INVALID_JITTER,
    INVALID_NONE,
    INVALID_OUT_OF_SCREEN,
    INVALID_SKY,
    DynamicObject,
    generate_motion_vectors,
)
from motion_vectors.transforms import (
    render_plane_depth,
    rotation_y,
    translation,
    view_projection,
)

WIDTH = 64
HEIGHT = 48
RESOLUTION = (WIDTH, HEIGHT)
ASPECT = WIDTH / HEIGHT
FOV_Y = np.deg2rad(60.0)
NEAR = 0.1
FAR = 100.0
PLANE_Z = -10.0


def camera_vp(eye, target=(0.0, 0.0, -10.0)):
    return view_projection(
        eye=np.asarray(eye, dtype=np.float64),
        target=np.asarray(target, dtype=np.float64),
        fov_y_rad=FOV_Y,
        aspect=ASPECT,
        near=NEAR,
        far=FAR,
    )


def uniform_plane_depth(vp):
    return render_plane_depth(PLANE_Z, vp, RESOLUTION)


class MotionVectorTests(unittest.TestCase):
    def test_static_scene_has_zero_motion(self):
        vp = camera_vp((0.0, 0.0, 0.0))
        depth = uniform_plane_depth(vp)
        result = generate_motion_vectors(depth, vp, vp, RESOLUTION)

        self.assertTrue(np.all(np.isfinite(result.motion)))
        np.testing.assert_allclose(result.motion, 0.0, atol=1e-9)
        np.testing.assert_array_equal(result.mask, INVALID_NONE)

    def test_camera_translation_reprojects_in_correct_direction(self):
        vp_curr = camera_vp((0.0, 0.0, 0.0))
        # Camera moves right; the previously sampled content sits to the left
        # of the current sample, so history lookup walks left (negative dx).
        vp_prev = camera_vp((-0.5, 0.0, 0.0))
        depth = uniform_plane_depth(vp_curr)

        result = generate_motion_vectors(depth, vp_curr, vp_prev, RESOLUTION)
        center = result.motion[HEIGHT // 2, WIDTH // 2, 0]

        self.assertTrue(np.all(np.isfinite(result.motion)))
        self.assertLess(center, 0.0)
        self.assertEqual(
            int(result.mask[HEIGHT // 2, WIDTH // 2]) & INVALID_DEPTH_BOUNDARY,
            0,
        )

    def test_camera_rotation_marks_points_behind_previous_camera(self):
        # The previous camera stands between the surface and the current
        # camera but looks the opposite way, so every surface point lands on
        # its backward hemisphere and produces a negative w_prev.
        vp_curr = camera_vp((0.0, 0.0, 0.0))
        vp_prev = view_projection(
            eye=np.array((-5.0, 0.0, 0.0)),
            target=np.array((-5.0, 0.0, 10.0)),
            fov_y_rad=FOV_Y,
            aspect=ASPECT,
            near=NEAR,
            far=FAR,
        )
        depth = render_plane_depth(PLANE_Z, vp_curr, RESOLUTION)

        result = generate_motion_vectors(depth, vp_curr, vp_prev, RESOLUTION)

        self.assertTrue(np.all(np.isfinite(result.motion)))
        self.assertTrue(np.all(result.mask & INVALID_BEHIND_CAMERA))
        flagged = result.mask & INVALID_BEHIND_CAMERA != 0
        np.testing.assert_array_equal(result.motion[flagged], 0.0)

    def test_near_plane_clipping_is_invalid_and_finite(self):
        # The plane is visible to both cameras but sits inside the previous
        # camera's near distance, so it projects past NDC z = -1 while its
        # camera-space w stays positive.
        close_vp = view_projection(
            eye=np.zeros(3),
            target=np.array((0.0, 0.0, -1.0)),
            fov_y_rad=FOV_Y,
            aspect=ASPECT,
            near=0.1,
            far=FAR,
        )
        far_vp = view_projection(
            eye=np.zeros(3),
            target=np.array((0.0, 0.0, -1.0)),
            fov_y_rad=FOV_Y,
            aspect=ASPECT,
            near=1.0,
            far=FAR,
        )
        depth = render_plane_depth(-0.5, close_vp, RESOLUTION)

        result = generate_motion_vectors(depth, close_vp, far_vp, RESOLUTION)

        self.assertTrue(np.all(np.isfinite(result.motion)))
        self.assertTrue(np.any(result.mask & INVALID_BEHIND_CAMERA))

    def test_sky_depth_produces_finite_zero_vectors(self):
        vp = camera_vp((0.0, 0.0, 0.0))
        depth = np.full((HEIGHT, WIDTH), 1.0, dtype=np.float64)

        result = generate_motion_vectors(depth, vp, vp, RESOLUTION)

        self.assertTrue(np.all(np.isfinite(result.motion)))
        np.testing.assert_array_equal(result.motion, 0.0)
        self.assertTrue(np.all(result.mask & INVALID_SKY))

    def test_empty_depth_is_flagged(self):
        vp = camera_vp((0.0, 0.0, 0.0))
        depth = uniform_plane_depth(vp)
        depth[10, 10] = np.nan
        depth[20, 20] = np.inf

        result = generate_motion_vectors(depth, vp, vp, RESOLUTION)

        self.assertTrue(np.all(np.isfinite(result.motion)))
        self.assertTrue(result.mask[10, 10] & INVALID_EMPTY)
        self.assertTrue(result.mask[20, 20] & INVALID_EMPTY)
        np.testing.assert_array_equal(result.motion[10, 10], 0.0)

    def test_subpixel_jitter_is_marked_and_baked_into_motion(self):
        vp = camera_vp((0.0, 0.0, 0.0))
        depth = uniform_plane_depth(vp)

        result = generate_motion_vectors(
            depth,
            vp,
            vp,
            RESOLUTION,
            jitter_curr=(0.25, -0.15),
            jitter_prev=(0.0, 0.0),
        )

        self.assertTrue(np.all(np.isfinite(result.motion)))
        np.testing.assert_allclose(result.motion[..., 0], -0.25, atol=1e-9)
        # Positive current y jitter moves the sample downward, so the previous
        # sample is above it and dy is positive.
        np.testing.assert_allclose(result.motion[..., 1], 0.15, atol=1e-9)
        self.assertTrue(np.all(result.mask & INVALID_JITTER))

    def test_depth_boundary_is_marked(self):
        vp = camera_vp((0.0, 0.0, 0.0))
        depth = uniform_plane_depth(vp)
        # Punch a hole: neighbouring surface pixels now border empty space.
        depth[20:28, 30:38] = np.nan

        result = generate_motion_vectors(depth, vp, vp, RESOLUTION)

        boundary = result.mask & INVALID_DEPTH_BOUNDARY != 0
        self.assertTrue(boundary[19, 34])
        self.assertTrue(boundary[28, 34])
        self.assertTrue(boundary[24, 29])
        self.assertTrue(boundary[24, 38])
        # Interior pixels far from the hole remain clean.
        self.assertFalse(boundary[5, 5])

    def test_out_of_screen_samples_are_flagged(self):
        vp_curr = camera_vp((0.0, 0.0, 0.0))
        vp_prev = camera_vp((-8.0, 0.0, 0.0))
        depth = uniform_plane_depth(vp_curr)

        result = generate_motion_vectors(depth, vp_curr, vp_prev, RESOLUTION)

        self.assertTrue(np.all(np.isfinite(result.motion)))
        self.assertTrue(np.any(result.mask & INVALID_OUT_OF_SCREEN))

    def test_dynamic_character_moves_relative_to_camera(self):
        # Camera and character translate together. A static-only reprojection
        # would report zero motion; the dynamic transform must instead recover
        # the character's relative movement on screen.
        vp_curr = camera_vp((0.0, 0.0, 0.0))
        vp_prev = camera_vp((-1.0, 0.0, 0.0))

        depth = uniform_plane_depth(vp_curr)
        object_ids = np.full((HEIGHT, WIDTH), -1, dtype=np.int64)
        object_ids[18:30, 26:38] = 7

        model_curr = translation(0.5, 0.0, 0.0)
        model_prev = translation(0.0, 0.0, 0.0)
        character = DynamicObject(
            object_id=7, model_prev=model_prev, model_curr=model_curr
        )

        # Static reference: same camera motion, no dynamic transforms.
        static_result = generate_motion_vectors(depth, vp_curr, vp_prev,
                                                RESOLUTION)
        dynamic_result = generate_motion_vectors(
            depth,
            vp_curr,
            vp_prev,
            RESOLUTION,
            dynamic_objects=[character],
            object_ids=object_ids,
        )

        row, col = 24, 32
        self.assertTrue(np.all(np.isfinite(dynamic_result.motion)))
        self.assertNotAlmostEqual(
            float(dynamic_result.motion[row, col, 0]),
            float(static_result.motion[row, col, 0]),
            places=5,
        )
        # Character moves right relative to the camera, so its history is to
        # the left of the current sample.
        self.assertLess(dynamic_result.motion[row, col, 0], 0.0)

    def test_rotation_transform_helper_is_orthonormal(self):
        matrix = rotation_y(0.7)
        rotation = matrix[:3, :3]
        np.testing.assert_allclose(rotation @ rotation.T, np.eye(3), atol=1e-12)


if __name__ == "__main__":
    unittest.main()
