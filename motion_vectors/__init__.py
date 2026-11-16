"""Screen-space motion vector generation for game rendering."""

from .generator import (
    INVALID_BEHIND_CAMERA,
    INVALID_DEPTH_BOUNDARY,
    INVALID_EMPTY,
    INVALID_JITTER,
    INVALID_NONE,
    INVALID_OUT_OF_SCREEN,
    INVALID_SKY,
    DynamicObject,
    MotionVectors,
    generate_motion_vectors,
)

__all__ = [
    "INVALID_BEHIND_CAMERA",
    "INVALID_DEPTH_BOUNDARY",
    "INVALID_EMPTY",
    "INVALID_JITTER",
    "INVALID_NONE",
    "INVALID_OUT_OF_SCREEN",
    "INVALID_SKY",
    "DynamicObject",
    "MotionVectors",
    "generate_motion_vectors",
]
