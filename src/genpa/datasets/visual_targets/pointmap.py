"""Encode robot-base XYZ coordinates as a three-channel image."""

from __future__ import annotations

import numpy as np

from .geometry import base_points, camera_points

LOW = np.array([-0.25, -0.45, -0.10], dtype=np.float64)
HIGH = np.array([1.00, 0.50, 0.50], dtype=np.float64)
CODE_LOW, CODE_HIGH = 16.0, 240.0


def encoding_spec():
    return dict(kind="smooth", low=LOW.tolist(), high=HIGH.tolist(),
                code_low=16, code_high=240, tail="rational")


def encode_points(points, *, quantize=True):
    xyz = np.asarray(points, dtype=np.float64)
    if xyz.shape[-1] != 3 or not np.isfinite(xyz).all():
        raise ValueError("Expected finite XYZ coordinates")
    slope = (CODE_HIGH - CODE_LOW) / (HIGH - LOW)
    below = np.maximum(LOW - xyz, 0)
    above = np.maximum(xyz - HIGH, 0)
    left = 1 + 15 / (1 + slope * below / 15)
    right = 255 - 15 / (1 + slope * above / 15)
    code = np.where(xyz < LOW, left,
                    np.where(xyz > HIGH, right, CODE_LOW + slope * (xyz - LOW)))
    return np.rint(np.clip(code, 1, 255)).astype(np.uint8) if quantize else code


def decode_points(code):
    """Diagnostic inverse; endpoint bins use finite half-bin representatives."""
    code = np.asarray(code, dtype=np.float64)
    if code.shape[-1] != 3 or not np.isfinite(code).all():
        raise ValueError("Expected finite RGB codes")
    slope = (CODE_HIGH - CODE_LOW) / (HIGH - LOW)
    capped = np.clip(code, 1.5, 254.5)
    left = LOW - 15 / slope * (15 / (capped - 1) - 1)
    right = HIGH + 15 / slope * (15 / (255 - capped) - 1)
    return np.where(code < CODE_LOW, left,
                    np.where(code > CODE_HIGH, right, LOW + (code - CODE_LOW) / slope))


def point_target(points, depth, geom):
    depth, geom = np.asarray(depth), np.asarray(geom)
    points = np.asarray(points)
    if points.shape != (*depth.shape, 3) or geom.shape != depth.shape:
        raise ValueError("Point/depth/geom shapes differ")
    valid = (np.isfinite(depth) & (depth > 0) & (geom > 0)
             & np.isfinite(points).all(axis=-1))
    rgb = np.zeros(points.shape, dtype=np.uint8)
    rgb[valid] = encode_points(points[valid])
    return rgb, valid


def point_map_target(depth, rays, camera_pose, base_pose, geom):
    with np.errstate(invalid="ignore", over="ignore"):
        points = base_points(camera_points(depth, rays), camera_pose, base_pose)
    rgb, valid = point_target(points, depth, geom)
    return rgb, valid, points
