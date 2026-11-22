"""Camera geometry and normals used to construct the paper targets."""

from __future__ import annotations

import numpy as np


NORMAL_JUMP_M = 0.05


NORMAL_CROSS_EPS_M2 = 1e-8


DEPTH_SAMPLE_GL_XY = (0.375, 0.125)


NORMAL_PATCH_MAX_ANGLE_DEG = 30.0


def assert_transform(transform: np.ndarray) -> np.ndarray:
    t = np.asarray(transform, dtype=np.float64)
    if t.shape != (4, 4) or not np.isfinite(t).all():
        raise ValueError("Expected finite 4x4 rigid transform")
    if not np.allclose(t[3], [0, 0, 0, 1], atol=1e-6):
        raise ValueError("Invalid homogeneous row")
    r = t[:3, :3]
    if not np.allclose(r.T @ r, np.eye(3), atol=2e-5) or abs(np.linalg.det(r) - 1) > 2e-5:
        raise ValueError("Camera/base rotation is not a proper rotation")
    return t


def camera_rays(intrinsic: np.ndarray, stored_to_raw: np.ndarray,
                height: int, width: int, *,
                sample_gl_xy: tuple[float, float] = DEPTH_SAMPLE_GL_XY) -> np.ndarray:
    """Use the validated canonical depth sample, not the nominal pixel center."""
    k = np.asarray(intrinsic, dtype=np.float64)
    mapping = np.asarray(stored_to_raw, dtype=np.float64)
    expected = np.array([[-1, 0, width - 1], [0, -1, height - 1], [0, 0, 1]])
    if k.shape != (3, 3) or not np.isfinite(k).all():
        raise ValueError("Invalid intrinsic matrix")
    if not np.array_equal(mapping, expected):
        raise ValueError("Expected LIBERO images stored with rotate_180")
    if not np.allclose(k[2], [0, 0, 1]) or not np.allclose(k[:2, 2], [width / 2, height / 2]):
        raise ValueError("K must use boundary-origin principal point W/2,H/2")
    if k[0, 0] <= 0 or k[1, 1] <= 0 or abs(k[0, 1]) + abs(k[1, 0]) > 1e-7:
        raise ValueError("Unsupported focal length/skew")
    sample = np.asarray(sample_gl_xy, dtype=np.float64)
    if sample.shape != (2,) or not np.isfinite(sample).all() or not ((sample >= 0) & (sample < 1)).all():
        raise ValueError("Expected a finite OpenGL subpixel sample in [0,1)^2")
    vv, uu = np.indices((height, width), dtype=np.float64)
    stored = np.stack((uu, vv, np.ones_like(uu)), axis=-1)
    gl_to_cv = np.array([[1, 0, 0], [0, -1, height - 1], [0, 0, 1]])
    # GL y points up; after the index flip its within-pixel offset is 1-sample_y.
    sample_offset = np.array([[1, 0, sample[0]], [0, 1, 1-sample[1]], [0, 0, 1]])
    transform = np.linalg.inv(k) @ sample_offset @ gl_to_cv @ mapping
    rays = np.einsum("ij,hwj->hwi", transform, stored)
    if not np.allclose(rays[..., 2], 1):
        raise ValueError("Rays must have Z=1; depth is axial depth, not radial range")
    return rays.astype(np.float32)


def camera_points(depth_m: np.ndarray, rays: np.ndarray) -> np.ndarray:
    depth = np.asarray(depth_m, dtype=np.float32)
    if depth.ndim != 2 or rays.shape != (*depth.shape, 3):
        raise ValueError("Expected depth [H,W] and rays [H,W,3]")
    return depth[..., None] * rays


def world_points(points_camera: np.ndarray, world_from_camera: np.ndarray) -> np.ndarray:
    t = assert_transform(world_from_camera).astype(np.float32)
    return np.einsum("ij,hwj->hwi", t[:3, :3], points_camera) + t[:3, 3]


def base_points(points_camera: np.ndarray, world_from_camera: np.ndarray,
                world_from_base: np.ndarray) -> np.ndarray:
    base = assert_transform(world_from_base).astype(np.float32)
    world = world_points(points_camera, world_from_camera)
    return np.einsum("ji,hwj->hwi", base[:3, :3], world - base[:3, 3])


def surface_normal_target(depth_m: np.ndarray, rays: np.ndarray, *,
                          diagnostics: dict[str, int] | None = None
                          ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Camera-frame, camera-facing normals; no world/base rotation is applied."""
    depth = np.asarray(depth_m, dtype=np.float32)
    good_depth = np.isfinite(depth) & (depth > 0)
    points = camera_points(np.where(good_depth, depth, 0), rays)
    center = points[1:-1, 1:-1]
    right, left = points[1:-1, 2:] - center, center - points[1:-1, :-2]
    down, up = points[2:, 1:-1] - center, center - points[:-2, 1:-1]
    dx, dy = right + left, down + up
    cross = np.cross(dx, dy)
    magnitude = np.linalg.norm(cross, axis=-1)
    center_depth = depth[1:-1, 1:-1]
    neighbors = np.stack((depth[1:-1, :-2], depth[1:-1, 2:],
                          depth[:-2, 1:-1], depth[2:, 1:-1]), axis=-1)
    with np.errstate(invalid="ignore"):
        jump = np.max(np.abs(neighbors - center_depth[..., None]), axis=-1)
    finite_patch = (good_depth[1:-1, 1:-1]
                    & np.isfinite(neighbors).all(axis=-1) & (neighbors > 0).all(axis=-1))
    reliable_cross = np.isfinite(magnitude) & (magnitude > NORMAL_CROSS_EPS_M2)
    small_jump = jump < NORMAL_JUMP_M
    # One central difference can bridge two surfaces even for a small depth jump.
    # All four one-sided triangle normals must describe the same local surface.
    triangle = np.stack((np.cross(right, down), np.cross(left, down),
                         np.cross(left, up), np.cross(right, up)), axis=-2)
    triangle_norm = np.linalg.norm(triangle, axis=-1)
    triangle_ok = np.isfinite(triangle_norm).all(axis=-1) & (
        triangle_norm > NORMAL_CROSS_EPS_M2 / 4).all(axis=-1)
    triangle /= np.maximum(triangle_norm[..., None], NORMAL_CROSS_EPS_M2 / 4)
    min_cosine = np.ones(center_depth.shape, dtype=np.float32)
    for i in range(4):
        for j in range(i+1, 4):
            min_cosine = np.minimum(min_cosine, np.sum(triangle[..., i, :] * triangle[..., j, :], axis=-1))
    consistent = triangle_ok & (min_cosine >= np.cos(np.deg2rad(NORMAL_PATCH_MAX_ANGLE_DEG)))
    valid_inner = finite_patch & reliable_cross & small_jump & consistent
    if diagnostics is not None:
        diagnostics.update(
            normal_invalid_border_pixels=int(depth.size-center_depth.size),
            normal_invalid_depth_pixels=int((~finite_patch).sum()),
            normal_degenerate_cross_pixels=int((finite_patch & ~reliable_cross).sum()),
            normal_depth_jump_pixels=int((finite_patch & reliable_cross & ~small_jump).sum()),
            normal_inconsistent_patch_pixels=int((finite_patch & reliable_cross & small_jump & ~consistent).sum()),
        )
    normals = np.zeros((*depth.shape, 3), dtype=np.float32)
    inner = cross / np.maximum(magnitude[..., None], NORMAL_CROSS_EPS_M2)
    flip = np.sum(inner * -points[1:-1, 1:-1], axis=-1) < 0
    inner[flip] *= -1
    inner[~valid_inner] = 0
    normals[1:-1, 1:-1] = inner
    valid = np.zeros(depth.shape, dtype=bool)
    valid[1:-1, 1:-1] = valid_inner
    rgb = np.zeros((*depth.shape, 3), dtype=np.uint8)
    rgb[valid] = np.rint(255 * np.clip((normals[valid] + 1) / 2, 0, 1)).astype(np.uint8)
    return rgb, valid, normals
