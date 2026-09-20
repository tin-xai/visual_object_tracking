"""Coordinate geometry for Section III-A/B of Ngoc, Tin and Tuan (2020).

Column convention: X_world = R_wc @ X_camera + camera_position_world.
Arrays of points are stored as rows. Lengths use the baseline's units (meters).
"""
from dataclasses import dataclass
import numpy as np


class LocalizationError(RuntimeError):
    """Insufficient or degenerate observations; no valid pose should be emitted."""


def rotation_matrix(value):
    r = np.asarray(value, dtype=float)
    if (r.shape != (3, 3) or not np.isfinite(r).all()
            or not np.allclose(r.T @ r, np.eye(3), atol=1e-5)
            or not np.isclose(np.linalg.det(r), 1, atol=1e-5)):
        raise ValueError("Expected a finite, proper 3x3 rotation matrix")
    return r


@dataclass(frozen=True)
class Camera:
    fx: float
    fy: float
    cx: float
    cy: float
    baseline: float

    def __post_init__(self):
        values = np.array([self.fx, self.fy, self.cx, self.cy, self.baseline])
        if not np.isfinite(values).all() or min(self.fx, self.fy, self.baseline) <= 0:
            raise ValueError("Camera requires finite intrinsics and positive focal lengths/baseline")

    @property
    def K(self):
        return np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1.]])

    def bearings(self, pixels, R_wc):
        rays = np.column_stack((np.asarray(pixels), np.ones(len(pixels)))) @ np.linalg.inv(self.K).T
        rays /= np.linalg.norm(rays, axis=1, keepdims=True)
        return rays @ R_wc.T

    def triangulate(self, left, right, min_depth=0.5, max_depth=100., epipolar_px=1.5):
        """Rectified, common-principal-point stereo; reject reversed disparity."""
        left, right = np.asarray(left), np.asarray(right)
        disparity = left[:, 0] - right[:, 0]
        depth = self.fx * self.baseline / np.maximum(disparity, 1e-9)
        good = ((disparity > 0) & (np.abs(left[:, 1] - right[:, 1]) <= epipolar_px)
                & (depth >= min_depth) & (depth <= max_depth)
                & np.isfinite(left).all(axis=1) & np.isfinite(right).all(axis=1))
        xyz = np.column_stack(((left[:, 0] - self.cx) * depth / self.fx,
                               (left[:, 1] - self.cy) * depth / self.fy, depth))
        return xyz, good

    def project(self, world, R_wc, position):
        local = (world - position) @ R_wc
        z = local[:, 2]
        pixels = local[:, :2] / np.maximum(z[:, None], 1e-9)
        return pixels * [self.fx, self.fy] + [self.cx, self.cy], z


def _solve(points, rays, distances):
    # Equation (11), including the paper's printed 1/d weighting.
    projectors = (np.eye(3)[None] - rays[:, :, None] * rays[:, None, :]) / distances[:, None, None]
    a = projectors.sum(axis=0)
    if not np.isfinite(a).all() or np.linalg.cond(a) > 1e8:
        raise LocalizationError("Degenerate bearing geometry")
    return np.linalg.solve(a, np.einsum("nij,nj->i", projectors, points))


def estimate_position(points, pixels, camera, R_wc, previous, rng,
                      iterations=200, threshold_px=3., min_inliers=8):
    """Known-rotation two-point RANSAC followed by consensus refitting."""
    if len(points) < min_inliers:
        raise LocalizationError(f"Only {len(points)} landmarks; need {min_inliers}")
    rays = camera.bearings(pixels, R_wc)
    distances = np.maximum(np.linalg.norm(points - previous, axis=1), 1e-6)
    best = np.zeros(len(points), dtype=bool)
    best_error = np.inf

    def score(position):
        projected, depth = camera.project(points, R_wc, position)
        errors = np.linalg.norm(projected - pixels, axis=1)
        return (errors <= threshold_px) & (depth > 0), errors

    for _ in range(iterations):
        sample = rng.choice(len(points), 2, replace=False)
        try:
            candidate = _solve(points[sample], rays[sample], distances[sample])
        except (LocalizationError, np.linalg.LinAlgError):
            continue
        inliers, errors = score(candidate)
        count = inliers.sum()
        error = np.mean(errors[inliers]) if count else np.inf
        if count > best.sum() or (count == best.sum() and error < best_error):
            best, best_error = inliers, error
    if best.sum() < min_inliers:
        raise LocalizationError(f"No translation consensus: {best.sum()} inliers")
    for _ in range(3):
        position = _solve(points[best], rays[best], distances[best])
        updated, _ = score(position)
        if updated.sum() < min_inliers:
            raise LocalizationError("Translation consensus collapsed during refit")
        if np.array_equal(updated, best):
            break
        best = updated
    position = _solve(points[best], rays[best], distances[best])
    best, errors = score(position)
    if best.sum() < min_inliers:
        raise LocalizationError("Insufficient final translation inliers")
    return position, best, float(np.median(errors[best]))


def triangulate_temporal(centers, rays, min_parallax_deg=1.):
    """Equation (13), with observability and positive-ray-depth checks."""
    centers, rays = np.asarray(centers), np.asarray(rays)
    rays = rays / np.linalg.norm(rays, axis=1, keepdims=True)
    angles = np.arccos(np.clip(rays @ rays.T, -1, 1))
    if np.max(angles) < np.deg2rad(min_parallax_deg):
        raise LocalizationError("Insufficient triangulation parallax")
    point = _solve(centers, rays, np.ones(len(rays)))
    if np.any(np.sum((point - centers) * rays, axis=1) <= 0):
        raise LocalizationError("Triangulated landmark is behind a camera")
    return point
