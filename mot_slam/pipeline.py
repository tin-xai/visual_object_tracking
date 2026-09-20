"""Stereo initialization, IMU-aided localization and local map maintenance."""
from dataclasses import dataclass
import cv2
import numpy as np
from .geometry import (LocalizationError, estimate_position, rotation_matrix,
                       triangulate_temporal)
from .vision import gray, detector, stereo_features, track, motion_mask, static_at


@dataclass
class Config:
    feature: str = "sift"
    max_features: int = 1800
    stereo_interval: int = 25
    min_landmarks: int = 100
    min_inliers: int = 8
    max_depth: float = 100.
    motion_threshold: int = 30
    remove_moving: bool = True
    ransac_iterations: int = 200
    reprojection_px: float = 3.
    gamma_tolerance: float = .15
    min_parallax_deg: float = 1.
    seed: int = 0

    def __post_init__(self):
        if (self.stereo_interval < 1 or self.min_inliers < 4
                or self.min_landmarks < self.min_inliers or self.max_features < self.min_landmarks
                or self.ransac_iterations < 1 or self.max_depth <= .5
                or not 0 < self.motion_threshold < 256 or self.reprojection_px <= 0
                or self.gamma_tolerance <= 0 or self.min_parallax_deg <= 0):
            raise ValueError("Invalid SLAM configuration")


@dataclass
class Result:
    timestamp: float
    R_wc: np.ndarray
    position: np.ndarray
    landmarks: int
    inliers: int
    reprojection_error: float
    refreshed: bool
    gamma: float | None
    motion_mask: np.ndarray

    @property
    def pose(self):
        pose = np.eye(4)
        pose[:3, :3], pose[:3, 3] = self.R_wc, self.position
        return pose


class StereoIMUSLAM:
    """Input orientations rotate rectified left-camera coordinates into world.

    First position is the world origin. Orientation is supplied, not estimated
    from raw accelerometer/gyroscope samples. Instances stop on localization loss;
    create a new instance for a new, disconnected trajectory segment.
    """
    def __init__(self, camera, config=None):
        self.camera = camera
        self.config = config or Config()
        self.detector = detector(self.config.feature, self.config.max_features)
        self.rng = np.random.default_rng(self.config.seed)
        self.previous = None
        self.position = np.zeros(3)
        self.R_wc = np.eye(3)
        self.pixels = np.empty((0, 2), np.float32)
        self.points = np.empty((0, 3))
        self.stereo_points = np.empty((0, 3))
        self.confirmed = np.zeros(0, bool)
        self.archive = []
        self.frame = 0
        self.last_refresh = 0
        self.timestamp = -np.inf
        self.failed = False

    def _stereo(self, left, right, R_wc, position, mask):
        pixels, local = stereo_features(left, right, self.camera, self.detector,
                                       cv2.bitwise_not(mask), self.config.max_depth)
        return pixels, local @ R_wc.T + position

    def process(self, left, right, R_wc, timestamp):
        if self.failed:
            raise LocalizationError("This trajectory has lost localization; start a new instance")
        left, right = gray(left), gray(right)
        R_wc = rotation_matrix(R_wc)
        if left.shape != right.shape or (self.previous is not None and left.shape != self.previous.shape):
            raise ValueError("All rectified images must have identical dimensions")
        if not np.isfinite(timestamp) or timestamp <= self.timestamp:
            raise ValueError("Timestamps must be finite and strictly increasing")
        try:
            return self._process(left, right, R_wc, float(timestamp))
        except LocalizationError:
            self.failed = True
            raise

    def _process(self, left, right, R_wc, timestamp):
        cfg = self.config
        mask = np.zeros_like(left)
        gamma = None
        error, inliers, refresh = 0., 0, False
        position = self.position.copy()
        if self.previous is None:
            pixels, points = self._stereo(left, right, R_wc, position, mask)
            if len(points) < cfg.min_inliers:
                raise LocalizationError("Too few stereo landmarks to initialize")
            anchors = points.copy()
            confirmed = np.zeros(len(points), bool)
            refresh = True
        else:
            current, valid = track(self.previous, left, self.pixels)
            before, pixels = self.pixels[valid], current[valid]
            points, anchors = self.points[valid], self.stereo_points[valid]
            if cfg.remove_moving:
                mask = motion_mask(self.previous, left, before, pixels, cfg.motion_threshold)
                static = static_at(mask, pixels)
                before, pixels, points, anchors = before[static], pixels[static], points[static], anchors[static]
            position, keep, error = estimate_position(
                points, pixels, self.camera, R_wc, self.position, self.rng,
                cfg.ransac_iterations, cfg.reprojection_px, cfg.min_inliers)
            before, pixels, points, anchors = before[keep], pixels[keep], points[keep], anchors[keep]
            inliers = len(points)
            confirmed = np.ones(len(points), bool)
            # Monocular map maintenance (Eq. 13). Stereo anchors stay immutable
            # until refresh, allowing the range-ratio check in Eq. 14.
            old_rays = self.camera.bearings(before, self.R_wc)
            new_rays = self.camera.bearings(pixels, R_wc)
            ratios = []
            for i in range(len(points)):
                try:
                    p = triangulate_temporal([self.position, position], [old_rays[i], new_rays[i]],
                                             cfg.min_parallax_deg)
                except LocalizationError:
                    continue
                if not self._valid_landmark(p, [before[i], pixels[i]], [self.R_wc, R_wc],
                                            [self.position, position]):
                    continue
                if np.isfinite(anchors[i]).all():
                    ratios.append(np.linalg.norm(p-position) / max(np.linalg.norm(anchors[i]-position), 1e-9))
                points[i] = p
            if len(ratios) >= cfg.min_inliers:
                gamma = float(np.mean(ratios))
            refresh = (self.frame - self.last_refresh >= cfg.stereo_interval
                       or (gamma is not None and abs(gamma - 1.) > cfg.gamma_tolerance))
            if not refresh and len(points) < cfg.min_landmarks:
                # Discover fresh features in the previous left frame; triangulate
                # them only after estimating this frame's pose from known points.
                extra_mask = np.full_like(left, 255)
                for uv in before:
                    cv2.circle(extra_mask, tuple(np.rint(uv).astype(int)), 10, 0, -1)
                corners = cv2.goodFeaturesToTrack(self.previous, cfg.max_features-len(points), .01, 10,
                                                  mask=extra_mask)
                if corners is not None:
                    old = corners.reshape(-1, 2)
                    new, ok = track(self.previous, left, old)
                    ok &= static_at(mask, new)
                    old, new = old[ok], new[ok]
                    ra = self.camera.bearings(old, self.R_wc)
                    rb = self.camera.bearings(new, R_wc)
                    added_uv, added_xyz = [], []
                    for a, b, u, v in zip(ra, rb, old, new):
                        try:
                            p = triangulate_temporal([self.position, position], [a, b], cfg.min_parallax_deg)
                        except LocalizationError:
                            continue
                        if self._valid_landmark(p, [u, v], [self.R_wc, R_wc], [self.position, position]):
                            added_uv.append(v)
                            added_xyz.append(p)
                    if added_xyz:
                        pixels = np.vstack((pixels, added_uv)).astype(np.float32)
                        points = np.vstack((points, added_xyz))
                        anchors = np.vstack((anchors, np.full((len(added_xyz), 3), np.nan)))
                        confirmed = np.r_[confirmed, np.ones(len(added_xyz), bool)]
                # Practical fallback when temporal parallax is insufficient.
                refresh = len(points) < cfg.min_landmarks
            if refresh:
                new_uv, new_xyz = self._stereo(left, right, R_wc, position, mask)
                if len(new_xyz) >= cfg.min_inliers:
                    self.archive.append(points[confirmed].copy())
                    pixels, points = new_uv, new_xyz
                    anchors = points.copy()
                    confirmed = np.zeros(len(points), bool)
                elif gamma is not None and abs(gamma-1.) > cfg.gamma_tolerance:
                    raise LocalizationError("Local map failed consistency check and stereo recovery failed")
                else:
                    refresh = False  # Keep a valid old map if stereo lacks texture.
        self.previous = left.copy()
        self.R_wc, self.position = R_wc.copy(), position
        self.pixels, self.points, self.stereo_points = pixels, points, anchors
        self.confirmed = confirmed
        self.timestamp = timestamp
        if refresh:
            self.last_refresh = self.frame
        self.frame += 1
        return Result(timestamp, R_wc.copy(), position.copy(), len(points), inliers, error, refresh, gamma, mask)

    def _valid_landmark(self, point, pixels, rotations, centers):
        for pixel, R, center in zip(pixels, rotations, centers):
            predicted, depth = self.camera.project(point[None], R, center)
            if not .5 < depth[0] < self.config.max_depth:
                return False
            if np.linalg.norm(predicted[0]-pixel) > self.config.reprojection_px:
                return False
        return True

    def map_points(self, voxel_size=.1):
        """Sparse historical cloud, without global optimization or loop closure.

        Newly initialized stereo points must survive a temporal pose consensus
        before export. Previously archived points cannot be retroactively removed.
        """
        points = np.concatenate(self.archive + [self.points[self.confirmed]], axis=0)
        if len(points) and voxel_size > 0:
            _, indices = np.unique(np.floor(points/voxel_size).astype(np.int64), axis=0, return_index=True)
            points = points[indices]
        return points
