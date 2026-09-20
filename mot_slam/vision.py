"""Stereo feature extraction, KLT and camera-compensated motion masks."""
import cv2
import numpy as np
from .geometry import LocalizationError


def gray(image):
    if image is None or image.dtype != np.uint8 or image.ndim not in (2, 3):
        raise ValueError("Expected a uint8 grayscale or BGR image")
    return image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def detector(name, count):
    if name == "sift":
        return cv2.SIFT_create(nfeatures=count)
    if name == "surf":
        try:
            return cv2.xfeatures2d.SURF_create(hessianThreshold=400)
        except (AttributeError, cv2.error) as exc:
            raise ValueError("SURF requires an OpenCV build with OPENCV_ENABLE_NONFREE; use sift with standard wheels") from exc
    raise ValueError("Feature detector must be sift or surf")


def stereo_features(left, right, camera, feature_detector, mask, max_depth):
    kl, dl = feature_detector.detectAndCompute(left, mask)
    kr, dr = feature_detector.detectAndCompute(right, None)
    if dl is None or dr is None or len(dr) < 2:
        return np.empty((0, 2), np.float32), np.empty((0, 3))
    matcher = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=64))
    pairs = matcher.knnMatch(dl, dr, k=2)
    matches = sorted([m[0] for m in pairs if len(m) == 2 and m[0].distance < .7*m[1].distance],
                     key=lambda m: m.distance)
    # Avoid several left landmarks sharing a right observation.
    used, unique = set(), []
    for m in matches:
        if m.trainIdx not in used:
            unique.append(m)
            used.add(m.trainIdx)
    if not unique:
        return np.empty((0, 2), np.float32), np.empty((0, 3))
    pl = np.float32([kl[m.queryIdx].pt for m in unique])
    pr = np.float32([kr[m.trainIdx].pt for m in unique])
    xyz, good = camera.triangulate(pl, pr, max_depth=max_depth)
    return pl[good], xyz[good]


def track(previous, current, pixels):
    if len(pixels) == 0:
        return np.empty((0, 2), np.float32), np.zeros(0, bool)
    pts = np.float32(pixels).reshape(-1, 1, 2)
    options = dict(winSize=(21, 21), maxLevel=3,
                   criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, .01))
    nxt, st, _ = cv2.calcOpticalFlowPyrLK(previous, current, pts, None, **options)
    if nxt is None:
        return np.asarray(pixels).copy(), np.zeros(len(pixels), bool)
    back, sb, _ = cv2.calcOpticalFlowPyrLK(current, previous, nxt, None, **options)
    if back is None:
        return nxt.reshape(-1, 2), np.zeros(len(pixels), bool)
    nxt = nxt.reshape(-1, 2)
    h, w = current.shape
    valid = ((st.ravel() == 1) & (sb.ravel() == 1)
             & (np.linalg.norm(back.reshape(-1, 2) - pixels, axis=1) < 1.)
             & np.isfinite(nxt).all(axis=1)
             & (nxt[:, 0] >= 2) & (nxt[:, 0] < w-2)
             & (nxt[:, 1] >= 2) & (nxt[:, 1] < h-2))
    return nxt, valid


def motion_mask(previous, current, before, after, threshold=30):
    """Warp previous into current coordinates; output 255 = excluded.

    This is the inverse registration direction to the paper, so masking current
    features needs no extra coordinate conversion. Invalid warp borders are
    excluded. A failed homography is reported, never silently treated as static.
    """
    if len(before) < 8:
        raise LocalizationError("Insufficient correspondences for motion compensation")
    H, inliers = cv2.findHomography(np.float32(before), np.float32(after), cv2.RANSAC, 3.)
    if H is None or inliers is None or inliers.sum() < 8 or not np.isfinite(H).all():
        raise LocalizationError("Motion-compensation homography failed")
    h, w = current.shape
    warped = cv2.warpPerspective(previous, H, (w, h))
    support = cv2.warpPerspective(np.full_like(previous, 255), H, (w, h), flags=cv2.INTER_NEAREST)
    support = cv2.erode(support, np.ones((5, 5), np.uint8), borderType=cv2.BORDER_CONSTANT, borderValue=0)
    difference = cv2.absdiff(cv2.GaussianBlur(warped, (5, 5), 0), cv2.GaussianBlur(current, (5, 5), 0))
    mask = np.uint8(difference > threshold) * 255
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.dilate(mask, kernel, iterations=2)
    mask[support == 0] = 255
    return mask


def static_at(mask, pixels):
    xy = np.rint(pixels).astype(int)
    h, w = mask.shape
    inside = (xy[:, 0] >= 0) & (xy[:, 0] < w) & (xy[:, 1] >= 0) & (xy[:, 1] < h)
    result = np.zeros(len(xy), bool)
    result[inside] = mask[xy[inside, 1], xy[inside, 0]] == 0
    return result
