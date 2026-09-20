import cv2
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from mot_slam import Camera, LocalizationError, StereoIMUSLAM
from mot_slam.geometry import estimate_position, triangulate_temporal
from mot_slam.vision import motion_mask, static_at


@pytest.fixture
def camera():
    return Camera(400., 410., 320., 240., .54)


def test_metric_stereo_and_bad_disparities(camera):
    xyz = np.array([[1., .2, 8.], [-2., -.3, 15.], [0., 0., 4.]])
    left, _ = camera.project(xyz, np.eye(3), np.zeros(3))
    right, _ = camera.project(xyz, np.eye(3), np.array([.54, 0, 0]))
    recovered, good = camera.triangulate(left, right)
    np.testing.assert_allclose(recovered, xyz, atol=1e-12)
    assert good.all()
    right[0, 0] = left[0, 0]+1
    right[1, 1] += 8
    _, good = camera.triangulate(left, right)
    assert good.tolist() == [False, False, True]


def test_position_with_rotation_and_moving_outliers(camera):
    rng = np.random.default_rng(2)
    points = rng.uniform([-4., -2., 5.], [4., 2., 20.], (120, 3))
    R = Rotation.from_euler("xyz", [.03, -.07, .04]).as_matrix()
    truth = np.array([.4, -.1, .2])
    pixels, _ = camera.project(points, R, truth)
    pixels += rng.normal(0, .1, pixels.shape)
    pixels[:35] += [50, -25]
    position, keep, error = estimate_position(points, pixels, camera, R, np.zeros(3), rng)
    np.testing.assert_allclose(position, truth, atol=.01)
    assert keep.sum() >= 80 and not keep[:35].any() and error < .5


def test_degenerate_translation_is_rejected(camera):
    points = np.tile([0., 0., 10.], (20, 1))
    pixels = np.tile([camera.cx, camera.cy], (20, 1))
    with pytest.raises(LocalizationError):
        estimate_position(points, pixels, camera, np.eye(3), np.zeros(3), np.random.default_rng(0))


def test_temporal_triangulation_and_zero_baseline():
    centers = np.array([[0., 0., 0.], [1., 0., 0.]])
    point = np.array([.2, .5, 8.])
    rays = point-centers
    np.testing.assert_allclose(triangulate_temporal(centers, rays), point, atol=1e-10)
    with pytest.raises(LocalizationError):
        triangulate_temporal(np.zeros((2, 3)), np.tile([0., 0., 1.], (2, 1)))


def test_compensated_mask_rejects_object_and_warp_border():
    rng = np.random.default_rng(0)
    previous = rng.integers(0, 256, (120, 180), dtype=np.uint8)
    H = np.array([[1., 0, 7], [0, 1., 0], [0, 0, 1.]])
    current = cv2.warpPerspective(previous, H, (180, 120))
    current[45:75, 85:115] = 255
    before = np.array([[x, y] for x in [20, 50, 120, 150] for y in [15, 35, 90, 105]], np.float32)
    mask = motion_mask(previous, current, before, before+[7, 0])
    assert mask[50:70, 90:110].mean() > 240
    assert static_at(mask, np.array([[40., 35.], [100., 60.], [1., 20.]])).tolist() == [True, False, False]




def test_loss_does_not_emit_fabricated_pose():
    slam = StereoIMUSLAM(Camera(400., 400., 320., 240., .54))
    with pytest.raises(LocalizationError):
        slam.process(np.zeros((100, 100), np.uint8), np.zeros((100, 100), np.uint8), np.eye(3), 0.)
    assert slam.frame == 0 and slam.failed
    with pytest.raises(LocalizationError):
        slam.process(np.zeros((100, 100), np.uint8), np.zeros((100, 100), np.uint8), np.eye(3), 1.)
