import cv2
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from mot_slam.io import Sequence, write_results
from mot_slam.pipeline import Result


def make_kitti(tmp_path):
    drive = tmp_path/"drive_sync"
    for side in ["image_00", "image_01"]:
        (drive/side/"data").mkdir(parents=True)
        (drive/side/"timestamps.txt").write_text("2011-09-26 13:02:45.123456789\n2011-09-26 13:02:45.223456789\n")
        for i in range(2):
            cv2.imwrite(str(drive/side/"data"/f"{i:010d}.png"), np.zeros((20, 30), np.uint8))
    (drive/"oxts"/"data").mkdir(parents=True)
    for i in range(2):
        # Latitude/longitude must have no effect on estimated translation.
        (drive/"oxts"/"data"/f"{i:010d}.txt").write_text("49 8 110 0.1 0.2 0.3")
    (tmp_path/"calib_cam_to_cam.txt").write_text("R_rect_00: 1 0 0 0 1 0 0 0 1\nP_rect_00: 400 0 320 0 0 400 240 0 0 0 1 0\nP_rect_01: 400 0 320 -216 0 400 240 0 0 0 1 0\n")
    Rci = Rotation.from_euler("z", .5).as_matrix()
    (tmp_path/"calib_velo_to_cam.txt").write_text("R: "+" ".join(map(str, Rci.ravel())))
    (tmp_path/"calib_imu_to_velo.txt").write_text("R: 1 0 0 0 1 0 0 0 1")
    return drive, Rci


def test_kitti_orientation_chain_and_timestamps(tmp_path):
    drive, Rci = make_kitti(tmp_path)
    sequence = Sequence(drive)
    frames = list(sequence)
    assert sequence.camera.baseline == pytest.approx(.54)
    assert frames[1].timestamp == pytest.approx(.1)
    expected = Rotation.from_euler("xyz", [.1, .2, .3]).as_matrix() @ Rci.T
    np.testing.assert_allclose(frames[0].R_wc, expected)
    (drive/"image_01"/"data"/"0000000001.png").unlink()
    with pytest.raises(ValueError, match="Missing right"):
        Sequence(drive)


def test_exports_use_initial_camera_coordinates(tmp_path):
    R = Rotation.from_euler("z", np.pi/2).as_matrix()
    results = [Result(i, R, np.array([0., float(i), 0.]), 20, 20, .1, False, None, None) for i in range(2)]
    write_results(tmp_path, results, np.array([[1., 2., 3.]]), ["a", "b"])
    poses = np.loadtxt(tmp_path/"poses.txt").reshape(-1, 3, 4)
    np.testing.assert_allclose(poses[0], np.eye(4)[:3], atol=1e-8)
    np.testing.assert_allclose(poses[1, :, 3], [1., 0., 0.], atol=1e-8)
    assert "element vertex 1" in (tmp_path/"map.ply").read_text()
